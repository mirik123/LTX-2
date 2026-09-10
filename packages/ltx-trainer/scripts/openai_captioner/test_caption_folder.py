import json
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import scripts.openai_captioner.caption_folder as caption_module
from scripts.openai_captioner.caption_folder import OpenAIFolderCaptioner, caption_folder, find_videos
from typer.testing import CliRunner


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text='{"combined_caption_english": "A combined caption."}')


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        message = SimpleNamespace(content="Dialogue, music, and quiet room ambience.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_caption_video_combines_exact_fps_frames_and_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    fake_client = _FakeClient()
    captioner = OpenAIFolderCaptioner(client=fake_client, fps=30)

    def fake_extract_frames(video_path: Path, frames_dir: Path) -> list[Path]:
        assert video_path == video.resolve()
        frames_dir.mkdir(parents=True)
        frames = []
        for index in range(300):
            frame = frames_dir / f"frame_{index + 1:06d}.jpg"
            frame.write_bytes(f"frame-{index}".encode())
            frames.append(frame)
        return frames

    def fake_run_ffmpeg(args: list[str]) -> None:
        with wave.open(str(args[-1]), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 160000)

    monkeypatch.setattr(captioner, "_extract_frames", fake_extract_frames)
    monkeypatch.setattr("scripts.openai_captioner.caption_folder._run_ffmpeg", fake_run_ffmpeg)

    assert captioner.caption_video(video) == "A combined caption."
    assert len(fake_client.chat.completions.calls) == 1
    request = fake_client.responses.calls[0]
    assert request["instructions"] == caption_module.DEFAULT_INSTRUCTION
    assert request["text"] == caption_module.CAPTION_RESPONSE_TEXT
    content = request["input"][0]["content"]
    image_parts = [part for part in content if part["type"] == "input_image"]
    timestamp_parts = [part for part in content if part["type"] == "input_text" and part["text"].startswith("Frame at")]
    assert len(image_parts) == 300
    assert timestamp_parts[0]["text"] == "Frame at 0.000 seconds:"
    assert timestamp_parts[-1]["text"] == "Frame at 9.967 seconds:"
    assert "Dialogue, music" in content[0]["text"]


def test_caption_video_without_audio_still_calls_vision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "silent.mov"
    video.write_bytes(b"video")
    fake_client = _FakeClient()
    captioner = OpenAIFolderCaptioner(client=fake_client)

    def fake_extract_frames(video_path: Path, frames_dir: Path) -> list[Path]:
        del video_path
        frames_dir.mkdir(parents=True)
        frame = frames_dir / "frame_000001.jpg"
        frame.write_bytes(b"frame")
        return [frame]

    def fail_audio_extraction(args: list[str]) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd=args)

    monkeypatch.setattr(captioner, "_extract_frames", fake_extract_frames)
    monkeypatch.setattr("scripts.openai_captioner.caption_folder._run_ffmpeg", fail_audio_extraction)

    assert captioner.caption_video(video) == "A combined caption."
    assert not fake_client.chat.completions.calls
    content = fake_client.responses.calls[0]["input"][0]["content"]
    assert "Soundtrack analysis unavailable" in content[0]["text"]


def test_caption_folder_is_recursive_fresh_and_deterministic(tmp_path: Path) -> None:
    input_dir = tmp_path / "videos"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)
    (input_dir / "b.mp4").write_bytes(b"b")
    (nested / "a.MOV").write_bytes(b"a")
    (nested / "ignored.png").write_bytes(b"image")
    output = input_dir / "dataset.json"
    output.write_text('[{"caption": "stale", "media_path": "old.mp4"}]', encoding="utf-8")

    class FakeCaptioner:
        @staticmethod
        def caption_video(video_path: str | Path) -> str:
            return f"caption for {Path(video_path).name}"

    result = caption_folder(input_dir, output, FakeCaptioner(), num_workers=2, resume=False)  # type: ignore[arg-type]
    records = json.loads(output.read_text(encoding="utf-8"))

    assert not result.failures
    assert [record["media_path"] for record in records] == ["b.mp4", "nested/a.MOV"]
    assert all(record["media_path"] != "old.mp4" for record in records)
    assert find_videos(input_dir) == [input_dir / "b.mp4", nested / "a.MOV"]


def test_caption_folder_saves_successes_and_reports_failures(tmp_path: Path) -> None:
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    (input_dir / "bad.mp4").write_bytes(b"bad")
    (input_dir / "good.mp4").write_bytes(b"good")
    output = input_dir / "dataset.json"

    class FakeCaptioner:
        @staticmethod
        def caption_video(video_path: str | Path) -> str:
            if Path(video_path).name == "bad.mp4":
                raise RuntimeError("API failed")
            return "good caption"

    result = caption_folder(input_dir, output, FakeCaptioner(), num_workers=2)  # type: ignore[arg-type]
    records = json.loads(output.read_text(encoding="utf-8"))

    assert records == [{"caption": "good caption", "media_path": "good.mp4"}]
    assert result.failures == {"bad.mp4": "API failed"}


def test_api_key_uses_explicit_value_then_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_keys: list[str] = []

    def fake_openai(*, api_key: str, timeout: float) -> _FakeClient:
        assert timeout == 600.0
        captured_keys.append(api_key)
        return _FakeClient()

    monkeypatch.setattr("scripts.openai_captioner.caption_folder.OpenAI", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")

    OpenAIFolderCaptioner(api_key="explicit-key")
    OpenAIFolderCaptioner()

    assert captured_keys == ["explicit-key", "environment-key"]


@pytest.mark.parametrize(
    ("image_limit", "request_limit", "expected_error"),
    [
        (1, 512 * 1024 * 1024, "exceeding OpenAI's 1-image request limit"),
        (1_500, 1, "exceeding OpenAI's 512 MiB request limit"),
    ],
)
def test_caption_video_reports_openai_request_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_limit: int,
    request_limit: int,
    expected_error: str,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    captioner = OpenAIFolderCaptioner(client=_FakeClient())

    def fake_extract_frames(video_path: Path, frames_dir: Path) -> list[Path]:
        del video_path
        frames_dir.mkdir(parents=True)
        frames = []
        for index in range(2):
            frame = frames_dir / f"frame_{index + 1:06d}.jpg"
            frame.write_bytes(b"frame")
            frames.append(frame)
        return frames

    monkeypatch.setattr(captioner, "_extract_frames", fake_extract_frames)
    monkeypatch.setattr(caption_module, "MAX_IMAGES_PER_REQUEST", image_limit)
    monkeypatch.setattr(caption_module, "MAX_REQUEST_BYTES", request_limit)

    with pytest.raises(ValueError, match=expected_error):
        captioner.caption_video(video)


def test_cli_forwards_openai_options_and_returns_failure_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCaptioner:
        def __init__(self, **kwargs: object) -> None:
            captured["captioner_options"] = kwargs

    def fake_caption_folder(
        input_dir: Path,
        output_path: Path,
        captioner: OpenAIFolderCaptioner,
        num_workers: int,
        resume: bool,
    ) -> caption_module.BatchResult:
        del captioner
        assert resume is True
        captured["input_dir"] = input_dir
        captured["output_path"] = output_path
        captured["num_workers"] = num_workers
        return caption_module.BatchResult(captions={"good.mp4": "caption"}, failures={"bad.mp4": "failed"})

    def fake_find_videos(input_dir: Path) -> list[Path]:
        return [input_dir / "good.mp4", input_dir / "bad.mp4"]

    monkeypatch.setattr(caption_module, "OpenAIFolderCaptioner", FakeCaptioner)
    monkeypatch.setattr(caption_module, "find_videos", fake_find_videos)
    monkeypatch.setattr(caption_module, "caption_folder", fake_caption_folder)

    result = CliRunner().invoke(
        caption_module.app,
        [
            str(tmp_path),
            "--api-key",
            "explicit-key",
            "--model",
            "vision-model",
            "--audio-model",
            "audio-model",
            "--fps",
            "30",
            "--num-workers",
            "7",
            "--max-tokens",
            "1234",
        ],
    )

    assert result.exit_code == 1
    assert captured["num_workers"] == 7
    assert captured["output_path"] == (tmp_path / "dataset.json").resolve()
    assert captured["captioner_options"] == {
        "api_key": "explicit-key",
        "model": "vision-model",
        "audio_model": "audio-model",
        "fps": 30,
        "max_tokens": 1234,
        "instruction": None,
        "audio_max_tokens": 384,
        "include_audio": True,
    }


@pytest.fixture(autouse=True)
def mock_video_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caption_module, "_video_duration", lambda _path: 10.0)


@pytest.mark.parametrize(
    "bad_audio",
    [
        "At 0:15 a bell rings.",
        "At 1:00:00 a bell rings.",
        "At 15 seconds a bell rings.",
        "15\u20132s: music.",
        "-2s: music.",
    ],
)
@pytest.mark.parametrize("retry_valid", [True, False])
def test_invalid_audio_retried_before_synthesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_audio: str,
    retry_valid: bool,
) -> None:
    video = tmp_path / "clip.mp4"
    video.touch()
    client = _FakeClient()
    calls: list[dict[str, Any]] = []

    def completion(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        text = "0\u20132s: music." if retry_valid and len(calls) == 2 else bad_audio
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def ffmpeg(args: list[str]) -> None:
        output = Path(args[-1])
        if output.suffix == ".wav":
            with wave.open(str(output), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 48000)
        else:
            (output.parent / "frame_000001.jpg").write_bytes(b"frame")

    monkeypatch.setattr(client.chat.completions, "create", completion)
    monkeypatch.setattr(caption_module, "_run_ffmpeg", ffmpeg)
    assert OpenAIFolderCaptioner(client=client).caption_video(video) == "A combined caption."
    assert len(calls) == 2
    assert "3.000 seconds" in calls[0]["messages"][0]["content"][0]["text"]
    assert calls[0]["max_completion_tokens"] == 384
    content = client.responses.calls[0]["input"][0]["content"]
    assert bad_audio not in content[0]["text"]
    assert ("0\u20132s: music." if retry_valid else "unavailable") in content[0]["text"]
    assert "Clip duration: 10.000" in content[1]["text"]
    assert client.responses.calls[0]["max_output_tokens"] == 768


def test_resume_preserves_existing_and_only_captions_missing(tmp_path: Path) -> None:
    for name in ("old.mp4", "new.mp4"):
        (tmp_path / name).touch()
    output = tmp_path / "dataset.json"
    output.write_text('[{"media_path":"old.mp4","caption":"existing"}]')
    calls: list[str] = []

    class Captioner:
        @staticmethod
        def caption_video(path: Path) -> str:
            calls.append(path.name)
            return "new caption"

    result = caption_folder(tmp_path, output, Captioner())
    assert calls == ["new.mp4"]
    assert result.captions == {"old.mp4": "existing", "new.mp4": "new caption"}


@pytest.mark.parametrize("raw", ["{}", "broken", '[{"media_path": "a.mp4"}]'])
def test_resume_rejects_malformed_dataset_without_overwriting(tmp_path: Path, raw: str) -> None:
    output = tmp_path / "dataset.json"
    output.write_text(raw)
    with pytest.raises(ValueError, match=r"Existing dataset|Invalid existing|Expecting value"):
        caption_folder(tmp_path, output, _FakeClient())
    assert output.read_text() == raw


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(output_text='{"combined_caption_english": "partial"}', status="incomplete"),
        SimpleNamespace(output_text='{"combined_caption_english": "unfinished'),
        SimpleNamespace(output_text='{"combined_caption_english": ""}'),
        SimpleNamespace(output_text=json.dumps({"combined_caption_english": "word " * 251})),
    ],
)
def test_no_audio_and_invalid_final_caption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: SimpleNamespace,
) -> None:
    video = tmp_path / "clip.mp4"
    video.touch()
    client = _FakeClient()

    def ffmpeg(args: list[str]) -> None:
        output = Path(args[-1])
        assert output.suffix == ".jpg"
        (output.parent / "frame_000001.jpg").write_bytes(b"frame")

    monkeypatch.setattr(caption_module, "_run_ffmpeg", ffmpeg)
    monkeypatch.setattr(client.responses, "create", lambda **_kwargs: response)
    with pytest.raises(ValueError, match="Caption"):
        OpenAIFolderCaptioner(client=client, include_audio=False).caption_video(video)
    assert not client.chat.completions.calls
