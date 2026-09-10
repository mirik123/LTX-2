# OpenAI Folder Captioner

This standalone tool recursively captions every supported video in a folder using OpenAI APIs. It does not depend on
or modify the trainer's existing captioning script.

For each video, the tool samples frames at the exact requested rate, asks an OpenAI audio model to describe the duration-bounded
soundtrack, and asks an OpenAI vision model to combine both sources into one chronological caption. Supported video
extensions are `.mp4`, `.avi`, `.mov`, `.mkv`, and `.webm`.

The default synthesis prompt is designed for concise, evidence-grounded training captions. It accounts for temporal
gaps between sampled frames, preserves dialogue only when the audio analysis provides it, and avoids inferring motion
or sound that is not supported by the inputs. The final response is constrained with Structured Outputs to contain
exactly one `combined_caption_english` string.

## Authentication

Set an API key in the environment (recommended):

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Alternatively, pass `--api-key`. Environment variables are safer because command-line arguments may be saved in shell
history or visible to other processes.

## Caption a folder

From `packages/ltx-trainer`:

```bash
uv run python scripts/openai_captioner/caption_folder.py /path/to/videos
```

This recursively scans the folder and updates `/path/to/videos/dataset.json` containing records such as:

```json
[
  {
    "caption": "A detailed combined visual and audio caption.",
    "media_path": "clips/example.mp4"
  }
]
```

Runs resume by default, preserving existing records and skipping videos with nonempty captions.
Use `--override` to start a fresh dataset and regenerate all captions. Invalid existing JSON is rejected without overwriting it. Successful captions are
checkpointed atomically while the batch is running. If individual videos fail, the remaining videos continue, the
successful records are saved, and the command exits with a nonzero status after printing the failures.

## Options

```bash
uv run python scripts/openai_captioner/caption_folder.py /path/to/videos \
  --output /path/to/dataset.json \
  --fps 2 \
  --num-workers 4 \
  --model gpt-5.6-sol \
  --audio-model gpt-audio-mini \
  --caption-max-tokens 768 \
  --audio-max-tokens 384
```

- `--fps` is honored exactly and has no hidden cap. A 10-second video at `--fps 30` sends approximately 300 frames.
- OpenAI accepts no more than 1,500 images or 512 MiB in one vision request. The tool reports a clear error asking you
  to lower `--fps` if the extracted frames exceed either limit.
- `--num-workers` controls how many videos make API requests concurrently. Reduce it if your account reaches rate
  limits.
- `--instruction` replaces the default final-caption prompt; the structured JSON output contract remains enforced.
- Videos without a decodable audio stream are captioned from their visual frames alone.

Run `uv run python scripts/openai_captioner/caption_folder.py --help` for the complete CLI reference.

Audio stays enabled by default for AV training; use `--no-audio` for visual-only captions.
The audio pass receives the extracted WAV duration and a 120-word limit. Invalid timestamps,
empty, overlong, or truncated audio analyses are retried once, then discarded if still invalid.
Unavailable audio is not treated as silence. The final model receives the video duration and
produces a chronological caption without timestamps, targeting 120–220 words (250 maximum).
Watermarks and overlays are ignored. Final captions that are empty, overlong, or reported incomplete
fail that video so a resumed run can retry it. `--max-tokens` remains an alias for `--caption-max-tokens`.
For AV preprocessing, keep audio enabled (do not pass `--skip-audio`).
