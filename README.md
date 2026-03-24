# echo-vault

Local meeting recorder for Linux. Records system audio + microphone via
PipeWire/PulseAudio, transcribes with Whisper, identifies speakers, saves
Markdown to your Obsidian vault. Everything runs on your machine — no cloud,
no bots, no API calls for transcription.

Works with any video call platform: Google Meet, Zoom, Teams, Telegram, Discord, Slack huddles.

<!-- TODO: Add demo GIF here -->
<!-- ![echo-vault demo](docs/demo.gif) -->

## Features

- **Platform-agnostic** — captures OS-level audio, works with any app
- **Local transcription** — faster-whisper, runs on CPU or CUDA
- **Speaker diarization** — pyannote identifies who said what
- **Stereo channel separation** — your mic (left) vs. others (right) for accurate "You" attribution
- **Obsidian-native output** — Markdown with YAML frontmatter, wikilinks to audio files
- **LLM summarization** — optional summaries via Anthropic, OpenAI, Groq, Gemini, DeepSeek, OpenRouter, Qwen
- **CLI-first** — `meetrec start`, Ctrl+C to stop, done

## Requirements

- Linux (PipeWire or PulseAudio)
- Python 3.13+
- ffmpeg
- parecord (usually comes with `pulseaudio-utils` or `pipewire-pulse`)
- NVIDIA GPU (optional, for faster transcription and diarization)

## Install

### From PyPI (recommended)

```bash
# Install as isolated CLI tool
uv tool install echo-vault

# Or with pip
pip install echo-vault
```

### From source

```bash
git clone https://github.com/yastcher/echo-vault
cd echo-vault
uv sync
```

### Arch Linux (AUR)

```bash
yay -S echo-vault
```

### Nix

```bash
nix run github:yastcher/echo-vault
```

## Quick start

```bash
# 1. Set your Obsidian vault path
export MEETREC_VAULT_PATH=~/Documents/Obsidian/Vault

# 2. Start recording (blocks, Ctrl+C to stop)
meetrec start

# 3. That's it — check your vault for the new meeting note
```

## Configuration

All settings via environment variables (prefix `MEETREC_`) or `.env` file.

| Variable | Default | Description |
|---|---|---|
| `MEETREC_VAULT_PATH` | *(required)* | Path to Obsidian vault |
| `MEETREC_WHISPER_MODEL` | `large-v3-turbo` | Whisper model size |
| `MEETREC_LANGUAGE` | `en` | Transcription language |
| `MEETREC_DEVICE` | `cuda` | `cuda` or `cpu` |
| `MEETREC_COMPUTE_TYPE` | `float16` | `float16`, `int8`, or `float32` |
| `MEETREC_DIARIZE` | `true` | Enable speaker diarization |
| `MEETREC_HF_TOKEN` | *(empty)* | HuggingFace token for pyannote models |
| `MEETREC_MONITOR_SOURCE` | `auto` | PulseAudio monitor source name |
| `MEETREC_MIC_SOURCE` | `auto` | PulseAudio mic source name |
| `MEETREC_SAMPLE_RATE` | `48000` | Recording sample rate |
| `MEETREC_SUMMARIZE` | `true` | Enable LLM summarization |
| `MEETREC_LLM_PROVIDER` | `anthropic` | LLM provider for summaries |
| `MEETREC_LLM_API_KEY` | *(empty)* | API key (or use provider-specific env var) |
| `MEETREC_PAUSE_THRESHOLD` | `1.0` | Seconds — split segments on silence gaps >= this |

### Speaker diarization setup

Speaker diarization requires a HuggingFace token with access to pyannote models:

1. Create account at [huggingface.co](https://huggingface.co)
2. Accept license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept license at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Create token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. Set `MEETREC_HF_TOKEN=hf_your_token_here`

Without a token, meetrec still works — it just skips diarization.

## Commands

```bash
meetrec start [NAME]          # Start recording (Ctrl+C to stop and transcribe)
meetrec stop                  # Stop recording from another terminal
meetrec process <FILE>        # Transcribe an existing audio file
meetrec summarize <FILE>      # Add LLM summary to existing transcript
meetrec status                # Show recording status
```

### Options

```bash
meetrec start --no-diarize      # Skip speaker diarization
meetrec start --no-summarize    # Skip LLM summarization
meetrec process file.mp3 --name "Weekly standup"
```

## Output format

```
vault/
├── meetings/
│   └── 2026-03-23_14-30-00.md
└── attachments/audio/
    └── 2026-03-23_14-30-00.wav
```

Markdown output:

```markdown
---
date: 2026-03-23
time: "14:30"
duration: "01:23:45"
language: en
tags:
  - meeting
  - transcript
---

## Summary

Brief overview of the meeting.

### Action Items

- [ ] **You:** Send the report by Friday
- [ ] **Speaker 1:** Review the PR

### Key Decisions

- Use PostgreSQL instead of MongoDB

---

# Meeting 2026-03-23 14:30

![[attachments/audio/2026-03-23_14-30-00.wav]]

[00:00:01] **You:** Hello, let's start with the backend changes.

[00:01:23] **Speaker 1:** Sure, I have the slides ready.

[00:02:45] **Speaker 2:** Can we start with the backend changes?
```

## Architecture

```
src/meetrec/
├── cli.py          # Click CLI commands
├── recorder.py     # PulseAudio recording control
├── audio.py        # ffmpeg audio processing
├── transcriber.py  # faster-whisper transcription
├── diarizer.py     # pyannote speaker diarization + channel analysis
├── formatter.py    # Markdown generation (pure formatting, no I/O)
├── vault.py        # Obsidian vault I/O
├── summarizer.py   # LLM summarization (multi-provider)
├── models.py       # Domain objects (Segment, Word, etc.)
└── settings.py     # pydantic-settings configuration
```

## Development

```bash
uv sync --group dev
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).
