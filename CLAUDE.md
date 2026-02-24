# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Single-script Python tool that generates audio files of scripture names using F5-TTS zero-shot voice cloning. Given a reference voice recording and a list of scripture names, it produces one WAV file per name in the cloned voice.

## Requirements

- Python 3.10 or 3.11
- NVIDIA GPU with CUDA
- FFmpeg on PATH
- PyTorch with CUDA must be installed separately before other deps: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`

## Setup and Run

```bash
# Install dependencies (after PyTorch+CUDA)
pip install -r requirements.txt

# Run generation
python generate_audio.py \
    --ref-audio reference/my_voice.wav \
    --ref-text "The words you spoke in the recording." \
    --names-file input/scripture_names.txt

# Chinese-optimized model
python generate_audio.py \
    --ref-audio reference/my_voice.wav \
    --ref-text "你在录音中说的话" \
    --names-file input/scripture_names.txt \
    --model F5TTS_v1_Base_zh
```

## Architecture

Everything lives in `generate_audio.py` — there are no modules, tests, or build system. The script:

1. Parses CLI args and validates inputs (ref audio exists, names file is non-empty)
2. Lazy-imports `f5_tts.api.F5TTS` after validation so bad inputs fail fast before the ~3GB model download
3. Iterates over scripture names, calling `tts.infer()` for each, writing WAV files to `output/`
4. Tracks successes/failures and prints a summary

Key directories:
- `input/` — text files with scripture names (one per line)
- `reference/` — voice sample recordings for cloning (WAV recommended, 5-10s)
- `output/` — generated WAV files (gitignored)

## Key Details

- The F5-TTS model auto-downloads on first run (~3GB)
- Reference audio >12s is automatically truncated by F5-TTS; ideal is 5-10s
- If `--ref-text` is omitted, F5-TTS auto-transcribes via Whisper
- Very short names (e.g., "Job", "Ruth") may need a trailing period for better results
- Two model choices: `F5TTS_v1_Base` (English+Chinese default) and `F5TTS_v1_Base_zh` (Chinese-optimized)
