# Scripture Name Audio Generator

Generate audio files of scripture names spoken in your cloned voice using [F5-TTS](https://github.com/SWivid/F5-TTS) zero-shot voice cloning.

## Prerequisites

- Python 3.10 or 3.11
- NVIDIA GPU with CUDA
- [FFmpeg](https://ffmpeg.org) on your PATH

## Setup

1. **Create and activate a virtual environment:**
   ```bash
   cd c:\Projects\audio_generation
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install PyTorch with CUDA** (adjust `cu124` to your CUDA version):
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify FFmpeg is installed:**
   ```bash
   ffmpeg -version
   ```

## Prepare Your Files

- **Voice sample:** Place your voice recording (WAV recommended, 5-10 seconds of clear speech) in `reference/`.
- **Scripture names:** Create `input/scripture_names.txt` with one name per line, e.g.:
  ```
  Genesis
  Exodus
  Leviticus
  ```
  Or in Chinese:
  ```
  创世记
  出埃及记
  利未记
  ```

## Usage

**English:**
```bash
python generate_audio.py \
    --ref-audio reference/my_voice.wav \
    --ref-text "The words you spoke in the recording." \
    --names-file input/scripture_names.txt
```

**Chinese (with Chinese-optimized model):**
```bash
python generate_audio.py \
    --ref-audio reference/my_voice.wav \
    --ref-text "你在录音中说的话" \
    --names-file input/scripture_names.txt \
    --model F5TTS_v1_Base_zh
```

Output WAV files will appear in the `output/` folder, one per scripture name.

## Options

| Flag | Description | Default |
|---|---|---|
| `--ref-audio` | Path to your voice recording (required) | — |
| `--ref-text` | Transcription of your recording (auto-transcribes via Whisper if omitted) | `""` |
| `--names-file` | Text file with one scripture name per line (required) | — |
| `--output-dir` | Output directory for generated WAV files | `output/` |
| `--model` | `F5TTS_v1_Base` (English+Chinese) or `F5TTS_v1_Base_zh` (Chinese-optimized) | `F5TTS_v1_Base` |
| `--speed` | Speech speed multiplier | `1.0` |
| `--seed` | Random seed for reproducible output | `None` |
| `--remove-silence` | Remove silence from generated audio | off |

## Notes

- The F5-TTS model (~3GB) is downloaded automatically on the first run.
- Reference audio longer than 12 seconds is automatically truncated. The ideal length is 5-10 seconds.
- Very short names (e.g., "Job", "Ruth") may produce better results with a trailing period ("Job.").
