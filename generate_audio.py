"""
Generate audio files of scripture names using F5-TTS zero-shot voice cloning.

Usage:
    python generate_audio.py \
        --ref-audio reference/my_voice.wav \
        --ref-text "The exact words spoken in the reference audio." \
        --names-file input/scripture_names.txt \
        --output-dir output/
"""

import argparse
import os
import re
import sys
import time


def sanitize_filename(name: str) -> str:
    """Convert a scripture name into a safe filename."""
    safe = re.sub(r'[<>:"/\\|?*]', '', name)
    safe = safe.strip().replace(' ', '_')
    return safe


def load_scripture_names(filepath: str) -> list[str]:
    """Read scripture names from a text file, one per line."""
    if not os.path.isfile(filepath):
        print(f"Error: Names file not found: {filepath}")
        sys.exit(1)

    names = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                names.append(stripped)

    if not names:
        print("Error: Names file is empty or contains only blank lines.")
        sys.exit(1)

    return names


def main():
    parser = argparse.ArgumentParser(
        description="Generate voice-cloned audio for scripture names using F5-TTS."
    )
    parser.add_argument(
        '--ref-audio', required=True,
        help='Path to reference voice audio file (WAV recommended, 5-10 seconds).'
    )
    parser.add_argument(
        '--ref-text', default='',
        help='Transcription of reference audio. Leave empty for auto-transcription via Whisper.'
    )
    parser.add_argument(
        '--names-file', required=True,
        help='Text file with one scripture name per line.'
    )
    parser.add_argument(
        '--output-dir', default='output',
        help='Directory for generated audio files (default: output/).'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--speed', type=float, default=1.0,
        help='Speech speed multiplier (default: 1.0).'
    )
    parser.add_argument(
        '--remove-silence', action='store_true',
        help='Remove silence from generated audio.'
    )
    parser.add_argument(
        '--model', default='F5TTS_v1_Base',
        choices=['F5TTS_v1_Base', 'F5TTS_Base', 'F5TTS_Small', 'E2TTS_Base', 'E2TTS_Small'],
        help='Model to use. Default: F5TTS_v1_Base.'
    )
    args = parser.parse_args()

    # Validate reference audio
    if not os.path.isfile(args.ref_audio):
        print(f"Error: Reference audio not found: {args.ref_audio}")
        sys.exit(1)

    # Load scripture names
    names = load_scripture_names(args.names_file)
    print(f"Loaded {len(names)} scripture names.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Import and initialize F5-TTS (done after validation to fail fast on bad inputs)
    print("Loading F5-TTS model (this may download ~3GB on first run)...")
    from f5_tts.api import F5TTS

    tts = F5TTS(model=args.model)
    print(f"Model loaded on device: {tts.device}")

    # Generate audio for each scripture name
    successful = 0
    failed = []
    start_time = time.time()

    for i, name in enumerate(names, start=1):
        safe_name = sanitize_filename(name)
        output_path = os.path.join(args.output_dir, f"{safe_name}.wav")

        print(f"[{i}/{len(names)}] Generating: {name!r} -> {output_path}")

        try:
            wav, sr, spec = tts.infer(
                ref_file=args.ref_audio,
                ref_text=args.ref_text,
                gen_text=name,
                speed=args.speed,
                remove_silence=args.remove_silence,
                file_wave=output_path,
                seed=args.seed,
            )
            successful += 1
        except Exception as e:
            print(f"  ERROR generating '{name}': {e}")
            failed.append((name, str(e)))

    elapsed = time.time() - start_time

    # Summary
    print(f"\nDone in {elapsed:.1f}s. {successful}/{len(names)} files generated in '{args.output_dir}'.")
    if failed:
        print(f"\n{len(failed)} failures:")
        for name, err in failed:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
