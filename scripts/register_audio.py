#!/usr/bin/env python3
"""
scripts/register_audio.py
--------------------------
Batch-register recorded audio files into asr_samples.json.

Scans a directory for audio files (.mp3, .wav, .webm, .ogg, .m4a)
and either:
  a) Reports which sample IDs have a matching audio file (dry-run), or
  b) Updates the audio_path field in asr_samples.json in-place.

Usage:
  # 1. Dry-run: list which samples are covered
  python3 scripts/register_audio.py --audio-dir /path/to/recordings

  # 2. Auto-name: files must be named asr_001.mp3, asr_002.wav, …
  python3 scripts/register_audio.py --audio-dir /path/to/recordings --apply

  # 3. Copy files into the dataset audio folder and apply
  python3 scripts/register_audio.py --audio-dir /path/to/recordings --apply --copy

Naming convention:
  The script matches files whose stem ends with the sample ID.
  Examples that all match sample id=3:
      asr_003.mp3   asr_3.wav   003.wav   3.mp3
"""

import argparse
import json
import re
import shutil
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent
ASR_JSON    = REPO_ROOT / "evaluation" / "dataset" / "asr_samples.json"
AUDIO_DIR   = REPO_ROOT / "evaluation" / "dataset" / "audio"

AUDIO_EXTS  = {".mp3", ".wav", ".webm", ".ogg", ".m4a", ".flac"}

_ID_RE = re.compile(r"(\d+)$")


def _extract_id(stem: str) -> int | None:
    """Extract the trailing numeric ID from a file stem (e.g. 'asr_003' → 3)."""
    m = _ID_RE.search(stem)
    return int(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Register audio files into asr_samples.json")
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR),
                        help="Directory containing recorded audio files (default: evaluation/dataset/audio/)")
    parser.add_argument("--apply", action="store_true",
                        help="Write audio_path back into asr_samples.json (default: dry-run only)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy audio files into evaluation/dataset/audio/ first")
    parser.add_argument("--dataset", default=str(ASR_JSON),
                        help="Path to asr_samples.json")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"[ERROR] Audio directory not found: {audio_dir}")
        return

    # Build id → file map from the audio directory
    id_to_file: dict[int, Path] = {}
    for f in sorted(audio_dir.iterdir()):
        if f.suffix.lower() in AUDIO_EXTS:
            eid = _extract_id(f.stem)
            if eid is not None:
                id_to_file[eid] = f

    # Load dataset
    dataset_path = Path(args.dataset)
    with open(dataset_path) as fh:
        samples = json.load(fh)

    # Copy to canonical location if requested
    if args.copy:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        for eid, src in id_to_file.items():
            dst = AUDIO_DIR / f"asr_{eid:03d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            print(f"  Copied {src.name} → {dst.relative_to(REPO_ROOT)}")
        # Re-scan from canonical dir
        id_to_file = {}
        for f in sorted(AUDIO_DIR.iterdir()):
            if f.suffix.lower() in AUDIO_EXTS:
                eid = _extract_id(f.stem)
                if eid is not None:
                    id_to_file[eid] = f

    # Report / apply
    print(f"\n{'─'*55}")
    print(f"  {'ID':<5} {'File':<30} {'Status'}")
    print(f"{'─'*55}")

    updated = 0
    for sample in samples:
        sid = sample.get("id")
        current_path = sample.get("audio_path")
        audio_file = id_to_file.get(sid)

        if audio_file:
            # Use path relative to dataset/audio/ for portability
            try:
                rel = audio_file.relative_to(AUDIO_DIR)
                new_path = str(rel)
            except ValueError:
                new_path = str(audio_file)

            status = "✅ FOUND"
            if args.apply and current_path != new_path:
                sample["audio_path"] = new_path
                updated += 1
                status = f"✅ UPDATED → {new_path}"
        else:
            status = "❌ MISSING — record audio for this sample"

        ref_preview = sample.get("reference", "")[:40]
        print(f"  {sid:<5} {ref_preview:<40}  {status}")

    print(f"{'─'*55}\n")

    if args.apply:
        with open(dataset_path, "w", encoding="utf-8") as fh:
            json.dump(samples, fh, ensure_ascii=False, indent=2)
        print(f"  Saved {updated} update(s) to {dataset_path}")
    else:
        print("  Dry-run complete. Pass --apply to write changes.")


if __name__ == "__main__":
    main()
