"""Pair KSC2 FLAC files with their .txt transcripts and write a manifest CSV.

Tries two layouts (in order):
  1. transcript next to audio: foo/bar/audio.flac  + foo/bar/audio.txt
  2. transcript in parallel "transcripts/" tree:
        Train/radio/audio.flac → Train/radio_transcripts/audio.txt   (less common)

Outputs:
  - manifest.csv with columns: audio_path, transcript_path, transcript_text, split
  - prints summary statistics (found / missing / char vocab)
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def find_transcript(flac_path: Path) -> Path | None:
    """Try common locations for a transcript file matching this FLAC."""
    stem = flac_path.stem
    candidates = [
        flac_path.with_suffix(".txt"),  # same dir, same stem
        flac_path.with_suffix(".TXT"),  # uppercase variant
        flac_path.parent / f"{stem}.lab",  # Kaldi-style .lab
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def read_transcript(txt_path: Path) -> str:
    text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    return " ".join(text.split())  # collapse internal whitespace/newlines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Root of ISSAI_KSC2 (containing Train/, Dev/, Test/)")
    p.add_argument("--out", type=Path, default=Path("data/KSC2/asr_manifest.csv"))
    p.add_argument("--max-show", type=int, default=3, help="Show this many missing examples")
    args = p.parse_args()

    splits = ["Train", "Dev", "Test"]
    rows: list[dict] = []
    missing: list[Path] = []
    char_counter: Counter[str] = Counter()
    text_lengths: list[int] = []

    for split in splits:
        split_dir = args.src / split
        if not split_dir.exists():
            print(f"  [skip] {split_dir} not found")
            continue

        flacs = sorted(split_dir.rglob("*.flac"))
        print(f"{split}: {len(flacs)} FLAC files")

        for flac in flacs:
            tx = find_transcript(flac)
            if tx is None:
                missing.append(flac)
                continue
            text = read_transcript(tx)
            rows.append({
                "audio_path": str(flac),
                "transcript_path": str(tx),
                "transcript_text": text,
                "split": split,
            })
            char_counter.update(text)
            text_lengths.append(len(text))

    print()
    print(f"=== SUMMARY ===")
    print(f"Paired:  {len(rows)}")
    print(f"Missing: {len(missing)}")
    if missing:
        print(f"  First {args.max_show} missing FLACs (no transcript found):")
        for m in missing[: args.max_show]:
            print(f"    {m}")

    if text_lengths:
        text_lengths.sort()
        print(f"\nTranscript length (chars):")
        print(f"  min/median/max: {text_lengths[0]} / {text_lengths[len(text_lengths)//2]} / {text_lengths[-1]}")
        print(f"  mean: {sum(text_lengths)/len(text_lengths):.1f}")

    if char_counter:
        print(f"\nVocab size: {len(char_counter)} unique characters")
        top = char_counter.most_common(50)
        print(f"  Top 50 chars: {' '.join(repr(c) + f'({n})' for c, n in top[:20])}")
        # Print rare/suspicious chars (likely noise: latin letters, digits, etc.)
        non_kz = [c for c in char_counter if c.isascii() and c.isalpha()]
        if non_kz:
            print(f"  Non-Cyrillic letters present ({len(non_kz)}): {''.join(sorted(non_kz))[:40]}")

    if rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript_path", "transcript_text", "split"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✅ Manifest written: {args.out}  ({len(rows)} rows)")
    else:
        print(f"\n⚠️  No (audio, transcript) pairs found. Check the dataset layout.")


if __name__ == "__main__":
    main()
