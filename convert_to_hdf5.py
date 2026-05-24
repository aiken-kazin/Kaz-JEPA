"""Convert KSC2 FLAC files into the HDF5 layout that AudioSetDataset expects.

Schema (matches src/data/components/audioset_dataset.py with mp3_dataset=False):
  - audio_name: variable-length bytes, one per clip
  - waveform: int16, shape (N, sr * clip_length)  (gets /32767 -> float32 at load)
  - target:   float32, shape (N, classes_num)     (zeros for self-supervised pretraining)

Usage:
  python convert_to_hdf5.py \
      --src ksc2_sample \
      --out data/KSC2 \
      --sr 32000 --clip-length 10 --classes-num 527
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf


def list_flacs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.flac") if p.is_file())


def load_clip(path: Path, sr: int, target_len: int) -> np.ndarray:
    """Load a FLAC, mono-mix, resample if needed, pad/truncate to target_len, return int16."""
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if file_sr != sr:
        # Lazy import so users without torchaudio can still run on already-32k data
        import torch
        import torchaudio

        resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=sr)
        audio = resampler(torch.from_numpy(audio)).numpy()

    if audio.shape[0] < target_len:
        audio = np.pad(audio, (0, target_len - audio.shape[0]))
    else:
        audio = audio[:target_len]

    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def write_split(files: list[Path], out_path: Path, sr: int, clip_length: int, classes_num: int) -> None:
    target_len = sr * clip_length
    n = len(files)
    if n == 0:
        print(f"  [skip] no FLACs for {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {out_path}  ({n} clips, {sr} Hz, {clip_length}s)")

    with h5py.File(out_path, "w") as f:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("audio_name", shape=(n,), dtype=str_dtype)
        f.create_dataset(
            "waveform",
            shape=(n, target_len),
            dtype=np.int16,
            chunks=(1, target_len),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "target",
            shape=(n, classes_num),
            dtype=np.float32,
        )

        for i, path in enumerate(files):
            try:
                wav = load_clip(path, sr, target_len)
            except Exception as e:
                print(f"    [warn] {path.name}: {e}", file=sys.stderr)
                wav = np.zeros(target_len, dtype=np.int16)
            f["audio_name"][i] = path.stem
            f["waveform"][i] = wav
            if (i + 1) % 50 == 0 or (i + 1) == n:
                print(f"    [{i + 1}/{n}]")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Root of ksc2_sample (containing Train/, Dev/)")
    p.add_argument("--out", type=Path, required=True, help="Output directory for *.h5 files")
    p.add_argument("--sr", type=int, default=32000)
    p.add_argument("--clip-length", type=int, default=10)
    p.add_argument("--classes-num", type=int, default=527)
    args = p.parse_args()

    train_root = args.src / "Train"
    dev_root = args.src / "Dev"
    train_files = list_flacs(train_root) if train_root.exists() else []
    dev_files = list_flacs(dev_root) if dev_root.exists() else []

    print(f"Train FLACs: {len(train_files)} under {train_root}")
    print(f"Dev   FLACs: {len(dev_files)} under {dev_root}")

    write_split(train_files, args.out / "ksc2_train.h5", args.sr, args.clip_length, args.classes_num)
    write_split(dev_files, args.out / "ksc2_dev.h5", args.sr, args.clip_length, args.classes_num)
    print("done.")


if __name__ == "__main__":
    main()
