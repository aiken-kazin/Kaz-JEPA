"""ASR data utilities: char tokenizer + Dataset class for KSC2.

Two components:

  1. KazakhCharTokenizer
     Char-level tokenizer that builds vocab from the manifest CSV and
     handles encode/decode + CTC-decode (collapse repeats, remove blanks).
     Reserves token 0 = <blank>, token 1 = <unk>.

  2. KSC2AsrDataset
     Loads (audio, transcript) pairs from asr_manifest.csv. Each item is:
         mel_spec: (1, n_mels, T_audio)
         input_lengths: int                (T_audio for CTC)
         target_ids: 1-D LongTensor        (encoded transcript)
         target_lengths: int               (len of transcript)
     Use collate_asr_batch as the DataLoader collate_fn.

Run as a script to build/save vocab from the manifest:
    python asr_data.py build-vocab \
        --manifest data/KSC2/asr_manifest.csv \
        --out data/KSC2/vocab.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset


BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"


@dataclass
class KazakhCharTokenizer:
    """Char-level tokenizer with reserved tokens for CTC.

    token 0  = <blank>   (CTC blank, never emitted in decoded output)
    token 1  = <unk>     (any char not in vocab)
    token 2+ = actual characters from training data
    """

    itos: list[str]                          # idx → char
    stoi: dict[str, int]                     # char → idx

    @classmethod
    def from_manifest(cls, manifest_path: Path, min_count: int = 5,
                       splits: tuple[str, ...] = ("Train",)) -> "KazakhCharTokenizer":
        """Build vocab from manifest. Only chars seen ≥ min_count times in `splits` are kept.

        Rare chars get mapped to <unk> at encode time.
        """
        counter: Counter[str] = Counter()
        with manifest_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] not in splits:
                    continue
                counter.update(row["transcript_text"])

        chars = sorted(c for c, n in counter.items() if n >= min_count)
        itos = [BLANK_TOKEN, UNK_TOKEN] + chars
        stoi = {c: i for i, c in enumerate(itos)}
        return cls(itos=itos, stoi=stoi)

    @classmethod
    def from_json(cls, path: Path) -> "KazakhCharTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        itos = data["itos"]
        stoi = {c: i for i, c in enumerate(itos)}
        return cls(itos=itos, stoi=stoi)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.unk_id) for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))

    def ctc_decode(self, ids: Iterable[int]) -> str:
        """Collapse consecutive repeats then drop blanks. Standard CTC greedy decode."""
        out: list[str] = []
        prev = -1
        for i in ids:
            if i != prev and i != self.blank_id:
                if 0 <= i < len(self.itos):
                    out.append(self.itos[i])
            prev = i
        return "".join(out)


class KSC2AsrDataset(Dataset):
    """Loads (audio, transcript) pairs from manifest CSV for a given split."""

    def __init__(
        self,
        manifest_path: Path,
        tokenizer: KazakhCharTokenizer,
        split: str,                          # "Train" | "Dev" | "Test"
        sr: int = 32000,
        clip_length: float = 10.0,           # seconds; pad/truncate to this
        n_mels: int = 96,
        target_time_bins: int = 512,
        n_fft: int = 2048,
        hop_length: int = 320,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.sr = sr
        self.target_samples = int(sr * clip_length)
        self.target_time_bins = target_time_bins

        # Load manifest rows for this split only
        self.rows: list[tuple[str, str]] = []  # (audio_path, text)
        with manifest_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.rows.append((row["audio_path"], row["transcript_text"]))

        # Pre-build the mel-spec transform (same params as pretraining for compat).
        self.mel = torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            ),
            T.AmplitudeToDB(),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _load_audio(self, path: str) -> np.ndarray:
        audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != self.sr:
            audio = self._resample(audio, file_sr)
        # Pad / truncate to target_samples
        if audio.shape[0] < self.target_samples:
            audio = np.pad(audio, (0, self.target_samples - audio.shape[0]))
        else:
            audio = audio[: self.target_samples]
        return audio

    def _resample(self, audio: np.ndarray, src_sr: int) -> np.ndarray:
        resampler = T.Resample(orig_freq=src_sr, new_freq=self.sr)
        return resampler(torch.from_numpy(audio)).numpy()

    def __getitem__(self, idx: int) -> dict:
        audio_path, text = self.rows[idx]
        audio = self._load_audio(audio_path)
        wav = torch.from_numpy(audio).float()           # (samples,)
        spec = self.mel(wav.unsqueeze(0)).squeeze(0)    # (n_mels, T)
        # Pad/truncate the time dimension to target_time_bins, same as pretraining
        if spec.shape[-1] < self.target_time_bins:
            pad = self.target_time_bins - spec.shape[-1]
            spec = torch.nn.functional.pad(spec, (0, pad))
        else:
            spec = spec[..., : self.target_time_bins]

        target_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return {
            "spec": spec.unsqueeze(0),                  # (1, n_mels, T) — add channel dim
            "input_length": self.target_time_bins,
            "target_ids": target_ids,
            "target_length": len(target_ids),
            "text": text,
        }


def collate_asr_batch(batch: list[dict]) -> dict:
    """Pad target_ids to max length in batch; stack specs."""
    specs = torch.stack([b["spec"] for b in batch])                 # (B, 1, n_mels, T)
    input_lengths = torch.tensor([b["input_length"] for b in batch], dtype=torch.long)
    target_lengths = torch.tensor([b["target_length"] for b in batch], dtype=torch.long)
    max_target = max(target_lengths).item()
    targets = torch.zeros(len(batch), max_target, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["target_length"]
        targets[i, :n] = b["target_ids"]
    return {
        "spec": specs,
        "input_lengths": input_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": [b["text"] for b in batch],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build-vocab", help="Build & save tokenizer vocab from manifest")
    pb.add_argument("--manifest", type=Path, required=True)
    pb.add_argument("--out", type=Path, default=Path("data/KSC2/vocab.json"))
    pb.add_argument("--min-count", type=int, default=5)

    pi = sub.add_parser("inspect", help="Print a sample from the dataset")
    pi.add_argument("--manifest", type=Path, required=True)
    pi.add_argument("--vocab", type=Path, default=Path("data/KSC2/vocab.json"))
    pi.add_argument("--split", default="Train")
    pi.add_argument("--n", type=int, default=2)

    args = p.parse_args()

    if args.cmd == "build-vocab":
        tok = KazakhCharTokenizer.from_manifest(args.manifest, min_count=args.min_count)
        tok.to_json(args.out)
        print(f"Vocab size: {tok.vocab_size}")
        print(f"First 20 tokens: {tok.itos[:20]}")
        print(f"Saved: {args.out}")

    elif args.cmd == "inspect":
        tok = KazakhCharTokenizer.from_json(args.vocab)
        ds = KSC2AsrDataset(manifest_path=args.manifest, tokenizer=tok, split=args.split)
        print(f"Dataset size ({args.split}): {len(ds)}")
        for i in range(min(args.n, len(ds))):
            item = ds[i]
            print(f"\n--- Sample {i} ---")
            print(f"spec shape    : {tuple(item['spec'].shape)}")
            print(f"input_length  : {item['input_length']}")
            print(f"target_length : {item['target_length']}")
            print(f"text          : {item['text'][:80]!r}{'...' if len(item['text'])>80 else ''}")
            ids = item["target_ids"].tolist()
            print(f"target_ids[:20]: {ids[:20]}")
            print(f"decoded       : {tok.decode(ids)[:80]!r}")


if __name__ == "__main__":
    main()
