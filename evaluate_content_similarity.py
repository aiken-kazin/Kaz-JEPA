"""Content-similarity evaluation: does the encoder capture transcript content?

For each of N sampled pairs of test clips:
  - Compute embedding cosine similarity from the (frozen) pretrained encoder
  - Compute transcript-text similarity (word-set Jaccard)
Then report the Pearson and Spearman correlations between these two measures.

A strong positive correlation (e.g. >0.4) means clips with similar spoken
content end up with similar embeddings — i.e., the encoder captured
semantic structure rather than just recording-condition acoustics.

This evaluation needs no manually annotated labels — transcripts are part
of the manifest and act as a self-supervised "content" signal.

Usage:
  python evaluate_content_similarity.py \
      --ckpt logs/train/runs/2026-05-25_20-58-02/checkpoints/last.ckpt \
      [--n-clips 1000] [--n-pairs 5000] [--device cuda] \
      [--mel-bands 128 --mel-time 256]

Compare on the same n-pairs / random seed:
  --random-init   to see what correlation random features give (the floor)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from asr_model import build_encoder, load_jepa_encoder_weights


# -----------------------------------------------------------------------------
# Manifest → (audio_path, transcript, split)
# -----------------------------------------------------------------------------
def load_manifest_pairs(manifest_path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with manifest_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["audio_path"], row["transcript_text"], row["split"]))
    return rows


# -----------------------------------------------------------------------------
# Text similarity (word-set Jaccard)
# -----------------------------------------------------------------------------
def tokenize_text(text: str) -> set[str]:
    """Lowercase, split on whitespace, strip basic punctuation."""
    tokens = []
    for tok in text.lower().split():
        tok = tok.strip(".,!?;:\"'()[]{}«»—–-")
        if tok:
            tokens.append(tok)
    return set(tokens)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = a & b
    union = a | b
    return len(inter) / len(union)


# -----------------------------------------------------------------------------
# Audio → mel-spec → embedding
# -----------------------------------------------------------------------------
class MelExtractor:
    def __init__(self, sr=32000, clip_length=10.0, n_mels=128, target_time_bins=256,
                 n_fft=2048, hop_length=320):
        self.sr = sr
        self.target_samples = int(sr * clip_length)
        self.target_time_bins = target_time_bins
        self.mel = nn.Sequential(
            T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
                             n_mels=n_mels, power=2.0),
            T.AmplitudeToDB(),
        )

    def __call__(self, audio_path: str) -> torch.Tensor:
        audio, file_sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != self.sr:
            audio = T.Resample(file_sr, self.sr)(torch.from_numpy(audio)).numpy()
        if audio.shape[0] < self.target_samples:
            audio = np.pad(audio, (0, self.target_samples - audio.shape[0]))
        else:
            audio = audio[: self.target_samples]
        wav = torch.from_numpy(audio).float().unsqueeze(0)
        spec = self.mel(wav).squeeze(0)
        if spec.shape[-1] < self.target_time_bins:
            spec = F.pad(spec, (0, self.target_time_bins - spec.shape[-1]))
        else:
            spec = spec[..., : self.target_time_bins]
        return spec.unsqueeze(0)


@torch.no_grad()
def compute_embeddings(encoder, mel, items, device, batch_size=16):
    embeds = []
    total = len(items)
    for start in range(0, total, batch_size):
        batch = items[start : start + batch_size]
        specs = torch.stack([mel(p) for p, _ in batch]).to(device)
        out = encoder(specs)
        if isinstance(out, tuple):
            out = out[0]
        pooled = out.mean(dim=1) if out.dim() == 3 else out
        embeds.append(pooled.cpu())
        if (start // batch_size) % 5 == 0:
            print(f"  embeddings {start + len(batch)}/{total}", flush=True)
    return torch.cat(embeds)


# -----------------------------------------------------------------------------
# Correlation
# -----------------------------------------------------------------------------
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--manifest", type=Path, default=Path("data/KSC2/asr_manifest.csv"))
    p.add_argument("--split", default="Test", choices=["Train", "Dev", "Test"])
    p.add_argument("--n-clips", type=int, default=1000)
    p.add_argument("--n-pairs", type=int, default=5000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mel-bands", type=int, default=128)
    p.add_argument("--mel-time", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.random_init and args.ckpt is None:
        sys.exit("Either pass --ckpt or --random-init.")

    rng = random.Random(args.seed)

    print(f"[1/4] Loading manifest from {args.manifest}")
    rows = load_manifest_pairs(args.manifest)
    split_rows = [(p_, t) for p_, t, s in rows if s == args.split and t.strip()]
    if not split_rows:
        print(f"  no {args.split} rows with non-empty transcripts; falling back to Dev")
        split_rows = [(p_, t) for p_, t, s in rows if s == "Dev" and t.strip()]

    rng.shuffle(split_rows)
    items = split_rows[: args.n_clips]
    print(f"      using {len(items)} clips from {args.split} (each has a transcript)")

    print(f"[2/4] Building encoder (input_size = {args.mel_time} x {args.mel_bands})")
    encoder = build_encoder(input_size=(args.mel_time, args.mel_bands))
    if args.random_init:
        print("      using RANDOM-init encoder (baseline)")
    else:
        load_jepa_encoder_weights(encoder, args.ckpt)
    encoder = encoder.to(args.device).eval()
    for p_ in encoder.parameters():
        p_.requires_grad = False

    mel = MelExtractor(n_mels=args.mel_bands, target_time_bins=args.mel_time)

    print(f"[3/4] Computing embeddings for {len(items)} clips")
    emb = compute_embeddings(encoder, mel, items, args.device, args.batch_size)
    emb = F.normalize(emb, dim=-1)                          # cosine ready
    token_sets = [tokenize_text(t) for _, t in items]

    print(f"[4/4] Sampling {args.n_pairs} pairs and computing similarities")
    n = len(items)
    pair_indices = []
    while len(pair_indices) < args.n_pairs:
        i, j = rng.sample(range(n), 2)
        pair_indices.append((i, j))

    text_sims = np.empty(args.n_pairs, dtype=np.float32)
    emb_sims = np.empty(args.n_pairs, dtype=np.float32)
    for k, (i, j) in enumerate(pair_indices):
        text_sims[k] = jaccard(token_sets[i], token_sets[j])
        emb_sims[k] = float(torch.dot(emb[i], emb[j]))

    p_corr = pearson(text_sims, emb_sims)
    s_corr = spearman(text_sims, emb_sims)

    # Stratified analysis: bucket by text similarity
    buckets = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.60), (0.60, 1.01)]
    print("\n" + "=" * 60)
    print(f"RESULTS — content-similarity correlation ({args.split})")
    print("=" * 60)
    print(f"  Pairs evaluated     : {args.n_pairs}")
    print(f"  Pearson correlation : {p_corr:+.4f}")
    print(f"  Spearman correlation: {s_corr:+.4f}")
    print()
    print("  Mean embedding similarity, bucketed by text Jaccard:")
    print(f"    {'text Jaccard':<18s}  {'N':>5s}  {'mean emb sim':>14s}")
    for lo, hi in buckets:
        mask = (text_sims >= lo) & (text_sims < hi)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        mean_emb = float(emb_sims[mask].mean())
        label = f"[{lo:.2f}, {hi:.2f})"
        print(f"    {label:<18s}  {n_b:>5d}  {mean_emb:>14.4f}")

    print()
    if p_corr > 0.35:
        print("✅ Strong content-aware encoding: text similarity predicts embedding similarity.")
    elif p_corr > 0.15:
        print("⚠️  Moderate content-aware signal — encoder partially tracks content.")
    else:
        print("⚠️  Weak content correlation — encoder mostly captures non-textual acoustics.")


if __name__ == "__main__":
    main()
