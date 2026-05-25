"""kNN evaluation of a pretrained JEPA encoder on Kazakh domain classification.

For each clip in the KSC2 dataset we know the source domain from the file
path: Train/<domain>/<file>.flac where domain ∈ {radio, podcasts, tv_news,
talkshow, crowdsourced, parliament, tts}.

Procedure:
  1. Load pretrained encoder from checkpoint
  2. Sample N train clips per domain (balanced)
  3. Sample M test clips per domain (balanced)
  4. Compute pooled embeddings for all clips
  5. For each test embedding, find K nearest train embeddings by cosine
     similarity → majority-vote the domain label
  6. Report accuracy + per-domain breakdown + confusion matrix

No training needed. The only thing that matters is whether pretraining
produced semantically meaningful embeddings.

Usage:
  python evaluate_knn.py --ckpt logs/train/runs/2026-05-25_06-33-57/checkpoints/last.ckpt \
      [--manifest data/KSC2/asr_manifest.csv] \
      [--per-domain-train 200] [--per-domain-test 50] [--k 5] [--device cuda]

Compare three encoders for the paper:
  --ckpt <our_pretrained>       (our model)
  --ckpt <none>  --random-init  (baseline: random weights, should be ~chance)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from asr_model import build_encoder, load_jepa_encoder_weights


# -----------------------------------------------------------------------------
# Manifest → (audio_path, domain) pairs
# -----------------------------------------------------------------------------
def extract_domain(audio_path: str) -> str:
    """Extract domain from path: .../Train/<domain>/file.flac → '<domain>'
    For Dev/Test where files may not have domain subdir, returns 'dev_or_test'.
    """
    parts = Path(audio_path).parts
    # find 'Train'/'Dev'/'Test' marker, return the next component if any
    for i, p in enumerate(parts):
        if p in {"Train", "Dev", "Test"} and i + 1 < len(parts):
            next_part = parts[i + 1]
            # if the next part is a file (ends with .flac), not a domain dir
            if next_part.endswith(".flac"):
                return p.lower()  # e.g., 'dev'
            return next_part
    return "unknown"


def load_manifest_pairs(manifest_path: Path) -> list[tuple[str, str, str]]:
    """Return list of (audio_path, domain, split)."""
    rows: list[tuple[str, str, str]] = []
    with manifest_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = extract_domain(row["audio_path"])
            rows.append((row["audio_path"], d, row["split"]))
    return rows


def sample_balanced(rows: list[tuple[str, str, str]], target_split: str,
                    per_domain: int, seed: int = 0) -> list[tuple[str, str]]:
    """Sample up to `per_domain` examples per domain from the given split."""
    rng = random.Random(seed)
    by_domain: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, dom, split in rows:
        if split == target_split:
            by_domain[dom].append((path, dom))

    selected: list[tuple[str, str]] = []
    for dom, items in by_domain.items():
        rng.shuffle(items)
        selected.extend(items[:per_domain])
    rng.shuffle(selected)
    return selected


# -----------------------------------------------------------------------------
# Audio → mel-spectrogram (same params as pretraining)
# -----------------------------------------------------------------------------
class MelExtractor:
    def __init__(self, sr: int = 32000, clip_length: float = 10.0,
                 n_mels: int = 128, target_time_bins: int = 256,
                 n_fft: int = 2048, hop_length: int = 320):
        self.sr = sr
        self.target_samples = int(sr * clip_length)
        self.target_time_bins = target_time_bins
        self.mel = torch.nn.Sequential(
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

        wav = torch.from_numpy(audio).float().unsqueeze(0)   # (1, samples)
        spec = self.mel(wav).squeeze(0)                       # (n_mels, T)
        if spec.shape[-1] < self.target_time_bins:
            spec = F.pad(spec, (0, self.target_time_bins - spec.shape[-1]))
        else:
            spec = spec[..., : self.target_time_bins]
        return spec.unsqueeze(0)                              # (1, n_mels, T)


# -----------------------------------------------------------------------------
# Embedding extraction
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_embeddings(encoder, mel_extractor, items: list[tuple[str, str]],
                       device: str, batch_size: int = 16,
                       desc: str = "embeddings") -> tuple[torch.Tensor, list[str]]:
    embeds: list[torch.Tensor] = []
    labels: list[str] = []
    total = len(items)
    for batch_start in range(0, total, batch_size):
        batch = items[batch_start : batch_start + batch_size]
        specs = torch.stack([mel_extractor(p) for p, _ in batch]).to(device)
        out = encoder(specs)
        if isinstance(out, tuple):
            out = out[0]
        # mean-pool over patch dim → (B, embed_dim)
        pooled = out.mean(dim=1) if out.dim() == 3 else out
        embeds.append(pooled.cpu())
        labels.extend([d for _, d in batch])
        if (batch_start // batch_size) % 5 == 0:
            print(f"  [{desc}] {batch_start + len(batch)}/{total}", flush=True)
    return torch.cat(embeds), labels


# -----------------------------------------------------------------------------
# kNN classifier (cosine similarity)
# -----------------------------------------------------------------------------
def knn_predict(train_emb: torch.Tensor, train_labels: list[str],
                test_emb: torch.Tensor, k: int = 5) -> list[str]:
    train_n = F.normalize(train_emb, dim=-1)
    test_n = F.normalize(test_emb, dim=-1)
    sim = test_n @ train_n.T                              # (N_test, N_train)
    top_k_idx = sim.topk(k, dim=-1).indices.tolist()      # (N_test, k)
    preds: list[str] = []
    for indices in top_k_idx:
        votes = Counter(train_labels[i] for i in indices)
        preds.append(votes.most_common(1)[0][0])
    return preds


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Path to pretrained JEPA checkpoint. Skip for random-init baseline.")
    p.add_argument("--random-init", action="store_true",
                   help="Use random weights (skip checkpoint loading) for baseline.")
    p.add_argument("--manifest", type=Path, default=Path("data/KSC2/asr_manifest.csv"))
    p.add_argument("--per-domain-train", type=int, default=200)
    p.add_argument("--per-domain-test", type=int, default=50)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mel-bands", type=int, default=128, help="n_mels (must match checkpoint)")
    p.add_argument("--mel-time", type=int, default=256, help="target_time_bins (must match checkpoint)")
    args = p.parse_args()

    if not args.random_init and args.ckpt is None:
        sys.exit("Either pass --ckpt or --random-init.")

    print(f"[1/5] Loading manifest from {args.manifest}")
    rows = load_manifest_pairs(args.manifest)

    train_items = sample_balanced(rows, "Train", args.per_domain_train, args.seed)
    test_items = sample_balanced(rows, "Test", args.per_domain_test, args.seed + 1)
    if not test_items:
        # fall back to Dev split if Test is missing in manifest
        print("  no Test split rows found → using Dev")
        test_items = sample_balanced(rows, "Dev", args.per_domain_test, args.seed + 1)

    print(f"      train: {len(train_items)}  test: {len(test_items)}")
    domains_train = Counter(d for _, d in train_items)
    domains_test = Counter(d for _, d in test_items)
    print(f"      train domains: {dict(domains_train)}")
    print(f"      test  domains: {dict(domains_test)}")

    print(f"[2/5] Building encoder (input_size = {args.mel_time} x {args.mel_bands})")
    encoder = build_encoder(input_size=(args.mel_time, args.mel_bands))
    if args.random_init:
        print("      using RANDOM-init encoder (baseline)")
    else:
        load_jepa_encoder_weights(encoder, args.ckpt)
    encoder = encoder.to(args.device).eval()

    mel = MelExtractor(n_mels=args.mel_bands, target_time_bins=args.mel_time)

    print(f"[3/5] Computing TRAIN embeddings")
    train_emb, train_labels = compute_embeddings(encoder, mel, train_items,
                                                 args.device, args.batch_size, desc="train")
    print(f"      train_emb: {tuple(train_emb.shape)}")

    print(f"[4/5] Computing TEST embeddings")
    test_emb, test_labels = compute_embeddings(encoder, mel, test_items,
                                               args.device, args.batch_size, desc="test")
    print(f"      test_emb: {tuple(test_emb.shape)}")

    print(f"[5/5] kNN (k={args.k}) with cosine similarity")
    preds = knn_predict(train_emb, train_labels, test_emb, k=args.k)

    correct = sum(p == y for p, y in zip(preds, test_labels))
    acc = correct / len(test_labels)

    per_domain_correct: dict[str, int] = defaultdict(int)
    per_domain_total: dict[str, int] = defaultdict(int)
    for p_, y in zip(preds, test_labels):
        per_domain_total[y] += 1
        if p_ == y:
            per_domain_correct[y] += 1

    print("\n" + "=" * 60)
    print(f"RESULTS — kNN domain classification (k={args.k})")
    print("=" * 60)
    print(f"  Overall accuracy: {acc*100:.2f}%  ({correct}/{len(test_labels)})")
    print(f"  Domains seen    : {len(domains_train)}")
    print(f"  Random baseline : {100/len(domains_train):.2f}%")
    print()
    print("  Per-domain accuracy:")
    for d in sorted(per_domain_total):
        t = per_domain_total[d]
        c = per_domain_correct[d]
        print(f"    {d:<15s} {c:>4d}/{t:<4d}  {c/t*100:6.2f}%")

    # Confusion-style summary: top-3 confusions
    conf: Counter[tuple[str, str]] = Counter()
    for p_, y in zip(preds, test_labels):
        if p_ != y:
            conf[(y, p_)] += 1
    if conf:
        print("\n  Top confusions (true → predicted):")
        for (y, p_), n in conf.most_common(5):
            print(f"    {y:<15s} → {p_:<15s}  {n}")


if __name__ == "__main__":
    main()
