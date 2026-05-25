"""Linear / MLP probing of a pretrained JEPA encoder on Kazakh domain classification.

Tuncay et al. (2025) note that Audio-JEPA's embedding space is not guaranteed
to be linearly separable — a small trained classifier on top of frozen
embeddings typically extracts more of the model's learned signal than kNN.

Procedure:
  1. Load pretrained encoder, FREEZE it.
  2. Compute pooled embeddings for train + dev/test clips (sampled balanced
     across domains).
  3. Train a 2-layer MLP (768 → 256 → n_classes) on the frozen embeddings.
  4. Report accuracy + per-domain breakdown on the held-out set.

Usage:
  python linear_probe.py --ckpt logs/train/runs/2026-05-25_06-33-57/checkpoints/last.ckpt \
      [--per-domain-train 500 --per-domain-test 100] \
      [--epochs 50 --lr 1e-3] [--device cuda]

  python linear_probe.py --random-init --per-domain-train 500 --per-domain-test 100

For the paper we compare:
  - random-init  (baseline)
  - our pretrained 30k checkpoint
  - our pretrained 100k checkpoint
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
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from asr_model import build_encoder, load_jepa_encoder_weights


# -----------------------------------------------------------------------------
# Manifest helpers (reused from evaluate_knn.py)
# -----------------------------------------------------------------------------
def extract_domain(audio_path: str) -> str:
    parts = Path(audio_path).parts
    for i, p in enumerate(parts):
        if p in {"Train", "Dev", "Test"} and i + 1 < len(parts):
            n = parts[i + 1]
            if n.endswith(".flac"):
                return p.lower()
            return n
    return "unknown"


def load_manifest_pairs(manifest_path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with manifest_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = extract_domain(row["audio_path"])
            rows.append((row["audio_path"], d, row["split"]))
    return rows


def sample_balanced(rows, target_split, per_domain, seed=0):
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for path, dom, split in rows:
        if split == target_split:
            by_domain[dom].append((path, dom))
    sel = []
    for items in by_domain.values():
        rng.shuffle(items)
        sel.extend(items[:per_domain])
    rng.shuffle(sel)
    return sel


# -----------------------------------------------------------------------------
# Mel-spec extractor
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
def compute_embeddings(encoder, mel_extractor, items, device, batch_size=16, desc="embeddings"):
    embeds: list[torch.Tensor] = []
    labels: list[str] = []
    total = len(items)
    for start in range(0, total, batch_size):
        batch = items[start : start + batch_size]
        specs = torch.stack([mel_extractor(p) for p, _ in batch]).to(device)
        out = encoder(specs)
        if isinstance(out, tuple):
            out = out[0]
        pooled = out.mean(dim=1) if out.dim() == 3 else out
        embeds.append(pooled.cpu())
        labels.extend([d for _, d in batch])
        if (start // batch_size) % 5 == 0:
            print(f"  [{desc}] {start + len(batch)}/{total}", flush=True)
    return torch.cat(embeds), labels


# -----------------------------------------------------------------------------
# Probe head
# -----------------------------------------------------------------------------
class MLPProbe(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(train_emb, train_labels, test_emb, test_labels,
                domains, epochs=50, lr=1e-3, batch_size=64, device="cuda"):
    label2id = {d: i for i, d in enumerate(domains)}
    n_classes = len(domains)

    train_y = torch.tensor([label2id[l] for l in train_labels], dtype=torch.long)
    test_y = torch.tensor([label2id[l] for l in test_labels], dtype=torch.long)

    train_emb = train_emb.to(device)
    train_y = train_y.to(device)
    test_emb = test_emb.to(device)
    test_y = test_y.to(device)

    probe = MLPProbe(train_emb.size(-1), n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    n_train = train_emb.size(0)
    best_acc = 0.0
    best_per_domain: dict[str, tuple[int, int]] = {}

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            x = train_emb[idx]
            y = train_y[idx]
            logits = probe(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        sched.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(test_emb).argmax(dim=-1)
            acc = (preds == test_y).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            per_dom_correct: dict[str, int] = defaultdict(int)
            per_dom_total: dict[str, int] = defaultdict(int)
            for p_, y in zip(preds.cpu().tolist(), test_y.cpu().tolist()):
                dom = domains[y]
                per_dom_total[dom] += 1
                if p_ == y:
                    per_dom_correct[dom] += 1
            best_per_domain = {d: (per_dom_correct[d], per_dom_total[d]) for d in domains if per_dom_total[d] > 0}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>3d}/{epochs}  train_loss={total_loss/n_train:.4f}  test_acc={acc*100:.2f}%  (best={best_acc*100:.2f}%)")

    return best_acc, best_per_domain


# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--manifest", type=Path, default=Path("data/KSC2/asr_manifest.csv"))
    p.add_argument("--per-domain-train", type=int, default=500)
    p.add_argument("--per-domain-test", type=int, default=100)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--probe-batch", type=int, default=64)
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
        print("  no Test rows → using Dev")
        test_items = sample_balanced(rows, "Dev", args.per_domain_test, args.seed + 1)

    print(f"      train: {len(train_items)}  test: {len(test_items)}")
    train_dom_counts = Counter(d for _, d in train_items)
    test_dom_counts = Counter(d for _, d in test_items)
    print(f"      train domains: {dict(train_dom_counts)}")
    print(f"      test  domains: {dict(test_dom_counts)}")

    # Keep ONLY domains that appear in BOTH train and test
    domains = sorted(set(train_dom_counts) & set(test_dom_counts))
    train_items = [t for t in train_items if t[1] in domains]
    test_items = [t for t in test_items if t[1] in domains]
    print(f"      using {len(domains)} domains in both splits: {domains}")

    print(f"[2/5] Building encoder (input_size = {args.mel_time} x {args.mel_bands})")
    encoder = build_encoder(input_size=(args.mel_time, args.mel_bands))
    if args.random_init:
        print("      using RANDOM-init encoder (baseline)")
    else:
        load_jepa_encoder_weights(encoder, args.ckpt)
    encoder = encoder.to(args.device).eval()
    for p_ in encoder.parameters():
        p_.requires_grad = False

    mel = MelExtractor(n_mels=args.mel_bands, target_time_bins=args.mel_time)

    print(f"[3/5] Computing TRAIN embeddings (frozen encoder)")
    train_emb, train_labels = compute_embeddings(encoder, mel, train_items,
                                                 args.device, args.batch_size, desc="train")
    print(f"      train_emb: {tuple(train_emb.shape)}")

    print(f"[4/5] Computing TEST embeddings (frozen encoder)")
    test_emb, test_labels = compute_embeddings(encoder, mel, test_items,
                                               args.device, args.batch_size, desc="test")
    print(f"      test_emb: {tuple(test_emb.shape)}")

    print(f"[5/5] Training MLP probe ({args.epochs} epochs)")
    best_acc, per_domain = train_probe(
        train_emb, train_labels, test_emb, test_labels,
        domains=domains, epochs=args.epochs, lr=args.lr,
        batch_size=args.probe_batch, device=args.device,
    )

    print("\n" + "=" * 60)
    print(f"RESULTS — MLP linear probe (domain classification)")
    print("=" * 60)
    print(f"  Best test accuracy: {best_acc*100:.2f}%")
    print(f"  Random baseline   : {100/len(domains):.2f}%")
    print()
    print("  Per-domain accuracy:")
    for d in domains:
        c, t = per_domain.get(d, (0, 0))
        if t:
            print(f"    {d:<15s} {c:>4d}/{t:<4d}  {c/t*100:6.2f}%")


if __name__ == "__main__":
    main()
