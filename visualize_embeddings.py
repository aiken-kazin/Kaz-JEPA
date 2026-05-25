"""Visualize JEPA embeddings via PCA + t-SNE.

Loads a pretrained encoder, computes pooled embeddings for ~1000 test
clips, projects them to 2D via PCA→t-SNE, and saves a scatter plot
colored by domain. Also reports the cumulative explained variance to
characterize the intrinsic dimensionality of the representations.

Outputs:
  - paper/embeddings_tsne_<tag>.png  — 2D scatter colored by domain
  - paper/explained_variance_<tag>.png — PCA explained variance curve

Usage:
  python visualize_embeddings.py --ckpt <path/to/ckpt> --tag paper_config

For random-init baseline:
  python visualize_embeddings.py --random-init --tag random_init
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from asr_model import build_encoder, load_jepa_encoder_weights


# -----------------------------------------------------------------------------
# Manifest sampling (balanced across domains)
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
    rows = []
    with manifest_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["audio_path"], extract_domain(row["audio_path"]), row["split"]))
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
    embeds, labels = [], []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        specs = torch.stack([mel(p) for p, _ in batch]).to(device)
        out = encoder(specs)
        if isinstance(out, tuple):
            out = out[0]
        pooled = out.mean(dim=1) if out.dim() == 3 else out
        embeds.append(pooled.cpu())
        labels.extend([d for _, d in batch])
        if (start // batch_size) % 5 == 0:
            print(f"  emb {start + len(batch)}/{len(items)}", flush=True)
    return torch.cat(embeds).numpy(), labels


# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--tag", required=True, help="Output filename suffix, e.g., 'paper_config_2ep'")
    p.add_argument("--manifest", type=Path, default=Path("data/KSC2/asr_manifest.csv"))
    p.add_argument("--split", default="Test")
    p.add_argument("--per-domain", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mel-bands", type=int, default=128)
    p.add_argument("--mel-time", type=int, default=256)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--output-dir", type=Path, default=Path("paper"))
    args = p.parse_args()

    if not args.random_init and args.ckpt is None:
        sys.exit("Either --ckpt or --random-init.")

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. Install: pip install scikit-learn matplotlib")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading manifest")
    rows = load_manifest_pairs(args.manifest)
    items = sample_balanced(rows, args.split, args.per_domain, seed=42)
    if not items:
        items = sample_balanced(rows, "Dev", args.per_domain, seed=42)
    print(f"      {len(items)} clips")

    print(f"[2/6] Building encoder ({args.mel_time}x{args.mel_bands})")
    encoder = build_encoder(input_size=(args.mel_time, args.mel_bands))
    if args.random_init:
        print("      random init")
    else:
        load_jepa_encoder_weights(encoder, args.ckpt)
    encoder = encoder.to(args.device).eval()
    for p_ in encoder.parameters():
        p_.requires_grad = False

    mel = MelExtractor(n_mels=args.mel_bands, target_time_bins=args.mel_time)

    print(f"[3/6] Computing embeddings")
    emb, labels = compute_embeddings(encoder, mel, items, args.device, args.batch_size)
    print(f"      emb shape: {emb.shape}")

    domains = sorted(set(labels))
    color_map = {d: plt.cm.tab10(i / max(1, len(domains))) for i, d in enumerate(domains)}

    # --- PCA + explained variance ----------------------------------------
    print(f"[4/6] PCA → {args.pca_dim} components")
    pca = PCA(n_components=min(args.pca_dim, emb.shape[1], emb.shape[0]))
    emb_pca = pca.fit_transform(emb)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"      first  5 components explain {cumvar[4]*100:5.1f}% of variance")
    print(f"      first 10 components explain {cumvar[9]*100:5.1f}% of variance")
    print(f"      first 50 components explain {cumvar[-1]*100:5.1f}% of variance")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar * 100, "-o", markersize=3)
    ax.axhline(95, ls="--", color="grey", lw=1)
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title(f"PCA explained variance — {args.tag}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_pca = args.output_dir / f"explained_variance_{args.tag}.png"
    fig.savefig(out_pca, dpi=150)
    plt.close(fig)
    print(f"      saved: {out_pca}")

    # --- t-SNE -----------------------------------------------------------
    print(f"[5/6] t-SNE (perplexity={args.tsne_perplexity})")
    tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, init="pca", random_state=42)
    emb_2d = tsne.fit_transform(emb_pca)

    fig, ax = plt.subplots(figsize=(7, 6))
    for d in domains:
        mask = np.array([l == d for l in labels])
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   s=18, alpha=0.7, color=color_map[d], label=f"{d} (n={mask.sum()})",
                   edgecolors="none")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Embeddings — {args.tag}")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_tsne = args.output_dir / f"embeddings_tsne_{args.tag}.png"
    fig.savefig(out_tsne, dpi=150)
    plt.close(fig)
    print(f"      saved: {out_tsne}")

    print(f"\n[6/6] Done.")
    print(f"  Use these figures in the paper:")
    print(f"    {out_tsne}")
    print(f"    {out_pca}")


if __name__ == "__main__":
    main()
