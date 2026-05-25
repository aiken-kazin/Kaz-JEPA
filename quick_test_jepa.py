"""Quick sanity check for a trained JEPA checkpoint.

Loads the encoder from a checkpoint, runs ~20 dev clips through it, and
prints embedding statistics + pairwise similarities. Purpose: detect
representation collapse without building full downstream eval.

Healthy signs:
  - embedding std > 0.01
  - pairwise cosine similarities spread across some range (not all ~1)
Collapse signs:
  - std near 0
  - all similarities ~1 (every clip looks identical to model)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="Path to .ckpt file")
    p.add_argument("--hdf5", type=Path, default=Path("data/KSC2/ksc2_dev.h5"))
    p.add_argument("--n-clips", type=int, default=20)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    repo_root = Path(__file__).parent.resolve()
    configs_dir = str(repo_root / "configs")

    print(f"[1/5] Loading config & instantiating model...")
    with initialize_config_dir(version_base=None, config_dir=configs_dir):
        cfg = compose(
            config_name="train",
            overrides=["data=ksc2", "trainer=cpu", "logger=null", "callbacks=default"],
        )
    model = instantiate(cfg.model)
    model.eval()

    print(f"[2/5] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location=args.device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  warn: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  warn: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    encoder = model.encoder.to(args.device).eval()

    print(f"[3/5] Loading {args.n_clips} clips from {args.hdf5}")
    with h5py.File(str(args.hdf5), "r") as f:
        n = min(args.n_clips, len(f["waveform"]))
        waves = f["waveform"][:n].astype(np.float32) / 32767.0
        names = [f["audio_name"][i].decode() for i in range(n)]

    print(f"[4/5] Computing mel-spectrograms & embeddings...")
    transforms = [instantiate(t) for t in cfg.data.transforms]

    embeddings = []
    with torch.no_grad():
        for wav in waves:
            x = torch.from_numpy(wav).unsqueeze(0).to(args.device)  # (1, samples)
            for t in transforms:
                x = t(x)
            if x.dim() == 3:
                x = x.unsqueeze(1)  # add channel dim if missing
            out = encoder(x)
            if isinstance(out, tuple):
                out = out[0]
            # out shape typically (1, num_patches, embed_dim) — pool to one vec
            pooled = out.mean(dim=1) if out.dim() == 3 else out
            embeddings.append(pooled.squeeze(0).cpu())

    emb = torch.stack(embeddings)  # (n, embed_dim)

    print(f"[5/5] Stats:")
    print(f"  embedding shape: {tuple(emb.shape)}")
    print(f"  abs mean       : {emb.abs().mean().item():.4f}")
    print(f"  std per-dim    : {emb.std(dim=0).mean().item():.4f}")
    print(f"  norm per-clip  : {emb.norm(dim=-1).mean().item():.4f}")

    normed = F.normalize(emb, dim=-1)
    sim = normed @ normed.T
    off = sim.masked_select(~torch.eye(len(sim), dtype=torch.bool))
    print(f"  pairwise cosine sim (off-diag):")
    print(f"    mean: {off.mean().item():+.4f}")
    print(f"    std : {off.std().item():.4f}")
    print(f"    min : {off.min().item():+.4f}")
    print(f"    max : {off.max().item():+.4f}")

    print()
    if off.std().item() < 0.01:
        print("⚠️  All clips look near-identical to the model — likely COLLAPSE.")
    elif off.mean().item() > 0.95:
        print("⚠️  All similarities very high — partial collapse possible.")
    elif emb.std(dim=0).mean().item() < 0.001:
        print("⚠️  Embedding dimensions barely vary — likely COLLAPSE.")
    else:
        print("✅ Embeddings look diverse — model learned something.")


if __name__ == "__main__":
    main()
