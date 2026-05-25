"""Quick sanity check for a trained JEPA checkpoint.

Inspects the saved weights directly — no Hydra, no forward pass, no model
instantiation. Catches obvious failures (collapse, dead weights, broken EMA)
in seconds.

Healthy signs:
  - Encoder weights have non-zero magnitude (model trained, not random init)
  - Encoder ≠ target_encoder (EMA momentum worked, two encoders diverged)
  - Weight magnitudes look reasonable for a Transformer (~0.01 to 1.0)

Collapse / failure signs:
  - All weights near zero or constant
  - Encoder == target_encoder exactly (EMA broken)
  - NaN / Inf in any tensor
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="Path to .ckpt file")
    args = p.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    print(f"\nCheckpoint keys: {list(ckpt.keys())[:10]}")
    print(f"Total state_dict tensors: {len(sd)}\n")

    # Group keys by component
    groups: dict[str, list[str]] = defaultdict(list)
    for k in sd:
        prefix = k.split(".")[0]
        groups[prefix].append(k)

    print("=" * 60)
    print("COMPONENTS")
    print("=" * 60)
    for prefix, keys in sorted(groups.items()):
        total_params = sum(sd[k].numel() for k in keys)
        print(f"  {prefix:<20s} {len(keys):>4d} tensors  {total_params/1e6:>7.2f} M params")

    # NaN / Inf check
    print("\n" + "=" * 60)
    print("HEALTH CHECKS")
    print("=" * 60)
    nan_keys = [k for k, v in sd.items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]
    if nan_keys:
        print(f"⚠️  NaN/Inf in {len(nan_keys)} tensors: {nan_keys[:3]}")
    else:
        print("✅ No NaN/Inf in any tensor")

    # Encoder weight statistics
    enc_weights = [v for k, v in sd.items() if k.startswith("encoder.") and "weight" in k and v.dim() >= 2]
    if enc_weights:
        all_vals = torch.cat([w.flatten() for w in enc_weights])
        print(f"✅ Encoder weight stats ({len(enc_weights)} matrices):")
        print(f"     mean abs: {all_vals.abs().mean().item():.4f}")
        print(f"     std     : {all_vals.std().item():.4f}")
        print(f"     min/max : {all_vals.min().item():+.4f} / {all_vals.max().item():+.4f}")

        if all_vals.abs().mean().item() < 1e-5:
            print("⚠️  Encoder weights near zero — model may not have trained!")
        elif all_vals.std().item() < 1e-4:
            print("⚠️  Encoder weights have very low variance — possible issue")

    # Encoder vs target_encoder divergence (EMA check)
    enc_keys = sorted(k for k in sd if k.startswith("encoder."))
    tgt_keys = sorted(k for k in sd if k.startswith("target_encoder."))

    if enc_keys and tgt_keys:
        diffs = []
        for ek, tk in zip(enc_keys, tgt_keys):
            e_suffix = ek[len("encoder.") :]
            t_suffix = tk[len("target_encoder.") :]
            if e_suffix == t_suffix and sd[ek].shape == sd[tk].shape:
                d = (sd[ek] - sd[tk]).abs().mean().item()
                diffs.append(d)
        if diffs:
            mean_diff = sum(diffs) / len(diffs)
            print(f"\n✅ Encoder ↔ Target Encoder divergence ({len(diffs)} matching tensors):")
            print(f"     mean abs diff: {mean_diff:.6f}")
            if mean_diff < 1e-7:
                print("⚠️  Encoders are identical — EMA may not have updated!")
            elif mean_diff > 0.1:
                print("⚠️  Encoders very different — EMA momentum may be too low")
            else:
                print("     ✓ Reasonable EMA divergence (encoders trained, but coupled)")

    # Training info from checkpoint
    print("\n" + "=" * 60)
    print("TRAINING METADATA")
    print("=" * 60)
    if "epoch" in ckpt:
        print(f"  Epoch         : {ckpt['epoch']}")
    if "global_step" in ckpt:
        print(f"  Global step   : {ckpt['global_step']}")
    if "callbacks" in ckpt:
        for k, v in ckpt["callbacks"].items():
            if isinstance(v, dict) and "best_model_score" in v:
                print(f"  Best val_loss : {v['best_model_score'].item():.6f}")
                print(f"  Best ckpt path: {v.get('best_model_path', 'n/a')}")
                break

    print()


if __name__ == "__main__":
    main()
