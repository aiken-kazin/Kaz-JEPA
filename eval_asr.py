"""Evaluate a fine-tuned ASR checkpoint on the KSC2 Test split.

Loads the trained ASR model, runs greedy CTC decoding over the Test split
(or any split passed via --split), and reports:

  - Word Error Rate (WER)
  - Character Error Rate (CER)
  - Sample REF / HYP pairs

Both metrics are computed with our own Levenshtein implementation (no
external dependency), so the script works without jiwer installed.

Usage:
  python eval_asr.py --ckpt logs/asr_pretrained/checkpoints/last.ckpt \
      [--manifest data/KSC2/asr_manifest.csv] \
      [--vocab data/KSC2/vocab.json] \
      [--split Test] [--n 1000] [--device cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from asr_data import KSC2AsrDataset, KazakhCharTokenizer, collate_asr_batch
from asr_model import KazJepaForCTC, build_encoder


# -----------------------------------------------------------------------------
# Levenshtein-based WER / CER (no external dep)
# -----------------------------------------------------------------------------
def _levenshtein(a: list, b: list) -> int:
    """Minimal edit distance between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,           # deletion
                curr[j - 1] + 1,       # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = curr
    return prev[-1]


def compute_wer(refs: list[str], hyps: list[str]) -> tuple[float, int, int]:
    total_words, total_errs = 0, 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        total_words += len(rw)
        total_errs += _levenshtein(rw, hw)
    return (total_errs / max(1, total_words)), total_errs, total_words


def compute_cer(refs: list[str], hyps: list[str]) -> tuple[float, int, int]:
    total_chars, total_errs = 0, 0
    for r, h in zip(refs, hyps):
        total_chars += len(r)
        total_errs += _levenshtein(list(r), list(h))
    return (total_errs / max(1, total_chars)), total_errs, total_chars


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def load_model_from_ckpt(ckpt_path: Path, vocab_size: int, device: str) -> KazJepaForCTC:
    encoder = build_encoder()
    model = KazJepaForCTC(encoder, vocab_size=vocab_size)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    # Strip Lightning's "model." prefix
    model_sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
    if not model_sd:
        model_sd = sd  # already raw model state
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing:
        print(f"  missing {len(missing)} keys: {missing[:3]}")
    if unexpected:
        print(f"  unexpected {len(unexpected)} keys: {unexpected[:3]}")
    return model.to(device).eval()


@torch.no_grad()
def evaluate(
    model: KazJepaForCTC,
    tokenizer: KazakhCharTokenizer,
    dataloader: DataLoader,
    device: str,
    n_show: int = 5,
) -> tuple[list[str], list[str]]:
    refs: list[str] = []
    hyps: list[str] = []

    for batch in dataloader:
        spec = batch["spec"].to(device)
        log_probs = model(spec)              # (B, T, V)
        preds = log_probs.argmax(dim=-1)     # (B, T)
        for pred_ids, ref_text in zip(preds, batch["texts"]):
            hyp = tokenizer.ctc_decode(pred_ids.tolist())
            refs.append(ref_text)
            hyps.append(hyp)

    print("\n=== SAMPLE PREDICTIONS ===")
    for i in range(min(n_show, len(refs))):
        print(f"\n[{i}] REF: {refs[i]}")
        print(f"    HYP: {hyps[i]}")

    return refs, hyps


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("data/KSC2/asr_manifest.csv"))
    p.add_argument("--vocab", type=Path, default=Path("data/KSC2/vocab.json"))
    p.add_argument("--split", default="Test", choices=["Train", "Dev", "Test"])
    p.add_argument("--n", type=int, default=None, help="Evaluate on N examples (default: full split)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print(f"[1/4] Loading tokenizer from {args.vocab}")
    tokenizer = KazakhCharTokenizer.from_json(args.vocab)

    print(f"[2/4] Loading model from {args.ckpt}")
    model = load_model_from_ckpt(args.ckpt, tokenizer.vocab_size, args.device)

    print(f"[3/4] Building {args.split} dataset")
    ds = KSC2AsrDataset(args.manifest, tokenizer, split=args.split)
    if args.n is not None:
        ds = Subset(ds, list(range(min(args.n, len(ds)))))
    print(f"      {len(ds)} examples")
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_asr_batch,
        pin_memory=True,
    )

    print(f"[4/4] Running inference & decoding")
    refs, hyps = evaluate(model, tokenizer, dl, args.device)

    wer, w_err, w_tot = compute_wer(refs, hyps)
    cer, c_err, c_tot = compute_cer(refs, hyps)

    print("\n" + "=" * 60)
    print(f"RESULTS on {args.split} ({len(refs)} examples)")
    print("=" * 60)
    print(f"  WER : {wer*100:6.2f}%  ({w_err} / {w_tot} words)")
    print(f"  CER : {cer*100:6.2f}%  ({c_err} / {c_tot} chars)")
    print()


if __name__ == "__main__":
    main()
