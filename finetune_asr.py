"""Fine-tune the pretrained JEPA encoder for Kazakh ASR via CTC.

End-to-end script:
  1. Loads char tokenizer from data/KSC2/vocab.json
  2. Builds KSC2 ASR Dataset for Train and Dev splits
  3. Constructs KazJepaForCTC = pretrained encoder + linear head
  4. Lightning trainer with CTCLoss
  5. Periodically prints WER/CER on a small dev subset

Usage:
  python finetune_asr.py --pretrained_ckpt path/to/last.ckpt \
      [--manifest data/KSC2/asr_manifest.csv] \
      [--vocab data/KSC2/vocab.json] \
      [--max-steps 20000] [--batch-size 16] [--lr 1e-4] \
      [--freeze-encoder]

Run with --random-init instead of --pretrained_ckpt to train the baseline
(same architecture but encoder weights are random — proves pretraining helps).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from torch.utils.data import DataLoader, Subset

from asr_data import KSC2AsrDataset, KazakhCharTokenizer, collate_asr_batch
from asr_model import KazJepaForCTC, build_encoder, load_jepa_encoder_weights


# -----------------------------------------------------------------------------
# Lightning module wrapping model + CTC training step
# -----------------------------------------------------------------------------
class AsrLightning(L.LightningModule):
    def __init__(
        self,
        model: KazJepaForCTC,
        tokenizer: KazakhCharTokenizer,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        log_decode_every_n: int = 200,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.log_decode_every_n = log_decode_every_n
        self.ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
        self.save_hyperparameters(ignore=["model", "tokenizer"])

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.model(spec)

    def _ctc_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = self(batch["spec"])                       # (B, T, V)
        log_probs_t = log_probs.transpose(0, 1).contiguous()  # (T, B, V) for CTCLoss
        T = log_probs_t.size(0)
        B = log_probs_t.size(1)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)
        loss = self.ctc_loss(log_probs_t, batch["targets"], input_lengths, batch["target_lengths"])
        return loss, log_probs

    def training_step(self, batch, batch_idx):
        loss, _ = self._ctc_step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_probs = self._ctc_step(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Greedy decode a couple of examples for visual sanity
        if batch_idx == 0 and self.global_step % self.log_decode_every_n == 0:
            preds = log_probs.argmax(dim=-1)
            for i in range(min(2, preds.size(0))):
                hyp = self.tokenizer.ctc_decode(preds[i].tolist())
                ref = batch["texts"][i]
                print(f"\n  REF: {ref[:80]}")
                print(f"  HYP: {hyp[:80]}")
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # Linear warmup then constant — simple, robust for ASR fine-tuning.
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            return 1.0
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/KSC2/asr_manifest.csv"))
    p.add_argument("--vocab", type=Path, default=Path("data/KSC2/vocab.json"))

    pre = p.add_mutually_exclusive_group(required=True)
    pre.add_argument("--pretrained_ckpt", type=Path, help="Path to pretrained JEPA checkpoint")
    pre.add_argument("--random-init", action="store_true", help="Baseline: random encoder weights")

    p.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder, train only the head")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--accelerator", default="gpu")
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--limit-train", type=int, default=None,
                   help="Use only N training examples (for fast iteration)")
    p.add_argument("--limit-dev", type=int, default=500,
                   help="Use only N dev examples for validation (faster val loop)")
    p.add_argument("--output-dir", type=Path, default=Path("logs/asr"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(42)

    print(f"[1/4] Loading tokenizer from {args.vocab}")
    tokenizer = KazakhCharTokenizer.from_json(args.vocab)
    print(f"      vocab size: {tokenizer.vocab_size}")

    print(f"[2/4] Building datasets from {args.manifest}")
    train_ds_full = KSC2AsrDataset(args.manifest, tokenizer, split="Train")
    dev_ds_full = KSC2AsrDataset(args.manifest, tokenizer, split="Dev")
    print(f"      train: {len(train_ds_full)} | dev: {len(dev_ds_full)}")

    train_ds = train_ds_full
    if args.limit_train:
        idx = random.sample(range(len(train_ds_full)), min(args.limit_train, len(train_ds_full)))
        train_ds = Subset(train_ds_full, idx)
        print(f"      using {len(train_ds)} train examples (--limit-train)")

    dev_ds = dev_ds_full
    if args.limit_dev:
        idx = random.sample(range(len(dev_ds_full)), min(args.limit_dev, len(dev_ds_full)))
        dev_ds = Subset(dev_ds_full, idx)
        print(f"      using {len(dev_ds)} dev examples (--limit-dev)")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_asr_batch,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    dev_dl = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_asr_batch,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    print(f"[3/4] Building model")
    encoder = build_encoder()
    if args.pretrained_ckpt:
        load_jepa_encoder_weights(encoder, args.pretrained_ckpt)
    else:
        print("      using random-init encoder (BASELINE)")

    model = KazJepaForCTC(encoder, vocab_size=tokenizer.vocab_size)
    if args.freeze_encoder:
        model.freeze_encoder()
        print("      encoder frozen — training only the head")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"      trainable: {n_train/1e6:.2f} M / total: {n_total/1e6:.2f} M")

    lit = AsrLightning(
        model=model,
        tokenizer=tokenizer,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
    )

    print(f"[4/4] Starting training")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(args.output_dir / "checkpoints"),
        filename="{step:06d}-{val/loss:.4f}",
        save_top_k=2,
        monitor="val/loss",
        mode="min",
        save_last=True,
        every_n_train_steps=2000,
    )

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_steps=args.max_steps,
        precision=args.precision,
        default_root_dir=str(args.output_dir),
        callbacks=[ckpt_cb, RichProgressBar()],
        val_check_interval=2000,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, train_dl, dev_dl)
    print("Done. Best ckpt:", ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()
