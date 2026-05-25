"""ASR model: pretrained JEPA encoder + linear CTC head.

Loads the encoder weights from a JEPA checkpoint, drops the predictor and
target_encoder, and adds a single Linear head that maps each patch
embedding to a vocab distribution. Outputs log-probabilities ready for
nn.CTCLoss.

Audio-JEPA's encoder gives 192 patches per 10-second clip (32 time
patches × 6 frequency patches at 16×16 patch size on a 512×96 mel-spec).
We treat all 192 patches as the CTC time axis — this gives ~52 ms per
frame, more than enough resolution for char-level CTC over Kazakh text
(median 74 chars per 10 s).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.components.vision_transformer import VisionTransformer


def build_encoder(
    input_size: tuple[int, int] = (512, 96),
    patch_size: tuple[int, int] = (16, 16),
    in_chans: int = 1,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
) -> VisionTransformer:
    """Construct the same VisionTransformer used in pretraining."""
    return VisionTransformer(
        input_size=input_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )


def load_jepa_encoder_weights(encoder: nn.Module, ckpt_path: Path) -> None:
    """Load encoder.* weights from a JEPA checkpoint into the given module.

    The JEPA Lightning checkpoint has keys like:
        encoder.patch_embed.proj.weight
        encoder.blocks.0.attn.mha.in_proj_weight
        target_encoder.*           ← skipped
        predictor.*                ← skipped
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    encoder_sd = {
        k[len("encoder.") :]: v
        for k, v in sd.items()
        if k.startswith("encoder.")
    }
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing:
        print(f"[load_jepa_encoder_weights] missing {len(missing)} keys (sample: {missing[:3]})")
    if unexpected:
        print(f"[load_jepa_encoder_weights] unexpected {len(unexpected)} keys (sample: {unexpected[:3]})")
    print(f"[load_jepa_encoder_weights] loaded {len(encoder_sd)} tensors from {ckpt_path}")


class KazJepaForCTC(nn.Module):
    """JEPA encoder + linear head over patch embeddings.

    Forward:
        spec       : (B, 1, time_bins, n_mels)
        returns    : (B, T_patches, vocab_size) log-softmaxed
                     where T_patches = num_patches_h * num_patches_w
    """

    def __init__(self, encoder: VisionTransformer, vocab_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        # Find embed_dim from the encoder
        embed_dim = encoder.patch_embed.proj.out_channels
        self.head = nn.Linear(embed_dim, vocab_size)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(spec)
        if isinstance(feats, tuple):
            feats = feats[0]
        # feats: (B, num_patches, embed_dim)
        logits = self.head(feats)
        return logits.log_softmax(dim=-1)
