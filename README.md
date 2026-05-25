# Kaz-JEPA: Audio-JEPA Pretrained for Kazakh Speech

The first self-supervised speech representation model for the **Kazakh** language, adapted from [Audio-JEPA](https://github.com/LudovicTuncay/Audio-JEPA) (Tuncay et al., ICME 2025).

We pretrain a Vision Transformer encoder on **1,744 hours** of unlabeled Kazakh speech from the [KSC2 corpus (ISSAI)](https://issai.nu.edu.kz/) using the Joint-Embedding Predictive Architecture (JEPA), then release the pretrained encoder as a foundation for downstream Kazakh speech tasks.

## Highlights

- 🇰🇿 **First** JEPA-based SSL model for Kazakh speech
- 📊 **+14 pp** improvement over random initialization on KSC2 domain classification (linear probe)
- 🧊 Trained on a single **RTX 5090** in ~3 hours
- 🔓 Open source code + checkpoints (MIT)

## Setup

```bash
git clone https://github.com/aiken-kazin/Kaz-JEPA.git
cd Kaz-JEPA

# Recommended: uv
uv sync --no-install-package flash-attn

# Or pip
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

For RTX 5090 (Blackwell, sm_120) install the CUDA 12.8 PyTorch build:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Reproducing the pretraining

### 1. Convert FLAC → HDF5

```bash
PROJECT_ROOT=$PWD python convert_to_hdf5.py \
    --src /path/to/ISSAI_KSC2 \
    --out data/KSC2
```

### 2. Run pretraining (paper-matching config)

```bash
PROJECT_ROOT=$PWD nohup python src/train.py data=ksc2 trainer=gpu \
    trainer.max_steps=100000 \
    data.batch_size=64 \
    +trainer.accumulate_grad_batches=4 \
    data.num_workers=8 \
    tags='[kazakh,jepa]' test=False > training.log 2>&1 &
```

This matches Audio-JEPA's original setup (effective batch 256, 128 mel × 256 time bins).

## Evaluation utilities

```bash
# Pair FLAC files with transcripts → manifest CSV
python prepare_ksc2_asr.py --src /path/to/ISSAI_KSC2

# Build the Kazakh char vocabulary
python asr_data.py build-vocab --manifest data/KSC2/asr_manifest.csv

# kNN domain classification
python evaluate_knn.py --ckpt <ckpt> --per-domain-train 500 --per-domain-test 100

# MLP linear probing
python linear_probe.py --ckpt <ckpt> --per-domain-train 500 --per-domain-test 100

# Content-similarity (label-free): does the encoder track transcript content?
python evaluate_content_similarity.py --ckpt <ckpt>

# PCA + t-SNE visualization
python visualize_embeddings.py --ckpt <ckpt> --tag mymodel

# Quick sanity check (weights healthy, no collapse, EMA worked)
python quick_test_jepa.py --ckpt <ckpt>
```

## Using the pretrained encoder in your own work

```python
import torch
from asr_model import build_encoder, load_jepa_encoder_weights

encoder = build_encoder(input_size=(256, 128))  # paper config
load_jepa_encoder_weights(encoder, "path/to/last.ckpt")
encoder.eval()

# Encode a mel-spectrogram of shape (B, 1, 256, 128):
with torch.no_grad():
    features = encoder(mel_spec)        # (B, 128, 768)
    pooled   = features.mean(dim=1)     # (B, 768) — one embedding per clip
```

Pre-computed embeddings work well for: domain / speaker / dialect classification, similarity retrieval, and as a feature extractor for downstream Kazakh speech tasks.

For character-level CTC ASR, the patch-grid temporal resolution is too coarse — see the discussion in our preprint (forthcoming).

## Citation

If you use this repo or the released checkpoints, please cite **both** the original Audio-JEPA paper (for the method) and this Kazakh adaptation:

```bibtex
@misc{kazin2026kazjepa,
  title  = {Kaz-JEPA: Audio-JEPA Pretrained for Kazakh Speech},
  author = {Aiken Kazin},
  year   = {2026},
  howpublished = {\url{https://github.com/aiken-kazin/Kaz-JEPA}}
}

@inproceedings{tuncay2025audiojepa,
  title     = {Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning},
  author    = {Tuncay, Ludovic and Labb\'e, Etienne and Benetos, Emmanouil and Pellegrini, Thomas},
  booktitle = {Proc.\ IEEE International Conference on Multimedia and Expo (ICME)},
  year      = {2025}
}
```

## Acknowledgments

- [Audio-JEPA](https://github.com/LudovicTuncay/Audio-JEPA) — Tuncay, Labbé, Benetos, Pellegrini (ICME 2025). The codebase this work is built on.
- [ISSAI](https://issai.nu.edu.kz/) — for releasing the Kazakh Speech Corpus 2 (KSC2).

## License

MIT, matching upstream Audio-JEPA.
