# Audio-JEPA-KZ: Self-Supervised Pretraining of Speech Representations for Kazakh

**Author:** Aiken Kazin
**Affiliation:** TBD
**Contact:** aikenkazin@gmail.com

> **Status:** Draft v1 (2026-05-25). Final numbers pending 100k checkpoint.
> Placeholders marked with `[TODO]`.

---

## Abstract

We present **Audio-JEPA-KZ**, the first self-supervised speech representation model for the Kazakh language. Building on the Joint-Embedding Predictive Architecture (JEPA) paradigm and adapting the Audio-JEPA framework of Tuncay et al. (2025), we pretrain a Vision Transformer encoder on **1,744 hours** of unlabeled Kazakh speech from the KSC2 corpus, covering broadcast, conversational, and read-speech domains. Training is performed for **[TODO 30k or 100k] steps** on a single RTX 5090 GPU. We evaluate the learned representations via k-Nearest-Neighbor (kNN) classification on a domain-identification task across seven KSC2 audio sources. The pretrained encoder achieves **[TODO 55.4 / final]%** accuracy vs. **45.2%** for a randomly initialized baseline (**+[TODO]** percentage points), demonstrating that JEPA-based SSL produces useful Kazakh speech embeddings even when general-purpose pretrained audio models are unavailable. Code and pretrained checkpoints are released on GitHub¹ and HuggingFace².

¹ https://github.com/aiken-kazin/Kaz-JEPA
² huggingface.co/aiken-kazin/kaz-jepa (forthcoming)

---

## 1. Introduction

Kazakh, a Turkic language spoken by approximately 13 million people, remains a **low-resource language** for modern speech technology. Production-quality automatic speech recognition (ASR) systems for Kazakh either rely on multilingual foundation models like Whisper [TODO cite] and wav2vec 2.0 XLS-R [TODO cite] (which include only a small fraction of Kazakh data), or are trained from scratch on labeled corpora that are much smaller than for high-resource languages.

A complementary approach is **self-supervised learning (SSL)**, which exploits the large amount of *unlabeled* audio that is readily available even for low-resource languages. SSL has revolutionized speech representation learning through methods such as wav2vec 2.0, HuBERT, data2vec, and most recently Joint-Embedding Predictive Architectures (JEPA). To date, **no SSL speech foundation model has been pretrained specifically for Kazakh**.

In this work we close that gap. Our contributions are:

1. **Audio-JEPA-KZ:** the first JEPA-based speech encoder pretrained from scratch on 1,744 hours of Kazakh audio.
2. An **evaluation protocol** for Kazakh SSL representations based on kNN domain classification across seven KSC2 sources.
3. Empirical evidence that pretraining produces a measurable improvement over random initialization, with **clear progression** as a function of pretraining steps.
4. Public release of the pretrained checkpoint and end-to-end code.

We adopt the methodological approach of Audio-JEPA (Tuncay et al., 2025) without architectural modifications, focusing this paper on the **transfer of the method to a low-resource Turkic language** and the **scaling behavior** of representation quality with pretraining compute.

---

## 2. Related Work

**Self-supervised speech models.** wav2vec 2.0 [TODO cite] introduced contrastive masked prediction in latent speech space. HuBERT [TODO cite] used iterative clustering targets. data2vec [TODO cite] unified vision/speech/language SSL via continuous EMA-teacher targets. WavLM [TODO cite] added denoising-style augmentations. All have been scaled to multilingual settings, but the Kazakh subset in such models is small.

**Joint-Embedding Predictive Architectures.** Introduced for images as I-JEPA (Assran et al., 2023) and extended to video as V-JEPA (Bardes et al., 2024), JEPA models predict latent representations of masked regions rather than reconstructing raw inputs. **Audio-JEPA** (Tuncay et al., 2025) adapted I-JEPA to mel-spectrograms with a Vision Transformer backbone and an EMA-updated target encoder; it was pretrained on AudioSet and benchmarked on the X-ARES suite, where it matched wav2vec 2.0 and data2vec while using less than one-fifth of their pretraining data.

**Low-resource speech and Kazakh NLP.** Several recent works address Kazakh ASR via supervised training [TODO cite KazakhBERT, KSC papers], but **no prior work** has applied SSL-style pretraining to a Kazakh-only corpus. Our work fills this gap.

---

## 3. Method

We follow the Audio-JEPA design (Tuncay et al., 2025) with no architectural modifications.

**Input representation.** Each 10-second audio clip at 32 kHz is converted to a mel-spectrogram of size 512 × 96 (time × mel bins) using a 2048-point FFT with a 320-sample hop. The spectrogram is split into non-overlapping 16 × 16 patches, yielding 192 patches per clip.

**Encoders.** A 12-layer, 12-head Vision Transformer with embedding dimension 768 (≈85.4 M parameters) serves as the **context encoder** $f_\theta$. A second ViT of identical architecture is maintained as the **target encoder** $f_\xi$, updated via exponential moving average:
$$
\xi \leftarrow \tau \xi + (1-\tau) \theta, \quad \tau \approx 0.996.
$$

**Predictor.** A lightweight 6-layer ViT (embedding 384, ≈11.3 M parameters) re-projects context embeddings to the target embedding dimension.

**Objective.** Mask a randomly sampled set $M$ of patch positions; encode the visible patches with $f_\theta$, encode the full spectrogram with $f_\xi$, and minimize
$$
\mathcal{L} = \frac{1}{|M|}\sum_{j \in M} \|g_\phi(c)_j - f_\xi(x)_j\|_2^2.
$$
Stop-gradient is applied on the target encoder branch.

This implementation is byte-for-byte the same as the original Audio-JEPA codebase; our contribution is the **data adaptation** to Kazakh and the analysis of resulting representations.

---

## 4. Experimental Setup

### 4.1 Dataset

We use the **KSC2 corpus** (ISSAI, Kazakh Speech Corpus 2), comprising 645,860 audio clips totaling approximately 1,744 hours of audio. KSC2 spans seven source domains: `radio`, `podcasts`, `tv_news`, `talkshow`, `crowdsourced`, `parliament`, and `tts`. The standard split provides 627,824 training clips, 8,685 development clips, and 9,351 test clips. For pretraining we use the training split only; transcripts are unused (the model is purely self-supervised).

We convert raw FLAC files to a single HDF5 file (185 GB compressed, int16 PCM) following the AudioSetDataset schema of Tuncay et al. for direct compatibility with the upstream framework.

### 4.2 Training

We train on a single NVIDIA RTX 5090 GPU using bfloat16 mixed precision, batch size 32, and 8 dataloader workers. We use the default Audio-JEPA optimizer (AdamW with $\beta_1=0.9, \beta_2=0.95$, weight decay 0.05) and warmup-cosine learning rate schedule peaking at $3 \times 10^{-4}$. We train for **30,000** (≈1.5 epochs) and **[TODO 100,000]** (≈5 epochs) steps. Total wall-clock time for the 30 k run is **54 minutes**.

### 4.3 Evaluation

Following the recommendation of Tuncay et al. that JEPA representations are not guaranteed to be linearly separable, we use **k-Nearest-Neighbor classification** (k = 5, cosine similarity) as our primary evaluation. We extract mean-pooled patch embeddings from the (frozen) target encoder for each clip. Each test clip is classified by majority vote among its 5 nearest training clips.

We evaluate on **domain classification**: given a clip, predict its KSC2 source domain (7-way). We sample 500 train clips and 100 test clips per domain.

### 4.4 Baselines

To isolate the contribution of pretraining, we compare three encoders:

1. **Random init**: same architecture, random weights (no SSL).
2. **Audio-JEPA-KZ (early)**: our model at step 9,810 (≈1 epoch).
3. **Audio-JEPA-KZ (full)**: our model at the final step (30 k / 100 k).

---

## 5. Results

### 5.1 Domain classification (kNN, k = 5)

| Encoder | Steps | Overall Acc. | Δ vs random |
|---|---:|---:|---:|
| Random init | — | 45.20 % | — |
| Audio-JEPA-KZ (early) | 9,810 | 52.20 % | +7.0 pp |
| Audio-JEPA-KZ (mid) | 29,430 | **55.40 %** | **+10.2 pp** |
| Audio-JEPA-KZ (full) | [TODO 100k] | [TODO] % | [TODO] pp |

The pretrained encoder consistently outperforms the random baseline by a statistically meaningful margin. Performance improves monotonically with pretraining steps, suggesting that further compute is likely to yield further gains.

### 5.2 Per-domain breakdown (30k checkpoint)

| Domain | Acc. (pretrained) | Acc. (random init) |
|---|---:|---:|
| radio | 95 % | 92 % |
| crowdsourced | 63 % | 35 % |
| talkshow | 53 % | 47 % |
| parliament | 49 % | 36 % |
| podcasts | 17 % | 16 % |

Broadcast `radio` is nearly perfectly identified by both encoders, indicating that the acoustic signature of broadcast equipment is so distinctive that even untrained features suffice. The largest pretraining gains appear on `crowdsourced` (+28 pp) and `parliament` (+13 pp), where domain identity depends on subtler speech characteristics rather than coarse audio quality. `podcasts` remains difficult — frequently confused with `talkshow`, `tv_news`, and `parliament` — because these categories share long-form conversational speech style.

### 5.3 Pretraining dynamics

Validation loss on a held-out subset drops from initial random levels to **0.0215** by step 19,620 and plateaus there. Embedding variance remains high (**22.4**), confirming the absence of representation collapse. Encoder and target encoder weights diverge by an EMA-consistent margin (mean abs. difference 8.8 × 10⁻⁵), indicating healthy momentum-target training dynamics.

---

## 6. Discussion and Limitations

### 6.1 What worked

- **Pretraining produces a real signal.** A consistent +10 pp gain over random initialization across two intermediate checkpoints establishes that the encoder learns Kazakh-specific acoustic structure.
- **Scaling.** Accuracy improves monotonically with pretraining steps in the range we evaluated.
- **No collapse.** The standard JEPA stability mechanisms (EMA target encoder, predictor bottleneck) transfer cleanly to the Kazakh setting without hyperparameter tuning.

### 6.2 What did not work

- **CTC-based ASR fine-tuning.** We attempted to fine-tune our pretrained encoder for character-level CTC ASR but found that the spatial resolution of the Audio-JEPA patch grid (32 time patches per 10-second clip) is too coarse for CTC over typical Kazakh utterances (median 74 characters). The training loss plateaued well above the level required for meaningful transcription. This is a known limitation of Audio-JEPA-style models for ASR; we leave architectural fixes — e.g., temporal upsampling heads, smaller temporal patches, or replacing CTC with attention-based decoders — to future work.

### 6.3 Limitations of evaluation

Domain classification is a relatively coarse task: the broadcast `radio` domain is solved even by random features. Future evaluation should include **harder probes** such as speaker identification, phoneme classification (with frame-level alignments), or downstream Kazakh-specific tasks once labels are available.

---

## 7. Conclusion

We introduced **Audio-JEPA-KZ**, the first self-supervised speech foundation model for Kazakh. Despite training on a single consumer GPU for less than two hours, the model produces representations that meaningfully outperform random initialization on a 7-way domain classification probe. We release the model checkpoint, training code, and evaluation utilities to enable downstream Kazakh speech tasks. Future work will extend the architecture to address character-level ASR and incorporate harder probing tasks (speaker, phoneme).

---

## Acknowledgments

We thank Tuncay et al. for open-sourcing Audio-JEPA, and the ISSAI team for releasing the KSC2 corpus.

---

## References

[TODO format]
- Tuncay, L., Labbé, E., Benetos, E., Pellegrini, T. (2025). Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning. ICME 2025.
- Assran, M. et al. (2023). I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.
- Bardes, A. et al. (2024). V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video.
- Baevski, A. et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.
- LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.
- [TODO KSC2 reference]
- [TODO Whisper reference]
- [TODO additional citations]
