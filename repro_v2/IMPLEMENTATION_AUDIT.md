# AIDE v2 implementation audit

This file maps the KSEM 2026 paper to the reconstructed code and separates implemented behavior from implementation assumptions that still require experiment-level validation.

## Paper-to-code map

| Paper component | Code location | Implemented behavior |
|---|---|---|
| Stage I input audio and target pool | aide_repro/audio.py, aide_repro/data.py, aide_repro/pgd.py | Fixed target pool, deterministic target choice per source ID, sample-rate/length normalization, source/target SHA256 |
| Encoder-latent steering and separation | aide_repro/pgd.py, aide_repro/backend.py | AudioLDM VAE posterior mode for clean, target and protected latents; mean-reduced squared-distance objective |
| Perturbation budget | aide_repro/pgd.py | Sign-PGD descent, L-infinity projection, waveform clipping |
| Psychoacoustic masking term | aide_repro/pgd.py | Differentiable local time-frequency masking envelope proxy |
| Prompt category, scalar gate, channel gate | aide_repro/psm.py | Shared RoBERTa representation; seven-way classifier; scalar gate; 8-channel gate |
| Eq. (4) latent fusion | aide_repro/losses.py, aide_repro/trainer.py, aide_repro/inference.py | Shape-checked clean/protected latent interpolation before forward noising and reverse DDIM |
| Eq. (5) classification loss | aide_repro/losses.py | Seven-class cross entropy |
| Eq. (6) gate calibration | aide_repro/losses.py | Benign scalar gate toward zero; harmful scalar gate toward one |
| Eq. (7) generation-coupled loss | aide_repro/losses.py, aide_repro/backend.py, aide_repro/trainer.py | Paired clean/gated generation with shared noise; CLAP cosine similarity; differentiable DDIM/VAE/vocoder/audio-CLAP path |
| Eq. (8) PSM objective | aide_repro/trainer.py | Weighted sum of classification, gate calibration and generation-coupled losses |
| OWM incremental learning | aide_repro/owm.py, aide_repro/psm.py, aide_repro/trainer.py | RLS-style projection update, saved projection state, classifier expansion, old/new class F1 |
| Deployment | aide_repro/inference.py, aide_repro/benchmark.py | Prompt-specific gates, same-seed paired outputs, JSON sidecars with checkpoint/input hashes |
| Evaluation | aide_repro/metrics.py | CLAP/H-CLAP, FD/IS/KL via vendored AudioLDM evaluator, SNR/LSD |

## Explicit implementation assumptions

The paper does not fully specify all of the following. These choices are made explicit rather than presented as confirmed paper facts:

1. The paper's psychoacoustic threshold estimator is not fully disclosed. The implementation uses a local max-envelope threshold proxy in the STFT domain.
2. The paper writes squared L2 distances but does not specify the reduction. The implementation uses mean-reduced squared distances for stable optimization.
3. The exact names and source-label mapping of all six harmful prompt subclasses are not fully enumerated in the available paper text. The taxonomy template therefore requires explicit AART and RED-INSTRUCT mappings and rejects unknown categories.
4. The paper defines a channel-wise gate but does not provide a fully specified tensor interface. The implementation uses one gate value per VAE channel (8 for the selected AudioLDM checkpoint), broadcast across time/frequency.
5. The paper does not specify the exact source-latent injection point in AudioLDM. The implementation uses the official AudioLDM style-transfer path: encode clean/protected audio, interpolate VAE latents with Eq. (4), forward-noise the result, then run text-conditioned DDIM.
6. Clean/protected VAE latents use the deterministic posterior mode. DDIM uses eta=0 and a shared noise tensor for paired comparisons.
7. The paper does not publish every gate-loss coefficient. The implementation defaults lambda_gate, lambda_generation, mu_benign and mu_harmful to 1.0. PGD defaults and masking-proxy coefficient are recorded in run_config.json.
8. OWM projection is applied to the PSM shared/head layers. During classifier expansion with OWM, the pretrained RoBERTa backbone is frozen to bound memory. The projection update uses ridge=1 and at most 128 activation rows per module update.

## Validation status

### Verified in repository CI

- Python AST parsing for every package and test file.
- Bash syntax checks for all launch scripts.
- JSON configuration parsing.
- Six CPU unit tests: Eq. (7) loss, Eq. (6) gate calibration, masking-proxy finiteness/gradient, waveform length fitting, OWM projection update, and Eq. (4) fusion endpoints/shape guard.

### Not yet verified on the target AutoDL/A100 environment

- Loading the actual AudioLDM checkpoint and confirming the exact latent tensor shape on the selected machine.
- One-sample VAE encode/decode and audio listening check.
- PGD one-sample objective trajectory and gradient checks with the real VAE.
- End-to-end CLAP gradient check through the real DDIM, VAE decoder, vocoder and CLAP audio branch.
- One PSM generation-coupled training batch on the target GPU.
- Full AudioCaps/VCTK experiments and FD/IS/KL evaluation with the selected pretrained evaluator weights.

Do not report the second list as passed until the corresponding AutoDL logs, JSON outputs and artifact hashes have been inspected.

## Recommended first execution gate

1. Install the pinned environment and verify imports.
2. Run Stage I with sample-limit 1.
3. Confirm PGD gradient is finite, perturbation L-infinity is within epsilon, and the per-step loss history is present.
4. Run gradient-check on that sample. Stop if the CLAP-to-latent gradient is missing, zero or non-finite.
5. Train one tiny PSM batch with generation coupling enabled. Confirm L_cls, L_gate, L_gen and scalar/channel gate gradients are finite.
6. Only then launch the complete train/test pipeline.
