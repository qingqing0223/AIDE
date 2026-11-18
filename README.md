# AIDE — KSEM 2026 clean reconstruction

This branch is the rebuild line for **Audio Immunization Against Harmful Audio Editing with Diffusion Models (AIDE)**.

The complete historical implementation is preserved under `archive/legacy/2026-11-original/`. New code never imports from that directory.

## Method implemented
- Encoder-latent immunization with PGD.
- Paper objective: `||z_p-z_tar||² - lambda_f ||z_p-z_c||² + lambda_m L_m`.
- Explicit `||delta||_inf <= epsilon` projection; default `epsilon=0.01`.
- Psychoacoustic masking penalty.
- Prompt Safety Module with seven classes: one benign + six harmful subsets.
- Scalar gate and channel-wise gate.
- Controlled condition: `(1-s*g) * z_c + (s*g) * z_p`.
- Frozen AudioLDM generator and DDIM generation.
- AudioCaps/VCTK manifests, PSM manifests, signal metrics, run records and tests.

## What is deliberately supplemented
The paper does not disclose every engineering detail. The rebuild exposes these as configuration rather than hiding them:
- target-pool construction and sampling;
- psychoacoustic threshold estimator;
- PSM hidden width and optimizer;
- scalar/channel gate stabilization weight;
- DDIM step count;
- exact dataset file layout.

## Data
Raw AudioCaps/VCTK audio and large model checkpoints are not committed to GitHub. The repository contains preparation code, manifests, source information and checksums.

## Quick start
```bash
pip install -r requirements.txt
pip install -r requirements-audioldm.txt
pip install -e .
python scripts/check_environment.py
pytest -q

# optional FD/IS/KL evaluation
pip install -r requirements-eval.txt
```

The first GPU pass should validate the AudioLDM checkpoint, differentiable first-stage encoder, PSM condition shape and DDIM loop before launching the full experiment grid.
