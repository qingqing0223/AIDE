# Reproducibility checklist

Record before each final run:
- Git commit SHA
- AudioLDM checkpoint SHA-256 and source revision
- AudioCaps split/manifest SHA-256
- VCTK 0.92 microphone and manifest SHA-256
- prompt manifest SHA-256 and six-class mapping
- epsilon, PGD alpha, PGD steps, masking gamma
- PSM architecture and optimizer
- DDIM steps, guidance scale, control mode and seed
- GPU/CUDA/PyTorch versions
- output audio manifest
- all metric JSON files

Never overwrite a prior run. Use a timestamped run directory.
