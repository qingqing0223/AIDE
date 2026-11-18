# Implementation notes

Paper-backed:
- frozen AudioLDM;
- PGD through encoder;
- epsilon=0.01;
- target, clean-separation and masking terms;
- seven prompt classes;
- scalar and channel gates;
- controlled condition interpolation.

Supplements:
1. The official AudioLDM wav_to_fbank path converts to NumPy and detaches, so the rebuild uses a differentiable torch STFT/mel path before the frozen first-stage encoder.
2. Target-pool construction is externalized.
3. RoBERTa-base + 256 hidden units is the default PSM engineering choice.
4. Channel-gate stabilization is an auxiliary implementation term.
5. No undocumented UNet timestep injection is made the default; condition fusion happens before the standard AudioLDM DDIM loop.
6. Raw datasets and checkpoints are excluded from Git history.
