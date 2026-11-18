# Implementation notes

## Paper-backed
- AudioLDM is the frozen generation backbone.
- Encoder-latent immunization uses target attraction, clean-separation and psychoacoustic masking terms.
- epsilon=0.01.
- PSM has seven prompt classes, scalar gate and channel-wise gate.
- PSM losses are L_cls, L_gate and generation-coupled L_gen.
- Clean/protected condition fusion follows z_tilde=(1-sg)z_c+sg z_p.

## Explicit supplements
1. The final paper abstracts E_a without fixing its exact encoder implementation. This rebuild defaults to the frozen AudioLDM CLAP audio-conditioning representation because it is the condition directly consumed by the AudioLDM denoiser.
2. The paper does not specify the psychoacoustic threshold estimator; the current code uses a configurable STFT-domain threshold estimator.
3. The paper does not expose every PSM backbone width; RoBERTa-base with a 256-dimensional projection and a 512-dimensional AudioLDM CLAP condition are explicit engineering defaults.
4. The paper's L_gen requires generation-coupled supervision. The rebuild includes a differentiable DDIM path so gradients can reach PSM parameters while AudioLDM parameters remain frozen.
5. The official AudioLDM DDIM sampler is used for normal inference to avoid a hand-written sampler mismatch.
6. The previous repository's post-hoc waveform gate is archived only. It is not part of the new AIDE path.
