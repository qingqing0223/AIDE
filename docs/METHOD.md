# AIDE method specification

## 1. Encoder-latent immunization

For clean speech x, perturbation delta and protected speech x_tilde=x+delta:

z_c = E_a(x)
z_p = E_a(x_tilde)
z_tar = E_a(x_tar)

The paper gives:

L_p = ||z_p-z_tar||_2^2
      - lambda_f ||z_p-z_c||_2^2
      + lambda_m L_m

subject to:

||delta||_infinity <= epsilon

with epsilon=0.01.

The masking term is:

L_m = sum_{f,tau} max(0, S_delta(f,tau)-gamma T_x(f,tau))^2.

The paper describes T_x as the psychoacoustic masking threshold estimated from clean speech. The exact threshold estimator is not disclosed, so the repository exposes the estimator as an implementation component rather than presenting its smoothing rule as paper text.

## 2. Prompt Safety Module

For prompt y, the module outputs:

p(y) = softmax(W_c r(y)+b_c)
s(y) = sigmoid(w_s^T r(y)+b_s)
g(y) = sigmoid(W_g r(y)+b_g)

where:
- p(y) has seven classes;
- s(y) is the scalar gate;
- g(y) is the channel-wise gate.

The paper defines:
c=0 as benign and c=1,...,6 as harmful.

The paper does not give semantic names for the six harmful subclasses in the final text, so the repository keeps them as harmful_1 ... harmful_6.

## 3. PSM objective

L_cls = CE(p(y),c)

L_gate =
  mu_b E_{y~Y_b}[s(y)^2]
  + mu_h E_{y~Y_h}[(1-s(y))^2]

L_gen =
  E_{Y_b}[sim(y,x_c)-sim(y,x_g)]_+
  + E_{Y_h}[sim(y,x_g)]

L_PSM = L_cls + lambda_g L_gate + lambda_r L_gen

The channel gate does not have an independent calibration loss in the paper; it participates in the gated generation condition and therefore receives supervision through L_gen.

## 4. Gated condition

The paper gives:

z_tilde(y) =
  (1-s(y)g(y)) odot z_c
  + s(y)g(y) odot z_p.

This repository implements the condition as a [B,1,D] AudioLDM cross-attention condition.

Important reproducibility note:
the final paper uses the abstract encoder E_a and does not explicitly state in the equation whether it is the AudioLDM CLAP audio condition or another encoder output. The rebuild therefore makes the representation choice explicit. The default implementation uses AudioLDM's frozen CLAP audio-conditioning embedding because it is the condition consumed by the AudioLDM denoiser and is compatible with the paper's gated-condition interface.

## 5. Diffusion generation

AudioLDM remains frozen.

Inference uses AudioLDM's official DDIM sampler rather than a separate hand-written sampler. This avoids timestep-index and alpha-schedule drift.

For PSM generation-coupled training, the repository contains a differentiable DDIM implementation. It freezes AudioLDM parameters while retaining gradients through the gated condition so L_gen can update the PSM.

The paper does not disclose a specific denoising timestep subset for gate computation in the final text. The default implementation therefore forms the gated condition once and feeds it to the AudioLDM DDIM process. Any alternative timestep-specific schedule is recorded as a supplement rather than silently presented as the paper method.
