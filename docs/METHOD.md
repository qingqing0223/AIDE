# AIDE method specification

## Encoder-latent immunization
For clean x, encode z_c and target z_tar. Optimize:
L_p = ||z_p-z_tar||_2^2 - lambda_f ||z_p-z_c||_2^2 + lambda_m L_m
subject to ||delta||_infinity <= epsilon.

L_m = sum max(0, S_delta(f,t) - gamma T_x(f,t))^2.

The paper reports epsilon=0.01. Target-pool construction, PGD initialization/steps and the exact threshold estimator are not disclosed and are configurable here.

## Prompt Safety Module
p(y)=softmax(W_c r+b_c)
s(y)=sigmoid(w_s^T r+b_s)
g(y)=sigmoid(W_g r+b_g)

Seven classes: one benign plus six harmful subsets.

## Controlled condition
z_tilde(y)=(1-s(y)g(y)) odot z_c + s(y)g(y) odot z_p.

AudioLDM remains frozen.
