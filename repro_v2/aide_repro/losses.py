from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels.long())


def scalar_gate_calibration(
    scalar_gate: torch.Tensor,
    is_harmful: torch.Tensor,
    mu_benign: float = 1.0,
    mu_harmful: float = 1.0,
) -> torch.Tensor:
    benign = is_harmful < 0.5
    harmful = ~benign
    zero = scalar_gate.new_zeros(())
    benign_term = scalar_gate[benign].square().mean() if benign.any() else zero
    harmful_term = (1.0 - scalar_gate[harmful]).square().mean() if harmful.any() else zero
    return float(mu_benign) * benign_term + float(mu_harmful) * harmful_term


def generation_coupled_loss(
    clean_similarity: torch.Tensor,
    gated_similarity: torch.Tensor,
    is_harmful: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Paper Eq. (7): benign hinge preservation plus harmful prompt alignment."""
    benign = is_harmful < 0.5
    harmful = ~benign
    zero = gated_similarity.new_zeros(())
    benign_term = F.relu(clean_similarity[benign] - gated_similarity[benign]).mean() if benign.any() else zero
    harmful_term = gated_similarity[harmful].mean() if harmful.any() else zero
    total = benign_term + harmful_term
    return total, {
        "l_gen_benign": float(benign_term.detach().cpu()),
        "l_gen_harmful": float(harmful_term.detach().cpu()),
        "l_gen": float(total.detach().cpu()),
    }


def combine_psm_losses(
    l_cls: torch.Tensor,
    l_gate: torch.Tensor,
    l_gen: torch.Tensor,
    lambda_gate: float = 1.0,
    lambda_generation: float = 1.0,
) -> torch.Tensor:
    return l_cls + float(lambda_gate) * l_gate + float(lambda_generation) * l_gen


def fuse_latents(
    z_clean: torch.Tensor,
    z_protected: torch.Tensor,
    scalar_gate: torch.Tensor,
    channel_gate: torch.Tensor,
) -> torch.Tensor:
    """Apply paper Eq. (4) with strict batch/channel/spatial shape checks."""
    if z_clean.shape != z_protected.shape:
        raise ValueError(f"Clean/protected latent shape mismatch: {tuple(z_clean.shape)} vs {tuple(z_protected.shape)}")
    if z_clean.ndim != 4:
        raise ValueError(f"Expected latent [B,C,H,W], got {tuple(z_clean.shape)}")
    batch, channels = z_clean.shape[:2]
    scalar_gate = scalar_gate.reshape(-1)
    if scalar_gate.shape[0] != batch:
        raise ValueError(f"Scalar gate batch mismatch: {scalar_gate.shape[0]} vs {batch}")
    if channel_gate.shape != (batch, channels):
        raise ValueError(
            f"Channel gate must be [B,C]=[{batch},{channels}], got {tuple(channel_gate.shape)}"
        )
    weight = scalar_gate[:, None] * channel_gate
    return z_clean + weight[:, :, None, None] * (z_protected - z_clean)
