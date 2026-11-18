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
