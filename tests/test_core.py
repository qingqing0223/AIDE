import torch

from aide.immunization import masking_loss
from aide.psm import (
    PromptSafetyModule,
    PSMOutput,
    generation_coupled_loss,
    psm_loss,
)
from aide.audioldm import AudioLDMAdapter


def test_masking_zero():
    x = torch.randn(1, 1, 16000)
    d = torch.zeros_like(x)
    assert torch.isfinite(masking_loss(x, d, 0.8))


def test_condition_fusion_endpoints():
    zc = torch.ones(2, 1, 8)
    zp = torch.zeros_like(zc)
    s = torch.zeros(2)
    g = torch.ones(2, 8)

    out = AudioLDMAdapter.fuse_conditions(zc, zp, s, g)
    assert torch.allclose(out, zc)

    s = torch.ones(2)
    out = AudioLDMAdapter.fuse_conditions(zc, zp, s, g)
    assert torch.allclose(out, zp)


def test_condition_fusion_shape_guard():
    zc = torch.ones(2, 1, 8)
    zp = torch.zeros_like(zc)
    s = torch.ones(2)
    g = torch.ones(2, 7)

    try:
        AudioLDMAdapter.fuse_conditions(zc, zp, s, g)
    except ValueError:
        return
    raise AssertionError("gate/condition dimension mismatch was not rejected")


def test_psm_gate_and_classification_loss():
    out = PSMOutput(
        logits=torch.randn(4, 7, requires_grad=True),
        scalar_gate=torch.sigmoid(torch.randn(4, requires_grad=True)),
        channel_gate=torch.sigmoid(torch.randn(4, 8, requires_grad=True)),
        hidden=torch.randn(4, 16, requires_grad=True),
    )
    labels = torch.tensor([0, 1, 2, 0])
    harmful = torch.tensor([0.0, 1.0, 1.0, 0.0])

    losses = psm_loss(
        out,
        labels,
        harmful,
        sim_clean=torch.tensor([0.8, 0.2, 0.2, 0.7]),
        sim_gated=torch.tensor([0.7, 0.3, 0.4, 0.6], requires_grad=True),
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()


def test_generation_coupled_loss_direction():
    sim_clean = torch.tensor([0.8, 0.2])
    sim_gated = torch.tensor([0.7, 0.5])
    harmful = torch.tensor([0.0, 1.0])
    loss = generation_coupled_loss(sim_clean, sim_gated, harmful)
    assert torch.allclose(loss, torch.tensor(0.1))
