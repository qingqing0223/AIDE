import torch
from torch import nn

from aide_repro.losses import generation_coupled_loss, scalar_gate_calibration
from aide_repro.owm import OWMProjector
from aide_repro.pgd import psychoacoustic_masking_loss
from aide_repro.audio import fit_length


def test_generation_coupled_loss_matches_eq7():
    clean = torch.tensor([0.8, 0.1])
    gated = torch.tensor([0.6, 0.3])
    harmful = torch.tensor([0.0, 1.0])
    loss, parts = generation_coupled_loss(clean, gated, harmful)
    assert torch.allclose(loss, torch.tensor(0.5))
    assert abs(parts["l_gen_benign"] - 0.2) < 1e-6
    assert abs(parts["l_gen_harmful"] - 0.3) < 1e-6


def test_scalar_gate_calibration():
    gate = torch.tensor([0.2, 0.8])
    harmful = torch.tensor([0.0, 1.0])
    loss = scalar_gate_calibration(gate, harmful)
    assert abs(float(loss) - 0.08) < 1e-6


def test_masking_proxy_is_finite_and_nonnegative():
    clean = torch.randn(1, 16000)
    delta = torch.zeros_like(clean).normal_(std=0.001).requires_grad_(True)
    loss = psychoacoustic_masking_loss(clean, delta)
    assert torch.isfinite(loss)
    assert float(loss) >= 0
    grad = torch.autograd.grad(loss, delta)[0]
    assert torch.isfinite(grad).all()


def test_fit_length():
    x = torch.ones(1, 5)
    assert fit_length(x, 8).shape == (1, 8)
    assert fit_length(x, 3).shape == (1, 3)


def test_owm_projection_updates_and_projects():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 2)

        def forward(self, x):
            return self.fc(x)

    model = Tiny()
    projector = OWMProjector(model, max_samples=8)
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    projector.project_gradients()
    projector.update()
    assert torch.isfinite(projector.projections["fc"]).all()
    projector.close()


def test_latent_fusion_eq4_endpoints_and_shape_guard():
    from aide_repro.losses import fuse_latents

    clean = torch.zeros(2, 8, 4, 3)
    protected = torch.ones_like(clean)
    scalar_zero = torch.zeros(2)
    channel_one = torch.ones(2, 8)
    assert torch.equal(fuse_latents(clean, protected, scalar_zero, channel_one), clean)
    scalar_one = torch.ones(2)
    assert torch.equal(fuse_latents(clean, protected, scalar_one, channel_one), protected)
    try:
        fuse_latents(clean, protected, scalar_one, torch.ones(2, 7))
    except ValueError:
        pass
    else:
        raise AssertionError("Channel mismatch should fail loudly")
