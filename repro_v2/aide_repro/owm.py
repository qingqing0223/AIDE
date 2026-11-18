from __future__ import annotations

import torch
from torch import nn


class OWMProjector:
    """Online weight-modification projection for selected PSM linear layers."""

    def __init__(self, model: nn.Module, ridge: float = 1.0, max_samples: int = 128):
        self.ridge = float(ridge)
        self.max_samples = int(max_samples)
        self.layers: dict[str, nn.Linear] = {}
        self.projections: dict[str, torch.Tensor] = {}
        self.inputs: dict[str, list[torch.Tensor]] = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not name.startswith("backbone."):
                self.layers[name] = module
                self.projections[name] = torch.eye(module.in_features, device=module.weight.device, dtype=torch.float32)
                self.inputs[name] = []
                self.handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                value = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                if value.numel():
                    self.inputs[name].append(value[: self.max_samples].to(torch.float32))
        return hook

    @torch.no_grad()
    def project_gradients(self) -> None:
        for name, module in self.layers.items():
            if module.weight.grad is not None:
                projection = self.projections[name].to(device=module.weight.grad.device, dtype=module.weight.grad.dtype)
                module.weight.grad.copy_(module.weight.grad @ projection)

    @torch.no_grad()
    def update(self) -> None:
        for name, chunks in self.inputs.items():
            if not chunks:
                continue
            features = torch.cat(chunks, dim=0)[: self.max_samples]
            projection = self.projections[name].to(features.device)
            xp = features @ projection
            gram = xp @ features.T
            identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            solved = torch.linalg.solve(identity + gram / self.ridge, xp)
            projection = projection - (projection @ features.T / self.ridge) @ solved
            self.projections[name] = 0.5 * (projection + projection.T)
            self.inputs[name].clear()

    def state_dict(self) -> dict:
        return {name: value.detach().cpu() for name, value in self.projections.items()}

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        for name, value in state.items():
            if name in self.projections and tuple(value.shape) == tuple(self.projections[name].shape):
                self.projections[name] = value.to(self.projections[name].device, dtype=torch.float32)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
