from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class PSMModel(nn.Module):
    """Seven-way prompt classifier with scalar and latent-channel gates."""

    def __init__(
        self,
        num_labels: int = 7,
        latent_channels: int = 8,
        text_model: str = "roberta-base",
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.text_model_name = text_model
        self.latent_channels = int(latent_channels)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model, local_files_only=local_files_only)
        self.backbone = AutoModel.from_pretrained(text_model, local_files_only=local_files_only)
        hidden = int(self.backbone.config.hidden_size)
        self.shared = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.classifier = nn.Linear(hidden, int(num_labels))
        self.scalar_head = nn.Linear(hidden, 1)
        self.channel_head = nn.Linear(hidden, self.latent_channels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        encoded = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoded.pooler_output if getattr(encoded, "pooler_output", None) is not None else encoded.last_hidden_state[:, 0]
        shared = self.shared(pooled)
        return {
            "logits": self.classifier(shared),
            "scalar_gate": torch.sigmoid(self.scalar_head(shared)).squeeze(-1),
            "channel_gate": torch.sigmoid(self.channel_head(shared)),
            "features": shared,
        }

    def expand_classifier(self, new_num_labels: int) -> None:
        """Append new classes while retaining all previously learned classifier rows."""
        old = self.classifier
        if new_num_labels <= old.out_features:
            raise ValueError("new_num_labels must exceed the current classifier size.")
        expanded = nn.Linear(old.in_features, int(new_num_labels)).to(old.weight.device, old.weight.dtype)
        with torch.no_grad():
            expanded.weight[: old.out_features].copy_(old.weight)
            expanded.bias[: old.out_features].copy_(old.bias)
        self.classifier = expanded

    def tokenize(self, texts: list[str], max_length: int = 128) -> dict[str, torch.Tensor]:
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
