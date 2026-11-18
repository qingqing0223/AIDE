from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


@dataclass
class PSMOutput:
    logits: torch.Tensor
    scalar_gate: torch.Tensor
    channel_gate: torch.Tensor
    hidden: torch.Tensor


class PromptSafetyModule(nn.Module):
    """
    Prompt Safety Module.

    Paper-defined outputs:
      p(y): seven-class prompt-category distribution
      s(y): scalar gate in [0,1]
      g(y): channel-wise gate in (0,1)^d

    The exact backbone depth/hidden width is not specified by the paper, so
    these remain explicit engineering parameters rather than hidden constants.
    """

    def __init__(
        self,
        backbone="roberta-base",
        num_classes=7,
        hidden_dim=256,
        channel_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone)

        backbone_dim = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.scalar_head = nn.Linear(hidden_dim, 1)
        self.channel_head = nn.Linear(hidden_dim, channel_dim)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0]

        r = self.proj(h)
        logits = self.classifier(r)
        scalar = torch.sigmoid(self.scalar_head(r)).squeeze(-1)
        channel = torch.sigmoid(self.channel_head(r))

        return PSMOutput(
            logits=logits,
            scalar_gate=scalar,
            channel_gate=channel,
            hidden=r,
        )

    def encode(self, prompts, device=None):
        batch = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        return self(**batch)


class PromptDataset(Dataset):
    """
    Required fields:
      text,label,is_harmful

    Optional generation-coupled fields:
      clean_path,protected_path
    """

    def __init__(self, manifest):
        self.rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
        required = {"text", "label", "is_harmful"}
        missing = required - set(self.rows[0]) if self.rows else required
        if missing:
            raise ValueError(f"PSM manifest missing fields: {sorted(missing)}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "text": r["text"],
            "label": int(r["label"]),
            "is_harmful": float(r["is_harmful"]),
            "clean_path": r.get("clean_path", ""),
            "protected_path": r.get("protected_path", ""),
        }


def psm_class_gate_loss(
    out: PSMOutput,
    labels: torch.Tensor,
    is_harmful: torch.Tensor,
    mu_benign: float = 1.0,
    mu_harmful: float = 1.0,
):
    """
    Paper:
      L_cls = CE(p(y), c)
      L_gate = mu_b E_b[s(y)^2]
              + mu_h E_h[(1-s(y))^2]
    """
    cls = F.cross_entropy(out.logits, labels.long())

    benign = is_harmful < 0.5
    harmful = ~benign

    gate_terms = []
    if benign.any():
        gate_terms.append(mu_benign * out.scalar_gate[benign].square().mean())
    if harmful.any():
        gate_terms.append(
            mu_harmful * (1.0 - out.scalar_gate[harmful]).square().mean()
        )

    gate = sum(gate_terms) if gate_terms else cls * 0.0
    return cls, gate


def generation_coupled_loss(
    sim_clean: torch.Tensor,
    sim_gated: torch.Tensor,
    is_harmful: torch.Tensor,
):
    """
    Paper:
      L_gen =
        E_Yb [ sim(y,x_c) - sim(y,x_g) ]_+
        + E_Yh [ sim(y,x_g) ]

    sim_clean is treated as the benign-generation reference.
    """
    benign = is_harmful < 0.5
    harmful = ~benign
    terms = []

    if benign.any():
        terms.append(
            F.relu(sim_clean[benign] - sim_gated[benign]).mean()
        )
    if harmful.any():
        terms.append(sim_gated[harmful].mean())

    if not terms:
        return sim_gated.mean() * 0.0
    return sum(terms)


def psm_loss(
    out: PSMOutput,
    labels: torch.Tensor,
    is_harmful: torch.Tensor,
    sim_clean: torch.Tensor | None = None,
    sim_gated: torch.Tensor | None = None,
    mu_benign: float = 1.0,
    mu_harmful: float = 1.0,
    lambda_g: float = 1.0,
    lambda_r: float = 1.0,
):
    cls, gate = psm_class_gate_loss(
        out,
        labels,
        is_harmful,
        mu_benign=mu_benign,
        mu_harmful=mu_harmful,
    )

    if sim_clean is None or sim_gated is None:
        gen = cls * 0.0
    else:
        gen = generation_coupled_loss(
            sim_clean,
            sim_gated,
            is_harmful,
        )

    total = cls + lambda_g * gate + lambda_r * gen
    return {
        "total": total,
        "classification": cls,
        "gate": gate,
        "generation": gen,
    }


def collate_prompts(batch, tokenizer, max_length=128):
    texts = [x["text"] for x in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "is_harmful": torch.tensor(
            [x["is_harmful"] for x in batch], dtype=torch.float32
        ),
        "texts": texts,
        "clean_paths": [x["clean_path"] for x in batch],
        "protected_paths": [x["protected_path"] for x in batch],
    }


@torch.no_grad()
def evaluate_psm(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for batch in loader:
        out = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        y_true.extend(batch["labels"].tolist())
        y_pred.extend(out.logits.argmax(dim=-1).cpu().tolist())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def save_psm_checkpoint(path, model, optimizer, metrics, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )
