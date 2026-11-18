from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .losses import classification_loss, scalar_gate_calibration, generation_coupled_loss, combine_psm_losses
from .owm import OWMProjector
from .psm import PSMModel


class PromptDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, latent_dir: str | Path, seed: int = 2025):
        self.records = [
            json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.latent_dir = Path(latent_dir)
        self.latent_ids = sorted(p.stem for p in self.latent_dir.glob("*.pt"))
        if not self.latent_ids:
            raise FileNotFoundError(f"No Stage-I latent cache (*.pt) in {self.latent_dir}")
        self.seed = int(seed)
        if not self.records:
            raise ValueError(f"No prompt records in {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        audio_id = row.get("audio_id")
        if not audio_id:
            audio_id = self.latent_ids[random.Random(self.seed + index).randrange(len(self.latent_ids))]
        return {
            "text": str(row["text"]),
            "label_id": int(row["label_id"]),
            "is_harmful": int(row.get("is_harmful", int(row["label_id"] > 0))),
            "audio_id": audio_id,
        }


def collate_prompts(batch, tokenizer, max_length: int):
    texts = [row["text"] for row in batch]
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": torch.tensor([row["label_id"] for row in batch], dtype=torch.long),
        "is_harmful": torch.tensor([row["is_harmful"] for row in batch], dtype=torch.float32),
        "audio_ids": [row["audio_id"] for row in batch],
        "texts": texts,
    }


def _load_latent_batch(latent_dir: Path, audio_ids: list[str], device: torch.device):
    clean, protected = [], []
    for audio_id in audio_ids:
        path = latent_dir / f"{audio_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage-I latent cache for {audio_id}: {path}")
        item = torch.load(path, map_location="cpu")
        clean.append(item["z_clean"].squeeze(0))
        protected.append(item["z_protected"].squeeze(0))
    return torch.stack(clean).to(device), torch.stack(protected).to(device)


class PSMTrainer:
    def __init__(
        self,
        model: PSMModel,
        backend,
        latent_dir: str | Path,
        output_dir: str | Path,
        device: str,
        learning_rate: float = 2e-5,
        lambda_gate: float = 1.0,
        lambda_generation: float = 1.0,
        generation_every: int = 1,
        generation_ddim_steps: int = 8,
        generation_t_start: int = 4,
        generation_guidance_scale: float = 1.0,
        max_length: int = 128,
        use_owm: bool = False,
        disable_generation_loss: bool = False,
        disable_channel_gate: bool = False,
        binary_risk: bool = False,
        seed: int = 2025,
        owm_state: dict | None = None,
    ):
        self.model = model
        self.backend = backend
        self.latent_dir = Path(latent_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.learning_rate = float(learning_rate)
        self.lambda_gate = float(lambda_gate)
        self.lambda_generation = float(lambda_generation)
        self.generation_every = max(1, int(generation_every))
        self.generation_ddim_steps = int(generation_ddim_steps)
        self.generation_t_start = int(generation_t_start)
        self.generation_guidance_scale = float(generation_guidance_scale)
        self.max_length = int(max_length)
        self.disable_generation_loss = bool(disable_generation_loss)
        self.disable_channel_gate = bool(disable_channel_gate)
        self.binary_risk = bool(binary_risk)
        self.seed = int(seed)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.owm = OWMProjector(self.model) if use_owm else None
        if self.owm is not None and owm_state:
            self.owm.load_state_dict(owm_state)
        self.global_step = 0

    def _forward_generation_loss(self, outputs, batch, z_clean, z_protected):
        scalar = outputs["scalar_gate"]
        channel = outputs["channel_gate"]
        if self.disable_channel_gate:
            channel = torch.ones_like(channel)
        weight = scalar.unsqueeze(1) * channel
        gated_latent = z_clean + weight[:, :, None, None] * (z_protected - z_clean)
        shared_noise = torch.randn_like(z_clean)
        clean_audio = self.backend.generate_from_latent(
            z_clean, batch["texts"], ddim_steps=self.generation_ddim_steps,
            t_start=self.generation_t_start, guidance_scale=self.generation_guidance_scale,
            noise=shared_noise,
        )
        gated_audio = self.backend.generate_from_latent(
            gated_latent, batch["texts"], ddim_steps=self.generation_ddim_steps,
            t_start=self.generation_t_start, guidance_scale=self.generation_guidance_scale,
            noise=shared_noise,
        )
        clean_sim = self.backend.clap_similarity(clean_audio, batch["texts"])
        gated_sim = self.backend.clap_similarity(gated_audio, batch["texts"])
        loss, details = generation_coupled_loss(clean_sim, gated_sim, batch["is_harmful"].to(self.device))
        details["clap_clean_mean"] = float(clean_sim.detach().mean().cpu())
        details["clap_gated_mean"] = float(gated_sim.detach().mean().cpu())
        return loss, details

    @torch.no_grad()
    def evaluate_classifier(self, loader):
        self.model.eval()
        labels, predictions, gates, harmful = [], [], [], []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention = batch["attention_mask"].to(self.device)
            y = batch["labels"].to(self.device)
            h = batch["is_harmful"].to(self.device)
            if self.binary_risk:
                y = h.long()
            output = self.model(input_ids, attention)
            labels.extend(y.cpu().tolist())
            predictions.extend(output["logits"].argmax(-1).cpu().tolist())
            gates.extend(output["scalar_gate"].cpu().tolist())
            harmful.extend(h.cpu().tolist())
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
            "gate_benign_mean": float(np.mean([g for g, h in zip(gates, harmful) if h < 0.5])) if any(h < 0.5 for h in harmful) else None,
            "gate_harmful_mean": float(np.mean([g for g, h in zip(gates, harmful) if h >= 0.5])) if any(h >= 0.5 for h in harmful) else None,
        }

    def fit(self, train_jsonl: str | Path, dev_jsonl: str | Path, epochs: int = 10, batch_size: int = 4):
        train_ds = PromptDataset(train_jsonl, self.latent_dir, self.seed)
        dev_ds = PromptDataset(dev_jsonl, self.latent_dir, self.seed + 1)
        tokenizer = self.model.tokenizer
        collate = lambda rows: collate_prompts(rows, tokenizer, self.max_length)

        # Inverse-frequency sampling approximates the paper's near-balanced mini-batches.
        class_ids = [int(row["label_id"] > 0) if self.binary_risk else int(row["label_id"]) for row in train_ds.records]
        counts = np.bincount(class_ids, minlength=max(class_ids) + 1)
        sample_weights = [1.0 / max(1, counts[c]) for c in class_ids]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate, num_workers=0)
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

        best_f1 = -1.0
        history_path = self.output_dir / "training.jsonl"
        for epoch in range(1, int(epochs) + 1):
            self.model.train()
            epoch_rows = []
            for batch_index, batch in enumerate(train_loader):
                self.global_step += 1
                input_ids = batch["input_ids"].to(self.device)
                attention = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                harmful = batch["is_harmful"].to(self.device)
                if self.binary_risk:
                    labels = harmful.long()

                outputs = self.model(input_ids, attention)
                l_cls = classification_loss(outputs["logits"], labels)
                l_gate = scalar_gate_calibration(outputs["scalar_gate"], harmful)

                l_gen = l_cls.new_zeros(())
                gen_info = {"l_gen": 0.0}
                if not self.disable_generation_loss and self.global_step % self.generation_every == 0:
                    z_clean, z_protected = _load_latent_batch(self.latent_dir, batch["audio_ids"], self.device)
                    l_gen, gen_info = self._forward_generation_loss(outputs, batch, z_clean, z_protected)

                total = combine_psm_losses(l_cls, l_gate, l_gen, self.lambda_gate, self.lambda_generation)
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                if self.owm is not None:
                    self.owm.project_gradients()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if self.owm is not None:
                    self.owm.update()

                epoch_rows.append({
                    "epoch": epoch, "step": self.global_step,
                    "l_cls": float(l_cls.detach().cpu()),
                    "l_gate": float(l_gate.detach().cpu()),
                    "l_total": float(total.detach().cpu()),
                    **gen_info,
                })

            dev_metrics = self.evaluate_classifier(dev_loader)
            train_loss = float(np.mean([r["l_total"] for r in epoch_rows])) if epoch_rows else float("nan")
            summary = {"epoch": epoch, "train_loss": train_loss, **dev_metrics}
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
            checkpoint = {
                "state_dict": self.model.state_dict(),
                "num_labels": self.model.classifier.out_features,
                "latent_channels": self.model.latent_channels,
                "text_model": self.model.text_model_name,
                "epoch": epoch,
                "global_step": self.global_step,
                "metrics": summary,
                "binary_risk": self.binary_risk,
                "owm_enabled": self.owm is not None,
                "owm_state": self.owm.state_dict() if self.owm is not None else None,
            }
            torch.save(checkpoint, self.output_dir / "last.pt")
            if dev_metrics["macro_f1"] > best_f1:
                best_f1 = dev_metrics["macro_f1"]
                torch.save(checkpoint, self.output_dir / "best.pt")
            print(json.dumps(summary, ensure_ascii=False))

        if self.owm is not None:
            self.owm.close()
        return {"best_macro_f1": best_f1, "checkpoint": str((self.output_dir / "best.pt").resolve())}
