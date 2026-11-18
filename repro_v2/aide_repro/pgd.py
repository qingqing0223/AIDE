from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .audio import list_audio_files, load_audio, save_audio, sha256_file, snr_db, lsd_db, write_jsonl


def psychoacoustic_masking_loss(
    clean: torch.Tensor,
    delta: torch.Tensor,
    offset_db: float = 20.0,
) -> torch.Tensor:
    """Differentiable local masking-threshold proxy.

    The paper specifies a psychoacoustic threshold T_x but does not publish its
    exact estimator or STFT settings. This implementation uses a documented
    local time-frequency masker envelope; it is an engineering proxy, not a
    claim to reproduce an undisclosed psychoacoustic implementation.
    """
    def magnitude(wav):
        window = torch.hann_window(1024, periodic=True, device=wav.device)
        return torch.stft(
            wav, n_fft=1024, hop_length=160, win_length=1024, window=window,
            center=True, pad_mode="reflect", return_complex=True
        ).abs()

    clean_mag = magnitude(clean).unsqueeze(1)
    delta_mag = magnitude(delta).unsqueeze(1)
    local_masker = F.max_pool2d(clean_mag, kernel_size=(5, 3), stride=1, padding=(2, 1))
    threshold = local_masker * (10.0 ** (-float(offset_db) / 20.0))
    return F.relu(delta_mag - threshold).square().mean()


class LatentPGD:
    def __init__(
        self,
        backend,
        epsilon: float = 0.01,
        step_size: float = 0.001,
        steps: int = 40,
        lambda_target: float = 1.0,
        lambda_far: float = 0.2,
        lambda_mask: float = 0.001,
        masking_offset_db: float = 20.0,
        seed: int = 2025,
    ):
        self.backend = backend
        self.epsilon = float(epsilon)
        self.step_size = float(step_size)
        self.steps = int(steps)
        self.lambda_target = float(lambda_target)
        self.lambda_far = float(lambda_far)
        self.lambda_mask = float(lambda_mask)
        self.masking_offset_db = float(masking_offset_db)
        self.seed = int(seed)

    def optimize_one(self, clean: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        clean = clean.to(self.backend.device).float()
        target = target.to(self.backend.device).float()
        with torch.no_grad():
            z_clean = self.backend.encode_waveform(clean)
            z_target = self.backend.encode_waveform(target)

        delta = torch.zeros_like(clean)
        history = []
        for step in range(self.steps):
            delta = delta.detach().requires_grad_(True)
            protected = (clean + delta).clamp(-1.0, 1.0)
            z_protected = self.backend.encode_waveform(protected)
            l_target = (z_protected - z_target).square().mean()
            l_far = -(z_protected - z_clean).square().mean()
            l_mask = psychoacoustic_masking_loss(clean, delta, self.masking_offset_db)
            total = self.lambda_target * l_target + self.lambda_far * l_far + self.lambda_mask * l_mask
            grad = torch.autograd.grad(total, delta, retain_graph=False, create_graph=False)[0]
            if grad is None or not torch.isfinite(grad).all():
                raise RuntimeError("PGD gradient is missing or non-finite.")
            with torch.no_grad():
                # Minimize the paper's objective. The old main branch used +sign(grad), i.e. ascent.
                delta = (delta - self.step_size * grad.sign()).clamp(-self.epsilon, self.epsilon)
            history.append({
                "step": step + 1,
                "l_target": float(l_target.detach().cpu()),
                "l_far": float(l_far.detach().cpu()),
                "l_mask": float(l_mask.detach().cpu()),
                "total": float(total.detach().cpu()),
                "delta_linf": float(delta.abs().max().cpu()),
            })

        delta = delta.detach()
        protected = (clean + delta).clamp(-1.0, 1.0).detach()
        with torch.no_grad():
            z_protected = self.backend.encode_waveform(protected)
        latent = {
            "z_clean": z_clean.detach().cpu(),
            "z_protected": z_protected.detach().cpu(),
            "z_target": z_target.detach().cpu(),
            "latent_shape": list(z_protected.shape),
        }
        metrics = {
            "snr_db": snr_db(clean, protected),
            "lsd_db": lsd_db(clean, protected),
            "linf": float((protected - clean).abs().max().cpu()),
            "loss_last": history[-1]["total"] if history else None,
            "history": history,
        }
        return delta, protected, {"latent": latent, "metrics": metrics}

    def run_manifest(
        self,
        manifest_jsonl: str | Path,
        target_root: str | Path,
        output_dir: str | Path,
        sample_limit: int | None = None,
        duration_seconds: float = 10.0,
    ) -> dict:
        manifest = [json.loads(line) for line in Path(manifest_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
        targets = list_audio_files(target_root)
        if sample_limit:
            manifest = manifest[:sample_limit]
        out = Path(output_dir)
        (out / "audio").mkdir(parents=True, exist_ok=True)
        (out / "latents").mkdir(parents=True, exist_ok=True)
        metric_rows = []
        target_rows = []
        history_rows = []
        run_config = {
            "seed": self.seed,
            "epsilon": self.epsilon,
            "step_size": self.step_size,
            "steps": self.steps,
            "lambda_target": self.lambda_target,
            "lambda_far": self.lambda_far,
            "lambda_mask": self.lambda_mask,
            "masking_offset_db": self.masking_offset_db,
            "duration_seconds": duration_seconds,
            "target_root": str(Path(target_root).resolve()),
            "target_pool_size": len(targets),
            "manifest": str(Path(manifest_jsonl).resolve()),
        }
        for row in manifest:
            audio_id = row["audio_id"]
            clean = load_audio(row["path"], duration_seconds=duration_seconds).to(self.backend.device)
            seed_int = int.from_bytes(audio_id.encode("utf-8")[:4].ljust(4, b"0"), "little")
            rng = random.Random(self.seed + seed_int)
            target_path = targets[rng.randrange(len(targets))]
            target = load_audio(target_path, duration_seconds=duration_seconds).to(self.backend.device)
            delta, protected, result = self.optimize_one(clean, target)
            audio_dir = out / "audio"
            save_audio(audio_dir / f"{audio_id}_raw.wav", clean)
            save_audio(audio_dir / f"{audio_id}_delta.wav", delta)
            save_audio(audio_dir / f"{audio_id}_adv.wav", protected)
            torch.save(result["latent"], out / "latents" / f"{audio_id}.pt")
            metrics = {
                "audio_id": audio_id,
                "source_sha256": row.get("sha256"),
                "target_path": str(target_path.resolve()),
                "target_sha256": sha256_file(target_path),
                **{k: v for k, v in result["metrics"].items() if k != "history"},
            }
            metric_rows.append(metrics)
            for item in result["metrics"]["history"]:
                history_rows.append({"audio_id": audio_id, **item})
            target_rows.append({
                "audio_id": audio_id,
                "target_path": str(target_path.resolve()),
                "target_sha256": metrics["target_sha256"],
                "seed": self.seed,
            })
            print(json.dumps(metrics, ensure_ascii=False))
        from .audio import write_jsonl, sha256_file
        write_jsonl(out / "pgd_metrics.jsonl", metric_rows)
        write_jsonl(out / "pgd_history.jsonl", history_rows)
        write_jsonl(out / "target_manifest.jsonl", target_rows)
        (out / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        hashes = {}
        for artifact in sorted(p for p in out.rglob("*") if p.is_file()):
            hashes[str(artifact.relative_to(out))] = sha256_file(artifact)
        (out / "artifacts_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        summary = {"num_samples": len(metric_rows), "output_dir": str(out.resolve()),
                   "metrics": {
                       "snr_db_mean": sum(x["snr_db"] for x in metric_rows) / max(1, len(metric_rows)),
                       "lsd_db_mean": sum(x["lsd_db"] for x in metric_rows) / max(1, len(metric_rows)),
                       "linf_max": max((x["linf"] for x in metric_rows), default=0.0),
                   }}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
