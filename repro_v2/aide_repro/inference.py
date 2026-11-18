from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from .audio import fit_length, load_audio, save_audio, sha256_file
from .psm import PSMModel
from .losses import fuse_latents


def load_psm_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    local_files_only: bool = False,
) -> tuple[PSMModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = PSMModel(
        num_labels=int(checkpoint["num_labels"]),
        latent_channels=int(checkpoint["latent_channels"]),
        text_model=str(checkpoint["text_model"]),
        local_files_only=local_files_only,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def load_delta(
    path: str | Path,
    sample_rate: int = 16000,
    duration_seconds: float = 10.0,
) -> torch.Tensor:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(audio.T.copy()).mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    target_frames = int(round(duration_seconds * 102.4))
    return fit_length(waveform, target_frames * 160)


@torch.no_grad()
def infer_one(
    backend,
    psm: PSMModel,
    audio_path: str | Path,
    delta_path: str | Path,
    prompt: str,
    output_path: str | Path,
    duration_seconds: float = 10.0,
    ddim_steps: int = 50,
    t_start: int | None = None,
    guidance_scale: float = 2.5,
    seed: int = 2025,
    mode: str = "aide",
    hard_gate: bool = False,
    disable_channel_gate: bool = False,
    save_paired: bool = True,
    local_files_only: bool = False,
    psm_checkpoint_path: str | Path | None = None,
) -> dict:
    device = backend.device
    clean = load_audio(audio_path, duration_seconds=duration_seconds).to(device)
    delta = load_delta(delta_path, duration_seconds=duration_seconds).to(device)
    if clean.shape != delta.shape:
        raise ValueError(f"Clean/delta shape mismatch: {tuple(clean.shape)} vs {tuple(delta.shape)}")
    protected = (clean + delta).clamp(-1.0, 1.0)
    z_clean = backend.encode_waveform(clean)
    z_protected = backend.encode_waveform(protected)
    if psm.latent_channels != backend.latent_channels:
        raise ValueError(
            f"PSM latent_channels={psm.latent_channels} does not match "
            f"AudioLDM latent_channels={backend.latent_channels}"
        )

    tokens = psm.tokenize([prompt], max_length=128)
    tokens = {key: value.to(device) for key, value in tokens.items()}
    output = psm(tokens["input_ids"], tokens["attention_mask"])
    probabilities = torch.softmax(output["logits"], dim=-1)
    predicted_label = int(probabilities.argmax(-1).item())
    harmful_probability = float((1.0 - probabilities[:, 0]).item())
    scalar_gate = float(output["scalar_gate"].item())
    channel_gate = output["channel_gate"]
    if disable_channel_gate:
        channel_gate = torch.ones_like(channel_gate)
    if hard_gate:
        scalar_gate_tensor = (1.0 - probabilities[:, 0] >= 0.5).to(output["scalar_gate"].dtype)
    else:
        scalar_gate_tensor = output["scalar_gate"]
    weight = scalar_gate_tensor[:, None] * channel_gate

    if mode == "clean":
        selected = z_clean
    elif mode in {"protected", "static_protection"}:
        selected = z_protected
    elif mode == "aide":
        selected = fuse_latents(z_clean, z_protected, scalar_gate_tensor, channel_gate)
    else:
        raise ValueError("mode must be one of: aide, clean, protected")

    generator = torch.Generator(device=device).manual_seed(int(seed))
    shared_noise = torch.randn(z_clean.shape, generator=generator, device=device, dtype=z_clean.dtype)
    gated_audio = backend.generate_from_latent(
        selected, [prompt], ddim_steps=ddim_steps, t_start=t_start,
        guidance_scale=guidance_scale, noise=shared_noise, seed=seed
    )
    output_path = Path(output_path)
    save_audio(output_path, gated_audio)

    meta = {
        "audio_path": str(Path(audio_path).resolve()),
        "audio_sha256": sha256_file(audio_path),
        "delta_path": str(Path(delta_path).resolve()),
        "delta_sha256": sha256_file(delta_path),
        "psm_checkpoint": str(Path(psm_checkpoint_path).resolve()) if psm_checkpoint_path else None,
        "psm_checkpoint_sha256": sha256_file(psm_checkpoint_path) if psm_checkpoint_path else None,
        "audio_model": backend.model_name,
        "audio_checkpoint": backend.checkpoint_path,
        "audio_checkpoint_sha256": backend.checkpoint_sha256,
        "sample_rate": backend.sample_rate,
        "latent_shape": list(z_clean.shape),
        "prompt": prompt,
        "mode": mode,
        "predicted_label_id": predicted_label,
        "class_probabilities": probabilities[0].detach().cpu().tolist(),
        "harmful_probability": harmful_probability,
        "scalar_gate": scalar_gate,
        "effective_scalar_gate": float(scalar_gate_tensor.item()),
        "channel_gate": channel_gate[0].detach().cpu().tolist(),
        "ddim_steps": int(ddim_steps),
        "t_start": int(t_start if t_start is not None else max(1, round(ddim_steps * 0.5))),
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),
        "output_path": str(output_path.resolve()),
    }

    if save_paired:
        paired_dir = output_path.parent / f"{output_path.stem}_paired"
        paired_dir.mkdir(parents=True, exist_ok=True)
        clean_noise = shared_noise
        clean_audio = backend.generate_from_latent(
            z_clean, [prompt], ddim_steps=ddim_steps, t_start=t_start,
            guidance_scale=guidance_scale, noise=clean_noise, seed=seed
        )
        protected_audio = backend.generate_from_latent(
            z_protected, [prompt], ddim_steps=ddim_steps, t_start=t_start,
            guidance_scale=guidance_scale, noise=clean_noise, seed=seed
        )
        save_audio(paired_dir / "clean.wav", clean_audio)
        save_audio(paired_dir / "protected.wav", protected_audio)
        save_audio(paired_dir / "gated.wav", gated_audio)
        meta["paired_dir"] = str(paired_dir.resolve())

    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
