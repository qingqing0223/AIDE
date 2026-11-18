from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import torchaudio

from .audio import fit_length, sha256_file


class AudioLDMBackend:
    """Adapter for the original haoheliu/AudioLDM implementation.

    The adapter uses the original AudioLDM VAE, CLAP conditioner, UNet and DDIM
    schedule. The differentiable DDIM loop mirrors the upstream deterministic
    eta=0 update without calling the upstream sampler's @torch.no_grad methods.
    """

    def __init__(
        self,
        model_name: str = "audioldm-s-full-v2",
        device: str = "auto",
        checkpoint: str | None = None,
    ) -> None:
        try:
            from audioldm import build_model
            from audioldm.utils import default_audioldm_config, get_metadata
        except Exception as exc:
            raise RuntimeError(
                "Original AudioLDM is not installed. Follow repro_v2/README.md "
                "to install the pinned upstream package."
            ) from exc

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        if checkpoint is not None and not Path(checkpoint).is_file():
            raise FileNotFoundError(f"AudioLDM checkpoint does not exist: {checkpoint}")
        self.checkpoint_path = str(Path(checkpoint).resolve()) if checkpoint else str(Path(get_metadata()[model_name]["path"]).expanduser())
        self.model = build_model(ckpt_path=checkpoint, model_name=model_name)
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path) if Path(self.checkpoint_path).is_file() else None
        self.model = self.model.to(self.device).eval()
        self.model.device = self.device
        if getattr(self.model, "parameterization", "eps") != "eps":
            raise RuntimeError("The differentiable DDIM adapter currently requires epsilon parameterization.")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        config = default_audioldm_config(model_name)
        pre = config["preprocessing"]
        self.sample_rate = int(pre["audio"]["sampling_rate"])
        self.target_frames = int(pre["mel"]["target_length"])
        self.hop_length = int(pre["stft"]["hop_length"])
        self.n_fft = int(pre["stft"]["filter_length"])
        self.win_length = int(pre["stft"]["win_length"])
        self.n_mels = int(pre["mel"]["n_mel_channels"])
        self.latent_channels = int(config["model"]["params"]["first_stage_config"]["params"]["embed_dim"])
        self.mel_fmin = float(pre["mel"]["mel_fmin"])
        self.mel_fmax = float(pre["mel"]["mel_fmax"])

        import librosa
        mel = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
        )
        self.mel_basis = torch.as_tensor(mel, dtype=torch.float32, device=self.device)
        self.window = torch.hann_window(
            self.win_length, periodic=True, dtype=torch.float32, device=self.device
        )

        self.model.cond_stage_key = "text"
        self.model.cond_stage_model.embed_mode = "text"
        self.clap_encoder = self.model.cond_stage_model
        self.clap_encoder.eval()
        self.clap_model = self.clap_encoder.model.eval()
        for parameter in self.clap_model.parameters():
            parameter.requires_grad_(False)
        self.vocoder = self.model.first_stage_model.vocoder.eval()
        for parameter in self.vocoder.parameters():
            parameter.requires_grad_(False)

    def waveform_to_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        """Differentiable counterpart of AudioLDM's 1024-frame log-mel frontend."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        target_samples = self.target_frames * self.hop_length
        waveform = fit_length(waveform, target_samples).to(self.device).float()
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs()
        mel = torch.einsum("mf,bft->btm", self.mel_basis, spec)
        log_mel = torch.log(mel.clamp_min(1e-5))
        log_mel = log_mel[:, : self.target_frames, :]
        if log_mel.shape[1] < self.target_frames:
            log_mel = F.pad(log_mel, (0, 0, 0, self.target_frames - log_mel.shape[1]))
        return log_mel.unsqueeze(1)

    def encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        fbank = self.waveform_to_fbank(waveform)
        posterior = self.model.first_stage_model.encode(fbank)
        latent = posterior.mode() if hasattr(posterior, "mode") else posterior
        if latent.ndim != 4 or latent.shape[1] != self.latent_channels:
            raise RuntimeError(
                f"Unexpected AudioLDM VAE latent shape {tuple(latent.shape)}; "
                f"expected channel dimension {self.latent_channels}."
            )
        return latent * self.model.scale_factor

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Differentiable VAE and HiFi-GAN decode; returns [B,T] at 16 kHz.

        The upstream AudioLDM style-transfer path removes the last three latent
        time frames before its final VAE decode to avoid unstable tail frames.
        We retain that crop while bypassing the upstream no-grad convenience API.
        """
        if latent.shape[2] <= 3:
            raise ValueError(f"Latent time axis is too short to apply AudioLDM decode crop: {tuple(latent.shape)}")
        latent = latent[:, :, :-3, :]
        mel = self.model.first_stage_model.decode(latent / self.model.scale_factor)
        mel = mel.squeeze(1).permute(0, 2, 1).contiguous()
        return self.vocoder(mel).squeeze(1)

    def encode_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        self.model.cond_stage_key = "text"
        self.clap_encoder.embed_mode = "text"
        return self.model.get_learned_conditioning(list(prompts)).to(self.device)

    def _clap_text_embeddings(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.clap_encoder.tokenizer(list(prompts))
        return self.clap_model.get_text_embedding(tokens)

    def clap_similarity(self, waveform: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        """Differentiable CLAP cosine similarity using AudioLDM's own CLAP weights."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device).float()
        if waveform.shape[0] != len(prompts):
            raise ValueError("Number of waveforms and prompts must match.")
        audio_48k = torchaudio.functional.resample(waveform, self.sample_rate, 48000)
        audio_48k = fit_length(audio_48k, 480000)
        audio_items = [{"waveform": audio_48k[i]} for i in range(audio_48k.shape[0])]
        audio_emb = self.clap_model.get_audio_embedding(audio_items)
        text_emb = self._clap_text_embeddings(prompts)
        return F.cosine_similarity(audio_emb, text_emb, dim=-1)

    def generate_from_latent(
        self,
        latent: torch.Tensor,
        prompts: Sequence[str],
        ddim_steps: int = 50,
        t_start: int | None = None,
        guidance_scale: float = 2.5,
        noise: torch.Tensor | None = None,
        seed: int = 2025,
    ) -> torch.Tensor:
        """Text-guided audio editing from a supplied AudioLDM VAE latent.

        z_t is formed with the DDIM forward-noising equation. Supply the same
        noise tensor to clean and gated calls for a paired comparison.
        """
        from audioldm.latent_diffusion.ddim import DDIMSampler

        latent = latent.to(self.device).float()
        if latent.shape[0] != len(prompts):
            raise ValueError("Latent batch and prompt batch sizes differ.")
        t_start = int(t_start if t_start is not None else max(1, round(ddim_steps * 0.5)))
        t_start = max(1, min(int(ddim_steps), t_start))
        sampler = DDIMSampler(self.model)
        sampler.make_schedule(ddim_num_steps=int(ddim_steps), ddim_eta=0.0, verbose=False)

        if noise is None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            noise = torch.randn(latent.shape, generator=generator, device=self.device, dtype=latent.dtype)
        else:
            noise = noise.to(self.device, dtype=latent.dtype)
        index = t_start - 1
        alpha_start = sampler.ddim_alphas[index].to(device=self.device, dtype=latent.dtype).clamp(1e-8, 1.0)
        current = alpha_start.sqrt() * latent + (1.0 - alpha_start).sqrt() * noise

        cond = self.encode_prompts(prompts)
        unconditional = None
        if guidance_scale != 1.0:
            unconditional = self.clap_encoder.get_unconditional_condition(len(prompts)).to(self.device)

        for step_index in reversed(range(t_start)):
            timestep = int(sampler.ddim_timesteps[step_index])
            ts = torch.full((latent.shape[0],), timestep, device=self.device, dtype=torch.long)
            if unconditional is None:
                eps = self.model.apply_model(current, ts, cond)
            else:
                model_in = torch.cat([current, current], dim=0)
                time_in = torch.cat([ts, ts], dim=0)
                cond_in = torch.cat([unconditional, cond], dim=0)
                eps_uncond, eps_cond = self.model.apply_model(model_in, time_in, cond_in).chunk(2)
                eps = eps_uncond + float(guidance_scale) * (eps_cond - eps_uncond)

            alpha_t = sampler.ddim_alphas[step_index].to(device=self.device, dtype=latent.dtype)
            alpha_prev = sampler.ddim_alphas_prev[step_index].to(device=self.device, dtype=latent.dtype)
            sqrt_one_minus = sampler.ddim_sqrt_one_minus_alphas[step_index].to(
                device=self.device, dtype=latent.dtype
            )
            pred_x0 = (current - sqrt_one_minus * eps) / alpha_t.clamp_min(1e-8).sqrt()
            direction = (1.0 - alpha_prev).clamp_min(0.0).sqrt() * eps
            current = alpha_prev.clamp_min(1e-8).sqrt() * pred_x0 + direction

        return self.decode_latent(current)

    def assert_gradient_path(self, latent: torch.Tensor, prompt: str = "A quiet rain sound.") -> dict:
        """Diagnostic proving that the generated CLAP score differentiates to latent."""
        z = latent[:1].detach().clone().requires_grad_(True)
        audio = self.generate_from_latent(
            z, [prompt], ddim_steps=2, t_start=1, guidance_scale=1.0, noise=torch.zeros_like(z)
        )
        score = self.clap_similarity(audio, [prompt]).mean()
        grad = torch.autograd.grad(score, z, allow_unused=True)[0]
        return {
            "audio_shape": list(audio.shape),
            "score": float(score.detach().cpu()),
            "gradient_exists": grad is not None,
            "gradient_norm": float(grad.norm().detach().cpu()) if grad is not None else 0.0,
            "finite": bool(torch.isfinite(grad).all()) if grad is not None else False,
        }
