from __future__ import annotations

from typing import Iterable
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


class AudioLDMAdapter:
    """
    AudioLDM adapter used by AIDE.

    Paper alignment:
      - the protected/clean/target representation is kept as an AudioLDM
        conditioning representation, not the first-stage VAE reconstruction latent;
      - the PSM gates are applied to the clean/protected conditioning pair;
      - generation uses the official AudioLDM DDIM sampler for inference.

    Implementation supplement:
      - AudioLDM's public conditioning wrapper detaches its CLAP output.
        For PGD/PSM generation-coupled training we therefore call the frozen
        underlying CLAP model directly with differentiable waveform tensors.
      - the exact paper text does not expose every checkpoint/config detail;
        those remain explicit constructor/config parameters.
    """

    def __init__(
        self,
        checkpoint=None,
        model_name="audioldm-s-full",
        device="cuda",
        sample_rate=16000,
        clap_sample_rate=48000,
        clap_max_samples=480000,
    ):
        from audioldm.pipeline import build_model

        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.sample_rate = int(sample_rate)
        self.clap_sample_rate = int(clap_sample_rate)
        self.clap_max_samples = int(clap_max_samples)

        self.model = build_model(
            ckpt_path=checkpoint,
            model_name=model_name,
            device=self.device,
        ).to(self.device).eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.cond_stage = self.model.cond_stage_model
        self.cond_stage.eval()
        for p in self.cond_stage.parameters():
            p.requires_grad_(False)

        self.clap = getattr(self.cond_stage, "model", None)
        if self.clap is None or not hasattr(self.clap, "get_audio_embedding"):
            raise RuntimeError(
                "AudioLDM checkpoint does not expose the expected frozen CLAP "
                "conditioning model. AIDE requires an audio-condition encoder."
            )

    # ------------------------------------------------------------------
    # Waveform -> AudioLDM CLAP condition
    # ------------------------------------------------------------------
    def _prepare_clap_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.float()
        if x.dim() == 3:
            if x.shape[1] != 1:
                x = x.mean(dim=1, keepdim=True)
            x = x[:, 0, :]
        elif x.dim() == 2:
            pass
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported waveform shape: {tuple(x.shape)}")

        if self.sample_rate != self.clap_sample_rate:
            x = torchaudio.functional.resample(
                x, self.sample_rate, self.clap_sample_rate
            )

        if x.shape[-1] > self.clap_max_samples:
            x = x[..., : self.clap_max_samples]
        elif x.shape[-1] < self.clap_max_samples:
            x = F.pad(x, (0, self.clap_max_samples - x.shape[-1]))

        return x.contiguous()

    def _clap_input_dicts(self, wav_48k: torch.Tensor):
        enable_fusion = bool(getattr(self.cond_stage, "enable_fusion", False))

        if not enable_fusion:
            return [{"waveform": wav_48k[i]} for i in range(wav_48k.shape[0])]

        # AudioLDM's fusion CLAP expects the same 4-view mel_fusion layout
        # used by its official conditioning wrapper.
        mel_transform = getattr(self.cond_stage, "mel_transform", None)
        if mel_transform is None:
            raise RuntimeError(
                "Fusion CLAP is enabled but AudioLDM exposes no mel_transform."
            )

        mel = mel_transform(wav_48k)
        # mel: [B, 64, T] -> [B, T, 64]
        mel = mel.transpose(1, 2)
        mel_fusion = torch.stack([mel, mel, mel, mel], dim=1)
        longer = torch.zeros(
            wav_48k.shape[0], dtype=torch.bool, device=wav_48k.device
        )

        return [
            {
                "waveform": wav_48k[i],
                "mel_fusion": mel_fusion[i],
                "longer": longer[i],
            }
            for i in range(wav_48k.shape[0])
        ]

    def encode_audio_condition(
        self, wav: torch.Tensor, differentiable: bool = False
    ) -> torch.Tensor:
        """
        Return the frozen AudioLDM audio-conditioning embedding.

        Shape:
            [B, D] when returned here.
        AudioLDM's standard CLAP condition path reshapes this to [B, 1, D]
        before cross-attention.
        """
        wav_48k = self._prepare_clap_waveform(wav)

        # Never disable gradients when AIDE is optimizing waveform perturbations.
        context = torch.enable_grad() if differentiable else torch.no_grad()
        with context:
            out = self.clap.get_audio_embedding(
                self._clap_input_dicts(wav_48k)
            )

        return out

    @torch.no_grad()
    def audio_condition(self, wav: torch.Tensor) -> torch.Tensor:
        return self.encode_audio_condition(wav, differentiable=False).unsqueeze(1)

    # ------------------------------------------------------------------
    # Text / unconditional AudioLDM conditions
    # ------------------------------------------------------------------
    @torch.no_grad()
    def text_condition(self, prompts: list[str]) -> torch.Tensor:
        tokenizer = getattr(self.cond_stage, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("AudioLDM CLAP tokenizer is unavailable.")
        data = tokenizer(prompts)
        emb = self.clap.get_text_embedding(data)
        return emb.unsqueeze(1)

    @torch.no_grad()
    def unconditional_condition(self, batch_size: int) -> torch.Tensor:
        return self.cond_stage.get_unconditional_condition(batch_size)

    # ------------------------------------------------------------------
    # AIDE gate fusion
    # ------------------------------------------------------------------
    @staticmethod
    def fuse_conditions(
        clean_condition: torch.Tensor,
        protected_condition: torch.Tensor,
        scalar_gate: torch.Tensor,
        channel_gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Paper:
            z_tilde = (1 - s*g) ⊙ z_c + (s*g) ⊙ z_p
                    = z_c + s*g ⊙ (z_p-z_c)

        Inputs:
            clean/protected: [B,D] or [B,1,D]
            scalar_gate:     [B]
            channel_gate:    [B,D]
        """
        zc = clean_condition
        zp = protected_condition
        if zc.dim() == 2:
            zc = zc.unsqueeze(1)
        if zp.dim() == 2:
            zp = zp.unsqueeze(1)

        s = scalar_gate.reshape(-1, 1, 1)
        g = channel_gate
        if g.dim() == 2:
            g = g.unsqueeze(1)

        if zc.shape != zp.shape:
            raise ValueError(
                f"clean/protected condition shape mismatch: {zc.shape} vs {zp.shape}"
            )
        if g.shape[-1] != zc.shape[-1]:
            raise ValueError(
                f"channel gate dimension {g.shape[-1]} does not match "
                f"condition dimension {zc.shape[-1]}"
            )

        gate = (s * g).clamp(0.0, 1.0)
        return (1.0 - gate) * zc + gate * zp

    # ------------------------------------------------------------------
    # Official inference sampler
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        steps: int = 50,
        guidance_scale: float = 2.5,
        seed: int = 42,
        x_T: torch.Tensor | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Use AudioLDM's own DDIMSampler/sample_log interface instead of a
        hand-written approximation of the sampler.
        """
        from audioldm.latent_diffusion.ddim import DDIMSampler

        torch.manual_seed(seed)
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)

        batch_size = condition.shape[0]
        uc = (
            self.unconditional_condition(batch_size)
            if guidance_scale != 1.0
            else None
        )

        shape = (
            self.model.channels,
            self.model.latent_t_size,
            self.model.latent_f_size,
        )
        sampler = DDIMSampler(self.model)
        samples, _ = sampler.sample(
            S=int(steps),
            batch_size=batch_size,
            shape=shape,
            conditioning=condition,
            unconditional_guidance_scale=float(guidance_scale),
            unconditional_conditioning=uc,
            eta=float(eta),
            x_T=x_T,
            verbose=False,
        )

        mel = self.model.decode_first_stage(samples)
        waveform = self.model.mel_spectrogram_to_waveform(mel)
        return torch.from_numpy(waveform).float().to(self.device)

    # ------------------------------------------------------------------
    # Differentiable DDIM for the PSM generation-coupled loss
    # ------------------------------------------------------------------
    def _ddim_schedule(self, steps: int, eta: float = 0.0):
        from audioldm.latent_diffusion.ddim import DDIMSampler

        sampler = DDIMSampler(self.model)
        sampler.make_schedule(
            ddim_num_steps=int(steps),
            ddim_eta=float(eta),
            verbose=False,
        )
        return sampler

    def generate_differentiable(
        self,
        condition: torch.Tensor,
        steps: int = 10,
        guidance_scale: float = 1.0,
        seed: int = 42,
        x_T: torch.Tensor | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Differentiable counterpart of AudioLDM DDIM sampling.

        The diffusion model and VAE remain frozen. Gradients are retained
        through the conditioning tensor so L_gen can update the PSM.
        """
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)

        sampler = self._ddim_schedule(steps, eta)
        b = condition.shape[0]
        device = condition.device

        torch.manual_seed(seed)
        shape = (
            b,
            self.model.channels,
            self.model.latent_t_size,
            self.model.latent_f_size,
        )
        x = torch.randn(shape, device=device) if x_T is None else x_T.to(device)

        uc = (
            self.unconditional_condition(b)
            if guidance_scale != 1.0
            else None
        )

        timesteps = np.flip(sampler.ddim_timesteps)
        total_steps = len(timesteps)

        for i, step in enumerate(timesteps):
            index = total_steps - i - 1
            t = torch.full(
                (b,), int(step), device=device, dtype=torch.long
            )

            if uc is None or guidance_scale == 1.0:
                eps = self.model.apply_model(x, t, condition)
            else:
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t, t], dim=0)
                c_in = torch.cat([uc, condition], dim=0)
                eps_u, eps_c = self.model.apply_model(
                    x_in, t_in, c_in
                ).chunk(2)
                eps = eps_u + float(guidance_scale) * (eps_c - eps_u)

            a_t = sampler.ddim_alphas[index].to(device).reshape(1, 1, 1, 1)
            a_prev = sampler.ddim_alphas_prev[index].to(device).reshape(
                1, 1, 1, 1
            )
            sigma_t = sampler.ddim_sigmas[index].to(device).reshape(
                1, 1, 1, 1
            )
            sqrt_one_minus = sampler.ddim_sqrt_one_minus_alphas[index].to(
                device
            ).reshape(1, 1, 1, 1)

            pred_x0 = (x - sqrt_one_minus * eps) / a_t.sqrt()
            direction = torch.sqrt(
                torch.clamp(1.0 - a_prev - sigma_t.square(), min=0.0)
            ) * eps

            if float(eta) == 0.0:
                noise = torch.zeros_like(x)
            else:
                noise = sigma_t * torch.randn_like(x)

            x = a_prev.sqrt() * pred_x0 + direction + noise

        # AudioLDM's public decode helper is @no_grad. Calling the underlying
        # frozen first-stage model directly preserves gradients to condition.
        z = x / self.model.scale_factor
        mel = self.model.first_stage_model.decode(z)
        waveform = self.model.first_stage_model.vocoder(
            mel.squeeze(1).permute(0, 2, 1)
        )
        return waveform

    # ------------------------------------------------------------------
    # CLAP similarity used by L_gen
    # ------------------------------------------------------------------
    def differentiable_text_audio_similarity(
        self, waveform: torch.Tensor, prompts: list[str]
    ) -> torch.Tensor:
        audio_emb = self.encode_audio_condition(
            waveform, differentiable=True
        )
        text_emb = self.text_condition(prompts).squeeze(1)
        return F.cosine_similarity(audio_emb, text_emb, dim=-1)
