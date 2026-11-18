from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import librosa

class AudioLDMAdapter:
    """
    Official AudioLDM wrapper plus a differentiable first-stage preprocessing path.
    The official wav_to_fbank helper converts to NumPy and detaches; that path is
    unsuitable for PGD gradients, so this module reproduces the STFT/mel geometry
    with torch operations before calling the frozen VAE encoder.
    """
    def __init__(self,checkpoint=None,model_name="audioldm-s-full",device="cuda"):
        from audioldm.pipeline import build_model
        self.device=torch.device(device if torch.cuda.is_available() else "cpu")
        self.model=build_model(ckpt_path=checkpoint,model_name=model_name).to(self.device).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        basis=librosa.filters.mel(sr=16000,n_fft=1024,n_mels=64,fmin=0,fmax=8000)
        self.mel_basis=torch.from_numpy(basis).float().to(self.device)

    def differentiable_fbank(self,wav):
        x=wav.squeeze(1) if wav.dim()==3 else wav
        window=torch.hann_window(1024,device=x.device,dtype=x.dtype)
        spec=torch.stft(x,n_fft=1024,hop_length=160,win_length=1024,window=window,
                        center=True,pad_mode="reflect",return_complex=True)
        mag=spec.abs()
        mel=torch.einsum("mf,bft->bmt",self.mel_basis,mag)
        mel=torch.log(mel.clamp_min(1e-5)).transpose(1,2).unsqueeze(1)
        if mel.shape[2]<1024: mel=F.pad(mel,(0,0,0,1024-mel.shape[2]))
        else: mel=mel[:,:,:1024,:]
        return mel

    def encode_waveform(self,wav):
        mel=self.differentiable_fbank(wav.to(self.device))
        posterior=self.model.encode_first_stage(mel)
        z=posterior.mode() if hasattr(posterior,"mode") else posterior
        return self.model.scale_factor*z

    @torch.no_grad()
    def audio_condition(self,wav):
        self.model.cond_stage_model.embed_mode="audio"
        x=wav.to(self.device); x=x.squeeze(1) if x.dim()==3 else x
        return self.model.cond_stage_model(x)

    @torch.no_grad()
    def unconditional_condition(self,batch):
        return self.model.cond_stage_model.get_unconditional_condition(batch)

    @torch.no_grad()
    def decode(self,z):
        mel=self.model.decode_first_stage(z)
        wav=self.model.mel_spectrogram_to_waveform(mel)
        return torch.from_numpy(wav).float().to(self.device)

    def ddim_schedule(self,steps=50,eta=1.0):
        alpha=self.model.alphas_cumprod.detach().float().cpu().numpy()
        idx=np.linspace(0,len(alpha)-1,steps,dtype=int)
        a=torch.tensor(alpha[idx],device=self.device); ap=torch.tensor(np.r_[1.0,alpha[idx[:-1]]],device=self.device)
        sigma=eta*torch.sqrt((1-ap)/(1-a)*(1-a/ap))
        return torch.tensor(idx,device=self.device),a,ap,sigma

    @torch.no_grad()
    def generate(self,condition,steps=50,guidance_scale=2.5,seed=42,x_T=None):
        torch.manual_seed(seed); b=condition.shape[0]
        shape=(b,self.model.channels,self.model.latent_t_size,self.model.latent_f_size)
        x=torch.randn(shape,device=self.device) if x_T is None else x_T.to(self.device)
        uc=self.unconditional_condition(b); ts,a,ap,sig=self.ddim_schedule(steps)
        for i in reversed(range(steps)):
            t=torch.full((b,),int(ts[i]),device=self.device,dtype=torch.long)
            eps_c=self.model.apply_model(x,t,condition); eps_u=self.model.apply_model(x,t,uc)
            eps=eps_u+guidance_scale*(eps_c-eps_u)
            aa=a[i]; prev=ap[i]; ss=sig[i]
            x0=(x-torch.sqrt(1-aa)*eps)/torch.sqrt(aa)
            direction=torch.sqrt(torch.clamp(1-prev-ss**2,min=0))*eps
            x=torch.sqrt(prev)*x0+direction
        return self.decode(x)
