from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from aide.audio import stft_magnitude

@dataclass
class PGDConfig:
    epsilon: float=0.01; alpha: float=0.001; steps: int=40
    lambda_target: float=1.0; lambda_far: float=0.2
    lambda_mask: float=0.1; gamma: float=0.8

def psychoacoustic_threshold(clean, smoothing=9):
    power=stft_magnitude(clean).pow(2)
    x=power.unsqueeze(1)
    return F.avg_pool2d(x,(1,smoothing),1,(0,smoothing//2)).squeeze(1).sqrt().clamp_min(1e-5)

def masking_loss(clean, delta, gamma):
    a=stft_magnitude(delta); b=psychoacoustic_threshold(clean)
    f=min(a.shape[-2],b.shape[-2]); t=min(a.shape[-1],b.shape[-1])
    excess=(a[...,:f,:t]-gamma*b[...,:f,:t]).clamp_min(0)
    return excess.pow(2).mean()

class LatentImmunizer:
    def __init__(self, encoder, cfg: PGDConfig):
        self.encoder=encoder; self.cfg=cfg

    def optimize(self, clean, target):
        clean=clean.detach()
        with torch.no_grad():
            z_clean=self.encoder(clean); z_target=self.encoder(target)
        delta=torch.zeros_like(clean); history=[]
        for step in range(self.cfg.steps):
            delta.requires_grad_(True)
            protected=(clean+delta).clamp(-1,1)
            z_p=self.encoder(protected)
            target_term=F.mse_loss(z_p,z_target)
            far_term=F.mse_loss(z_p,z_clean)
            mask_term=masking_loss(clean,delta,self.cfg.gamma)
            loss=(self.cfg.lambda_target*target_term
                  -self.cfg.lambda_far*far_term
                  +self.cfg.lambda_mask*mask_term)
            grad=torch.autograd.grad(loss,delta)[0]
            with torch.no_grad():
                delta=delta-self.cfg.alpha*grad.sign()
                delta=delta.clamp(-self.cfg.epsilon,self.cfg.epsilon)
                delta=(clean+delta).clamp(-1,1)-clean
            history.append({"step":step,"loss":float(loss.detach()),
                            "target":float(target_term.detach()),
                            "far":float(far_term.detach()),
                            "mask":float(mask_term.detach()),
                            "linf":float(delta.abs().max())})
        return delta.detach(),(clean+delta).clamp(-1,1).detach(),history
