from __future__ import annotations
import json
from pathlib import Path
import torch
from aide.audio import stft_magnitude
from aide.immunization import psychoacoustic_threshold

def snr_db(clean,protected,eps=1e-12):
    d=protected-clean
    return float(10*torch.log10(clean.pow(2).mean()/(d.pow(2).mean()+eps)))

def lsd_db(clean,protected,eps=1e-10):
    a=stft_magnitude(clean).clamp_min(eps); b=stft_magnitude(protected).clamp_min(eps)
    f=min(a.shape[-2],b.shape[-2]); t=min(a.shape[-1],b.shape[-1])
    d=torch.log10(a[...,:f,:t])-torch.log10(b[...,:f,:t])
    return float(10*torch.sqrt(d.pow(2).mean()))

def masking_violation(clean,delta,gamma=0.8):
    a=stft_magnitude(delta); b=psychoacoustic_threshold(clean)
    f=min(a.shape[-2],b.shape[-2]); t=min(a.shape[-1],b.shape[-1])
    return float((a[...,:f,:t]-gamma*b[...,:f,:t]).clamp_min(0).mean())

def signal_report(clean,protected,gamma=0.8):
    d=protected-clean
    return {"snr_db":snr_db(clean,protected),"lsd_db":lsd_db(clean,protected),
            "masking_violation":masking_violation(clean,d,gamma),
            "linf":float(d.abs().max()),"l2":float(d.pow(2).sum().sqrt())}

def save_report(path,report):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2),encoding="utf-8")
