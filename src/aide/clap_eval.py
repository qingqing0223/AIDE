from __future__ import annotations
from pathlib import Path
import torch
import torchaudio
import torch.nn.functional as F
from transformers import ClapModel,ClapProcessor

class CLAPScorer:
    def __init__(self,model_name="laion/clap-htsat-unfused",device="cuda"):
        self.device=torch.device(device if torch.cuda.is_available() else "cpu")
        self.model=ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.processor=ClapProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def score_files(self,audio_paths:list[str|Path],texts:list[str]) -> list[float]:
        audios=[]
        for p in audio_paths:
            wav,sr=torchaudio.load(str(p))
            if wav.shape[0]>1: wav=wav.mean(0,keepdim=True)
            wav=torchaudio.functional.resample(wav,sr,48000)
            audios.append(wav.squeeze(0).cpu().numpy())
        inputs=self.processor(text=texts,audios=audios,return_tensors="pt",padding=True,sampling_rate=48000)
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        out=self.model(**inputs)
        return F.cosine_similarity(out.audio_embeds,out.text_embeds,dim=-1).cpu().tolist()
