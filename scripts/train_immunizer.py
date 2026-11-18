import argparse,json
from pathlib import Path
import torch
from aide.audio import load_audio,save_audio,pad_or_trim
from aide.audioldm import AudioLDMAdapter
from aide.immunization import LatentImmunizer,PGDConfig

p=argparse.ArgumentParser()
p.add_argument("--clean",required=True); p.add_argument("--target",required=True); p.add_argument("--checkpoint",default=None)
p.add_argument("--out",default="outputs/immunized.wav"); p.add_argument("--steps",type=int,default=40); p.add_argument("--alpha",type=float,default=.001); p.add_argument("--epsilon",type=float,default=.01)
a=p.parse_args()
device="cuda" if torch.cuda.is_available() else "cpu"; adapter=AudioLDMAdapter(a.checkpoint,device=device)
clean=load_audio(a.clean).to(adapter.device); target=pad_or_trim(load_audio(a.target).to(adapter.device),clean.shape[-1])
attack=LatentImmunizer(adapter.encode_waveform,PGDConfig(epsilon=a.epsilon,alpha=a.alpha,steps=a.steps))
delta,protected,history=attack.optimize(clean,target)
save_audio(a.out,protected); Path(a.out+".json").write_text(json.dumps({"history":history,"linf":float(delta.abs().max())},indent=2),encoding="utf-8")
print("saved",a.out)
