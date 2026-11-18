import argparse,torch
from aide.audio import load_audio,save_audio
from aide.audioldm import AudioLDMAdapter
from aide.psm import PromptSafetyModule,fuse_conditions
p=argparse.ArgumentParser()
p.add_argument("--clean",required=True); p.add_argument("--protected",required=True); p.add_argument("--prompt",required=True); p.add_argument("--psm",required=True); p.add_argument("--checkpoint",default=None); p.add_argument("--out",default="outputs/generated.wav"); p.add_argument("--steps",type=int,default=50)
a=p.parse_args(); device="cuda" if torch.cuda.is_available() else "cpu"; adapter=AudioLDMAdapter(a.checkpoint,device=device)
psm=PromptSafetyModule().to(adapter.device); psm.load_state_dict(torch.load(a.psm,map_location=adapter.device)); psm.eval()
clean=load_audio(a.clean).to(adapter.device); protected=load_audio(a.protected).to(adapter.device)
with torch.no_grad():
    cc=adapter.audio_condition(clean); cp=adapter.audio_condition(protected); po=psm.encode([a.prompt],adapter.device); cond=fuse_conditions(cc,cp,po.scalar_gate,po.channel_gate); wav=adapter.generate(cond,steps=a.steps)
save_audio(a.out,wav.squeeze(0),16000); print("saved",a.out)
