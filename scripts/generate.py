import argparse,torch
from aide.audio import load_audio,save_audio
from aide.audioldm import AudioLDMAdapter
from aide.psm import PromptSafetyModule

p=argparse.ArgumentParser()
p.add_argument("--clean",required=True)
p.add_argument("--protected",required=True)
p.add_argument("--prompt",required=True)
p.add_argument("--psm",required=True)
p.add_argument("--checkpoint",default=None)
p.add_argument("--out",default="outputs/generated.wav")
p.add_argument("--steps",type=int,default=50)
p.add_argument("--guidance",type=float,default=2.5)
p.add_argument("--mode",choices=["text_plus_residual","blend_audio"],default="text_plus_residual")
p.add_argument("--seed",type=int,default=42)
a=p.parse_args()

device="cuda" if torch.cuda.is_available() else "cpu"
adapter=AudioLDMAdapter(a.checkpoint,device=device)
psm=PromptSafetyModule().to(adapter.device)
psm.load_state_dict(torch.load(a.psm,map_location=adapter.device)); psm.eval()
clean=load_audio(a.clean).to(adapter.device)
protected=load_audio(a.protected).to(adapter.device)

with torch.no_grad():
    psm_out=psm.encode([a.prompt],adapter.device)
    wav=adapter.generate_controlled(
        a.prompt,clean,protected,psm_out.scalar_gate,psm_out.channel_gate,
        steps=a.steps,guidance_scale=a.guidance,seed=a.seed,mode=a.mode
    )

save_audio(a.out,wav.squeeze(0),16000)
print("saved",a.out)
