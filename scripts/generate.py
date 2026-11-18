import argparse
import torch

from aide.audio import load_audio, save_audio, pad_or_trim
from aide.audioldm import AudioLDMAdapter
from aide.psm import PromptSafetyModule


p = argparse.ArgumentParser()
p.add_argument("--clean", required=True)
p.add_argument("--protected", required=True)
p.add_argument("--prompt", required=True)
p.add_argument("--psm", required=True)
p.add_argument("--checkpoint", default=None)
p.add_argument("--model-name", default="audioldm-s-full")
p.add_argument("--out", default="outputs/generated.wav")
p.add_argument("--steps", type=int, default=50)
p.add_argument("--guidance", type=float, default=1.0)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

adapter = AudioLDMAdapter(
    checkpoint=a.checkpoint,
    model_name=a.model_name,
    device=device,
)

psm = PromptSafetyModule().to(adapter.device)
ckpt = torch.load(a.psm, map_location=adapter.device, weights_only=False)
state = ckpt.get("model_state_dict", ckpt)
psm.load_state_dict(state)
psm.eval()

clean = pad_or_trim(load_audio(a.clean), 160000).to(adapter.device)
protected = pad_or_trim(load_audio(a.protected), 160000).to(adapter.device)

with torch.no_grad():
    out = psm.encode([a.prompt], adapter.device)

    z_clean = adapter.encode_audio_condition(
        clean, differentiable=False
    )
    z_protected = adapter.encode_audio_condition(
        protected, differentiable=False
    )

    z_gate = adapter.fuse_conditions(
        z_clean,
        z_protected,
        out.scalar_gate,
        out.channel_gate,
    )

    waveform = adapter.generate(
        z_gate,
        steps=a.steps,
        guidance_scale=a.guidance,
        seed=a.seed,
        eta=0.0,
    )

save_audio(a.out, waveform.squeeze(0), 16000)
print("saved", a.out)
