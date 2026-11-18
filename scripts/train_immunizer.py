import argparse
import json
from pathlib import Path

import torch

from aide.audio import load_audio, save_audio, pad_or_trim
from aide.audioldm import AudioLDMAdapter
from aide.immunization import LatentImmunizer, PGDConfig


p = argparse.ArgumentParser()
p.add_argument("--clean", required=True)
p.add_argument("--target", required=True)
p.add_argument("--checkpoint", default=None)
p.add_argument("--model-name", default="audioldm-s-full")
p.add_argument("--out", default="outputs/immunized.wav")
p.add_argument("--steps", type=int, default=40)
p.add_argument("--alpha", type=float, default=0.001)
p.add_argument("--epsilon", type=float, default=0.01)
p.add_argument("--lambda-target", type=float, default=1.0)
p.add_argument("--lambda-far", type=float, default=0.2)
p.add_argument("--lambda-mask", type=float, default=0.1)
p.add_argument("--masking-gamma", type=float, default=0.8)
a = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
adapter = AudioLDMAdapter(
    checkpoint=a.checkpoint,
    model_name=a.model_name,
    device=device,
)

clean = pad_or_trim(load_audio(a.clean), 160000).to(adapter.device)
target = pad_or_trim(load_audio(a.target), clean.shape[-1]).to(adapter.device)

immunizer = LatentImmunizer(
    encoder=lambda x: adapter.encode_audio_condition(
        x, differentiable=True
    ),
    cfg=PGDConfig(
        epsilon=a.epsilon,
        alpha=a.alpha,
        steps=a.steps,
        lambda_target=a.lambda_target,
        lambda_far=a.lambda_far,
        lambda_mask=a.lambda_mask,
        gamma=a.masking_gamma,
    ),
)

delta, protected, history = immunizer.optimize(clean, target)
save_audio(a.out, protected, 16000)

Path(a.out + ".json").write_text(
    json.dumps(
        {
            "history": history,
            "linf": float(delta.abs().max()),
            "epsilon": a.epsilon,
            "model_name": a.model_name,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("saved", a.out)
