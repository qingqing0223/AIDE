import argparse,json
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument("--generated",required=True)
p.add_argument("--reference",required=True)
p.add_argument("--out",default="outputs/generation_metrics.json")
p.add_argument("--backbone",default="cnn14")
a=p.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    from audioldm_eval import EvaluationHelper
except Exception as e:
    raise SystemExit("Install optional requirements-eval.txt first.") from e

evaluator=EvaluationHelper(16000,device)
metrics=evaluator.main(a.generated,a.reference,backbone=a.backbone,limit_num=None)
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps(metrics,indent=2),encoding="utf-8")
print(json.dumps(metrics,indent=2))
