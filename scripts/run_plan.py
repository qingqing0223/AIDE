import argparse,json
from pathlib import Path
import yaml

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/base.yaml"); p.add_argument("--experiments",default="configs/experiments.yaml"); p.add_argument("--out",default="outputs/run_plan.json"); a=p.parse_args()
cfg=yaml.safe_load(Path(a.config).read_text()); exp=yaml.safe_load(Path(a.experiments).read_text())
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"config":cfg,"experiments":exp,"status":"planned"},ensure_ascii=False,indent=2),encoding="utf-8")
print(out)
