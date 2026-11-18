import argparse,json
from pathlib import Path
import torchaudio
p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--mic",choices=["mic1","mic2"],default="mic1"); p.add_argument("--out",default="data/manifests"); p.add_argument("--download",action="store_true"); a=p.parse_args()
ds=torchaudio.datasets.VCTK_092(a.root,mic_id=a.mic,download=a.download); rows=[]
for i in range(len(ds)):
    wav,sr,text,speaker,utterance=ds[i]; rows.append({"id":utterance,"speaker":speaker,"text":text,"sample_rate":sr})
out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/f"vctk_{a.mic}.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8"); print("records",len(rows))
