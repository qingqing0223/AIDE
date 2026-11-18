import argparse,csv
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--csv",required=True,help="AudioCaps metadata CSV")
p.add_argument("--audio-root",required=True)
p.add_argument("--split",required=True)
p.add_argument("--out",required=True)
a=p.parse_args()

rows=[]
with open(a.csv,encoding="utf-8") as f:
    for r in csv.DictReader(f):
        aid=str(r["audiocap_id"])
        wav=Path(a.audio_root)/f"{aid}.wav"
        rows.append({
            "id":aid,
            "path":str(wav),
            "exists":wav.exists(),
            "split":a.split,
            "text":r.get("caption",""),
            "youtube_id":r.get("youtube_id",""),
            "start_time":r.get("start_time",""),
        })

out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
with open(out,"w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["id","path","exists","split","text","youtube_id","start_time"])
    w.writeheader(); w.writerows(rows)

print("records",len(rows),"missing",sum(not x["exists"] for x in rows))
