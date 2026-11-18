import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--input",nargs="+",required=True); p.add_argument("--mapping",required=True); p.add_argument("--out",required=True); a=p.parse_args()
mapping=json.loads(Path(a.mapping).read_text(encoding="utf-8")); rows=[]; idx=0
for src in a.input:
    obj=json.loads(Path(src).read_text(encoding="utf-8"))
    items=[("",x) for x in obj] if isinstance(obj,list) else [(k,v) for k,vals in obj.items() if isinstance(vals,list) for v in vals]
    for category,item in items:
        text=item if isinstance(item,str) else item.get("text",item.get("prompt","")); text=str(text).strip()
        if not text: continue
        label=int(mapping.get(category,0))
        if not 0<=label<=6: raise ValueError(f"invalid label {label}")
        rows.append({"id":f"p_{idx:07d}","text":text,"label":label,"is_harmful":int(label>0),"source":src,"split":"train"}); idx+=1
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
with open(out,"w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["id","text","label","is_harmful","source","split"]); w.writeheader(); w.writerows(rows)
print("records",len(rows))
