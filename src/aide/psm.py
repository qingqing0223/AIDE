from __future__ import annotations
from dataclasses import dataclass
import csv,json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModel
from sklearn.metrics import accuracy_score,f1_score

@dataclass
class PSMOutput:
    logits: torch.Tensor
    scalar_gate: torch.Tensor
    channel_gate: torch.Tensor
    hidden: torch.Tensor

class PromptSafetyModule(nn.Module):
    def __init__(self,backbone="roberta-base",num_classes=7,hidden_dim=256,channel_dim=512,dropout=0.1):
        super().__init__()
        self.tokenizer=AutoTokenizer.from_pretrained(backbone)
        self.encoder=AutoModel.from_pretrained(backbone)
        h=self.encoder.config.hidden_size
        self.proj=nn.Sequential(nn.Linear(h,hidden_dim),nn.GELU(),nn.Dropout(dropout))
        self.classifier=nn.Linear(hidden_dim,num_classes)
        self.scalar_head=nn.Linear(hidden_dim,1)
        self.channel_head=nn.Linear(hidden_dim,channel_dim)

    def forward(self,input_ids,attention_mask):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0]
        r=self.proj(h)
        return PSMOutput(self.classifier(r),torch.sigmoid(self.scalar_head(r)).squeeze(-1),
                         torch.sigmoid(self.channel_head(r)),r)

    def encode(self,prompts,device):
        b=self.tokenizer(prompts,padding=True,truncation=True,max_length=128,return_tensors="pt").to(device)
        return self(**b)

def psm_loss(out,labels,is_harmful,mu_benign=1.0,mu_harmful=1.0,mu_channel=0.05):
    cls=F.cross_entropy(out.logits,labels); safe=is_harmful<0.5; harmful=~safe
    terms=[]
    if safe.any(): terms.append(mu_benign*out.scalar_gate[safe].pow(2).mean())
    if harmful.any(): terms.append(mu_harmful*(1-out.scalar_gate[harmful]).pow(2).mean())
    gate=sum(terms) if terms else cls*0
    channel=mu_channel*F.mse_loss(out.channel_gate.mean(-1),out.scalar_gate.detach())
    return {"total":cls+gate+channel,"classification":cls,"gate":gate,"channel":channel}

class PromptDataset(Dataset):
    def __init__(self,manifest,tokenizer,max_length=128):
        self.rows=list(csv.DictReader(open(manifest,encoding="utf-8"))); self.tokenizer=tokenizer; self.max_length=max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; e=self.tokenizer(r["text"],truncation=True,max_length=self.max_length)
        return {"input_ids":e["input_ids"],"attention_mask":e["attention_mask"],
                "label":int(r["label"]),"is_harmful":int(r["is_harmful"]),"text":r["text"]}

class PromptCollator:
    def __init__(self,tokenizer,max_length=128): self.tokenizer=tokenizer; self.max_length=max_length
    def __call__(self,batch):
        texts=[x["text"] for x in batch]
        e=self.tokenizer(texts,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt")
        return {"input_ids":e["input_ids"],"attention_mask":e["attention_mask"],
                "label":torch.tensor([x["label"] for x in batch]),
                "is_harmful":torch.tensor([x["is_harmful"] for x in batch]).float()}

def train_psm(train_manifest,val_manifest,out_dir,epochs=10,batch_size=32,lr=2e-5,seed=2026):
    torch.manual_seed(seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=PromptSafetyModule().to(device); collator=PromptCollator(model.tokenizer)
    tr=DataLoader(PromptDataset(train_manifest,model.tokenizer),batch_size=batch_size,shuffle=True,collate_fn=collator)
    va=DataLoader(PromptDataset(val_manifest,model.tokenizer),batch_size=batch_size,shuffle=False,collate_fn=collator)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01); best=-1
    import pathlib; pathlib.Path(out_dir).mkdir(parents=True,exist_ok=True)
    for epoch in range(epochs):
        model.train()
        for b in tr:
            b={k:v.to(device) for k,v in b.items()}; o=model(b["input_ids"],b["attention_mask"])
            loss=psm_loss(o,b["label"],b["is_harmful"])["total"]
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        m=evaluate_psm(model,va,device); print(epoch+1,m)
        if m["macro_f1"]>best:
            best=m["macro_f1"]; torch.save(model.state_dict(),pathlib.Path(out_dir)/"psm_best.pt")
            pathlib.Path(out_dir,"metrics.json").write_text(json.dumps(m,indent=2),encoding="utf-8")
    return pathlib.Path(out_dir)/"psm_best.pt"

@torch.no_grad()
def evaluate_psm(model,loader,device):
    model.eval(); y=[]; p=[]
    for b in loader:
        o=model(b["input_ids"].to(device),b["attention_mask"].to(device))
        y+=b["label"].tolist(); p+=o.logits.argmax(-1).cpu().tolist()
    return {"accuracy":float(accuracy_score(y,p)),"macro_f1":float(f1_score(y,p,average="macro")),
            "weighted_f1":float(f1_score(y,p,average="weighted"))}

def fuse_conditions(clean,protected,scalar_gate,channel_gate):
    s=scalar_gate
    while s.dim()<protected.dim(): s=s.unsqueeze(-1)
    g=channel_gate
    if g.dim()==2 and protected.dim()==3: g=g.unsqueeze(1)
    gate=(s*g).clamp(0,1)
    return (1-gate)*clean+gate*protected
