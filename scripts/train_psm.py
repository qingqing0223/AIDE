import argparse
from aide.psm import train_psm
p=argparse.ArgumentParser(); p.add_argument("--train",required=True); p.add_argument("--val",required=True); p.add_argument("--out",default="checkpoints/psm"); p.add_argument("--epochs",type=int,default=10)
a=p.parse_args(); train_psm(a.train,a.val,a.out,epochs=a.epochs)
