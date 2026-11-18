import argparse
from aide.audio import load_audio
from aide.evaluation import signal_report,save_report
p=argparse.ArgumentParser(); p.add_argument("--clean",required=True); p.add_argument("--protected",required=True); p.add_argument("--out",default="outputs/signal_metrics.json"); a=p.parse_args()
x=load_audio(a.clean); y=load_audio(a.protected); n=min(x.shape[-1],y.shape[-1]); r=signal_report(x[...,:n],y[...,:n]); save_report(a.out,r); print(r)
