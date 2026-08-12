#!/usr/bin/env python3
"""Friendly input/output TENSOR sizes for the user's real regime: small channels, large
rectangular spatial, N=1, non-winograd. Base = conv1 in[1,18,288,384] W[16,18,3,3] s1.
Sweeps Cin, Cout, W (innermost), H independently. buffer/fp16. -> conv_bench/real_regime_sweep.csv
"""
import csv, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
from gen_conv import make_conv
from bench import convert, run_on_device, parse, ensure_device
W = REPO / "conv_bench" / "sweepwork"; W.mkdir(exist_ok=True)
OUT = REPO / "conv_bench" / "real_regime_sweep.csv"
BASE = dict(N=1, Cin=18, Cout=16, H=288, Wd=384, k=3, stride=1)

def pt(sweep, **kw):
    d = dict(BASE); d.update(kw); d["sweep"]=sweep; return d

cases  = [pt("Cin",  Cin=c)  for c in [1,4,8,16,17,18,19,20,24,32,36,48,64]]
cases += [pt("Cout", Cout=c) for c in [8,12,15,16,17,20,24,32,48,64,96]]
cases += [pt("W",    Wd=w)   for w in [376,380,382,383,384,385,386,388,392,400]]
cases += [pt("H",    H=h)    for h in [280,284,286,287,288,289,290,292,296]]

FIELDS = ["sweep","N","Cin","Cout","H","W","conv_us","total_us","path","tag"]
def done_tags():
    if not OUT.exists(): return set()
    import csv as _c
    with open(OUT) as f: return {r["tag"] for r in _c.DictReader(f)}

def main():
    ensure_device(push=True)
    have = done_tags()
    new = not OUT.exists()
    f = open(OUT, "a", newline=""); wtr = csv.DictWriter(f, fieldnames=FIELDS)
    if new: wtr.writeheader(); f.flush()
    for i,c in enumerate(cases):
        tag=f"r_{c['sweep']}_{i}"
        if tag in have: continue
        N,Cin,Cout,H,Wd,k,stride = c["N"],c["Cin"],c["Cout"],c["H"],c["Wd"],c["k"],c["stride"]
        onnx=W/f"{tag}.onnx"; mnn=W/f"{tag}.mnn"; pad=k//2
        make_conv(str(onnx),N,Cin,Cout,H,Wd,k,k,stride,pad,1,1,"none")
        convert(str(onnx),str(mnn))
        out=run_on_device(str(mnn),"input",[N,Cin,H,Wd],"output",loops=50,gpu_mode=68,prec_mem_mask=2,tuning_cache=f"{tag}.cache")
        r=parse(out); path="winograd" if "winograd" in out else ("gemm" if "gemm" in out else "direct")
        row=dict(sweep=c["sweep"],N=N,Cin=Cin,Cout=Cout,H=H,W=Wd,conv_us=r.get("conv_us_med"),
                 total_us=r.get("total_kernel_us_med"),path=path,tag=tag)
        wtr.writerow(row); f.flush()
        print(f"[{i+1:2}/{len(cases)}] {c['sweep']:4} Cin{Cin} Cout{Cout} {H}x{Wd}: conv={r.get('conv_us_med')} tot={r.get('total_kernel_us_med')} {path}", flush=True)
    f.close()
    print(f"\ndone -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
