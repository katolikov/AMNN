#!/usr/bin/env python3
"""Isolate INPUT/OUTPUT activation-tensor friendly sizes (beyond weight params).

Sweeps width and height INDEPENDENTLY (my earlier sweep coupled H=W), plus a stride-2 case,
to find spatial-alignment cliffs on the innermost (W) vs outer (H) dims. buffer/fp16, k3.
Writes conv_bench/tensor_sweep.csv.
"""
import csv, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
from gen_conv import make_conv
from bench import convert, run_on_device, parse, ensure_device

W = REPO / "conv_bench" / "sweepwork"; W.mkdir(exist_ok=True)
OUT = REPO / "conv_bench" / "tensor_sweep.csv"
N, C = 8, 96

cases = []
# width alone (H fixed 64)
for w in [60,62,63,64,65,66,68,70,72,80,96,127,128]:  cases.append(("W_only", N, C, C, 64, w, 3, 1))
# height alone (W fixed 64)
for h in [60,62,63,64,65,66,68,70,72,80,96,127,128]:  cases.append(("H_only", N, C, C, h, 64, 3, 1))
# stride 2 (output = input/2): does cost track input or output spatial?
for s_in in [64,65,66,96,128]:                        cases.append(("stride2", N, C, C, s_in, s_in, 3, 2))

def main():
    ensure_device(push=True)
    rows = []
    for i,(sweep,n,ci,co,h,w,k,stride) in enumerate(cases):
        tag=f"t_{sweep}_{i}"; onnx=W/f"{tag}.onnx"; mnn=W/f"{tag}.mnn"; pad=k//2
        make_conv(str(onnx), n, ci, co, h, w, k, k, stride, pad, 1, 1, "none")
        convert(str(onnx), str(mnn))
        Ho=(h+2*pad-k)//stride+1; Wo=(w+2*pad-k)//stride+1
        out=run_on_device(str(mnn),"input",[n,ci,h,w],"output",loops=60,gpu_mode=68,prec_mem_mask=2,tuning_cache=f"{tag}.cache")
        us=parse(out).get("conv_us_med")
        path="winograd" if "winograd" in out else ("gemm" if "gemm" in out else "direct")
        rows.append(dict(sweep=sweep,N=n,C=ci,Hin=h,Win=w,stride=stride,Hout=Ho,Wout=Wo,conv_us=us,path=path))
        print(f"[{i+1:2}/{len(cases)}] {sweep:8} in {h}x{w} s{stride} -> out {Ho}x{Wo}: {us}us {path}")
    with open(OUT,"w",newline="") as f:
        wtr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()
