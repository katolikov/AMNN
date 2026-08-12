#!/usr/bin/env python3
"""Broad conv shape sweep on the OpenCL buffer/fp16 path -> CSV for later plotting.

One variable swept at a time around the hero point (N8 96x96 64x64 k3). Steady-state conv GPU
time (median). Writes conv_bench/shape_sweep.csv with one row per shape.
"""
import csv, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
from gen_conv import make_conv
from bench import convert, run_on_device, parse, ensure_device

W = REPO / "conv_bench" / "sweepwork"; W.mkdir(exist_ok=True)
OUT = REPO / "conv_bench" / "shape_sweep.csv"

# (sweep_name, N, Cin, Cout, H, Wd, k)   hero = N8 Cin96 Cout96 64x64 k3
HERO = dict(N=8, Cin=96, Cout=96, H=64, Wd=64, k=3)
def pt(sweep, **kw):
    d = dict(HERO); d.update(kw); d["sweep"] = sweep; return d

cases = []
for co in [32,48,64,80,96,97,104,112,128,160,192,256]:           cases.append(pt("Cout", Cout=co))
for ci in [32,48,64,80,96,97,112,128,192,256]:                   cases.append(pt("Cin",  Cin=ci))
for s  in [8,16,24,32,48,64,80,96,128]:                          cases.append(pt("HW",   H=s, Wd=s))
for n  in [1,2,4,8,16,32]:                                       cases.append(pt("N",    N=n))
for k  in [1,3,5,7]:                                             cases.append(pt("kernel", H=32, Wd=32, k=k))

def main():
    ensure_device(push=True)
    rows = []
    for i, c in enumerate(cases):
        N,Cin,Cout,H,Wd,k = c["N"],c["Cin"],c["Cout"],c["H"],c["Wd"],c["k"]
        tag = f"{c['sweep']}_{i}"
        onnx = W / f"{tag}.onnx"; mnn = W / f"{tag}.mnn"
        pad = k // 2
        make_conv(str(onnx), N, Cin, Cout, H, Wd, k, k, 1, pad, 1, 1, "none")
        convert(str(onnx), str(mnn))
        out = run_on_device(str(mnn), "input", [N,Cin,H,Wd], "output",
                            loops=60, gpu_mode=68, prec_mem_mask=2, tuning_cache=f"sw_{tag}.cache")
        r = parse(out)
        Ho = (H + 2*pad - k) + 1; Wo = (Wd + 2*pad - k) + 1
        macs = N * Cout * Ho * Wo * Cin * k * k
        us = r.get("conv_us_med")
        eff = (2*macs/(us*1e-6)/1e12) if us else None     # nominal TFLOP/s (pre-winograd MACs)
        path = "winograd" if "winograd" in out else ("gemm" if "gemm" in out else "direct/other")
        row = dict(sweep=c["sweep"], N=N, Cin=Cin, Cout=Cout, H=H, W=Wd, k=k,
                   conv_us=us, macs=macs, eff_tflops=round(eff,2) if eff else None, path=path)
        rows.append(row)
        print(f"[{i+1:2}/{len(cases)}] {c['sweep']:6} N{N} {Cin}->{Cout} {H}x{Wd} k{k}: "
              f"{us}us eff={row['eff_tflops']} {path}")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()
