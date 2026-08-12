#!/usr/bin/env python3
"""Correctness + timing for the LDS input-halo tiled 3x3 s1 conv (MNN_CONV_LDS).

Correctness: single core conv (32->32 @72x96, 3x3 s1 pad1 +PReLU) with random signed
input. Compares OpenCL-LDS vs CPU-fp32 reference AND vs the stock OpenCL path. The shape
satisfies the LDS gate (w%16==0, h%4==0). Timing: 6-deep homogeneous chain (the real core),
stock vs LDS, per-conv kernel us.

Usage: python3 lds_test.py [C H W]   (default 32 72 96)
"""
import json, os, re, sys, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
from gen_conv import make_conv
from mk_chain import make_chain
from bench import convert, DEV, adb
from c0_ceiling import per_kernel

LOCAL = REPO / "conv_bench" / "ldswork"; LOCAL.mkdir(parents=True, exist_ok=True)


def run(mnn, in_shape, fwd, mask, loops=1, env="", cache="l.bin"):
    tdir = LOCAL / "tdir"; tdir.mkdir(exist_ok=True)
    (tdir / "input.json").write_text(json.dumps({
        "inputs": [{"name": "input", "shape": in_shape}], "outputs": ["output"],
        "shapeMutable": False}))
    base = os.path.basename(mnn)
    adb(f"push {mnn} {DEV}/{base}")
    adb(f"push {tdir/'input.json'} {DEV}/tdir/input.json")
    if (tdir / "input.txt").exists():
        adb(f"push {tdir/'input.txt'} {DEV}/tdir/input.txt")
    adb(f"shell rm -rf {DEV}/output && adb shell mkdir -p {DEV}/output 2>/dev/null")
    adb(f"shell mkdir -p {DEV}/output")
    cmd = (f"shell 'cd {DEV} && {env} LD_LIBRARY_PATH={DEV} ./ModuleBasic.out {base} tdir "
           f"0 {fwd} {loops} 68 {mask} {cache} 2>&1'")
    r = adb(cmd)
    out = r.stdout + r.stderr
    arr = None
    p = LOCAL / f"out_{fwd}_{'lds' if env else 'base'}.txt"
    adb(f"pull {DEV}/output/0_0.txt {p}")
    if p.exists():
        arr = np.loadtxt(p, dtype=np.float32)
    return arr, out


def cos(a, b):
    n = min(a.size, b.size); a, b = a.reshape(-1)[:n], b.reshape(-1)[:n]
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    denom = max(np.abs(a).max(), 1e-6)
    return c, float(np.abs(a - b).max()), float(np.abs(a - b).max() / denom)


def main():
    C, H, W = (int(x) for x in (sys.argv[1:4] or [32, 72, 96]))

    # ---------- correctness: single conv, random signed input, +PReLU (fused) ----------
    onnx = LOCAL / "c.onnx"; mnn = LOCAL / "c.mnn"
    make_conv(str(onnx), 1, C, C, H, W, 3, 3, 1, 1, 1, 1, "prelu")
    convert(str(onnx), str(mnn), fuse_prelu=True)
    rng = np.random.default_rng(1)
    x = rng.standard_normal([1, C, H, W]).astype(np.float32)
    (LOCAL / "tdir").mkdir(exist_ok=True)
    np.savetxt(LOCAL / "tdir" / "input.txt", x.reshape(-1), fmt="%.6f")
    cpu, _ = run(str(mnn), [1, C, H, W], 0, 1)
    base, _ = run(str(mnn), [1, C, H, W], 3, 2)
    lds, ldsout = run(str(mnn), [1, C, H, W], 3, 2, env="MNN_CONV_LDS=1")
    print(f"== correctness {C}->{C} @{H}x{W} 3x3 s1 +PReLU (fused) ==")
    for name, arr in [("OCL-base vs CPU", base), ("OCL-LDS  vs CPU", lds)]:
        c, mad, rel = cos(cpu, arr)
        print(f"  {name}: cosine={c:.6f} max_abs={mad:.4g} rel={rel:.4g}")
    c, mad, rel = cos(base, lds)
    print(f"  OCL-LDS  vs OCL-base: cosine={c:.6f} max_abs={mad:.4g} rel={rel:.4g}")

    # ---------- timing: 6-deep homogeneous core chain, stock vs LDS ----------
    print(f"\n== timing: 6x {C}->{C} @{H}x{W} core chain (sustained) ==")
    con1 = LOCAL / "chain.onnx"; cmnn = LOCAL / "chain.mnn"
    make_chain(str(con1), 1, C, H, W, depth=6, act="prelu", k=3, stride=1)
    convert(str(con1), str(cmnn), fp16=False, fuse_prelu=True)
    if (LOCAL / "tdir" / "input.txt").exists():
        os.remove(LOCAL / "tdir" / "input.txt")  # use default filled input for timing
    for env, tag in [("", "stock"), ("MNN_CONV_LDS=1", "LDS  ")]:
        _, out = run(str(cmnn), [1, C, H, W], 3, 2, loops=120, env=env, cache=f"chain_{tag.strip()}.bin")
        pk, nwin = per_kernel(out)
        conv_sum = sum(v for k, v in pk.items() if k.startswith("ConvBuf2D"))
        print(f"  {tag}: per_conv={conv_sum/6:.1f} us  conv_sum={conv_sum}  (windows {nwin})")


if __name__ == "__main__":
    main()
