#!/usr/bin/env python3
"""CPU vs OpenCL(buffer,fp16) numerical check for a conv(+act) op.
Random signed input (exercises PReLU negative branch). Compares cosine + max-abs.
Establishes the reference any fused-PReLU kernel change must match within fp16 tol."""
import json, os, subprocess, sys, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
from gen_conv import make_conv
from bench import convert, DEV, BUILD, adb

LOCAL = REPO / "conv_bench" / "cwork"; LOCAL.mkdir(parents=True, exist_ok=True)


def run_backend(mnn, in_name, in_shape, out_name, fwd, mask):
    tdir = LOCAL / "tdir"; tdir.mkdir(exist_ok=True)
    (tdir / "input.json").write_text(json.dumps({
        "inputs": [{"name": in_name, "shape": in_shape}], "outputs": [out_name],
        "shapeMutable": False}))
    base = os.path.basename(mnn)
    adb(f"push {mnn} {DEV}/{base}")
    adb(f"push {tdir/'input.json'} {DEV}/tdir/input.json")
    adb(f"push {tdir/'input.txt'} {DEV}/tdir/input.txt")
    adb(f"shell rm -rf {DEV}/output && adb shell mkdir -p {DEV}/output 2>/dev/null")
    adb(f"shell mkdir -p {DEV}/output")
    # runMask 0, loops 1; precision Low (mask=2) for OpenCL, Normal for CPU baseline
    cmd = (f"shell 'cd {DEV} && LD_LIBRARY_PATH={DEV} ./ModuleBasic.out {base} tdir "
           f"0 {fwd} 1 68 {mask} c.bin 2>&1'")
    adb(cmd)
    adb(f"pull {DEV}/output/0_0.txt {LOCAL}/out_{fwd}.txt")
    arr = np.loadtxt(LOCAL / f"out_{fwd}.txt", dtype=np.float32)
    return arr


def main():
    N, Cin, Cout, H, W = 1, 96, 96, 16, 16
    act = sys.argv[1] if len(sys.argv) > 1 else "prelu"
    onnx = LOCAL / "c.onnx"; mnn = LOCAL / "c.mnn"
    make_conv(str(onnx), N, Cin, Cout, H, W, 3, 3, 1, 1, 1, 1, act)
    convert(str(onnx), str(mnn))
    # random signed input -> exercises PReLU negative branch
    (LOCAL / "tdir").mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    x = (rng.standard_normal([N, Cin, H, W]).astype(np.float32))
    np.savetxt(LOCAL / "tdir" / "input.txt", x.reshape(-1), fmt="%.6f")
    cpu = run_backend(str(mnn), "input", [N, Cin, H, W], "output", 0, 1)   # CPU, High prec
    ocl = run_backend(str(mnn), "input", [N, Cin, H, W], "output", 3, 2)   # OpenCL buffer fp16
    n = min(cpu.size, ocl.size)
    a, b = cpu.reshape(-1)[:n], ocl.reshape(-1)[:n]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    denom = np.maximum(np.abs(a).max(), 1e-6)
    print(f"act={act} N={n} cosine={cos:.6f} max_abs_diff={np.abs(a-b).max():.4g} "
          f"rel={np.abs(a-b).max()/denom:.4g} neg_frac={(a<0).mean():.2f}")


if __name__ == "__main__":
    main()
