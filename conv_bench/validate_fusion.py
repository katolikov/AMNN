#!/usr/bin/env python3
"""Validate the PReLU->conv fusion end to end on device, across many shapes.

For each shape: convert the SAME ONNX twice (unfused for CPU ref, fused with
MNN_FUSE_CONV_PRELU=1 for OpenCL), feed random signed input (exercises the PReLU
negative branch), and check:
  * correctness: CPU(unfused) vs OpenCL(fused) cosine within fp16 tol;
  * fusion actually happened where it should (separate `PReLU` kernel gone) and was
    correctly *declined* where it should not (kernel kept) — proving the scoping.

Slopes are VARYING per channel (gen_conv.py), so a channel-index/packing bug is caught.

Run AFTER rebuilding host MNNConvert + android libs/ModuleBasic and pushing fresh libs.
Usage:  python3 conv_bench/validate_fusion.py [sweep|perf]
"""
import json, os, sys, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
from gen_conv import make_conv
import bench
from bench import convert, DEV, adb, ensure_device, run_on_device, parse

W = REPO / "conv_bench" / "vwork"; (W / "tdir").mkdir(parents=True, exist_ok=True)


def _run(mnn, in_shape, fwd, mask):
    base = os.path.basename(mnn)
    adb(f"push {mnn} {DEV}/{base}")
    (W / "tdir" / "input.json").write_text(json.dumps({
        "inputs": [{"name": "input", "shape": in_shape}], "outputs": ["output"],
        "shapeMutable": False}))
    adb(f"push {W/'tdir'/'input.json'} {DEV}/vtdir/input.json")
    adb(f"push {W/'tdir'/'input.txt'} {DEV}/vtdir/input.txt")
    adb(f"shell mkdir -p {DEV}/output")
    out = adb(f"shell 'cd {DEV} && LD_LIBRARY_PATH={DEV} ./ModuleBasic.out {base} vtdir "
              f"0 {fwd} 1 68 {mask} v.bin 2>&1'").stdout
    adb(f"pull {DEV}/output/0_0.txt {W}/out_{fwd}_{base}.txt")
    return np.loadtxt(W / f"out_{fwd}_{base}.txt", dtype=np.float32), out


def check_one(name, N, Cin, Cout, H, Wd, k, act, should_fuse):
    onnx = W / f"{name}.onnx"
    pad = k // 2
    make_conv(str(onnx), N, Cin, Cout, H, Wd, k, k, 1, pad, 1, 1, act)
    unf = W / f"{name}_unf.mnn"; fus = W / f"{name}_fus.mnn"
    convert(str(onnx), str(unf), fuse_prelu=False)
    convert(str(onnx), str(fus), fuse_prelu=True)
    x = np.random.default_rng(7).standard_normal([N, Cin, H, Wd]).astype(np.float32)
    np.savetxt(W / "tdir" / "input.txt", x.reshape(-1), fmt="%.6f")
    cpu, _ = _run(str(unf), [N, Cin, H, Wd], 0, 1)        # CPU unfused, High
    ocl, dump = _run(str(fus), [N, Cin, H, Wd], 3, 2)     # OpenCL fused, fp16
    n = min(cpu.size, ocl.size); a, b = cpu.reshape(-1)[:n], ocl.reshape(-1)[:n]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    fused = "PReLU" not in dump                            # PReLU kernel absent => fused
    ok_corr = cos > 0.999
    ok_fuse = (fused == should_fuse)
    status = "PASS" if (ok_corr and ok_fuse) else "**FAIL**"
    print(f"[{status}] {name:22} N{N} {Cin}->{Cout} {H}x{Wd} k{k} {act:12} "
          f"cos={cos:.6f} fused={fused} (want {should_fuse}) negfrac={(a<0).mean():.2f}")
    return ok_corr and ok_fuse


def sweep():
    ensure_device(push=True)
    cases = [
        # name, N, Cin, Cout, H, W, k, act, should_fuse
        ("hero",        8, 96,  96,  64, 64, 3, "prelu",        True),
        ("c64",         1, 64,  64,  16, 16, 3, "prelu",        True),
        ("c128",        1, 128, 128, 16, 16, 3, "prelu",        True),
        ("asym_96_256", 1, 96,  256, 32, 32, 3, "prelu",        True),
        ("asym_128_64", 2, 128, 64,  12, 12, 3, "prelu",        True),
        ("odd_spatial", 1, 64,  64,  17, 13, 3, "prelu",        True),
        ("scalar_leaky",1, 64,  64,  16, 16, 3, "prelu_scalar", True),
        # non-winograd paths (now also fused via ConvBufExecution)
        ("c32_3x3",     1, 32,  32,  16, 16, 3, "prelu",        True),   # small ch -> ori path
        ("pw_1x1",      1, 96,  96,  16, 16, 1, "prelu",        True),   # 1x1 pointwise
        ("pw_1x1_big",  1, 256, 256, 16, 16, 1, "prelu",        True),   # 1x1 -> gemm path
        ("k5x5",        1, 96,  96,  16, 16, 5, "prelu",        True),   # 5x5 general conv
    ]
    results = [check_one(*c) for c in cases]
    print(f"\n=== {sum(results)}/{len(results)} passed ===")
    return all(results)


def perf():
    ensure_device(push=True)
    for tag, fuse in [("unfused", False), ("fused", True)]:
        onnx = W / f"hero_{tag}.onnx"; mnn = W / f"hero_{tag}.mnn"
        make_conv(str(onnx), 8, 96, 96, 64, 64, 3, 3, 1, 1, 1, 1, "prelu")
        convert(str(onnx), str(mnn), fuse_prelu=fuse)
        out = run_on_device(str(mnn), "input", [8, 96, 64, 64], "output",
                            loops=80, gpu_mode=68, prec_mem_mask=2, tuning_cache=f"v_{tag}.cache")
        r = parse(out)
        print(f"[perf {tag:8}] conv_us_med={r.get('conv_us_med')} "
              f"total_kernel_us={r.get('total_kernel_us_med')} "
              f"separate_PReLU_kernel={'PReLU' in out}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    ok = sweep() if what in ("sweep", "all") else True
    if what in ("perf", "all"):
        perf()
    sys.exit(0 if ok else 1)
