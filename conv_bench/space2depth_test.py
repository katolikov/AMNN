#!/usr/bin/env python3
"""space2depth on the stride-2 HEAD convs of Block1/Block2 (Session A lever).

For each block: build the exact chain ONNX, rewrite its stride-2 convs to
SpaceToDepth(2)+stride-1 (exact, self-checked transform in rewrite_stride2.py), then on device
compare TOTAL block kernel time before/after under sustained load. The input-raster fixture
artifact is identical in both, so the delta isolates (s2d + new head convs) vs (original heads).
Also: correctness (rewritten OCL vs original OCL, random signed input) + SpaceToDepth-on-GPU check.

Usage: python3 space2depth_test.py [Block1|Block2|all]
"""
import os, re, sys, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
import bench
from block_fixture import build_onnx, load_blocks
from rewrite_stride2 import rewrite_onnx
from c0_ceiling import per_kernel
import lds_test

LOCAL = REPO / "conv_bench" / "s2dwork"; (LOCAL / "tdir").mkdir(parents=True, exist_ok=True)
lds_test.LOCAL = LOCAL  # reuse lds_test.run with this working dir


def total_and_kernels(mnn, in_shape, cache):
    _, out = lds_test.run(str(mnn), in_shape, 3, 2, loops=120, cache=cache)
    pk, nwin = per_kernel(out)
    tots = [int(t) for t in re.findall(r"total kernel time = (\d+)  us", out)]
    tmed = sorted(tots[3:])[len(tots[3:]) // 2] if len(tots) > 3 else (tots[-1] if tots else 0)
    cpu_fallback = ("CPU" in out and "SpaceToDepth" in out and "fallback" in out.lower())
    return tmed, pk, nwin, cpu_fallback, out


def measure(name, convs):
    onnx0 = LOCAL / f"{name}.onnx"; mnn0 = LOCAL / f"{name}.mnn"
    onnx1 = LOCAL / f"{name}_s2d.onnx"; mnn1 = LOCAL / f"{name}_s2d.mnn"
    build_onnx(str(onnx0), convs)
    n_rw = rewrite_onnx(str(onnx0), str(onnx1), verbose=True)
    bench.convert(str(onnx0), str(mnn0), fp16=False, fuse_prelu=True)
    bench.convert(str(onnx1), str(mnn1), fp16=False, fuse_prelu=True)
    c0 = convs[0]; in_shape = [1, c0["cin"], c0["H"], c0["W"]]

    # ---- correctness: rewritten OCL vs original OCL vs CPU, random signed input ----
    rng = np.random.default_rng(3)
    x = rng.standard_normal(in_shape).astype(np.float32)
    np.savetxt(LOCAL / "tdir" / "input.txt", x.reshape(-1), fmt="%.6f")
    cpu0, _ = lds_test.run(str(mnn0), in_shape, 0, 1, loops=1, cache="cpu.bin")
    ocl0, _ = lds_test.run(str(mnn0), in_shape, 3, 2, loops=1, cache="o0.bin")
    ocl1, _ = lds_test.run(str(mnn1), in_shape, 3, 2, loops=1, cache="o1.bin")
    c_rw, mad_rw, rel_rw = lds_test.cos(ocl0, ocl1)
    os.remove(LOCAL / "tdir" / "input.txt")  # timing uses default filled input

    # ---- timing ----
    t0, pk0, nwin, _, _ = total_and_kernels(mnn0, in_shape, f"{name}_o.bin")
    t1, pk1, _, cpu_fb, out1 = total_and_kernels(mnn1, in_shape, f"{name}_r.bin")
    print(f"\n===== {name}: rewrote {n_rw} stride-2 head conv(s) =====")
    print(f"  correctness rewritten-OCL vs orig-OCL: cosine={c_rw:.6f} max_abs={mad_rw:.4g} rel={rel_rw:.4g}")
    if cpu_fb:
        print("  !! WARNING: SpaceToDepth may be CPU-fallback (check profiler)")
    print(f"  total kernel: orig={t0} us  rewritten={t1} us  delta={t1-t0:+d} us ({100*(t1-t0)/t0:+.1f}%)")
    print("  --- rewritten per-kernel (top) ---")
    for k, v in sorted(pk1.items(), key=lambda kv: -kv[1])[:8]:
        print(f"     {v:>6} us  {k}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    blocks = load_blocks()
    print("blocks:", {k: len(v) for k, v in blocks.items()})
    for name, convs in blocks.items():
        if which != "all" and name != which:
            continue
        measure(name, convs)
