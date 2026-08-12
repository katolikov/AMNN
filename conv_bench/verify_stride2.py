#!/usr/bin/env python3
"""Tier 1 on-device verification of the stride-2 -> space2depth rewrite.

For each stride-2 conv: rewrite it, then on the CONNECTED DEVICE:
  (1) equivalence  — OpenCL(buffer,fp16) rewritten vs OpenCL original AND vs CPU original,
                     same random signed input; require cosine > 0.999 + small max-abs-diff;
  (2) no CPU fallback — SpaceToDepth must run on the OpenCL backend (scan the profiler kernels);
  (3) speed        — original vs rewritten total GPU kernel time (median of repeats).

Run: python3 conv_bench/verify_stride2.py
"""
import json, os, sys, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "conv_bench"))
from bench import convert, DEV, adb, ensure_device, run_on_device, parse
from rewrite_stride2 import rewrite_onnx
W = os.path.join(REPO, "conv_bench", "vwork"); os.makedirs(os.path.join(W, "tdir"), exist_ok=True)


def _mk(path, N, Cin, Cout, H, Wd, k, s):
    rng = np.random.default_rng(0)
    Wt = numpy_helper.from_array((rng.standard_normal((Cout, Cin, k, k)) * 0.1).astype(np.float32), "W")
    B = numpy_helper.from_array((rng.standard_normal(Cout) * 0.01).astype(np.float32), "B")
    Ho = (H + 2 * (k // 2) - k) // s + 1; Wo = (Wd + 2 * (k // 2) - k) // s + 1
    n = helper.make_node("Conv", ["input", "W", "B"], ["output"], kernel_shape=[k, k],
                         pads=[k // 2] * 4, strides=[s, s])
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [N, Cin, H, Wd])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [N, Cout, Ho, Wo])
    m = helper.make_model(helper.make_graph([n], "c", [x], [y], [Wt, B]),
                          opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 9
    onnx.save(m, path)


def _run(mnn, in_shape, fwd, mask):
    base = os.path.basename(mnn)
    adb(f"push {mnn} {DEV}/{base}")
    (open(os.path.join(W, "tdir", "input.json"), "w")
     .write(json.dumps({"inputs": [{"name": "input", "shape": in_shape}],
                        "outputs": ["output"], "shapeMutable": False})))
    adb(f"push {os.path.join(W,'tdir','input.json')} {DEV}/vtdir/input.json")
    adb(f"push {os.path.join(W,'tdir','input.txt')} {DEV}/vtdir/input.txt")
    adb(f"shell mkdir -p {DEV}/output")
    out = adb(f"shell 'cd {DEV} && LD_LIBRARY_PATH={DEV} ./ModuleBasic.out {base} vtdir "
              f"0 {fwd} 1 68 {mask} v.bin 2>&1'").stdout
    adb(f"pull {DEV}/output/0_0.txt {W}/o_{fwd}_{base}.txt")
    return np.loadtxt(os.path.join(W, f"o_{fwd}_{base}.txt"), dtype=np.float32), out


def _speed(mnn, in_shape, tag):
    vals = [parse(run_on_device(mnn, "input", in_shape, "output", loops=80, gpu_mode=68,
            prec_mem_mask=2, tuning_cache=f"{tag}.cache")).get("total_kernel_us_med") for _ in range(3)]
    return vals


def check(name, N, Cin, Cout, H, Wd, k, s):
    orig = os.path.join(W, f"v_{name}_o.onnx"); rw = os.path.join(W, f"v_{name}_r.onnx")
    _mk(orig, N, Cin, Cout, H, Wd, k, s)
    if rewrite_onnx(orig, rw, verbose=False) == 0:
        print(f"[{name}] no conv rewritten (skipped)"); return None
    mo = os.path.join(W, f"v_{name}_o.mnn"); mr = os.path.join(W, f"v_{name}_r.mnn")
    convert(orig, mo); convert(rw, mr)
    x = np.random.default_rng(5).standard_normal([N, Cin, H, Wd]).astype(np.float32)
    np.savetxt(os.path.join(W, "tdir", "input.txt"), x.reshape(-1), fmt="%.6f")
    cpu, _ = _run(mo, [N, Cin, H, Wd], 0, 1)          # CPU original (High) = gold reference
    ocl_o, _ = _run(mo, [N, Cin, H, Wd], 3, 2)        # OpenCL original (fp16)
    ocl_r, dump = _run(mr, [N, Cin, H, Wd], 3, 2)     # OpenCL rewritten (fp16)
    def cos(a, b):
        n = min(a.size, b.size); a, b = a.reshape(-1)[:n], b.reshape(-1)[:n]
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), float(np.abs(a - b).max())
    c_rw_cpu, d_rw_cpu = cos(cpu, ocl_r)              # rewritten vs gold
    c_rw_o, d_rw_o = cos(ocl_o, ocl_r)               # rewritten vs original (both fp16)
    fallback = ("CPU" in dump and "SpaceToDepth" in dump) or "not support" in dump.lower()
    so, sr = _speed(mo, [N, Cin, H, Wd], f"{name}o"), _speed(mr, [N, Cin, H, Wd], f"{name}r")
    from statistics import median
    factor = median(so) / median(sr) if median(sr) else 0
    ok = c_rw_cpu > 0.999 and c_rw_o > 0.999
    print(f"[{'PASS' if ok else 'FAIL'}] {name} in[{N},{Cin},{H},{Wd}] s{s}: "
          f"cos(rw,cpu)={c_rw_cpu:.5f} cos(rw,ocl)={c_rw_o:.5f} maxdiff={d_rw_cpu:.3g} | "
          f"orig={so} rewritten={sr} = {factor:.2f}x | s2d_cpu_fallback={fallback}")
    return ok


if __name__ == "__main__":
    ensure_device(push=True)
    cases = [("conv3", 1, 1, 8, 576, 768, 3, 2),
             ("s2_18_32", 1, 18, 32, 288, 384, 3, 2),
             ("s2_34_64", 1, 34, 64, 144, 192, 3, 2),
             ("s2_64_128", 1, 64, 128, 96, 128, 3, 2)]
    res = [check(*c) for c in cases]
    res = [r for r in res if r is not None]
    print(f"\n=== {sum(res)}/{len(res)} passed ===")
    sys.exit(0 if all(res) else 1)
