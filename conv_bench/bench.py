#!/usr/bin/env python3
"""Convert a conv ONNX -> MNN, push to device, run ModuleBasic on OpenCL buffer/fp16,
parse wall-clock + per-kernel GPU time. One experiment = one (shape, act, gpuMode, prec)."""
import argparse, json, os, re, subprocess, sys, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONVERT = REPO / "build_host" / "MNNConvert"
BUILD = REPO / "build_android_profile"
MODULE = BUILD / "ModuleBasic.out"
LIBS = [
    BUILD / "OFF/arm64-v8a/libMNN.so",
    BUILD / "express/OFF/arm64-v8a/libMNN_Express.so",
    BUILD / "source/backend/opencl/OFF/arm64-v8a/libMNN_CL.so",
]
DEV = "/data/local/tmp/mnnopt"
LOCAL = REPO / "conv_bench" / "work"


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def adb(cmd):
    return sh(f"adb {cmd}")


def ensure_device(push=True):
    if not push:
        return
    for L in LIBS + [MODULE]:
        adb(f"push {L} {DEV}/")  # idempotent; cheap


def convert(onnx_path, mnn_path, fp16=False, fuse_prelu=False):
    extra = " --fp16" if fp16 else ""
    env = "MNN_FUSE_CONV_PRELU=1 " if fuse_prelu else ""
    r = sh(f'{env}"{CONVERT}" -f ONNX --modelFile "{onnx_path}" --MNNModel "{mnn_path}" '
           f'--bizCode biz{extra}')
    if not Path(mnn_path).exists():
        print(r.stdout[-2000:]); print(r.stderr[-2000:]); raise SystemExit("convert failed")
    return r.stdout


def run_on_device(mnn_path, in_name, in_shape, out_name, fwd=3, loops=200, gpu_mode=68,
                  prec_mem_mask=2, clear_cache=True, tuning_cache="conv.cache"):
    LOCAL.mkdir(parents=True, exist_ok=True)
    tdir = LOCAL / "tdir"; tdir.mkdir(exist_ok=True)
    (tdir / "input.json").write_text(json.dumps({
        "inputs": [{"name": in_name, "shape": in_shape, "value": 0.05}],
        "outputs": [out_name],
        "shapeMutable": False,
    }))
    base = os.path.basename(mnn_path)
    adb(f"push {mnn_path} {DEV}/{base}")
    adb(f"push {tdir/'input.json'} {DEV}/tdir/input.json")
    if clear_cache:
        adb(f"shell rm -f {DEV}/{tuning_cache}")
    libpath = DEV
    # args: model Dir runMask fwd loops gpuMode mask cache
    cmd = (f"shell 'cd {DEV} && LD_LIBRARY_PATH={libpath} ./ModuleBasic.out {base} tdir "
           f"0 {fwd} {loops} {gpu_mode} {prec_mem_mask} {tuning_cache} 2>&1'")
    r = adb(cmd)
    return r.stdout + r.stderr


CONV_RE = re.compile(r"conv time = (\d+) us \(gemm2:(\d+) us, gemm1:(\d+) us, 1x1:(\d+) us, ori:(\d+) us, wino: (\d+) us")
KT_RE = re.compile(r"kernel time = (\d+)\s+us (\S+)")
AVG_RE = re.compile(r"Avg= ([\d.]+) ms, min= ([\d.]+) ms, max= ([\d.]+) ms")
TOTAL_RE = re.compile(r"total kernel time = (\d+)  us")


def _stats(xs):
    if not xs:
        return {}
    xs = sorted(xs)
    return {"med": statistics.median(xs), "min": xs[0], "max": xs[-1], "n": len(xs)}


def parse(out, warmup=3):
    avg = AVG_RE.search(out)
    convs = CONV_RE.findall(out)          # one tuple per profiling window (≈ one inference)
    totals = TOTAL_RE.findall(out)
    kernels = KT_RE.findall(out)
    res = {}
    if avg:
        res["wall_avg_ms"] = float(avg.group(1))
        res["wall_min_ms"] = float(avg.group(2))
        res["wall_max_ms"] = float(avg.group(3))
    # steady-state conv GPU time: drop warmup windows, take median/min/max
    conv_us = [int(c[0]) for c in convs][warmup:]
    if conv_us:
        s = _stats(conv_us)
        res["conv_us_med"] = s["med"]; res["conv_us_min"] = s["min"]
        res["conv_us_max"] = s["max"]; res["conv_windows"] = s["n"]
        # which path: report nonzero buckets from last window
        last = convs[-1]
        res["path_us"] = {"gemm2": int(last[1]), "gemm1": int(last[2]), "1x1": int(last[3]),
                          "ori": int(last[4]), "wino": int(last[5])}
    tot = [int(t) for t in totals][warmup:]
    if tot:
        res["total_kernel_us_med"] = _stats(tot)["med"]
    conv_kernels = sorted({k for _, k in kernels if "Conv" in k})
    if conv_kernels:
        res["conv_kernels"] = conv_kernels
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="hero")
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--Cin", type=int, default=96); p.add_argument("--Cout", type=int, default=96)
    p.add_argument("--H", type=int, default=64); p.add_argument("--W", type=int, default=64)
    p.add_argument("--kH", type=int, default=3); p.add_argument("--kW", type=int, default=3)
    p.add_argument("--stride", type=int, default=1); p.add_argument("--pad", type=int, default=1)
    p.add_argument("--dil", type=int, default=1); p.add_argument("--group", type=int, default=1)
    p.add_argument("--act", default="none")
    p.add_argument("--gpu_mode", type=int, default=68); p.add_argument("--mask", type=int, default=2)
    p.add_argument("--loops", type=int, default=200)
    p.add_argument("--fp16conv", action="store_true")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--no-push", action="store_true")
    a = p.parse_args()

    LOCAL.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "conv_bench"))
    from gen_conv import make_conv
    onnx_path = LOCAL / f"{a.tag}.onnx"; mnn_path = LOCAL / f"{a.tag}.mnn"
    info = make_conv(str(onnx_path), a.N, a.Cin, a.Cout, a.H, a.W, a.kH, a.kW, a.stride,
                     a.pad, a.dil, a.group, a.act)
    convert(str(onnx_path), str(mnn_path), fp16=a.fp16conv)
    ensure_device(push=not a.no_push)
    out = run_on_device(str(mnn_path), "input", [a.N, a.Cin, a.H, a.W], "output",
                        loops=a.loops, gpu_mode=a.gpu_mode, prec_mem_mask=a.mask)
    if a.raw:
        print(out)
    res = parse(out)
    res.update(info); res["gpu_mode"] = a.gpu_mode; res["mask"] = a.mask
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
