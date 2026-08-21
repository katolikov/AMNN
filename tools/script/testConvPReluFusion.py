#!/usr/bin/env python3
"""End-to-end check of the PReLU -> Convolution fusion (Convolution2DCommon.leakyReluSlope).

For every case the SAME ONNX graph is converted twice:
  * unfused  -- plain MNNConvert, the PReLU stays a separate op;
  * fused    -- MNNConvert --fusePreluToConv, the pass folds the slopes onto the conv.

Two things are then asserted:
  * scoping  -- inspecting the converted model (MNNDump2Json) shows the separate PReLU op gone
                and the exact slope values on the conv, exactly where the pass must fire, and
                left untouched where it must decline (depthwise / grouped conv);
  * numerics -- CPU on the UNFUSED model (ground truth: no CPU backend reads leakyReluSlope,
                so CPU on a fused model would itself be the wrong answer) vs OpenCL on the
                FUSED model, cosine similarity within fp16 tolerance.

Slopes vary per channel, so a channel-index or packing bug fails the cosine check instead of
hiding behind a constant.

Requires: onnx + numpy on the host, and a host build with MNNConvert + MNNDump2Json. The
numeric half additionally needs an adb-connected device and an Android build providing
libMNN.so / libMNN_Express.so / libMNN_CL.so / ModuleBasic.out; pass --scoping-only to skip it.

    python3 tools/script/testConvPReluFusion.py --mode both

Paths are auto-discovered under the repo; override with MNN_CONVERT / MNN_ANDROID_BUILD.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "build_prelu_fusion_check"
DEV = "/data/local/tmp/mnn_prelu_fusion"

# MNNGpuMode bits consumed by ModuleBasic's "numberThread" argument.
GPU_MODE = {"buffer": 64 | 4, "image": 128 | 4}   # memory mode | tuning wide
FORWARD_CPU, FORWARD_OPENCL = 0, 3
PRECISION_HIGH, PRECISION_LOW = 1, 2


def sh(cmd):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


def adb(cmd):
    return sh("adb " + cmd)


def discover():
    convert = os.environ.get("MNN_CONVERT")
    if not convert:
        hits = sorted(REPO.glob("build*/MNNConvert"), key=lambda p: len(p.parts))
        convert = str(hits[0]) if hits else None
    root = os.environ.get("MNN_ANDROID_BUILD")
    roots = [Path(root)] if root else sorted(REPO.glob("build_android*"))
    need = ("libMNN.so", "libMNN_Express.so", "libMNN_CL.so", "ModuleBasic.out")
    for r in roots:
        found = []
        for n in need:
            # build layouts differ (flat vs MNN_SEP_BUILD's OFF/<abi>/), so take the shallowest
            hits = sorted(r.glob("**/" + n), key=lambda p: len(p.parts))
            if not hits:
                break
            found.append(hits[0])
        if len(found) == len(need):
            return convert, found
    return convert, None


def make_onnx(path, N, Cin, Cout, H, W, k, stride, pad, group, act, seed=0):
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [N, Cin, H, W])
    w = rng.standard_normal([Cout, Cin // group, k, k]).astype(np.float32) / np.sqrt(Cin * k * k)
    inits.append(numpy_helper.from_array(w, "W"))
    b = (rng.standard_normal([Cout]).astype(np.float32) * 0.01)
    inits.append(numpy_helper.from_array(b, "B"))
    nodes.append(helper.make_node("Conv", ["input", "W", "B"], ["conv_out"],
                                  kernel_shape=[k, k], strides=[stride, stride],
                                  pads=[pad, pad, pad, pad], dilations=[1, 1], group=group))
    if act == "relu":
        nodes.append(helper.make_node("Relu", ["conv_out"], ["output"]))
    elif act == "relu6":
        inits.append(numpy_helper.from_array(np.array(0.0, np.float32), "lo"))
        inits.append(numpy_helper.from_array(np.array(6.0, np.float32), "hi"))
        nodes.append(helper.make_node("Clip", ["conv_out", "lo", "hi"], ["output"]))
    else:
        if act == "prelu":
            # varying per channel: a permutation or packing bug cannot pass unnoticed
            slope = 0.05 + 0.40 * (np.arange(Cout, dtype=np.float32) / max(Cout - 1, 1))
        elif act == "prelu_scalar":
            slope = np.array([0.123], np.float32)  # 1-elem PRelu -> lowered to LeakyReLU
        else:
            raise ValueError(act)
        inits.append(numpy_helper.from_array(slope, "slope"))
        nodes.append(helper.make_node("PRelu", ["conv_out", "slope"], ["output"]))

    Ho = (H + 2 * pad - (k - 1) - 1) // stride + 1
    Wo = (W + 2 * pad - (k - 1) - 1) // stride + 1
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [N, Cout, Ho, Wo])
    model = helper.make_model(helper.make_graph(nodes, "conv_prelu", [x], [y], initializer=inits),
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def model_json(convert_bin, mnn_path):
    """Decode the converted model. MNNDump2Json sits next to MNNConvert in every layout."""
    dump = Path(convert_bin).with_name("MNNDump2Json")
    out = Path(str(mnn_path) + ".json")
    r = sh(f'"{dump}" "{mnn_path}" "{out}"')
    if not out.exists():
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("MNNDump2Json failed")
    return json.loads(out.read_text())


def inspect(convert_bin, mnn_path, expect):
    """True when the converted model is in the expected state. `expect` is either
      * an ndarray -- the exact slope vector the conv must carry (PReLU / LeakyReLU fused);
      * "relu" / "relu6" -- must take the cheaper common->relu flag, with NO slope vector;
      * None -- the fusion must NOT have happened: activation still its own op, no slopes."""
    net = model_json(convert_bin, mnn_path)
    slope, common = None, None
    for op in net["oplists"]:
        if op.get("type") in ("Convolution", "ConvolutionDepthwise"):
            common = op["main"]["common"]
            got = common.get("leakyReluSlope")
            if got:
                slope = np.asarray(got, dtype=np.float32)
    # a separate activation op: PReLU, LeakyReLU (ReLU carrying a slope), plain ReLU or ReLU6
    act_ops = [op for op in net["oplists"] if op.get("type") in ("PReLU", "ReLU", "ReLU6")]
    if expect is None:
        return bool(act_ops) and slope is None
    if isinstance(expect, str):
        return not act_ops and slope is None and common is not None \
            and common.get(expect, False) is True
    return not act_ops and slope is not None and slope.shape == expect.shape \
        and np.allclose(slope, expect, atol=1e-6)


def convert(convert_bin, onnx_path, mnn_path, fuse):
    flag = " --fusePreluToConv" if fuse else ""
    r = sh(f'"{convert_bin}" -f ONNX --modelFile "{onnx_path}" '
           f'--MNNModel "{mnn_path}" --bizCode biz{flag}')
    if not Path(mnn_path).exists():
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("MNNConvert failed")


def run_on_device(mnn_path, in_shape, x, forward, gpu_mode, precision):
    base = os.path.basename(mnn_path)
    tdir = WORK / "tdir"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "input.json").write_text(json.dumps({
        "inputs": [{"name": "input", "shape": list(in_shape)}],
        "outputs": ["output"], "shapeMutable": False}))
    np.savetxt(tdir / "input.txt", x.reshape(-1), fmt="%.6f")
    adb(f"push {mnn_path} {DEV}/{base}")
    adb(f"push {tdir/'input.json'} {DEV}/tdir/input.json")
    adb(f"push {tdir/'input.txt'} {DEV}/tdir/input.txt")
    adb(f"shell 'rm -rf {DEV}/output && mkdir -p {DEV}/output'")
    # ModuleBasic.out <model> <dir> <runMask> <forwardType> <loops> <numberThread> <precision>
    out = adb(f"shell 'cd {DEV} && LD_LIBRARY_PATH={DEV} ./ModuleBasic.out {base} tdir 0 "
              f"{forward} 1 {gpu_mode} {precision} 2>&1'").stdout
    local = WORK / f"out_{forward}_{gpu_mode}_{base}.txt"
    adb(f"pull {DEV}/output/0_0.txt {local}")
    if not local.exists():
        print(out[-3000:])
        raise SystemExit("no output pulled from device")
    return np.loadtxt(local, dtype=np.float32), out


def expectation(Cout, act, fused):
    """What the fused model must look like, given the activation and whether it should fuse."""
    if fused is not True:
        return fused or None                            # "relu" / "relu6" flag, or None
    if act == "prelu":
        return 0.05 + 0.40 * (np.arange(Cout, dtype=np.float32) / max(Cout - 1, 1))
    return np.full(Cout, 0.123, dtype=np.float32)       # LeakyReLU / 1-slope PReLU broadcast


def build(convert_bin, case):
    """Convert one case twice and assert the fusion fired / declined as specified."""
    name, N, Cin, Cout, H, W, k, group, act, should_fuse = case
    onnx_path = WORK / f"{name}.onnx"
    make_onnx(onnx_path, N, Cin, Cout, H, W, k, 1, k // 2, group, act)
    unfused, fused = WORK / f"{name}_unf.mnn", WORK / f"{name}_fus.mnn"
    convert(convert_bin, onnx_path, unfused, fuse=False)
    convert(convert_bin, onnx_path, fused, fuse=True)
    want = expectation(Cout, act, should_fuse)
    # the plain relu/relu6 folding is stock MNN, so it is already there in the unfused model too
    unfused_want = want if isinstance(want, str) else None
    ok = inspect(convert_bin, fused, want) and inspect(convert_bin, unfused, unfused_want)
    print(f"[{'PASS' if ok else '**FAIL**'}] scoping {name:16} {Cin}->{Cout} {H}x{W} k{k} "
          f"g{group} {act:12} fused={should_fuse}")
    return ok, unfused, fused


COS_TOL = 0.999


def cosine(a, b):
    n = min(a.size, b.size)
    a, b = a.reshape(-1)[:n], b.reshape(-1)[:n]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), a


def signed_input(N, Cin, H, W):
    """Signed, so the PReLU negative branch is actually exercised."""
    return np.random.default_rng(7).standard_normal([N, Cin, H, W]).astype(np.float32)


def numerics(mode, case, unfused, fused):
    """CPU on the unfused model is the reference; OpenCL runs the fused one."""
    name, N, Cin, Cout, H, W, k, group, act, _ = case
    x = signed_input(N, Cin, H, W)
    ref, _ = run_on_device(unfused, [N, Cin, H, W], x, FORWARD_CPU,
                           GPU_MODE[mode], PRECISION_HIGH)
    got, _ = run_on_device(fused, [N, Cin, H, W], x, FORWARD_OPENCL,
                           GPU_MODE[mode], PRECISION_LOW)
    cos, a = cosine(ref, got)
    ok = cos > COS_TOL
    print(f"[{'PASS' if ok else '**FAIL**'}] {mode:7} {name:16} {Cin}->{Cout} {H}x{W} k{k} "
          f"g{group} {act:12} cos={cos:.6f} neg={(a < 0).mean():.2f}")
    return ok


def negative_control(case, unfused, fused):
    """Prove the comparison has teeth: no CPU backend reads leakyReluSlope, so running the
    FUSED model on CPU must land well BELOW the tolerance. If it does not, the slopes are too
    close to 1 (or some backend started honouring them) and every PASS below is vacuous."""
    name, N, Cin, Cout, H, W, k, group, act, _ = case
    x = signed_input(N, Cin, H, W)
    ref, _ = run_on_device(unfused, [N, Cin, H, W], x, FORWARD_CPU,
                           GPU_MODE["buffer"], PRECISION_HIGH)
    dropped, _ = run_on_device(fused, [N, Cin, H, W], x, FORWARD_CPU,
                               GPU_MODE["buffer"], PRECISION_HIGH)
    cos, _ = cosine(ref, dropped)
    ok = cos < COS_TOL
    print(f"[{'PASS' if ok else '**FAIL**'}] control {name:16} activation dropped on CPU "
          f"reads cos={cos:.6f} (must be < {COS_TOL})")
    return ok


CASES = [
    #  name,           N, Cin, Cout,  H,  W, k, group, act,            should_fuse
    ("wino_64",        1,  64,   64, 16, 16, 3, 1, "prelu",        True),   # winograd path
    ("wino_128",       1, 128,  128, 16, 16, 3, 1, "prelu",        True),
    ("wino_asym",      1,  96,  256, 32, 32, 3, 1, "prelu",        True),
    ("wino_odd",       1,  64,   64, 17, 13, 3, 1, "prelu",        True),   # ragged spatial
    # A 1-element ONNX PRelu is lowered to OpType_ReLU(slope) by OnnxPrelu.cpp, i.e. it reaches
    # the pass as a LeakyReLU, not a PReLU -- these two cover that branch on both conv paths.
    ("wino_leaky",     1,  64,   64, 16, 16, 3, 1, "prelu_scalar", True),
    ("direct_leaky",   1,  32,   32, 16, 16, 3, 1, "prelu_scalar", True),
    ("direct_c32",     1,  32,   32, 16, 16, 3, 1, "prelu",        True),   # below winograd gate
    ("direct_k5",      1,  96,   96, 16, 16, 5, 1, "prelu",        True),   # general conv
    ("pointwise_1x1",  1,  96,   96, 16, 16, 1, 1, "prelu",        True),   # 1x1 path
    ("pointwise_gemm", 1, 256,  256, 16, 16, 1, 1, "prelu",        True),   # 1x1 -> gemm path
    ("batched",        2, 128,   64, 12, 12, 3, 1, "prelu",        True),
    ("depthwise",      1,  64,   64, 16, 16, 3, 64, "prelu",       False),  # must DECLINE
    ("depthwise_leaky",1,  64,   64, 16, 16, 3, 64, "prelu_scalar", False),  # must DECLINE
    # plain ReLU/ReLU6 must keep taking the stock common->relu flag: it is cheaper than a slope
    # vector, and stealing it here would be a silent regression rather than a failure.
    ("wino_relu",      1,  64,   64, 16, 16, 3, 1, "relu",         "relu"),
    ("wino_relu6",     1,  64,   64, 16, 16, 3, 1, "relu6",        "relu6"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["buffer", "image", "both"])
    ap.add_argument("--scoping-only", action="store_true",
                    help="skip the on-device numeric check (no adb device needed)")
    ap.add_argument("--filter", default="", help="only run cases whose name contains this")
    args = ap.parse_args()

    convert_bin, libs = discover()
    if not convert_bin or not Path(convert_bin).exists():
        raise SystemExit("MNNConvert not found; set MNN_CONVERT")
    WORK.mkdir(parents=True, exist_ok=True)
    cases = [c for c in CASES if args.filter in c[0]]

    results = []
    built = []
    for case in cases:
        ok, unfused, fused = build(convert_bin, case)
        results.append(ok)
        built.append((case, unfused, fused))

    if not args.scoping_only:
        if not libs:
            raise SystemExit("android build not found; set MNN_ANDROID_BUILD or --scoping-only")
        if "device" not in adb("devices").stdout:
            raise SystemExit("no adb device; use --scoping-only")
        adb(f"shell mkdir -p {DEV}/tdir")
        for lib in libs:
            adb(f"push {lib} {DEV}/")
        # one control per fused activation kind (per-channel PReLU, scalar LeakyReLU): each
        # takes a different branch of the pass, so each needs its own proof of teeth
        for act in ("prelu", "prelu_scalar"):
            rep = next((b for b in built if b[0][-1] is True and b[0][8] == act), None)
            if rep:
                results.append(negative_control(*rep))
        for mode in (["buffer", "image"] if args.mode == "both" else [args.mode]):
            for case, unfused, fused in built:
                results.append(numerics(mode, case, unfused, fused))

    print(f"\n=== {sum(results)}/{len(results)} passed ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
