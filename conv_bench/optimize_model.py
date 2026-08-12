#!/usr/bin/env python3
"""Tier 1 auto-optimizer: apply the safe, exact transforms to a model for OpenCL buffer/fp16.

  1. stride-2 -> space2depth rewrite, but ONLY on convs MEASURED faster on the device (the rewrite
     adds ~1.78x MACs, so it only wins for occupancy-starved low-Cin convs such as an image stem;
     it is exact either way). Per-conv keep/skip decided by on-device timing.
  2. PReLU fusion (converter flag --fuseConvPrelu) on the final model.

Handles models containing ANY other ops (concat, add, resize, matmul, attention, ...): it only
rewrites stride-2 Conv nodes and leaves every other node untouched; conv+PReLU fusion likewise
only touches conv->PReLU pairs. Convs whose input H,W cannot be shape-inferred are safely SKIPPED.

Outputs <model>_opt.onnx and prints the convert command + per-conv decisions.

Usage:
  python3 conv_bench/optimize_model.py model.onnx [--device SERIAL] [--keep-margin 1.05]
          [--max-cin N] [--gpu-mode 68] [--prec-mask 2] [--out path.onnx]
"""
import argparse, os, sys, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
from statistics import median
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "conv_bench"))
import rewrite_stride2 as R
W = os.path.join(REPO, "conv_bench", "vwork"); os.makedirs(W, exist_ok=True)


def _infer_shapes(path):
    m = onnx.load(path)
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception:
        pass
    shapes = {}
    for vi in list(m.graph.value_info) + list(m.graph.input):
        shapes[vi.name] = [(d.dim_value if d.HasField("dim_value") else None)
                           for d in vi.type.tensor_type.shape.dim]
    return shapes


def list_stride2(path):
    """Return (candidates, skipped, optypes). Loads the model ONCE. Ignores all non-conv ops."""
    m = onnx.load(path)
    shapes = _infer_shapes(path)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    optypes = {}
    cands, skipped = [], []
    for nd in m.graph.node:
        optypes[nd.op_type] = optypes.get(nd.op_type, 0) + 1
        if nd.op_type != "Conv":
            continue
        st = R._attr(nd, "strides", [1, 1]); gr = R._attr(nd, "group", 1); dil = R._attr(nd, "dilations", [1, 1])
        if not (len(st) == 2 and st[0] == st[1] and st[0] >= 2 and gr == 1 and dil == [1, 1]):
            continue
        name = nd.name or nd.output[0]
        if nd.input[1] not in inits:
            skipped.append((name, "weight not constant")); continue
        d = shapes.get(nd.input[0])
        if not d or len(d) != 4 or d[2] is None or d[3] is None:
            skipped.append((name, "input H,W not inferable")); continue
        if d[2] % st[0] or d[3] % st[0]:
            skipped.append((name, f"H/W not divisible by {st[0]}")); continue
        Cout, Cin, Kh, _ = inits[nd.input[1]].shape
        cands.append(dict(name=name, Cin=Cin, Cout=Cout, H=d[2], W=d[3], k=Kh, s=st[0]))
    return cands, skipped, optypes


def isolated_speed(c, tag, gpu_mode, prec_mask):
    from bench import convert, run_on_device, parse
    rng = np.random.default_rng(0)
    Wt = numpy_helper.from_array((rng.standard_normal((c["Cout"], c["Cin"], c["k"], c["k"])) * 0.1).astype(np.float32), "W")
    B = numpy_helper.from_array((rng.standard_normal(c["Cout"]) * 0.01).astype(np.float32), "B")
    n = helper.make_node("Conv", ["input", "W", "B"], ["output"], kernel_shape=[c["k"], c["k"]],
                         pads=[c["k"] // 2] * 4, strides=[c["s"], c["s"]])
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, c["Cin"], c["H"], c["W"]])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)
    mm = helper.make_model(helper.make_graph([n], "c", [x], [y], [Wt, B]),
                           opset_imports=[helper.make_opsetid("", 13)]); mm.ir_version = 9
    o = os.path.join(W, f"opt_{tag}_o.onnx"); r = os.path.join(W, f"opt_{tag}_r.onnx")
    onnx.save(mm, o)
    if R.rewrite_onnx(o, r, verbose=False) == 0:
        return None
    convert(o, os.path.join(W, f"opt_{tag}_o.mnn")); convert(r, os.path.join(W, f"opt_{tag}_r.mnn"))
    def sp(mnn, t):
        return median([parse(run_on_device(mnn, "input", [1, c["Cin"], c["H"], c["W"]], "output",
            loops=60, gpu_mode=gpu_mode, prec_mem_mask=prec_mask, tuning_cache=f"{t}.cache")).get("total_kernel_us_med")
            for _ in range(2)])
    return sp(os.path.join(W, f"opt_{tag}_o.mnn"), f"{tag}o"), sp(os.path.join(W, f"opt_{tag}_r.mnn"), f"{tag}r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", help="adb serial (sets ANDROID_SERIAL so all adb calls target it)")
    ap.add_argument("--keep-margin", type=float, default=1.05, help="keep a rewrite only if >= this x faster")
    ap.add_argument("--max-cin", type=int, default=None, help="skip (don't even measure) convs with Cin > this")
    ap.add_argument("--gpu-mode", type=int, default=68, help="ModuleBasic gpuMode (68 = buffer+WIDE)")
    ap.add_argument("--prec-mask", type=int, default=2, help="precision mask (2 = Low/fp16)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.device:
        os.environ["ANDROID_SERIAL"] = a.device
        print(f"targeting device ANDROID_SERIAL={a.device}")
    from bench import ensure_device
    ensure_device(push=True)

    cands, skipped, optypes = list_stride2(a.model)
    print(f"model op types: {dict(sorted(optypes.items(), key=lambda kv: -kv[1]))}")
    print(f"stride-2 conv candidates: {len(cands)} | skipped: {len(skipped)}")
    for nm, why in skipped:
        print(f"  skip {nm}: {why}")

    keep = []
    for c in cands:
        if a.max_cin is not None and c["Cin"] > a.max_cin:
            print(f"  {c['name']} [Cin{c['Cin']} {c['H']}x{c['W']}]: Cin>max-cin -> skip (heuristic)"); continue
        res = isolated_speed(c, c["name"].replace("/", "_").replace(":", "_"), a.gpu_mode, a.prec_mask)
        if res is None:
            print(f"  {c['name']}: not rewritable exactly -> skip"); continue
        so, sr = res; factor = so / sr if sr else 0
        dec = "KEEP" if factor >= a.keep_margin else "skip"
        if dec == "KEEP": keep.append(c["name"])
        print(f"  {c['name']} [Cin{c['Cin']} {c['H']}x{c['W']} s{c['s']}]: orig {so:.0f}us -> rw {sr:.0f}us = {factor:.2f}x -> {dec}")

    out = a.out or a.model.replace(".onnx", "_opt.onnx")
    n = R.rewrite_onnx(a.model, out, verbose=False, only_names=set(keep)) if keep else 0
    if not keep:
        onnx.save(onnx.load(a.model), out)
    print(f"\napplied {n} stride-2 rewrite(s) -> {out}")
    print("Now convert for the device WITH PReLU fusion (other ops pass through unchanged):")
    print(f"  MNN_FUSE_CONV_PRELU=1 ./build_host/MNNConvert -f ONNX --modelFile {out} "
          f"--MNNModel {os.path.basename(out).replace('.onnx','.mnn')} --bizCode biz --fp16")


if __name__ == "__main__":
    main()
