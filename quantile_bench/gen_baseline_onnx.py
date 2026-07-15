"""
Build an ONNX model that reproduces torch.quantile(x, q, interpolation='linear')
over the FULLY FLATTENED input tensor, using only ops ONNX/MNN already have:
TopK (smallest-k) + Slice + Sub/Mul/Add.

q and n (=numel of the input) are fixed at graph-build time, so per the
torch.quantile linear-interpolation formula

    idx = q * (n - 1); lo = floor(idx); hi = ceil(idx); frac = idx - lo
    result = sorted(x)[lo] + frac * (sorted(x)[hi] - sorted(x)[lo])

lo/hi/frac collapse to compile-time constants. We only need the lo-th and
hi-th smallest elements, not a full sort, so each quantile uses one
TopK(largest=False, k=hi+1) call instead of sorting all n elements.
"""
import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def build_model(shape, qs, opset=13):
    n = int(np.prod(shape))
    nodes = []
    initializers = []

    def const(name, arr):
        t = numpy_helper.from_array(np.asarray(arr), name=name)
        initializers.append(t)
        return name

    x_name = "x"
    flat_name = "x_flat"
    nodes.append(helper.make_node(
        "Reshape", [x_name, const("flat_shape", np.array([-1], dtype=np.int64))],
        [flat_name], name="flatten"))

    result_names = []
    for i, q in enumerate(qs):
        idx = q * (n - 1)
        lo = int(np.floor(idx))
        hi = int(np.ceil(idx))
        frac = float(idx - lo)
        k = hi + 1

        vals = f"q{i}_vals"
        idxs = f"q{i}_idxs"
        nodes.append(helper.make_node(
            "TopK", [flat_name, const(f"q{i}_k", np.array([k], dtype=np.int64))],
            [vals, idxs], name=f"topk_q{i}", axis=0, largest=0, sorted=1))

        lower = f"q{i}_lower"
        nodes.append(helper.make_node(
            "Slice", [vals,
                      const(f"q{i}_lo_start", np.array([lo], dtype=np.int64)),
                      const(f"q{i}_lo_end", np.array([lo + 1], dtype=np.int64)),
                      const(f"q{i}_axes", np.array([0], dtype=np.int64))],
            [lower], name=f"slice_lo_q{i}"))

        if hi == lo:
            result_names.append(lower)
            continue

        upper = f"q{i}_upper"
        nodes.append(helper.make_node(
            "Slice", [vals,
                      const(f"q{i}_hi_start", np.array([hi], dtype=np.int64)),
                      const(f"q{i}_hi_end", np.array([hi + 1], dtype=np.int64)),
                      const(f"q{i}_axes2", np.array([0], dtype=np.int64))],
            [upper], name=f"slice_hi_q{i}"))

        diff = f"q{i}_diff"
        nodes.append(helper.make_node("Sub", [upper, lower], [diff], name=f"sub_q{i}"))
        weighted = f"q{i}_weighted"
        nodes.append(helper.make_node(
            "Mul", [diff, const(f"q{i}_frac", np.array([frac], dtype=np.float32))],
            [weighted], name=f"mul_q{i}"))
        result = f"q{i}_result"
        nodes.append(helper.make_node("Add", [lower, weighted], [result], name=f"add_q{i}"))
        result_names.append(result)

    out_name = "quantiles"
    nodes.append(helper.make_node("Concat", result_names, [out_name], name="concat_out", axis=0))

    graph = helper.make_graph(
        nodes,
        "quantile_baseline",
        [helper.make_tensor_value_info(x_name, TensorProto.FLOAT, list(shape))],
        [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [len(qs)])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs=4, default=[1, 1, 1440, 1920])
    ap.add_argument("--q", type=float, nargs="+", default=[0.05, 0.25, 0.5, 0.75, 0.95])
    ap.add_argument("--out", default="quantile_bench/quantile_baseline.onnx")
    args = ap.parse_args()

    model = build_model(tuple(args.shape), args.q)
    onnx.save(model, args.out)
    n = int(np.prod(args.shape))
    print(f"wrote {args.out}  shape={args.shape} n={n} q={args.q}")
    for q in args.q:
        idx = q * (n - 1)
        lo, hi = int(np.floor(idx)), int(np.ceil(idx))
        print(f"  q={q:<5} lo={lo:>9} hi={hi:>9} k=hi+1={hi+1:>9}  (OpenCL TopKV2 buf cap=1024 -> {'OK' if hi+1<=1024 else 'EXCEEDS CAP'})")
