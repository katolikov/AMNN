#!/usr/bin/env python3
"""
make_dualrangehist_onnx.py

Create an ONNX model that uses MNN's custom **DualRangeHist** operator (dual
masked range-histogram), so it can be converted with MNNConvert (-f ONNX) and
run on the MNN/AMNN DualRangeHist op.

DualRangeHist has no standard ONNX operator, so this emits a custom node:

    op_type = "DualRangeHist"     (default ONNX domain "")
    attrs:
        binNum         : int   bins per histogram (each output length)
        low, high      : float shared inclusive validity range on the raw values
        sampleStride   : int   >1 downsamples the last two dims (approx histogram)
        emitValidCount : int   0/1 - emit a 3rd int32 output = sum(keep)
    inputs:
        0: A     (float frame, ~[0,1])
        1: B     (float frame, ~[0,1])
        2: (optional) base  per-element validity mask (float or int32; !=0 kept)
    outputs:
        histA[binNum], histB[binNum]  (int32 counts)
        (+ validCount[1] int32 when emitValidCount=1)

Semantics reproduced by the op:
    keep = base & (low<=A<=high) & (low<=B<=high)      # raw values, inclusive
    histA[rint(A*(binNum-1))] += keep;  histB[rint(B*(binNum-1))] += keep
    validCount = sum(keep) == sum(histA) == sum(histB)

The MNN converter also accepts attribute aliases "minlength" (==binNum),
"sample_stride" (==sampleStride) and "valid_count" (==emitValidCount).

Only `onnx` (and `numpy`) are required:  pip install onnx numpy

--------------------------------------------------------------------------
Reuse inside your own graph (instead of a standalone model):

    from make_dualrangehist_onnx import make_dualrangehist_node
    node = make_dualrangehist_node(["A", "B", "base"], ["histA", "histB"],
                                   bin_num=16, low=0.05, high=0.95,
                                   sample_stride=8, name="hist0")
    # ... append `node` to your GraphProto, wire the tensors ...
--------------------------------------------------------------------------
"""

import argparse
import onnx
from onnx import helper, TensorProto

_DTYPES = {"int32": TensorProto.INT32, "float32": TensorProto.FLOAT}


def make_dualrangehist_node(inputs, outputs, bin_num, low=0.0, high=1.0,
                            sample_stride=1, emit_valid_count=False,
                            name="dualrangehist"):
    """Build a single custom DualRangeHist NodeProto.

    inputs:  [A, B] or [A, B, base]
    outputs: [histA, histB] or [histA, histB, validCount]
    """
    if not (2 <= len(inputs) <= 3):
        raise ValueError("DualRangeHist takes 2 or 3 inputs, got %d" % len(inputs))
    if not (2 <= len(outputs) <= 3):
        raise ValueError("DualRangeHist has 2 or 3 outputs, got %d" % len(outputs))
    if emit_valid_count and len(outputs) != 3:
        raise ValueError("emit_valid_count requires 3 output names")
    if (not emit_valid_count) and len(outputs) != 2:
        raise ValueError("without emit_valid_count there are 2 output names")
    attrs = {
        "binNum": int(bin_num),
        "low": float(low),
        "high": float(high),
    }
    if sample_stride and int(sample_stride) > 1:
        attrs["sampleStride"] = int(sample_stride)
    if emit_valid_count:
        attrs["emitValidCount"] = 1
    return helper.make_node("DualRangeHist", inputs=list(inputs), outputs=list(outputs),
                            name=name, **attrs)


def build_model(shape, bin_num, low=0.0, high=1.0, with_base=False,
                base_dtype="float32", sample_stride=1, emit_valid_count=False,
                domain="", opset=13, ir_version=9):
    if bin_num <= 0:
        raise ValueError("bin_num must be > 0")

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, list(shape)),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, list(shape)),
    ]
    node_inputs = ["A", "B"]
    if with_base:
        inputs.append(helper.make_tensor_value_info("base", _DTYPES[base_dtype], list(shape)))
        node_inputs.append("base")

    node_outputs = ["histA", "histB"]
    outputs = [
        helper.make_tensor_value_info("histA", TensorProto.INT32, [bin_num]),
        helper.make_tensor_value_info("histB", TensorProto.INT32, [bin_num]),
    ]
    if emit_valid_count:
        node_outputs.append("validCount")
        outputs.append(helper.make_tensor_value_info("validCount", TensorProto.INT32, [1]))

    node = make_dualrangehist_node(node_inputs, node_outputs, bin_num, low=low, high=high,
                                   sample_stride=sample_stride,
                                   emit_valid_count=emit_valid_count, name="dualrangehist")
    if domain:
        node.domain = domain

    graph = helper.make_graph([node], "dualrangehist_model", inputs, outputs)
    opset_imports = [helper.make_opsetid("", opset)]
    if domain:
        opset_imports.append(helper.make_opsetid(domain, 1))
    model = helper.make_model(graph, opset_imports=opset_imports,
                              producer_name="make_dualrangehist_onnx")
    model.ir_version = ir_version
    return model


def _parse_shape(s):
    return [int(x) for x in s.replace("[", "").replace("]", "").split(",") if x.strip() != ""]


def main():
    p = argparse.ArgumentParser(description="Create an ONNX model with MNN's custom DualRangeHist op.")
    p.add_argument("--bins", type=int, required=True, help="binNum: bins per histogram (each output length)")
    p.add_argument("--shape", default="1,1,1440,1920",
                   help="frame shape, comma-separated (default 1,1,1440,1920)")
    p.add_argument("--low", type=float, default=0.05, help="shared inclusive range low (default 0.05)")
    p.add_argument("--high", type=float, default=0.95, help="shared inclusive range high (default 0.95)")
    p.add_argument("--base", action="store_true", help="add a 3rd base-mask input (!=0 kept)")
    p.add_argument("--base-dtype", choices=list(_DTYPES), default="float32",
                   help="dtype of the base-mask input when --base (default float32)")
    p.add_argument("--stride", type=int, default=1,
                   help="sampleStride: >1 downsamples last two dims A[..,::s,::s] (default 1)")
    p.add_argument("--valid-count", action="store_true",
                   help="emit a 3rd int32 output validCount = sum(keep)")
    p.add_argument("--domain", default="",
                   help='custom op domain (default "" - matches the MNN converter)')
    p.add_argument("--opset", type=int, default=13, help="ONNX opset (default 13)")
    p.add_argument("--ir-version", type=int, default=9, help="ONNX IR version (default 9)")
    p.add_argument("--check", action="store_true",
                   help="run onnx.checker (will complain about the custom op in the default domain)")
    p.add_argument("-o", "--out", default="dualrangehist.onnx", help="output .onnx path")
    args = p.parse_args()

    shape = _parse_shape(args.shape)
    model = build_model(shape, args.bins, low=args.low, high=args.high, with_base=args.base,
                        base_dtype=args.base_dtype, sample_stride=args.stride,
                        emit_valid_count=args.valid_count, domain=args.domain,
                        opset=args.opset, ir_version=args.ir_version)

    if args.check:
        try:
            onnx.checker.check_model(model)
            print("onnx.checker: OK")
        except Exception as e:  # custom op is expected to trip the strict checker
            print("onnx.checker (expected for a custom op): %s" % e)

    onnx.save(model, args.out)
    print("Wrote %s" % args.out)
    print("  node    : DualRangeHist (domain '%s')" % (args.domain or ""))
    print("  inputs  : A,B float %s%s" % (shape, ("  + base %s %s" % (args.base_dtype, shape)) if args.base else ""))
    print("  attrs   : binNum=%d, low=%g, high=%g%s%s" % (
        args.bins, args.low, args.high,
        (", sampleStride=%d" % args.stride) if args.stride > 1 else "",
        ", emitValidCount=1" if args.valid_count else ""))
    print("  outputs : histA[%d], histB[%d] int32%s" % (
        args.bins, args.bins, ", validCount[1] int32" if args.valid_count else ""))
    print("Convert with:  MNNConvert -f ONNX --modelFile %s --MNNModel dualrangehist.mnn --bizCode test" % args.out)


if __name__ == "__main__":
    main()
