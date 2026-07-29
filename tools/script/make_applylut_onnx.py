#!/usr/bin/env python3
"""
make_applylut_onnx.py

Create an ONNX model that uses MNN's custom **ApplyLUT** operator (per-element
1-D lookup-table remap with linear interpolation), so it can be converted with
MNNConvert (-f ONNX) and run on the MNN ApplyLUT op.

ApplyLUT has no standard ONNX operator, so this emits a custom node:

    op_type = "ApplyLUT"          (default ONNX domain "")
    attrs:
        interp : int   1 = linear (default, only mode implemented), 0 = nearest
    inputs:
        0: input   (float image, ~[0,1], e.g. [1,1,1440,1920])
        1: lut     (float [N], N in [8,256]) - initializer by default, or a
                   graph input with --lut-input so it can change at runtime
    outputs:
        0: output  (float, same shape/dtype as input 0)

Semantics reproduced by the op (clamp-index-only):
    pos  = input * (N-1)
    i0   = clamp(floor(pos), 0, N-1);  i1 = min(i0+1, N-1)
    frac = pos - i0
    output = lut[i0] + frac * (lut[i1] - lut[i0])

Only `onnx` (and `numpy`) are required:  pip install onnx numpy

--------------------------------------------------------------------------
NumPy reference (independent of the op, for cross-checking):

    from make_applylut_onnx import apply_lut_reference
    out = apply_lut_reference(img, lut)     # img any shape, lut 1-D [N]
--------------------------------------------------------------------------
"""

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def apply_lut_reference(img, lut):
    """Clamp-index-only linear-interp LUT remap. img: ndarray, lut: 1-D [N]."""
    lut = np.asarray(lut, dtype=np.float32).reshape(-1)
    N = lut.shape[0]
    pos = np.asarray(img, dtype=np.float32) * (N - 1)
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, N - 1)
    i1 = np.minimum(i0 + 1, N - 1)
    frac = pos - i0.astype(np.float32)
    lo = lut[i0]
    hi = lut[i1]
    return (lo + frac * (hi - lo)).astype(np.float32)


def make_applylut_node(inputs, outputs, interp=1, name="applylut"):
    """Build a single custom ApplyLUT NodeProto. inputs: [image, lut]."""
    if len(inputs) != 2:
        raise ValueError("ApplyLUT takes 2 inputs [image, lut], got %d" % len(inputs))
    if len(outputs) != 1:
        raise ValueError("ApplyLUT has 1 output, got %d" % len(outputs))
    return helper.make_node("ApplyLUT", inputs=list(inputs), outputs=list(outputs),
                            name=name, interp=int(interp))


def build_model(shape, lut, interp=1, lut_as_input=False, domain="",
                opset=13, ir_version=9):
    lut = np.asarray(lut, dtype=np.float32).reshape(-1)
    N = lut.shape[0]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(shape))]
    initializers = []
    if lut_as_input:
        inputs.append(helper.make_tensor_value_info("lut", TensorProto.FLOAT, [N]))
    else:
        initializers.append(numpy_helper.from_array(lut, name="lut"))

    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(shape))]
    node = make_applylut_node(["input", "lut"], ["output"], interp=interp, name="applylut")
    if domain:
        node.domain = domain

    graph = helper.make_graph([node], "applylut_model", inputs, outputs, initializer=initializers)
    opset_imports = [helper.make_opsetid("", opset)]
    if domain:
        opset_imports.append(helper.make_opsetid(domain, 1))
    model = helper.make_model(graph, opset_imports=opset_imports,
                              producer_name="make_applylut_onnx")
    model.ir_version = ir_version
    return model


def _parse_shape(s):
    return [int(x) for x in s.replace("[", "").replace("]", "").split(",") if x.strip() != ""]


def main():
    p = argparse.ArgumentParser(description="Create an ONNX model with MNN's custom ApplyLUT op.")
    p.add_argument("--lut-size", type=int, default=16, help="LUT length N (default 16)")
    p.add_argument("--shape", default="1,1,1440,1920",
                   help="image shape, comma-separated (default 1,1,1440,1920)")
    p.add_argument("--interp", type=int, default=1, choices=[0, 1],
                   help="1=linear (default), 0=nearest (reserved)")
    p.add_argument("--lut-input", action="store_true",
                   help="expose the LUT as a runtime graph input instead of an initializer")
    p.add_argument("--lut-kind", choices=["ramp", "gamma", "random"], default="gamma",
                   help="how to fill the baked LUT initializer (default gamma)")
    p.add_argument("--domain", default="",
                   help='custom op domain (default "" - matches the MNN converter)')
    p.add_argument("--opset", type=int, default=13, help="ONNX opset (default 13)")
    p.add_argument("--ir-version", type=int, default=9, help="ONNX IR version (default 9)")
    p.add_argument("--check", action="store_true",
                   help="run onnx.checker (will complain about the custom op in the default domain)")
    p.add_argument("-o", "--out", default="applylut.onnx", help="output .onnx path")
    args = p.parse_args()

    shape = _parse_shape(args.shape)
    N = args.lut_size
    t = np.linspace(0.0, 1.0, N, dtype=np.float32)
    if args.lut_kind == "ramp":
        lut = t
    elif args.lut_kind == "gamma":
        lut = np.power(t, 2.2, dtype=np.float32)   # a representative tone curve
    else:
        lut = np.random.RandomState(0).rand(N).astype(np.float32)

    model = build_model(shape, lut, interp=args.interp, lut_as_input=args.lut_input,
                        domain=args.domain, opset=args.opset, ir_version=args.ir_version)

    if args.check:
        try:
            onnx.checker.check_model(model)
            print("onnx.checker: OK")
        except Exception as e:  # custom op is expected to trip the strict checker
            print("onnx.checker (expected for a custom op): %s" % e)

    onnx.save(model, args.out)
    print("Wrote %s" % args.out)
    print("  node    : ApplyLUT (domain '%s')" % (args.domain or ""))
    print("  input   : input float %s" % shape)
    print("  lut     : %s N=%d (%s)" % ("graph input" if args.lut_input else "initializer",
                                        N, args.lut_kind))
    print("  attrs   : interp=%d" % args.interp)
    print("  output  : output float %s" % shape)
    print("Convert with:  MNNConvert -f ONNX --modelFile %s --MNNModel applylut.mnn --bizCode test" % args.out)


if __name__ == "__main__":
    main()
