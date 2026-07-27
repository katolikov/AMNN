#!/usr/bin/env python3
"""
make_bincount_onnx.py

Create an ONNX model that uses MNN's custom **BinCount** operator, so it can be
converted with MNNConvert (-f ONNX) and run on the MNN/AMNN BinCount op.

BinCount has no standard ONNX operator, so this emits a custom node:

    op_type = "BinCount"          (default ONNX domain "")
    attrs:
        binNum       : int   fixed number of output bins (output length)
        binaryMask   : int   0/1 - if 1 and a 2nd input is present, that input
                             is a per-element binary mask (mask != 0 kept),
                             output stays int32 counts
        sampleStride : int   >1 downsamples the last two dims,
                             input[..., ::stride, ::stride] (approx histogram)
    inputs:
        0: values  (int32 or float; float is truncated to a bin index)
        1: (optional) weights (float -> float weight-sums, output float32)
                      OR mask  (with --mask: binary mask, output int32 counts)
    output:
        1-D tensor of length binNum. int32 for counts / masked-counts,
        float32 for weighted sums.

The MNN converter also accepts the attribute aliases "minlength" (== binNum)
and "sample_stride" (== sampleStride); this script emits the canonical names.

Only `onnx` (and `numpy`) are required:  pip install onnx numpy

--------------------------------------------------------------------------
Reuse inside your own graph (instead of a standalone model):

    from make_bincount_onnx import make_bincount_node
    node = make_bincount_node(["features"], "hist", bin_num=16,
                              binary_mask=True, sample_stride=8, name="hist0")
    # ... append `node` to your GraphProto, wire "features"/"hist" tensors ...
--------------------------------------------------------------------------
"""

import argparse
import onnx
from onnx import helper, TensorProto

_DTYPES = {"int32": TensorProto.INT32, "float32": TensorProto.FLOAT}


def make_bincount_node(inputs, output, bin_num, binary_mask=False,
                       sample_stride=1, name="bincount"):
    """Build a single custom BinCount NodeProto.

    inputs: list of 1 or 2 input tensor names ([values] or [values, weights_or_mask])
    output: output tensor name
    """
    if not (1 <= len(inputs) <= 2):
        raise ValueError("BinCount takes 1 or 2 inputs, got %d" % len(inputs))
    attrs = {"binNum": int(bin_num)}
    if binary_mask:
        attrs["binaryMask"] = 1
    if sample_stride and int(sample_stride) > 1:
        attrs["sampleStride"] = int(sample_stride)
    return helper.make_node("BinCount", inputs=list(inputs), outputs=[output],
                            name=name, **attrs)


def build_model(shape, bin_num, value_dtype="int32", second=None,
                second_dtype="float32", sample_stride=1, domain="",
                opset=13, ir_version=9,
                value_name="input", second_name="aux", output_name="output"):
    """Build a standalone single-op ModelProto.

    second: None | "mask" | "weights"
    """
    if bin_num <= 0:
        raise ValueError("bin_num must be > 0")
    if second not in (None, "mask", "weights"):
        raise ValueError("second must be None, 'mask' or 'weights'")

    binary_mask = (second == "mask")
    weighted = (second == "weights")
    out_dtype = TensorProto.FLOAT if weighted else TensorProto.INT32

    inputs = [helper.make_tensor_value_info(value_name, _DTYPES[value_dtype], list(shape))]
    node_inputs = [value_name]
    if second is not None:
        inputs.append(helper.make_tensor_value_info(second_name, _DTYPES[second_dtype], list(shape)))
        node_inputs.append(second_name)

    node = make_bincount_node(node_inputs, output_name, bin_num,
                              binary_mask=binary_mask, sample_stride=sample_stride,
                              name="bincount")
    if domain:
        node.domain = domain

    output = helper.make_tensor_value_info(output_name, out_dtype, [bin_num])

    graph = helper.make_graph([node], "bincount_model", inputs, [output])
    opset_imports = [helper.make_opsetid("", opset)]
    if domain:
        opset_imports.append(helper.make_opsetid(domain, 1))
    model = helper.make_model(graph, opset_imports=opset_imports,
                              producer_name="make_bincount_onnx")
    model.ir_version = ir_version
    return model


def _parse_shape(s):
    return [int(x) for x in s.replace("[", "").replace("]", "").split(",") if x.strip() != ""]


def main():
    p = argparse.ArgumentParser(description="Create an ONNX model with MNN's custom BinCount op.")
    p.add_argument("--bins", type=int, required=True, help="binNum: number of output bins (output length)")
    p.add_argument("--shape", default="1,1,1440,1920",
                   help="input shape, comma-separated (default 1,1,1440,1920)")
    p.add_argument("--dtype", choices=list(_DTYPES), default="int32",
                   help="value-input dtype (default int32; float is truncated to a bin index)")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--mask", action="store_true",
                     help="add a 2nd binary-mask input (binaryMask=1); output stays int32 counts")
    grp.add_argument("--weights", action="store_true",
                     help="add a 2nd float weights input; output is float32 weight-sums")
    p.add_argument("--mask-dtype", choices=list(_DTYPES), default="float32",
                   help="dtype of the mask input when --mask (default float32)")
    p.add_argument("--stride", type=int, default=1,
                   help="sampleStride: >1 downsamples last two dims input[..,::s,::s] (default 1)")
    p.add_argument("--domain", default="",
                   help='custom op domain (default "" - matches the MNN converter; '
                        'set only if your toolchain requires a non-default domain)')
    p.add_argument("--opset", type=int, default=13, help="ONNX opset (default 13)")
    p.add_argument("--ir-version", type=int, default=9, help="ONNX IR version (default 9)")
    p.add_argument("--check", action="store_true",
                   help="run onnx.checker (will complain about the custom op in the default domain)")
    p.add_argument("-o", "--out", default="bincount.onnx", help="output .onnx path (default bincount.onnx)")
    args = p.parse_args()

    second = "mask" if args.mask else ("weights" if args.weights else None)
    shape = _parse_shape(args.shape)

    model = build_model(shape, args.bins, value_dtype=args.dtype, second=second,
                        second_dtype=args.mask_dtype, sample_stride=args.stride,
                        domain=args.domain, opset=args.opset, ir_version=args.ir_version)

    if args.check:
        try:
            onnx.checker.check_model(model)
            print("onnx.checker: OK")
        except Exception as e:  # custom op is expected to trip the strict checker
            print("onnx.checker (expected for a custom op): %s" % e)

    onnx.save(model, args.out)

    out_dtype = "float32" if second == "weights" else "int32"
    print("Wrote %s" % args.out)
    print("  node       : BinCount (domain '%s')" % (args.domain or ""))
    print("  inputs     : %s%s" % (
        "%s %s" % (args.dtype, shape),
        "" if second is None else "  +  %s %s (%s)" % (args.mask_dtype if second == "mask" else "float32", shape, second)))
    print("  attrs      : binNum=%d%s%s" % (
        args.bins,
        ", binaryMask=1" if second == "mask" else "",
        ", sampleStride=%d" % args.stride if args.stride > 1 else ""))
    print("  output     : %s [%d]" % (out_dtype, args.bins))
    print("Convert with:  MNNConvert -f ONNX --modelFile %s --MNNModel bincount.mnn --bizCode test" % args.out)


if __name__ == "__main__":
    main()
