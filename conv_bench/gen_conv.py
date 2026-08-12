#!/usr/bin/env python3
"""Generate single-op Conv(+activation) ONNX graphs for the MNN OpenCL buffer/fp16 sweep.

No torch dependency — builds ONNX via the onnx helper API directly.
Activation is emitted as a SEPARATE node (Relu/Clip[=ReLU6]/PRelu) so we can measure
fusion vs non-fusion; MNN's converter/optimizer may fuse it into the conv — that's exactly
what we want to observe.
"""
import argparse, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper


def make_conv(path, N, Cin, Cout, H, W, kH=3, kW=3, stride=1, pad=1, dil=1, group=1,
              act="none", bias=True, seed=0):
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [N, Cin, H, W])

    wshape = [Cout, Cin // group, kH, kW]
    w = rng.standard_normal(wshape).astype(np.float32) * (1.0 / np.sqrt(Cin * kH * kW))
    inits.append(numpy_helper.from_array(w, "W"))
    conv_inputs = ["input", "W"]
    if bias:
        b = (rng.standard_normal([Cout]).astype(np.float32) * 0.01)
        inits.append(numpy_helper.from_array(b, "B"))
        conv_inputs.append("B")

    conv_out = "conv_out" if act != "none" else "output"
    nodes.append(helper.make_node(
        "Conv", conv_inputs, [conv_out],
        kernel_shape=[kH, kW], strides=[stride, stride],
        pads=[pad, pad, pad, pad], dilations=[dil, dil], group=group))

    if act == "relu":
        nodes.append(helper.make_node("Relu", [conv_out], ["output"]))
    elif act == "relu6":
        # Clip 0..6  (ReLU6)
        lo = numpy_helper.from_array(np.array(0.0, np.float32), "clip_lo"); inits.append(lo)
        hi = numpy_helper.from_array(np.array(6.0, np.float32), "clip_hi"); inits.append(hi)
        nodes.append(helper.make_node("Clip", [conv_out, "clip_lo", "clip_hi"], ["output"]))
    elif act == "prelu":
        # VARYING per-channel slopes so a channel-index/packing bug is actually caught
        # (a constant slope would pass even if channels were permuted).
        slope = (0.05 + 0.40 * (np.arange(Cout, dtype=np.float32) / max(Cout - 1, 1)))
        inits.append(numpy_helper.from_array(slope, "slope"))
        nodes.append(helper.make_node("PRelu", [conv_out, "slope"], ["output"]))
    elif act == "prelu_scalar":
        # scalar LeakyReLU (slopeCount==1) -> tests the broadcast branch of the fusion pass
        slope = np.array([0.123], np.float32)
        inits.append(numpy_helper.from_array(slope, "slope"))
        nodes.append(helper.make_node("PRelu", [conv_out, "slope"], ["output"]))
    elif act != "none":
        raise ValueError(act)

    # output H/W
    Ho = (H + 2 * pad - dil * (kH - 1) - 1) // stride + 1
    Wo = (W + 2 * pad - dil * (kW - 1) - 1) // stride + 1
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [N, Cout, Ho, Wo])

    graph = helper.make_graph(nodes, "conv_test", [x], [y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    macs = N * Cout * Ho * Wo * (Cin // group) * kH * kW
    print(f"{path}  X=[{N},{Cin},{H},{W}] W=[{Cout},{Cin//group},{kH},{kW}] g{group} "
          f"s{stride} p{pad} d{dil} act={act} -> Y=[{N},{Cout},{Ho},{Wo}]  MACs={macs:,}")
    return dict(N=N, Cin=Cin, Cout=Cout, H=H, W=W, kH=kH, kW=kW, stride=stride, pad=pad,
                dil=dil, group=group, act=act, Ho=Ho, Wo=Wo, macs=macs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--Cin", type=int, default=96)
    p.add_argument("--Cout", type=int, default=96)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--kH", type=int, default=3)
    p.add_argument("--kW", type=int, default=3)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--pad", type=int, default=1)
    p.add_argument("--dil", type=int, default=1)
    p.add_argument("--group", type=int, default=1)
    p.add_argument("--act", default="none", choices=["none", "relu", "relu6", "prelu", "prelu_scalar"])
    p.add_argument("--no-bias", action="store_true")
    a = p.parse_args()
    make_conv(a.out, a.N, a.Cin, a.Cout, a.H, a.W, a.kH, a.kW, a.stride, a.pad, a.dil,
              a.group, a.act, not a.no_bias)
