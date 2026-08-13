#!/usr/bin/env python3
"""Build the REAL model's conv blocks (from model_convs_updated.csv) as faithful
ONNX chains, run on OpenCL buffer/fp16 under sustained load, dump per-kernel GPU time.

Each block is an independent linear chain (conv[+PReLU])* at fixed shapes. We build the
exact chain (Cin/Cout/H/W/stride/pad/prelu per conv) so timing reflects the real graph,
not a homogeneous approximation. Reuses bench.py (convert/push/run) + c0_ceiling.per_kernel.

Usage:  python3 block_fixture.py [Block1|Block2|all]
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
import bench
# c0_ceiling is imported lazily inside run_block(): make_bundle only needs
# load_blocks()/build_onnx(), and a top-level import made that a hard dependency.

CSV = REPO / "conv_bench" / "model_convs_updated.csv"
LOCAL = REPO / "conv_bench" / "work"


def load_blocks():
    """Parse the CSV into {block_name: [conv spec dicts in order]}."""
    blocks, cur = {}, None
    with open(CSV) as f:
        for row in csv.reader(f):
            if not row or not row[0].strip().isdigit():
                continue  # skip comments / header / blank-idx separators
            idx = int(row[0]); label = row[1].strip()
            cin = row[2].strip()
            if not cin:
                continue  # gap row (idx present, no data)
            spec = dict(idx=idx, cin=int(cin), cout=int(row[3]), H=int(row[4]),
                        W=int(row[5]), stride=int(row[6]), pad=int(row[7]),
                        prelu=int(row[8]))
            if label.lower().startswith("block"):
                cur = label; blocks[cur] = []
            if cur is None:
                cur = "Block?"; blocks[cur] = []
            blocks[cur].append(spec)
    return blocks


def build_onnx(path, convs):
    """Linear chain: input -> conv0[+prelu] -> conv1[+prelu] -> ... -> output."""
    inits, nodes = [], []
    c0 = convs[0]
    prev = "input"
    for j, c in enumerate(convs):
        w = numpy_helper.from_array(
            (np.random.randn(c["cout"], c["cin"], 3, 3).astype(np.float32) * 0.05), f"w{j}")
        b = numpy_helper.from_array(
            (np.random.randn(c["cout"]).astype(np.float32) * 0.01), f"b{j}")
        inits += [w, b]
        conv_out = f"c{j}"
        nodes.append(helper.make_node(
            "Conv", [prev, f"w{j}", f"b{j}"], [conv_out],
            kernel_shape=[3, 3], pads=[c["pad"]] * 4, strides=[c["stride"]] * 2))
        last = (j == len(convs) - 1)
        if c["prelu"]:
            slope = numpy_helper.from_array(
                (0.05 + 0.4 * np.arange(c["cout"]) / max(c["cout"] - 1, 1)).astype(np.float32),
                f"s{j}")
            inits.append(slope)
            act_out = "output" if last else f"p{j}"
            nodes.append(helper.make_node("PRelu", [conv_out, f"s{j}"], [act_out]))
            prev = act_out
        else:
            if last:
                nodes[-1].output[0] = "output"
            prev = conv_out
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, c0["cin"], c0["H"], c0["W"]])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 0, 0, 0])
    m = helper.make_model(helper.make_graph(nodes, "blk", [x], [y], inits),
                          opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.save(m, path)


def macs(c):  # multiply-accumulates for one conv (uses padded output size)
    oh = (c["H"] + 2 * c["pad"] - 3) // c["stride"] + 1
    ow = (c["W"] + 2 * c["pad"] - 3) // c["stride"] + 1
    return c["cin"] * c["cout"] * oh * ow * 9


def run_block(name, convs, fuse_prelu=False):
    LOCAL.mkdir(parents=True, exist_ok=True)
    tag = name + ("_fused" if fuse_prelu else "")
    onnx_p = LOCAL / f"{tag}.onnx"; mnn_p = LOCAL / f"{tag}.mnn"
    build_onnx(str(onnx_p), convs)
    bench.convert(str(onnx_p), str(mnn_p), fp16=False, fuse_prelu=fuse_prelu)
    bench.ensure_device(push=True)
    c0 = convs[0]
    out = bench.run_on_device(str(mnn_p), "input", [1, c0["cin"], c0["H"], c0["W"]], "output",
                              loops=120, gpu_mode=68, prec_mem_mask=2,
                              tuning_cache=f"{tag}.cache")
    from c0_ceiling import per_kernel
    pk, nwin = per_kernel(out)
    tots = [int(t) for t in re.findall(r"total kernel time = (\d+)  us", out)]
    tot_med = sorted(tots[3:])[len(tots[3:]) // 2] if len(tots) > 3 else (tots[-1] if tots else 0)
    total_macs = sum(macs(c) for c in convs)
    print(f"\n===== {name}: {len(convs)} convs, total_kernel_med = {tot_med} us "
          f"(windows kept {nwin}) =====")
    for k, v in sorted(pk.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>7} us  {k}")
    if tot_med:
        gflop = 2 * total_macs / 1e9
        print(f"  --> block MACs={total_macs/1e6:.1f}M, {gflop:.3f} GFLOP, "
              f"achieved={gflop/(tot_med/1e6)/1e3:.2f} TFLOP/s over total_kernel")
    return {"name": name, "convs": convs, "total_kernel_us_med": tot_med,
            "per_kernel_us": pk, "total_macs": total_macs}, out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = args[0] if args else "all"
    fuse = "--fuse" in sys.argv
    blocks = load_blocks()
    print("Parsed blocks:", {k: len(v) for k, v in blocks.items()}, "fuse_prelu=", fuse)
    results, raw = {}, {}
    for name, convs in blocks.items():
        if which != "all" and name != which:
            continue
        r, out = run_block(name, convs, fuse_prelu=fuse)
        results[name] = {k: v for k, v in r.items() if k != "convs"}
        raw[name] = out
    (REPO / "conv_bench" / "block_baseline.json").write_text(json.dumps(results, indent=2, default=str))
    (REPO / "conv_bench" / "block_baseline_raw.txt").write_text(
        "\n\n".join(f"##### {k}\n{v}" for k, v in raw.items()))
    print("\nwrote conv_bench/block_baseline.json")
