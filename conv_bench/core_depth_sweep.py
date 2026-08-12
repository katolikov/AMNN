#!/usr/bin/env python3
"""Quick empirical fusion-ceiling check (no custom kernel / no rebuild).

Sweep the exact Block1 core conv (32->32 @72x96, 3x3 s1 +PReLU) across chain depths.
Each added conv in the chain writes its output to global and the next reads it -- exactly
the traffic a fused megakernel would eliminate. If the per-conv kernel time is FLAT vs depth,
there is no super-linear inter-layer penalty, and the fusion ceiling = the (tiny) intermediate
write+read already baked into each conv. Bounds option-2/3 payoff before building anything.

Usage: python3 core_depth_sweep.py [C H W]   (default 32 72 96)
"""
import re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
import bench
from mk_chain import make_chain
from c0_ceiling import per_kernel

LOCAL = REPO / "conv_bench" / "work"


def run(C, H, W, depth):
    LOCAL.mkdir(parents=True, exist_ok=True)
    onnx_p = LOCAL / f"core_d{depth}.onnx"; mnn_p = LOCAL / f"core_d{depth}.mnn"
    make_chain(str(onnx_p), 1, C, H, W, depth=depth, act="prelu", k=3, stride=1)
    bench.convert(str(onnx_p), str(mnn_p), fp16=False, fuse_prelu=True)
    bench.ensure_device(push=True)
    out = bench.run_on_device(str(mnn_p), "input", [1, C, H, W], "output",
                              loops=120, gpu_mode=68, prec_mem_mask=2,
                              tuning_cache=f"core_d{depth}.cache")
    pk, nwin = per_kernel(out)
    conv_sum = sum(v for k, v in pk.items() if k.startswith("ConvBuf2D"))
    tots = [int(t) for t in re.findall(r"total kernel time = (\d+)  us", out)]
    tot_med = sorted(tots[3:])[len(tots[3:]) // 2] if len(tots) > 3 else (tots[-1] if tots else 0)
    return dict(depth=depth, conv_sum=conv_sum, per_conv=conv_sum / depth, total=tot_med, nwin=nwin)


if __name__ == "__main__":
    C, H, W = (int(x) for x in (sys.argv[1:4] or [32, 72, 96]))
    print(f"Core conv 3x3 s1 +PReLU(fused), C={C} {H}x{W}, sustained load")
    print(f"{'depth':>5} {'per_conv_us':>12} {'conv_sum_us':>12} {'total_us':>9} {'win':>4}")
    rows = []
    for d in (1, 2, 3, 4, 6):
        r = run(C, H, W, d); rows.append(r)
        print(f"{r['depth']:>5} {r['per_conv']:>12.1f} {r['conv_sum']:>12} {r['total']:>9} {r['nwin']:>4}")
    pc = [r['per_conv'] for r in rows]
    print(f"\nper-conv spread: min {min(pc):.1f}  max {max(pc):.1f}  "
          f"=> inter-layer penalty {(max(pc)-min(pc)):.1f} us/conv "
          f"({100*(max(pc)-min(pc))/min(pc):.1f}% of per-conv) = fusion ceiling upper bound")
