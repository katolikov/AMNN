#!/usr/bin/env python3
"""Phase C0 — measure the multi-conv fusion CEILING before writing any kernel.
Builds hero-shape conv chains of depth 1 and 2 (N=8, C=96, 64x64, 3x3 s1),
runs on OpenCL buffer/fp16, dumps the full per-kernel GPU-time breakdown.
The removable-by-fusion traffic = conv0 output-write + conv1 intermediate-read.
Compares depth-2 total vs 2x depth-1 to bound how much fusion could ever save."""
import json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
import bench
from mk_chain import make_chain

KT_RE = re.compile(r"kernel time = (\d+)\s+us (\S+)")


def per_kernel(out, warmup_frac=0.5):
    """Aggregate 'kernel time = X us NAME' across windows -> median per distinct name."""
    from collections import defaultdict
    import statistics
    windows = []
    cur = []
    for ln in out.splitlines():
        m = KT_RE.search(ln)
        if m:
            cur.append((m.group(2), int(m.group(1))))
        elif "total kernel time" in ln and cur:
            windows.append(cur); cur = []
    if cur:
        windows.append(cur)
    if not windows:
        return {}, 0
    # drop warmup windows
    keep = windows[int(len(windows) * warmup_frac):] or windows[-3:]
    agg = defaultdict(list)
    for w in keep:
        per = defaultdict(int)
        for name, us in w:
            per[name] += us
        for name, us in per.items():
            agg[name].append(us)
    return {k: int(statistics.median(v)) for k, v in agg.items()}, len(keep)


def run(depth, tag, N=8, C=96, H=64, W=64):
    LOCAL = REPO / "conv_bench" / "work"
    LOCAL.mkdir(parents=True, exist_ok=True)
    onnx_p = LOCAL / f"{tag}.onnx"; mnn_p = LOCAL / f"{tag}.mnn"
    make_chain(str(onnx_p), N, C, H, W, depth=depth, act="none", k=3, stride=1)
    bench.convert(str(onnx_p), str(mnn_p), fp16=False)
    bench.ensure_device(push=True)
    out = bench.run_on_device(str(mnn_p), "input", [N, C, H, W], "output",
                              loops=120, gpu_mode=68, prec_mem_mask=2)
    pk, nwin = per_kernel(out)
    tot = re.findall(r"total kernel time = (\d+)  us", out)
    tot = [int(t) for t in tot]
    tot_med = sorted(tot[3:])[len(tot[3:]) // 2] if len(tot) > 3 else (tot[-1] if tot else 0)
    return {"depth": depth, "per_kernel_us": pk, "total_kernel_us_med": tot_med,
            "windows_kept": nwin}, out


REGIMES = {
    # name: (N, C, H, W)  -- hero (winograd) vs small-channel (non-winograd, direct/gemm)
    "hero_C96": (8, 96, 64, 64),
    "small_C32": (1, 32, 144, 192),
}
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "hero_C96"
    N, C, H, W = REGIMES[which]
    results = {"regime": which, "shape": [N, C, H, W]}
    raw_dump = {}
    for depth in (1, 2):
        r, out = run(depth, f"c0_{which}_d{depth}", N, C, H, W)
        results[f"depth{depth}"] = r
        raw_dump[f"depth{depth}"] = out
        print(f"\n===== {which} depth={depth} shape={[N,C,H,W]} =====")
        print(f"total_kernel_us_med = {r['total_kernel_us_med']}  (windows kept {r['windows_kept']})")
        for k, v in sorted(r["per_kernel_us"].items(), key=lambda kv: -kv[1]):
            print(f"  {v:>7} us  {k}")
    (REPO / "conv_bench" / f"c0_ceiling_{which}.json").write_text(json.dumps(results, indent=2))
    (REPO / "conv_bench" / f"c0_raw_{which}.txt").write_text(
        "\n\n".join(f"##### {k}\n{v}" for k, v in raw_dump.items()))
    print(f"\nwrote conv_bench/c0_ceiling_{which}.json")
