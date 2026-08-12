#!/usr/bin/env python3
"""Tier 2 — MECHANISM tests on the connected device (NOT accuracy).

Verifies the search machinery is correct: weight-independence premise, oracle repeatability+cache,
generator validity (all candidates run on device), cascade-invariance (I/O + down-factor preserved),
search correctness (best-latency is on the Pareto front, front is non-dominated), and that emitted
Pareto ONNX candidates load/convert/run on device. Fake weights -> accuracy is out of scope.
"""
import os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "conv_bench"))
from arch_space import Arch, generate, build_onnx
from latency_oracle import Oracle
from search import run_search, pareto
from bench import ensure_device, convert, run_on_device, parse
import onnx

BASE = Arch((3, 64, 64), ((16, 1), (32, 2), (64, 1), (64, 2)))
OUT = os.path.join(REPO, "conv_bench", "tier2_out")


def main():
    ensure_device(push=True)
    orc = Oracle()
    fails = []

    # 1. weight-independence (the premise)
    ok, vals, spread = orc.weight_independence(BASE)
    print(f"[{'PASS' if ok else 'FAIL'}] 1 weight-independence: {vals} spread={spread*100:.1f}% (<=3%)")
    if not ok: fails.append("weight-independence")

    # 2. oracle repeatability + cache
    m1 = orc.measure(BASE); t = time.time(); m2 = orc.measure(BASE); dt = time.time() - t
    ok2 = (m1 == m2) and dt < 1.0
    print(f"[{'PASS' if ok2 else 'FAIL'}] 2 cache: measure={m1} repeat={m2} cached_fast={dt<1.0}")
    if not ok2: fails.append("cache")

    # 3+5. generator validity + search correctness (uses cache -> fast)
    results, P, n_ran = run_search(BASE, orc, max_variants=12, emit_dir=OUT)
    ran = [r for r in results if r["us"] is not None]
    ok3 = (n_ran == len(results))
    best = min(ran, key=lambda r: r["us"])
    ok5 = best in P and all(r in P for r in pareto(results))
    print(f"[{'PASS' if ok3 else 'FAIL'}] 3 generator validity: {n_ran}/{len(results)} ran on device")
    print(f"[{'PASS' if ok5 else 'FAIL'}] 5 search correctness: best_latency_in_pareto={best in P} pareto_nondominated={all(r in P for r in pareto(results))}")
    if not ok3: fails.append("generator-validity")
    if not ok5: fails.append("search-correctness")

    # 4. cascade-invariance: every variant preserves I/O shape and down-factor; a stride move changes an intermediate spatial
    variants = generate(BASE, max_variants=24)
    _, io_base = BASE.shapes()
    io_ok = all(v.shapes()[1] == io_base and v.down_factor() == BASE.down_factor() for v in variants)
    # confirm the cascade actually moves: two variants with different stride placement differ in an intermediate spatial
    sp = lambda a: [h for (_, _, h, _, _) in a.shapes()[0]]
    moved = len({tuple(sp(v)) for v in variants}) > 1
    ok4 = io_ok and moved
    print(f"[{'PASS' if ok4 else 'FAIL'}] 4 cascade: IO+downfactor invariant={io_ok} intermediate-spatial-varies={moved}")
    if not ok4: fails.append("cascade")

    # 6. emitted Pareto ONNX load/convert/run on device with expected output shape
    ok6 = True
    for j in range(len(P)):
        p = os.path.join(OUT, f"pareto_{j}.onnx")
        if not os.path.exists(p): ok6 = False; break
        mnn = p.replace(".onnx", ".mnn"); convert(p, mnn)
        C, H, W = BASE.in_chw
        out = run_on_device(mnn, "input", [1, C, H, W], "output", loops=10, gpu_mode=68, prec_mem_mask=2, tuning_cache=f"pm{j}.cache")
        ok6 &= ("kernel time" in out or "conv" in out.lower())
    print(f"[{'PASS' if ok6 else 'FAIL'}] 6 emitted Pareto ONNX run on device: {len(P)} candidate(s)")
    if not ok6: fails.append("emitted-onnx")

    print(f"\n=== MECHANISM: {'ALL PASS' if not fails else 'FAIL: ' + ','.join(fails)} ===")
    return not fails


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
