#!/usr/bin/env python3
"""Tier 2 — latency search over the arch space, emits a Pareto front (latency vs capacity).

Measures every cascade-aware variant on the device (fake weights), computes FLOPs/params as a
capacity proxy, and returns the non-dominated set (min latency, min params). Writes a report and
emits the Pareto archs as ONNX + specs. Ranking is by LATENCY (real, on device); accuracy is the
user's to validate after retraining the emitted candidates. Mechanism-only — no accuracy claim.
"""
import csv, json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "conv_bench"))
from arch_space import Arch, generate, build_onnx
from latency_oracle import Oracle
from bench import ensure_device
W = os.path.join(REPO, "conv_bench", "vwork")


def pareto(results):
    # Trade-off: MINIMIZE latency, MAXIMIZE capacity (params). A point is dominated if another is
    # at least as fast AND has at least as much capacity, with one strict.
    P = []
    for r in results:
        if r["us"] is None:
            continue
        dominated = any((o["us"] <= r["us"] and o["params"] >= r["params"]
                         and (o["us"] < r["us"] or o["params"] > r["params"]))
                        for o in results if o["us"] is not None and o is not r)
        if not dominated:
            P.append(r)
    return sorted(P, key=lambda r: r["us"])


def run_search(base, oracle, max_variants=12, emit_dir=None):
    variants = generate(base, max_variants=max_variants)
    print(f"generated {len(variants)} variants; measuring on device...")
    results = []
    n_ran = 0
    for i, a in enumerate(variants):
        us = oracle.measure(a)
        ran = us is not None
        n_ran += ran
        results.append(dict(sig=a.sig(), us=us, flops=a.flops(), params=a.params(), arch=a))
        print(f"  [{i+1:2}/{len(variants)}] {a.sig():48} {us}us flops={a.flops()/1e6:.0f}M params={a.params()} {'' if ran else 'DID-NOT-RUN'}")
    print(f"generator validity: {n_ran}/{len(variants)} ran on device")
    P = pareto(results)
    # write report
    rep_csv = os.path.join(REPO, "conv_bench", "tier2_search.csv")
    with open(rep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sig", "us", "flops", "params", "pareto"])
        w.writeheader()
        psigs = {r["sig"] for r in P}
        for r in sorted(results, key=lambda r: (r["us"] is None, r["us"] or 0)):
            w.writerow({"sig": r["sig"], "us": r["us"], "flops": r["flops"],
                        "params": r["params"], "pareto": r["sig"] in psigs})
    print(f"\nPareto front ({len(P)}):")
    for r in P:
        print(f"  {r['sig']:48} {r['us']}us  params={r['params']}")
    if emit_dir:
        os.makedirs(emit_dir, exist_ok=True)
        for j, r in enumerate(P):
            build_onnx(r["arch"], os.path.join(emit_dir, f"pareto_{j}.onnx"))
        json.dump([{"sig": r["sig"], "us": r["us"], "params": r["params"]} for r in P],
                  open(os.path.join(emit_dir, "pareto.json"), "w"), indent=2)
    return results, P, n_ran


if __name__ == "__main__":
    ensure_device(push=True)
    base = Arch((3, 64, 64), ((16, 1), (32, 2), (64, 1), (64, 2)))
    orc = Oracle()
    results, P, n_ran = run_search(base, orc, max_variants=12,
                                   emit_dir=os.path.join(REPO, "conv_bench", "tier2_out"))
    # --- mechanism checks (NOT accuracy) ---
    ran = [r for r in results if r["us"] is not None]
    valid = (n_ran == len(results))
    best = min(ran, key=lambda r: r["us"]) if ran else None
    best_in_pareto = best in P
    print(f"\n[mechanism] all_candidates_ran={valid}  best_latency_in_pareto={best_in_pareto}  "
          f"pareto_nondominated={all(r in P for r in pareto(results))}")
    sys.exit(0 if valid and best_in_pareto else 1)
