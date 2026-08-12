#!/usr/bin/env python3
"""Tier 2 — on-device latency oracle for fake-weight architectures.

measure(arch) builds a random-weight ONNX for the arch, converts it (stock MNNConvert, OpenCL
buffer/fp16 config), runs it on the connected device under sustained load, and returns the median
total-kernel GPU time. Results are cached by arch signature (persisted to json). Latency is
weight-independent (verified by weight_independence), which is what makes fake-weight search valid.
"""
import json, os, sys
from statistics import median
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "conv_bench"))
from bench import convert, run_on_device, parse, ensure_device
from arch_space import Arch, build_onnx
W = os.path.join(REPO, "conv_bench", "vwork"); os.makedirs(W, exist_ok=True)
CACHE = os.path.join(REPO, "conv_bench", "oracle_cache.json")


class Oracle:
    def __init__(self, gpu_mode=68, prec_mask=2, loops=50, repeats=3, cache_path=CACHE):
        self.cfg = dict(gpu_mode=gpu_mode, prec_mask=prec_mask, loops=loops, repeats=repeats)
        self.cache_path = cache_path
        self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        self.n_measured = 0

    def _key(self, arch):
        return f"{arch.sig()}|g{self.cfg['gpu_mode']}p{self.cfg['prec_mask']}"

    def _save(self):
        json.dump(self.cache, open(self.cache_path, "w"))

    def _run_once(self, arch, seed, tag):
        onnx_p = os.path.join(W, f"orc_{tag}.onnx"); mnn_p = os.path.join(W, f"orc_{tag}.mnn")
        build_onnx(arch, onnx_p, seed=seed); convert(onnx_p, mnn_p)
        C, H, Wd = arch.in_chw
        vals = [parse(run_on_device(mnn_p, "input", [1, C, H, Wd], "output", loops=self.cfg["loops"],
                gpu_mode=self.cfg["gpu_mode"], prec_mem_mask=self.cfg["prec_mask"],
                tuning_cache=f"orc_{tag}.cache")).get("total_kernel_us_med")
                for _ in range(self.cfg["repeats"])]
        return median([v for v in vals if v]) if any(vals) else None

    def measure(self, arch, force=False):
        k = self._key(arch)
        if not force and k in self.cache:
            return self.cache[k]
        us = self._run_once(arch, seed=0, tag=abs(hash(arch.sig())) % 100000)
        self.cache[k] = us; self.n_measured += 1; self._save()
        return us

    def weight_independence(self, arch, seeds=(1, 2, 3), tol=0.03):
        """Premise check: same arch, different random weights -> same latency. Returns (ok, vals)."""
        vals = [self._run_once(arch, seed=s, tag=f"wi{s}_{abs(hash(arch.sig()))%10000}") for s in seeds]
        vals = [v for v in vals if v]
        spread = (max(vals) - min(vals)) / min(vals) if vals else 1.0
        return (spread <= tol, vals, spread)


if __name__ == "__main__":
    from arch_space import Arch
    ensure_device(push=True)
    orc = Oracle()
    a = Arch((3, 64, 64), ((16, 1), (32, 2), (64, 1), (64, 2)))
    print("=== GATE 1: weight-independence (same arch, 3 random-weight seeds) ===")
    ok, vals, spread = orc.weight_independence(a)
    print(f"  latencies {vals}  spread={spread*100:.1f}%  {'PASS' if ok else 'FAIL'}")
    print("=== GATE 2: oracle repeatability + cache ===")
    m1 = orc.measure(a); import time; t = time.time(); m2 = orc.measure(a); cached = (time.time() - t) < 1.0
    print(f"  measure={m1}us  repeat={m2}us (from_cache={cached})")
    sys.exit(0 if ok else 1)
