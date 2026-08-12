#!/usr/bin/env python3
"""Tier 2 — architecture representation, ONNX builder, and cascade-aware variant generator.

An arch is a fixed input + an ordered list of stages; each stage = a 3x3 conv (stride s, pad 1)
followed by PReLU. The search knobs are (a) WHERE the stride-2 downsamples sit and (b) per-stage
channel WIDTH. Variants keep the model's input shape, the total downsample factor (=> final
spatial fixed), and the final output-channel count fixed — so every candidate has the SAME I/O
shape (no accidental output change) while the internal stride-1 convs run at different
resolution/width operating points. Weights are random (latency is weight-independent)."""
import itertools, os
from dataclasses import dataclass
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper


@dataclass(frozen=True)
class Arch:
    in_chw: tuple          # (C, H, W)
    stages: tuple          # ((out_ch, stride), ...)

    def sig(self):
        return f"{self.in_chw}|" + ",".join(f"{o}s{s}" for o, s in self.stages)

    def shapes(self):
        # returns list of (Cin,Cout,H,W,stride) per stage and final (C,H,W)
        C, H, W = self.in_chw; out = []
        for (oc, s) in self.stages:
            out.append((C, oc, H, W, s)); C, H, W = oc, H // s, W // s
        return out, (C, H, W)

    def down_factor(self):
        f = 1
        for _, s in self.stages: f *= s
        return f

    def flops(self):
        f = 0
        for (Cin, Cout, H, Wd, s) in self.shapes()[0]:
            f += 2 * Cout * (H // s) * (Wd // s) * Cin * 9   # 3x3
        return f

    def params(self):
        return sum(Cout * Cin * 9 + Cout for (Cin, Cout, _, _, _) in self.shapes()[0])


def build_onnx(arch: Arch, path, seed=0):
    rng = np.random.default_rng(seed)
    nodes, inits = [], []; prev = "input"
    stg = arch.shapes()[0]
    for i, (Cin, Cout, H, Wd, s) in enumerate(stg):
        wn, bn, sn = f"w{i}", f"b{i}", f"s{i}"
        inits.append(numpy_helper.from_array((rng.standard_normal((Cout, Cin, 3, 3)) * 0.05).astype(np.float32), wn))
        inits.append(numpy_helper.from_array((rng.standard_normal(Cout) * 0.01).astype(np.float32), bn))
        inits.append(numpy_helper.from_array((0.05 + 0.1 * rng.standard_normal(Cout)).astype(np.float32), sn))
        cout_t = f"c{i}"; pout_t = "output" if i == len(stg) - 1 else f"p{i}"
        nodes.append(helper.make_node("Conv", [prev, wn, bn], [cout_t], kernel_shape=[3, 3],
                                      pads=[1, 1, 1, 1], strides=[s, s], name=f"conv{i}"))
        nodes.append(helper.make_node("PRelu", [cout_t, sn], [pout_t], name=f"prelu{i}"))
        prev = pout_t
    C, H, W = arch.in_chw
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, H, W])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)
    m = helper.make_model(helper.make_graph(nodes, "arch", [x], [y], inits),
                          opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 9
    onnx.save(m, path)
    return path


def _r4(x):
    return max(4, int(round(x / 4)) * 4)


def generate(base: Arch, width_mults=(0.75, 1.0, 1.5), max_variants=24):
    """Cascade-aware variants: same input, same total down-factor, same final out_ch.
    Vary (a) placement of the stride-2 stages, (b) width of the intermediate stages."""
    n = len(base.stages)
    strides = [s for _, s in base.stages]
    widths = [o for o, _ in base.stages]
    k = sum(1 for s in strides if s == 2)          # number of stride-2 stages (down-factor 2^k)
    final_oc = widths[-1]
    seen, out = set(), []
    stride_placements = list(itertools.combinations(range(n), k)) if k else [()]
    # width variants: multiply each intermediate stage (not first? keep last=final_oc), small set
    width_choices = []
    inner = list(range(0, n - 1))                  # stages whose width we may scale (last fixed)
    # limit combinatorics: apply ONE global multiplier, plus the all-1.0 baseline
    for mult in width_mults:
        wv = [(_r4(widths[i] * mult) if i in inner else (final_oc if i == n - 1 else widths[i])) for i in range(n)]
        width_choices.append(wv)
    for places in stride_placements:
        st = [2 if i in places else 1 for i in range(n)]
        for wv in width_choices:
            stages = tuple((wv[i], st[i]) for i in range(n))
            a = Arch(base.in_chw, stages)
            if a.down_factor() != base.down_factor(): continue
            if a.stages[-1][0] != final_oc: continue
            _, (C, H, W) = a.shapes()
            if H <= 0 or W <= 0: continue
            if a.sig() in seen: continue
            seen.add(a.sig()); out.append(a)
            if len(out) >= max_variants: return out
    return out


if __name__ == "__main__":
    # smoke test (no device): shapes consistent, ONNX builds, I/O shape invariant across variants
    base = Arch((3, 64, 64), ((16, 1), (32, 2), (64, 1), (64, 2)))
    _, out0 = base.shapes()
    print("base:", base.sig(), "-> out", out0, "down", base.down_factor(),
          "flops", base.flops(), "params", base.params())
    vs = generate(base)
    print(f"{len(vs)} variants")
    ok = True
    for a in vs:
        _, o = a.shapes()
        same = (o == out0)
        ok &= same
        build_onnx(a, "/tmp/_arch_smoke.onnx")
        print(f"  {a.sig():55} out={o} {'OK' if same else 'BAD-IO'}")
    print("I/O invariant:", ok)
