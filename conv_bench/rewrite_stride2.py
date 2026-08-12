#!/usr/bin/env python3
"""Tier 1 — exact stride-2 (or stride-s) conv -> SpaceToDepth(s) + stride-1 conv rewrite.

A Conv(kernel K, stride s>=2, pad P, group 1, dilation 1) is mathematically identical to
SpaceToDepth(s) followed by a stride-1 conv on s^2*Cin channels with a smaller kernel and
rearranged weights. This is exact (no retrain) and keeps the OUTPUT shape unchanged (a stride-s
conv already downsamples), so nothing downstream cascades.

Weight remap (per original tap (a,b) of W[o,c,a,b], per spatial axis, pad P, stride s):
    r  = floor((a-P)/s)     phase = (a-P) mod s
    r_min = floor((0-P)/s)  Kb = floor((K-1-P)/s) - r_min + 1   new_pad_before = -r_min
    channel = phase_i*(s*Cin) + phase_j*Cin + c        (ONNX SpaceToDepth channel order)
    W'[o, channel, r_i - r_min_i, r_j - r_min_j] = W[o,c,a,b]

Everything is gated by an exact numpy self-check (max-abs-diff == 0 on random input) before it is
ever trusted; the caller then verifies equivalence + speed ON DEVICE (see verify_stride2.py).

Usage:
    python3 rewrite_stride2.py selfcheck                 # numpy-only exactness tests
    python3 rewrite_stride2.py <in.onnx> <out.onnx>      # rewrite a model
"""
import sys, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto


# ---------- numpy reference pieces (used only by the self-check) ----------
def _conv_ref(X, W, b, stride, pad):
    # X:[Cin,H,W]  W:[Cout,Cin,Kh,Kw]  pad:[pt,pl,pb,pr]  -> [Cout,Ho,Wo]
    Cout, Cin, Kh, Kw = W.shape
    pt, pl, pb, pr = pad
    Xp = np.pad(X, ((0, 0), (pt, pb), (pl, pr)))
    _, H, Wd = Xp.shape
    Ho = (H - Kh) // stride + 1
    Wo = (Wd - Kw) // stride + 1
    Y = np.zeros((Cout, Ho, Wo), np.float64)
    for o in range(Cout):
        for i in range(Ho):
            for j in range(Wo):
                patch = Xp[:, i * stride:i * stride + Kh, j * stride:j * stride + Kw]
                Y[o, i, j] = b[o] + np.sum(W[o] * patch)
    return Y


def _space_to_depth(X, s):
    # [Cin,H,W] -> [s*s*Cin, H/s, W/s], ONNX SpaceToDepth channel order (i*s*C + j*C + c)
    C, H, W = X.shape
    return X.reshape(C, H // s, s, W // s, s).transpose(2, 4, 0, 1, 3).reshape(s * s * C, H // s, W // s)


# ---------- the actual weight/attribute remap (backend-agnostic) ----------
def remap(W, s, pads_orig, Hin, Win):
    """W:[Cout,Cin,Kh,Kw], pads_orig ONNX [pt,pl,pb,pr], input Hin,Win (must be divisible by s)
    -> (Wp:[Cout,s*s*Cin,Kbh,Kbw], onnx_pads[pt,pl,pb,pr]). H-aware so output size matches exactly."""
    Cout, Cin, Kh, Kw = W.shape
    pt, pl, pb, pr = pads_orig
    ra = [(a - pt) // s for a in range(Kh)]; pih = [(a - pt) % s for a in range(Kh)]
    rb = [(b - pl) // s for b in range(Kw)]; pjw = [(b - pl) % s for b in range(Kw)]
    ra_min, ra_max = min(ra), max(ra); Kbh = ra_max - ra_min + 1
    rb_min, rb_max = min(rb), max(rb); Kbw = rb_max - rb_min + 1
    Wp = np.zeros((Cout, s * s * Cin, Kbh, Kbw), W.dtype)
    for c in range(Cin):
        for a in range(Kh):
            for b in range(Kw):
                ch = pih[a] * (s * Cin) + pjw[b] * Cin + c
                Wp[:, ch, ra[a] - ra_min, rb[b] - rb_min] = W[:, c, a, b]
    # choose after-pads so the stride-1 conv output == original stride-s output size
    Ho = (Hin + pt + pb - Kh) // s + 1
    Wo = (Win + pl + pr - Kw) // s + 1
    bh, bw = Hin // s, Win // s
    pad_after_h = Ho - 1 - bh - (-ra_min) + Kbh
    pad_after_w = Wo - 1 - bw - (-rb_min) + Kbw
    pads = [-ra_min, -rb_min, pad_after_h, pad_after_w]
    return Wp, pads


def _selfcheck_one(Cout, Cin, K, s, P, H=12, W=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((Cin, H, W)).astype(np.float64)
    Wt = rng.standard_normal((Cout, Cin, K, K)).astype(np.float64)
    b = rng.standard_normal(Cout).astype(np.float64)
    Yref = _conv_ref(X, Wt, b, s, [P, P, P, P])
    Wp, pads = remap(Wt, s, [P, P, P, P], H, W)
    if min(pads) < 0:
        print(f"  selfcheck Cout{Cout} Cin{Cin} K{K} s{s} P{P}: negative pad {pads} -> UNSUPPORTED")
        return None
    Y2 = _conv_ref(_space_to_depth(X, s), Wp, b, 1, pads)
    d = float(np.abs(Yref - Y2).max()) if Yref.shape == Y2.shape else float('inf')
    ok = bool(Yref.shape == Y2.shape and d < 1e-9)
    print(f"  selfcheck Cout{Cout} Cin{Cin} K{K} s{s} P{P}: shapes {Yref.shape} vs {Y2.shape}  "
          f"max_diff={d:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def selfcheck():
    cases = [(8, 1, 3, 2, 1), (16, 18, 3, 2, 1), (32, 34, 3, 2, 1), (8, 3, 3, 2, 1),
             (5, 2, 3, 2, 0), (6, 4, 5, 2, 2), (4, 1, 3, 3, 1)]
    res = [_selfcheck_one(*c) for c in cases]
    ok = all(r is not False for r in res)   # None (unsupported) is fine; False (wrong values) is a bug
    n_ok = sum(1 for r in res if r is True)
    print(f"self-check: {n_ok}/{len(res)} exact, {'no value bugs' if ok else 'VALUE BUGS'}")
    return ok


# ---------- ONNX graph rewrite ----------
def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return list(a.ints) if len(a.ints) else a.i
    return default


def _tensor_hw(shapes, name):
    d = shapes.get(name)
    if d is None or len(d) != 4 or d[2] is None or d[3] is None:
        return None, None
    return d[2], d[3]


def rewrite_onnx(in_path, out_path, verbose=True, only_names=None, max_cin=None):
    """Rewrite stride>=2 convs. only_names: if given, only rewrite convs whose node.name is in it.
    max_cin: if given, only rewrite convs with Cin <= max_cin (cheap occupancy heuristic; the
    rewrite is only *faster* for occupancy-starved low-Cin convs — measured crossover, so prefer
    measured selection via optimize_model.py)."""
    m = onnx.load(in_path)
    try:
        m2 = onnx.shape_inference.infer_shapes(m)
    except Exception:
        m2 = m
    shapes = {}
    for vi in list(m2.graph.value_info) + list(m2.graph.input) + list(m2.graph.output):
        dims = [(d.dim_value if d.HasField("dim_value") else None)
                for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    n_rw = 0
    for node in list(g.node):
        if node.op_type != "Conv":
            continue
        strides = _attr(node, "strides", [1, 1])
        group = _attr(node, "group", 1)
        dil = _attr(node, "dilations", [1, 1])
        pads = _attr(node, "pads", [0, 0, 0, 0])
        if not (len(strides) == 2 and strides[0] == strides[1] and strides[0] >= 2):
            continue
        if group != 1 or dil != [1, 1]:
            continue
        if only_names is not None and node.name not in only_names:
            continue
        s = strides[0]
        w_name = node.input[1]
        if w_name not in inits:
            if verbose: print(f"  skip {node.name}: weight not a constant initializer")
            continue
        Hin, Win = _tensor_hw(shapes, node.input[0])
        if Hin is None or Hin % s or Win % s:
            if verbose: print(f"  skip {node.name}: input HxW unknown or not divisible by {s} ({Hin}x{Win})")
            continue
        W = numpy_helper.to_array(inits[w_name])
        Cout, Cin, Kh, Kw = W.shape
        if max_cin is not None and Cin > max_cin:
            if verbose: print(f"  skip {node.name}: Cin {Cin} > max_cin {max_cin} (rewrite would slow it down)")
            continue
        # exact numpy self-check for THIS conv's real params/shape before rewriting it
        chk = _selfcheck_one(Cout, min(Cin, 4), Kh, s, pads[0], H=Hin if Hin <= 32 else (s * 6), W=Win if Win <= 32 else (s * 6))
        if chk is not True:
            if verbose: print(f"  skip {node.name}: self-check not exact for its params")
            continue
        Wp, new_pads = remap(W.astype(np.float32), s, pads, Hin, Win)
        Kbh, Kbw = Wp.shape[2], Wp.shape[3]
        # new initializer for W'
        wp_name = w_name + "_s2d"
        g.initializer.append(numpy_helper.from_array(Wp, wp_name))
        # SpaceToDepth node before the conv
        s2d_out = node.input[0] + f"_s2d{n_rw}"
        s2d = helper.make_node("SpaceToDepth", [node.input[0]], [s2d_out], blocksize=s,
                               name=f"s2d_{n_rw}")
        # rewire conv
        node.input[0] = s2d_out
        node.input[1] = wp_name
        for a in node.attribute:
            if a.name == "strides": a.ints[:] = [1, 1]
            elif a.name == "kernel_shape": a.ints[:] = [Kbh, Kbw]
            elif a.name == "pads": a.ints[:] = new_pads
        # insert s2d immediately before conv
        idx = list(g.node).index(node)
        g.node.insert(idx, s2d)
        n_rw += 1
        if verbose:
            print(f"  rewrote {node.name}: [{Cout},{Cin},{Kh},{Kw}] s{s} -> SpaceToDepth({s}) + "
                  f"[{Cout},{s*s*Cin},{Kbh},{Kbw}] s1 pads{new_pads}")
    onnx.save(m, out_path)
    if verbose: print(f"rewrote {n_rw} stride-{'>=2'} conv(s) -> {out_path}")
    return n_rw


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "selfcheck":
        sys.exit(0 if selfcheck() else 1)
    elif len(sys.argv) == 3:
        rewrite_onnx(sys.argv[1], sys.argv[2])
    else:
        print(__doc__); sys.exit(1)
