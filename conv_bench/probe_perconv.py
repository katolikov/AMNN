#!/usr/bin/env python3
"""Time MANY convs per launch by reading the shape tag MNN puts in each conv kernel's name.

Before: one conv per launch, 13 launches to cover the reduced shape set. The block models already
chain the real convs, so five of them cover all 13 distinct shapes -- and now that the image
backend tags its kernels too (Conv2D-ori-.../Conv2D-wino-...), the same five models can be timed
per conv in EITHER memory mode. That is 5 launches per arm instead of 13.

Parsing rules that matter:
  * Both backends are matched: ConvBuf2D-ori-b1ci32... (buffer) and Conv2D-ori-b1ci32... (image).
  * A Winograd conv dispatches THREE kernels (-src/-gemm/-dst) which all carry the same shape tag.
    They are summed into one number for that conv. MNN's own `conv time` counter omits the two
    transforms; scoring a Winograd arm with it credits the gemm only, which once turned a +15%
    regression into an apparent -45% win.
  * A shape appearing N times in a chain (the 6-deep cores) contributes N dispatches per
    inference; the per-conv figure divides by that count, not by the number of inferences alone.
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict

# ci/hi/wi/co identifies one conv shape unambiguously in both backends' kernel names.
# Four naming shapes exist across the two backends and two algorithms; all now carry the same
# b..ci..hi..wi..co..ho..wo..kh..kw shape field, which is what identifies the conv:
#   buffer direct    ConvBuf2D-ori-b1ci32hi24wi32co48ho12wo16kh3kw3-total:...
#   buffer winograd  Conv-winograd-batchgemm-b16m192n32k32-shape-b1ci32...-total:...
#   image  direct    Conv2D-ori-b1ci18hi96wi128co16ho48wo64kh3kw3
#   image  winograd  Conv2D-wino-b1ci96hi6wi8co96ho6wo8kh3kw3-src|-gemm|-dst
TAG = re.compile(r"kernel time = (\d+)\s+us (\S*[Cc]onv\S*?"
                 r"(b\d+ci\d+hi\d+wi\d+co\d+ho\d+wo\d+kh\d+kw\d+))(\S*)")
LOOP = re.compile(r"total kernel time = \d+  us")
# The one dispatch every conv has exactly once, whatever the implementation. Everything else that
# carries the same shape tag is a helper whose time belongs to the conv but which must not be
# counted as another instance of it.
PRIMARY = re.compile(r"(?:ConvBuf2D-ori-|ConvBuf2D-conv1x1-|Conv2D-ori-|"
                     r"Conv-winograd-batchgemm-|Conv2D-wino-\S*-gemm)")
# Any conv-ish kernel that carries NO shape tag. Some strategies switch the conv IMPLEMENTATION and
# emit untagged helper kernels -- MNN_CONV_NCHW and MNN_CONV_SPLITK add ConvBuf2D-gemm2-0 (layout
# in) and -gemm2-2 (layout out). Per-conv attribution cannot see those, so it silently counts a
# FRACTION of the conv's real cost and the arm looks dramatically faster than it is: splitK=4 read
# -70% on 64->96@12x16 this way, against a strategy measured dead in every prior run.
UNTAGGED = re.compile(r"kernel time = (\d+)\s+us (\S*[Cc]onv\S*)")


def untagged_fraction(out: str) -> float:
    """Share of conv-kernel time that per_conv() cannot attribute to any shape.

    Above a few percent, per-conv numbers for this arm are incomplete and must not be compared
    against arms whose cost is fully attributed. Use whole-model timing for those instead."""
    tagged = untagged = 0
    for m in UNTAGGED.finditer(out):
        us, name = int(m.group(1)), m.group(2)
        if re.search(r"b\d+ci\d+hi\d+wi\d+co\d+", name):
            tagged += us
        else:
            untagged += us
    total = tagged + untagged
    return (untagged / total) if total else 0.0


def per_conv(out: str) -> dict[str, float]:
    """{shape_tag: microseconds per conv per inference}, counting EVERY kernel the conv costs.

    A conv dispatches one PRIMARY kernel plus zero or more HELPERS:

        direct            ori (primary)
        winograd          batchgemm/gemm (primary) + rearrange/src/dst (helpers)
        NCHW, split-K     ori (primary) + gemm2-0 layout-in, gemm2-2 layout-out (helpers)

    Instances are counted from PRIMARY dispatches only, while the time sums over primaries and
    helpers alike. Inferring instances from the raw dispatch count instead -- which is what this
    did first -- divides by the helper multiplicity too, and reports a conv that gained expensive
    layout conversions as though it had got faster. That is how NCHW read -42% while actually
    being 48% slower."""
    loops = len(LOOP.findall(out))
    if not loops:
        return {}
    total: dict[str, int] = defaultdict(int)
    primaries: dict[str, int] = defaultdict(int)
    for m in TAG.finditer(out):
        # The role suffix (-gemm, -src, -dst) sits AFTER the shape, so it lands in group(4).
        # Testing PRIMARY against group(2) alone never matched image-mode Winograd, whose primary
        # is "Conv2D-wino-<shape>-gemm" -- those convs recorded zero primaries and were dropped
        # from the result entirely, which silently removed image mode as a candidate on exactly
        # the three Winograd cores where it wins.
        us, tag = int(m.group(1)), m.group(3)
        full = m.group(2) + m.group(4)
        total[tag] += us
        if PRIMARY.search(full):
            primaries[tag] += 1
    res = {}
    for tag, us in total.items():
        n = primaries.get(tag, 0)
        if not n:
            # Time was attributed to this shape but no primary dispatch was recognised, so the
            # instance count is unknown. Dropping it silently is what hid image-mode Winograd;
            # surface it instead so a naming change fails loudly rather than deleting a candidate.
            import sys as _s
            print(f"[probe_perconv] WARNING: {tag} has {us:.0f}us of kernel time but no "
                  f"recognised PRIMARY dispatch -- conv dropped; PRIMARY needs updating",
                  file=_s.stderr)
            continue
        res[tag] = us / n              # total cost of this shape / number of conv instances
    return res


def label_of(tag: str) -> str:
    """b1ci32hi24wi32co48ho12wo16kh3kw3 -> '32->48@24x32'."""
    m = re.match(r"b\d+ci(\d+)hi(\d+)wi(\d+)co(\d+)", tag)
    if not m:
        return tag
    ci, hi, wi, co = m.groups()
    return f"{ci}->{co}@{hi}x{wi}"


# The five block models that between them contain all 13 distinct convs of the shape set.
# Block5's two convs duplicate shapes already covered by Block1/Block4, so it is not needed.
#
# The INPUT SHAPES are read from the bundle manifest rather than written here, because they differ
# per shape family: Block3 takes [1,1,192,256] reduced and [1,1,576,768] full. Hardcoding the
# reduced ones meant CONV_BENCH_SHAPES=full fed every probe model the wrong input size.
_WANT = ["Block3", "Block4", "Block1", "Block2", "Block96"]


def _probe_models():
    import json
    from pathlib import Path as _P
    man = _P(__file__).resolve().parent / "conv_probe_bundle" / "manifest.json"
    if not man.exists():
        return []                       # bundle not built yet; callers surface the real error
    blocks = {b["key"]: b for b in json.loads(man.read_text()).get("blocks", [])}
    return [(blocks[k]["model"], blocks[k]["shape"]) for k in _WANT if k in blocks]


PROBE_MODELS = _probe_models()
