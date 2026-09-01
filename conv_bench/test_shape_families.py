#!/usr/bin/env python3
"""Shape-family tests. No device, no bundle.

Families are a divisor applied to model_convs_updated.csv. The risk that replaces is drift between
hand-maintained per-family files -- which really happened: CONV_BENCH_SHAPES=full built full-size
blocks alongside 1/3-size cores and heads, and nothing said so.

The div3 case is pinned to the 13 convs that were actually measured, so if the derivation ever
stops reproducing them the tests fail rather than the results quietly changing meaning.

    python3 conv_bench/test_shape_families.py
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASS, FAIL = [], []


def check(name, fn):
    try:
        fn(); PASS.append(name); print(f"  ok    {name}")
    except AssertionError as e:
        FAIL.append(name); print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAIL.append(name); print(f"  ERROR {name}: {type(e).__name__}: {e}")


def convs_for(family):
    """(cin, cout, H, W, stride) for every conv, as a subprocess so the env var takes effect."""
    code = ("import sys, json; sys.path.insert(0, %r);"
            "from block_fixture import load_blocks;"
            "print(json.dumps([[c['cin'], c['cout'], c['H'], c['W'], c['stride']]"
            " for b in load_blocks().values() for c in b]))" % str(HERE))
    env = dict(os.environ, CONV_BENCH_SHAPES=family)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr.strip().splitlines()[-1:] or out.stderr
    import json
    return [tuple(x) for x in json.loads(out.stdout.strip().splitlines()[-1])]


# the 13 distinct convs the reduced-shape investigation actually measured
MEASURED_DIV3 = {
    (1, 8, 192, 256, 2), (8, 16, 96, 128, 2), (18, 16, 96, 128, 2), (16, 32, 48, 64, 2),
    (34, 32, 48, 64, 2), (32, 48, 24, 32, 2), (64, 64, 24, 32, 2), (64, 96, 12, 16, 2),
    (8, 8, 96, 128, 1), (16, 16, 48, 64, 1), (32, 32, 24, 32, 1), (48, 48, 12, 16, 1),
    (96, 96, 6, 8, 1),
}


def test_div3_reproduces_the_measured_shapes():
    got = set(convs_for("3"))
    assert got == MEASURED_DIV3, f"missing {MEASURED_DIV3 - got}, unexpected {got - MEASURED_DIV3}"


def test_reduced_is_an_alias_for_3():
    assert set(convs_for("reduced")) == set(convs_for("3"))


def test_full_is_the_csv_unchanged():
    got = set(convs_for("full"))
    assert (96, 96, 18, 24, 1) in got and (1, 8, 576, 768, 2) in got, sorted(got)[:3]


def test_div1p5_is_two_thirds_of_full():
    full = {(ci, co, s): (h, w) for ci, co, h, w, s in convs_for("full")}
    for (ci, co, s), (h, w) in {(ci, co, s): (h, w)
                                for ci, co, h, w, s in convs_for("1.5")}.items():
        fh, fw = full[(ci, co, s)]
        assert (h, w) == (fh * 2 // 3, fw * 2 // 3), f"{ci}->{co} s{s}: {h}x{w} vs {fh}x{fw}"


def test_every_family_keeps_dimensions_even():
    """Stride-2 halving must stay exact, or a block's pyramid does not survive the reduction."""
    for fam in ("full", "1.5", "3"):
        for ci, co, h, w, s in convs_for(fam):
            assert h % 2 == 0 and w % 2 == 0, f"{fam}: {ci}->{co} {h}x{w} is odd"


def test_inexact_divisor_is_refused():
    try:
        convs_for("5")
    except AssertionError as e:
        assert "not an integer" in str(e), e
        return
    raise AssertionError("a divisor that does not divide every conv exactly must be refused")


def test_odd_result_is_refused():
    """/9 divides 576 and 288 exactly but yields odd 18x24 -> 2x3; stride-2 would stop halving."""
    try:
        convs_for("9")
    except AssertionError as e:
        assert "odd" in str(e) or "not an integer" in str(e), e
        return
    raise AssertionError("a divisor producing an odd dimension must be refused")


def test_nonsense_family_is_refused():
    try:
        convs_for("banana")
    except AssertionError:
        return
    raise AssertionError("a non-numeric, non-'full' family must be refused")


if __name__ == "__main__":
    print("shape families\n")
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            check(n[5:].replace("_", " "), f)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
