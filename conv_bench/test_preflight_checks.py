#!/usr/bin/env python3
"""Offline tests for preflight's parser-integrity check. No device.

C1 compares the sum of per-kernel times against the profiler's own total. It must separate two
very different things:

  * a dispatch the parser cannot see -- missing from EVERY window, the failure C1 exists to catch;
  * the device output stream mangled for one inference -- the OpenCL driver splices blobcache
    warnings into stdout mid-line, e.g.
        "kernel time = 71    us ConvBuf2D-ori-b1ci8hi288wWARN: CLPlatformVk.cpp:435 ..."
    which can swallow a whole line. Observed on three unrelated cases at full size, always exactly
    one window of eight. That costs one inference out of many and is absorbed by the medians every
    metric here uses.

Treating the second as the first blocked three runs that were structurally sound. Treating the
first as the second would wave through a genuinely broken parser, so both directions are tested.

    python3 conv_bench/test_preflight_checks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight as P  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn(); PASS.append(name); print(f"  ok    {name}")
    except AssertionError as e:
        FAIL.append(name); print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAIL.append(name); print(f"  ERROR {name}: {type(e).__name__}: {e}")


def _out(windows):
    """windows = list of per-window kernel-time lists; each closes with its true total."""
    s = ""
    for kernels, total in windows:
        for k in kernels:
            s += f"kernel time = {k}    us ConvBuf2D-ori-b1ci8hi288wi384co16ho144wo192kh3kw3\n"
        s += f"total kernel time = {total}  us, conv time = {total} us\n"
    return s


def _run(out):
    P.FAIL.clear(); P.WARNS.clear()
    P._check_C({"key": "case", "convs": [1, 2]}, out, {})
    return len(P.FAIL), len(P.WARNS)


def test_all_windows_agree_passes():
    f, w = _run(_out([([10, 90], 100)] * 8))
    assert (f, w) == (0, 0), f"{f} fails, {w} warns"


def test_one_mangled_window_warns_but_does_not_fail():
    """The real signature: 1 of 8 windows short, the rest exact."""
    wins = [([10, 90], 100)] * 8
    wins[4] = ([10], 100)              # one line swallowed by a spliced WARN
    f, w = _run(_out(wins))
    assert f == 0, "a single mangled window must not block the run"
    assert w == 1, "it must still be reported"


def test_two_of_eight_still_only_warns():
    wins = [([10, 90], 100)] * 8
    wins[2] = ([10], 100); wins[5] = ([10], 100)
    f, w = _run(_out(wins))
    assert f == 0 and w == 1, f"{f} fails, {w} warns"


def test_half_the_windows_is_systematic_and_fails():
    wins = [([10, 90], 100)] * 8
    for i in (0, 1, 2, 3):
        wins[i] = ([10], 100)
    f, _ = _run(_out(wins))
    assert f >= 1, "a drop in half the windows is systematic and must fail"


def test_every_window_short_fails():
    f, _ = _run(_out([([10], 100)] * 8))
    assert f >= 1, "a dispatch missing from every window must fail"


def test_spliced_line_still_closes_its_window():
    """A WARN can join a kernel-time line to the window-total line. With `elif`+`search` the
    window never closed and its dispatches rolled into the next one."""
    out = ("kernel time = 10    us ConvBuf2D-ori-x\n"
           "kernel time = 90    us ConvBuf2D-ori-yWARN: blob total kernel time = 100  us, conv\n"
           ) * 8
    f, w = _run(out)
    assert (f, w) == (0, 0), f"spliced totals must still close their window ({f} fails, {w} warns)"


if __name__ == "__main__":
    print("preflight C1 -- systematic drop vs mangled output\n")
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            check(n[5:].replace("_", " "), f)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
