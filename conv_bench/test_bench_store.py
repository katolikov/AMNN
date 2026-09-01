#!/usr/bin/env python3
"""Offline tests for bench_store. No device, no adb.

The interesting cases are not synthetic: each of the first three replays a real failure from the
2026-08-27/28 investigation using its actual recorded numbers, and asserts the store would have
stopped it. If a future refactor makes any of these pass silently, the guard is gone.

    python3 conv_bench/test_bench_store.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_store import (ResultStore, CrossBatchComparison, DegenerateMeasurement,
                         MissingBaseline, UncheckedKernel, is_deployed, NOISE_FLOOR_PCT)

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name); print(f"  ok    {name}")
    except AssertionError as e:
        FAIL.append((name, e)); print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAIL.append((name, e)); print(f"  ERROR {name}: {type(e).__name__}: {e}")


def new_store():
    d = tempfile.mkdtemp()
    s = ResultStore(Path(d) / "t.db")
    s.begin_run(device="TESTDEV", shape_family="reduced", clock_start=980)
    return s


# --------------------------------------------------------------------------- real failure 1
def test_nowg_baseline_is_labelled():
    """Section 5's numbers: every arm, INCLUDING its baseline, ran under MNN_NO_WINOGRAD=1.
    On 96->96@6x8 that baseline is a path MNN never deploys, which is how "-34%" was reported
    and then re-measured at +5%. The store must mark the batch as not-deployed and say so."""
    s = new_store()
    with s.batch(section="5", label="shape hardcoding (as section 5 measured it)", reps=5) as b:
        b.baseline("MNN default")
        b.record("96->96@6x8", "MNN default", 92.0, env="MNN_NO_WINOGRAD=1 ")
        b.record("96->96@6x8", "c4h1w1_hc+HARD", 61.0,
                 env="MNN_NO_WINOGRAD=1 MNN_CONV_HARD=1 MNN_CONV_SPEC=1 "
                     "MNN_CONV_FORCE=conv_2d_c4h1w1_hc ")
        bid = b.batch_id
    row = s.rows(bid)[0]
    assert row["baseline_env"].strip() == "MNN_NO_WINOGRAD=1", row["baseline_env"]
    assert row["deployed"] == 0, "a MNN_NO_WINOGRAD baseline must not be marked deployed"
    md = s.render(bid)
    assert "not the deployed configuration" in md, "render() must warn in the table itself"
    assert "MNN_NO_WINOGRAD=1" in md, "render() must print the baseline env"


def test_deployed_baseline_has_no_warning():
    """The corrected re-measurement used no flags, so its table must NOT carry the warning."""
    s = new_store()
    with s.batch(section="h2h", label="hardcoded vs deployed default", reps=5) as b:
        b.baseline("default")
        b.record("96->96@6x8", "default", 75.0, env="")
        b.record("96->96@6x8", "c4h1w1_hc+HARD", 79.0,
                 env="MNN_CONV_HARD=1 MNN_CONV_SPEC=1 MNN_CONV_FORCE=conv_2d_c4h1w1_hc ")
        bid = b.batch_id
    md = s.render(bid)
    assert s.rows(bid)[0]["deployed"] == 1
    assert "not the deployed configuration" not in md
    cmp = s.compare("96->96@6x8", "default", "c4h1w1_hc+HARD")
    assert cmp["pct"] > 0, "hardcoding is SLOWER here; the sign must survive"
    assert not cmp["significant"], f"+5% is inside the {NOISE_FLOOR_PCT}% floor"


# --------------------------------------------------------------------------- real failure 2
def test_cross_batch_comparison_refused():
    """32->32@24x32 read 99us in section 4 and 33us in section 17 -- both correct in their own
    section. Differencing them is what produced a fabricated ranking. Must raise."""
    s = new_store()
    with s.batch(section="4", label="kernel strategies", reps=5) as b:
        b.baseline("free choice")
        b.record("32->32@24x32", "free choice", 99.0, env="")
        b.record("32->32@24x32", "c8h2w1", 78.9,
                 env="MNN_NO_WINOGRAD=1 MNN_CONV_SPEC=1 MNN_CONV_FORCE=conv_2d_c8h2w1 ")
    with s.batch(section="17", label="memory mode", reps=5) as b:
        b.baseline("buffer default")
        b.record("32->32@24x32", "buffer default", 33.0, env="")
        b.record("32->32@24x32", "image", 29.0, env="", mode=132)
    try:
        s.compare("32->32@24x32", "c8h2w1", "image")
    except CrossBatchComparison as e:
        assert "same batch" in str(e)
        return
    raise AssertionError("comparing a section-4 arm against a section-17 arm must raise")


def test_same_batch_comparison_allowed():
    """The fix for the above is to measure them together -- which must then work."""
    s = new_store()
    with s.batch(section="h2h", label="image vs kernel, one batch", reps=5) as b:
        b.baseline("default")
        b.record("32->32@24x32", "default", 32.0, env="")
        b.record("32->32@24x32", "image", 28.5, env="", mode=132)
        b.record("32->32@24x32", "kernel", 32.0,
                 env="MNN_CONV_SPEC=1 MNN_CONV_FORCE=conv_2d_c8h2w1 ")
    c = s.compare("32->32@24x32", "default", "image")
    assert abs(c["pct"] + 10.9) < 0.5, c["pct"]
    assert c["significant"]
    k = s.compare("32->32@24x32", "default", "kernel")
    assert k["pct"] == 0.0 and not k["significant"], "forcing the kernel changed nothing here"


# --------------------------------------------------------------------------- real failure 3
def test_degenerate_cell_raises():
    """A leftover {DEV}/output dir degenerated the timed loops and recorded 0.98us for a conv.
    It was averaged into a published table. Must now blow up the batch."""
    s = new_store()
    try:
        with s.batch(section="s2", label="stride-2 modes (corrupted run)", reps=4) as b:
            b.baseline("BUFdef")
            b.record("64->96@12x16", "BUFdef", 204.0, env="")
            b.record("64->96@12x16", "IMGdef", 0.9838709677419355, env="", mode=132)
    except DegenerateMeasurement as e:
        assert "plausibility floor" in str(e)
        return
    raise AssertionError("a 0.98us conv must raise DegenerateMeasurement")


def test_degenerate_cell_is_still_stored_and_rendered_as_invalid():
    """Storing it (rather than dropping it) keeps the broken run debuggable."""
    s = new_store()
    with s.batch(section="s2", label="corrupted", reps=4, tolerate_invalid=True) as b:
        b.baseline("BUFdef")
        b.record("64->96@12x16", "BUFdef", 204.0, env="")
        b.record("64->96@12x16", "IMGdef", 0.98, env="", mode=132)
        bid = b.batch_id
    md = s.render(bid)
    assert "INVALID" in md
    try:
        s.compare("64->96@12x16", "BUFdef", "IMGdef")
    except DegenerateMeasurement:
        return
    raise AssertionError("comparing against an invalid cell must raise")


def test_inflated_baseline_would_not_be_caught_but_is_recorded():
    """Honest limit: the 104us vs 22.5us inflated baseline is PLAUSIBLE, so no trap fires on the
    value alone. What the store gives is the batch_id -- the clean re-run lands in a different
    batch, so the two can never be silently differenced, and the run's clock_valid flag travels
    with every row."""
    s = new_store()
    with s.batch(section="s2", label="corrupted run", reps=4) as b:
        b.baseline("BUFdef"); b.record("1->8@192x256", "BUFdef", 104.0, env="")
    with s.batch(section="s2", label="clean run", reps=4) as b:
        b.baseline("BUFdef"); b.record("1->8@192x256", "BUFdef", 22.5, env="")
    rows = s.find("1->8@192x256", "BUFdef")
    assert len({r["batch_id"] for r in rows}) == 2, "must remain two separable batches"


# --------------------------------------------------------------------------- structural guards
def test_missing_baseline_raises():
    s = new_store()
    try:
        with s.batch(section="x", label="no baseline declared") as b:
            b.record("32->32@24x32", "some arm", 40.0)
    except MissingBaseline:
        return
    raise AssertionError("a batch with no declared baseline must raise")


def test_baseline_declared_but_not_recorded_raises():
    s = new_store()
    try:
        with s.batch(section="x", label="phantom baseline") as b:
            b.baseline("default")
            b.record("32->32@24x32", "c8h1w1", 40.0)
    except MissingBaseline as e:
        assert "never recorded" in str(e)
        return
    raise AssertionError("declaring a baseline that was never measured must raise")


def test_correctness_gate_blocks_reporting():
    """fused2 timed as 'FASTER -52%' while returning cosine -0.0056. Timings for a kernel with a
    failing (or missing) gate must not render."""
    s = new_store()
    with s.batch(section="10", label="fused2", reps=3) as b:
        b.baseline("1 conv")
        b.record("32->32@24x32", "1 conv", 43.0, env="")
        b.record("32->32@24x32", "fused2", 41.0, env="MNN_CONV_FUSED2=1 ")
        b.correctness("32->32@24x32", "1 conv", 1.0)
        b.correctness("32->32@24x32", "fused2", -0.005629)
        bid = b.batch_id
    s.render(bid)                      # without the gate it renders fine
    try:
        s.render(bid, check_correctness=True)
    except UncheckedKernel as e:
        assert "FAILED its correctness gate" in str(e)
        return
    raise AssertionError("a kernel failing its correctness gate must not be reportable")


def test_missing_correctness_gate_blocks_reporting():
    s = new_store()
    with s.batch(section="x", label="ungated", reps=1) as b:
        b.baseline("default")
        b.record("32->32@24x32", "default", 32.0)
        b.record("32->32@24x32", "mystery", 20.0, env="MNN_CONV_FORCE=mystery ")
        b.correctness("32->32@24x32", "default", 1.0)
        bid = b.batch_id
    try:
        s.render(bid, check_correctness=True)
    except UncheckedKernel as e:
        assert "no correctness gate" in str(e)
        return
    raise AssertionError("an ungated kernel must not be reportable")


def test_samples_reduce_to_median_and_spread():
    s = new_store()
    with s.batch(section="x", label="samples", reps=5) as b:
        b.baseline("default")
        r = b.record("c", "default", 0, samples=[30.0, 32.0, 31.0, 90.0, 31.5])
    assert r["us"] == 31.5, r["us"]
    assert r["n"] == 5 and r["spread"] == 60.0


def test_noise_floor_marks_small_deltas():
    s = new_store()
    with s.batch(section="x", label="noise", reps=5) as b:
        b.baseline("default")
        b.record("c", "default", 100.0)
        b.record("c", "tiny win", 96.0)      # -4%, inside the floor
        b.record("c", "real win", 80.0)      # -20%
        bid = b.batch_id
    md = s.render(bid)
    assert "-4% (noise, floor 6%)" in md, md   # render now states WHICH floor it applied
    assert "-20% |" in md, md


def test_clock_validity_recorded():
    s = new_store()
    assert s.finish_run(clock_end=980) is True
    s2 = ResultStore(Path(tempfile.mkdtemp()) / "t2.db")
    s2.begin_run(device="D", shape_family="reduced", clock_start=980)
    assert s2.finish_run(clock_end=557) is False, "a 43% drop must be flagged invalid"


def test_is_deployed_semantics():
    assert is_deployed("", 68)
    assert not is_deployed("", 132), "image mode is an intervention"
    assert not is_deployed("MNN_NO_WINOGRAD=1 ", 68)
    assert not is_deployed("MNN_CONV_FORCE=x ", 68)


def test_batch_running_too_long_raises():
    """full_sweep's batch took ~2h, over which the clock fell 980 -> 899 MHz, and its verdicts
    contradicted three shorter runs. Duration is what decides comparability, so it is what is
    checked -- the same arm count can take a minute warm or an hour cold."""
    import time as _t
    from bench_store import OversizedBatch
    s = new_store()
    try:
        with s.batch(section="x", label="two-hour batch", reps=3, max_seconds=0.05) as b:
            b.baseline("a0")
            b.record("c", "a0", 30.0)
            b.record("c", "a1", 31.0)
            _t.sleep(0.06)
    except OversizedBatch as e:
        assert "did not see the same clock" in str(e)
        return
    raise AssertionError("a batch that ran past its time limit must raise")


def test_duration_guard_uses_measurement_start_not_record_time():
    """A caller that measures first and records afterwards would otherwise show ~0s elapsed and
    never trip the guard -- which is exactly how the two-hour batch was recorded as valid."""
    import time as _t
    from bench_store import OversizedBatch
    s = new_store()
    measured_at = _t.time() - 7200          # measurement began two hours ago
    try:
        with s.batch(section="x", label="recorded after the fact", reps=3,
                     started=measured_at, max_seconds=600) as b:
            b.baseline("a0"); b.record("c", "a0", 30.0); b.record("c", "a1", 31.0)
    except OversizedBatch as e:
        assert "120 min" in str(e), str(e)
        return
    raise AssertionError("a batch whose MEASUREMENT ran two hours must raise")


def test_measured_clock_validity_supersedes_the_duration_proxy():
    """Duration is a proxy for 'did every arm see the same clock'. When a Watchdog has measured
    that directly, the evidence wins: a 13-minute batch at a held clock is comparable, and
    rejecting it would throw away good data (the cold-cache case)."""
    import time as _t
    s = new_store()
    with s.batch(section="x", label="long but stable", reps=3,
                 started=_t.time() - 1800, max_seconds=600, clock_valid=True) as b:
        b.baseline("a0"); b.record("c", "a0", 30.0); b.record("c", "a1", 31.0)
    assert len(s.rows(b.batch_id)) == 2


def test_throttled_batch_still_fails_on_duration():
    import time as _t
    from bench_store import OversizedBatch
    s = new_store()
    try:
        with s.batch(section="x", label="long and throttled", reps=3,
                     started=_t.time() - 1800, max_seconds=600, clock_valid=False) as b:
            b.baseline("a0"); b.record("c", "a0", 30.0)
    except OversizedBatch:
        return
    raise AssertionError("a long batch whose clock did NOT hold must still raise")


def test_oversized_arm_count_raises():
    from bench_store import OversizedBatch
    s = new_store()
    try:
        with s.batch(section="x", label="too many arms", reps=3, max_arms=5) as b:
            b.baseline("a0")
            for i in range(6):
                b.record("c", f"a{i}", 30.0 + i)
    except OversizedBatch as e:
        assert "split it per probe model" in str(e)
        return
    raise AssertionError("a batch above the arm cap must raise")


def test_batch_within_cap_is_fine():
    s = new_store()
    with s.batch(section="x", label="one model", reps=3, max_arms=5) as b:
        b.baseline("a0")
        for i in range(4):
            b.record("c", f"a{i}", 30.0 + i)
    assert len(s.rows(b.batch_id)) == 4


if __name__ == "__main__":
    print("bench_store — replaying the investigation's real failures\n")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name[5:].replace("_", " "), fn)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


# --------------------------------------------------------------------------- variance / floors
def test_stability_is_the_one_legal_cross_batch_query():
    """compare() refuses two arms across batches; stability() of ONE arm across batches is the
    measurement we actually want, and must work. Uses the real 34->32@48x64 image numbers."""
    import bench_store as BS
    s = new_store()
    for us in (71.3, 46.2, 46.2):
        with s.batch(section="v", label=f"repeat {us}", reps=3) as b:
            b.baseline("image")
            b.record("34->32@48x64 s2", "image", us, env="", mode=132)
    st = s.stability("34->32@48x64 s2", "image")
    assert st["n"] == 3
    assert 53 < st["spread_pct"] < 55, st["spread_pct"]


def test_measured_floor_suppresses_a_false_claim():
    """With the measured 54% spread loaded, a 20% difference on that conv is NOT a result --
    even though it clears the global 6% floor."""
    import bench_store as BS, json as _j, tempfile as _t
    from pathlib import Path as _P
    f = _P(_t.mkdtemp()) / "noise_floors.json"
    f.write_text(_j.dumps({"floors": {"34->32@48x64 s2|image": {"spread_pct": 54.0},
                                      "34->32@48x64 s2|buffer": {"spread_pct": 4.0}}}))
    assert BS.load_noise_floors(f) == 2
    try:
        assert BS.noise_floor("34->32@48x64 s2", "image") == 54.0
        assert BS.noise_floor("34->32@48x64 s2", "buffer") == 6.0, "never below the fallback"
        s = new_store()
        with s.batch(section="v", label="modes", reps=3) as b:
            b.baseline("buffer")
            b.record("34->32@48x64 s2", "buffer", 51.0, env="", mode=68)
            b.record("34->32@48x64 s2", "image", 41.0, env="", mode=132)   # -20%
        c = s.compare("34->32@48x64 s2", "buffer", "image")
        assert abs(c["pct"] + 19.6) < 0.5, c["pct"]
        assert not c["significant"], "-20% must NOT be a claim against a 54% floor"
        assert c["floor"] == 54.0
        md = s.render(b.batch_id)
        assert "floor 54%" in md, md
    finally:
        BS._FLOORS = {}


def test_stable_config_still_allows_claims():
    import bench_store as BS, json as _j, tempfile as _t
    from pathlib import Path as _P
    f = _P(_t.mkdtemp()) / "nf.json"
    f.write_text(_j.dumps({"floors": {"c|buffer": {"spread_pct": 3.0},
                                      "c|image": {"spread_pct": 4.0}}}))
    BS.load_noise_floors(f)
    try:
        s = new_store()
        with s.batch(section="v", label="stable", reps=3) as b:
            b.baseline("buffer")
            b.record("c", "buffer", 100.0, env="", mode=68)
            b.record("c", "image", 82.0, env="", mode=132)
        c = s.compare("c", "buffer", "image")
        assert c["significant"] and c["floor"] == 6.0
    finally:
        BS._FLOORS = {}