#!/usr/bin/env python3
"""Render a run's results as three recommendation tables.

gpuMode is set per Interpreter, so a model gets ONE memory mode even when its convs disagree. A
single "best per conv" table therefore answers a question nobody can act on directly: it mixes
buffer and image winners that cannot coexist. These three answer the questions that map onto a
decision:

    1. BUFFER   -- if this model runs in buffer mode, what is the best kernel for each conv?
    2. IMAGE    -- if it runs in image mode, likewise.
    3. OVERALL  -- the best arm regardless of mode, against the deployed baseline. Useful for
                   sizing the prize, and for convs that sit in their own submodel.

Every percentage is measured against the baseline named in the table and checked against that
configuration's own measured noise floor; anything smaller is reported as noise, not a win.
Runnable directly against a results.db:

    python3 conv_bench/recommend.py                     # latest run
    python3 conv_bench/recommend.py --runs              # list runs, then --run <id>
    python3 conv_bench/recommend.py --run <id> --db ... # a specific run / database
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_store as BS

BUFFER, IMAGE = 68, 132


def _rows_by_conv(st, run_ids):
    """Rows for one or several runs.

    A conv measured across two runs (a model re-measured on its own after its first batch was
    rejected) still has all of ITS arms inside one batch, which is what comparability requires --
    so merging runs is safe here, while differencing arms across batches remains refused."""
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    out = {}
    q = ("SELECT batch_id, baseline_arm FROM batches WHERE run_id IN (%s) ORDER BY created"
         % ",".join("?" * len(run_ids)))
    for b in st._conn.execute(q, tuple(run_ids)):
        for r in st.rows(b["batch_id"]):
            if r["valid"]:
                out.setdefault(r["conv"], []).append(r)
    return out


def _floor(conv):
    return max(BS.noise_floor(conv, "buffer"), BS.noise_floor(conv, "image"))


def _table(title, convs, pick, baseline_arm, note=""):
    """pick(rows) -> the candidate rows for this table; baseline_arm names the reference."""
    lines = [f"\n{title}", "-" * len(title)]
    if note:
        lines.append(note)
    lines.append(f"{'conv':<20}{'baseline':>10}{'best':>10}  {'arm':<24}{'gain':>7}  verdict")
    wins = 0
    for conv in sorted(convs):
        rows = pick(convs[conv])
        base = next((r for r in rows if r["arm"] == baseline_arm), None)
        if not base or not rows:
            continue
        best = min(rows, key=lambda r: r["us"])
        gain = 100 * (best["us"] - base["us"]) / base["us"] if base["us"] else 0.0
        fl = _floor(conv)
        sig = abs(gain) >= fl and best["arm"] != baseline_arm
        wins += sig
        arm = best["arm"] if sig else baseline_arm
        shown = best["us"] if sig else base["us"]
        lines.append(f"{conv:<20}{base['us']:>10.1f}{shown:>10.1f}  {arm:<24}"
                     f"{gain:>6.0f}%  {'USE IT' if sig else 'keep default'}")
    lines.append(f"({wins} of {len(convs)} convs improve on the default in this mode)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Winograd table (--winograd)
#
# The three tables above print only the WINNING arm per conv, so an arm that never wins is never
# named. Winograd is exactly that case: on a conv the gate already admits, `buf force-winograd`
# measures the same thing as the baseline and cannot win; on a conv the gate refuses, forcing it
# usually loses. Its microseconds were always recorded -- they were just never displayed, which
# reads as "Winograd is not tested" when it is.
#
# This table shows the algorithm arms side by side and says, per conv, which of three states it is
# in. The states are not interchangeable and conflating them is the trap:
#
#   stride != 1   ConvBufWinograd::valid() rejects on stride BEFORE reading MNN_FORCE_WINOGRAD, so
#                 both flags are no-ops. Any delta here is drift, not an algorithm result -- which
#                 makes these rows a free control: they must read ~0%, and a run where they do not
#                 has a batch-stability problem that invalidates its other deltas too.
#   gate admits   the default IS Winograd. `no-winograd` shows what turning it off would cost.
#   gate refuses  the default is direct. `force-winograd` shows what admitting it would cost/save.

def _gate(ci, co, in_w):
    """MNN's admission rule for 3x3 stride-1 (ConvBufWinograd.cpp). Mirrored, not imported --
    recommend.py reads a database and must not need the MNN tree present."""
    return (ci >= 32 and co >= 32 and in_w <= 2 * co) or (ci >= 64 and co >= 64)


def _parse(conv):
    """'32->48@24x32' -> (32, 48, 24, 32). Returns None if the label is not a conv shape."""
    import re
    m = re.match(r"(\d+)->(\d+)@(\d+)x(\d+)$", conv)
    return tuple(int(x) for x in m.groups()) if m else None


def _strides(shape_family):
    """{label: stride} from model_convs_updated.csv, scaled to the run's shape family.

    Stride is not in the database (the kernel tag carries input dims only), and it decides whether
    the Winograd arms could do anything at all. Read from the same CSV the bundle was built from,
    so a family mismatch shows up as a missing entry rather than a wrong verdict."""
    import csv as _csv
    fam = (shape_family or "").strip().lower()
    div = 1.0 if fam in ("full", "1", "1.0") else 3.0 if fam == "reduced" else None
    if div is None:
        try:
            div = float(fam.removeprefix("div"))
        except ValueError:
            return {}
    csv_path = Path(__file__).resolve().parent / "model_convs_updated.csv"
    if not csv_path.exists():
        return {}
    out = {}
    with open(csv_path) as f:
        for row in _csv.reader(f):
            if not row or not row[0].strip().isdigit() or not row[2].strip():
                continue
            try:
                ci, co = int(row[2]), int(row[3])
                h, w, s = int(row[4]) / div, int(row[5]) / div, int(row[6])
            except (ValueError, IndexError):
                continue
            if h != int(h) or w != int(w):
                continue
            out[f"{ci}->{co}@{int(h)}x{int(w)}"] = s
    return out


def winograd_table(convs, shape_family=None):
    strides = _strides(shape_family)
    head = "\n4. WINOGRAD  (algorithm arms, buffer mode)"
    lines = [head, "-" * len(head.strip()),
             "what MNN's gate does per conv, and what the other choice would have cost.",
             f"{'conv':<20}{'default':>9}{'force-wino':>11}{'no-wino':>9}{'floor':>7}  "
             f"{'gate':<8}verdict"]
    drift = []
    for conv in sorted(convs):
        row = {r["arm"]: r["us"] for r in convs[conv] if r["mode"] == BUFFER}
        b = row.get("buffer default")
        fw = row.get("buf force-winograd")
        nw = row.get("buf no-winograd")
        if not (b and fw and nw):
            continue
        fl = _floor(conv)
        d_f = 100 * (fw - b) / b
        d_n = 100 * (nw - b) / b
        shp = _parse(conv)
        stride = strides.get(conv)
        if stride is not None and stride != 1:
            gate = "n/a"
            verdict = "no Winograd path (stride %d) -- both arms inert; CONTROL" % stride
            if max(abs(d_f), abs(d_n)) >= fl:
                verdict = ("!! inert arms moved %+.0f%%/%+.0f%% -- batch drift, "
                           "distrust this run's deltas" % (d_f, d_n))
                drift.append(conv)
        elif shp is None:
            gate, verdict = "?", "unrecognised conv label"
        else:
            ci, co, _hi, wi = shp
            admits = _gate(ci, co, wi)
            gate = "admits" if admits else "refuses"
            if admits:
                verdict = ("MNN uses Winograd; turning it off costs %+.0f%%" % d_n if d_n >= fl
                           else "MNN uses Winograd, but off is within noise (%+.0f%%)" % d_n)
            elif d_f <= -fl:
                verdict = ("forcing WINS %+.0f%% -- gate is too strict for this shape" % d_f)
            elif d_f >= fl:
                verdict = "forcing loses %+.0f%% -- gate is right to refuse" % d_f
            else:
                verdict = "forcing changes nothing (%+.0f%%) -- either choice is fine" % d_f
        lines.append(f"{conv:<20}{b:>9.1f}{fw:>11.1f}{nw:>9.1f}{fl:>6.0f}%  {gate:<8}{verdict}")
    if drift:
        lines.append("")
        lines.append("!! %d conv(s) where Winograd cannot run still moved more than the noise "
                     "floor." % len(drift))
        lines.append("   Those arms are provably no-ops, so the movement is thermal/batch drift.")
        lines.append("   Re-run before trusting any delta in this run.")
    lines.append("")
    lines.append("stride-2 rows are the control: they MUST read ~0%. gate=admits means the")
    lines.append("'default' column already IS Winograd, so force-wino measures the same thing.")
    return "\n".join(lines)


def render(st, run_ids, floors_path=None, winograd=False, shape_family=None):
    if floors_path:
        BS.load_noise_floors(floors_path)
    convs = _rows_by_conv(st, run_ids)
    if not convs:
        return "(no results for this run)"
    out = []
    out.append(_table(
        "1. BUFFER MODE  (gpuMode 68)", convs,
        lambda rs: [r for r in rs if r["mode"] == BUFFER], "buffer default",
        "the best kernel per conv if the model runs in buffer mode"))
    out.append(_table(
        "2. IMAGE MODE  (gpuMode 132)", convs,
        lambda rs: [r for r in rs if r["mode"] == IMAGE], "image default",
        "the best kernel per conv if the model runs in image mode.\n"
        "NOTE: MNN_CONV_FORCE/SPEC/HARD are read only by the buffer backend, so image mode has no\n"
        "per-kernel choice -- the arms here differ by algorithm (Winograd) only."))
    out.append(_table(
        "3. OVERALL  (best arm, either mode)", convs,
        lambda rs: rs, "buffer default",
        "against the DEPLOYED baseline (buffer, no flags). gpuMode is per Interpreter, so a\n"
        "buffer and an image winner in this table cannot both be used unless the convs live in\n"
        "different submodels -- confirm at whole-model wall-clock before shipping."))
    if winograd:
        out.append(winograd_table(convs, shape_family))
    return "\n".join(out)


def main():
    import argparse
    import datetime
    from bench_store import ResultStore

    HERE = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="three recommendation tables from a results.db")
    ap.add_argument("--db", default=str(HERE / "results.db"))
    ap.add_argument("--run", default=None,
                    help="run id, or several comma-separated to merge (default: most recent "
                         "full_sweep). Merging is how a model re-measured on its own rejoins the "
                         "run whose batch was rejected.")
    ap.add_argument("--runs", action="store_true", help="list runs and exit")
    ap.add_argument("--floors", default=str(HERE / "noise_floors.json"))
    ap.add_argument("--winograd", action="store_true",
                    help="add a fourth table comparing the algorithm arms (buffer default vs "
                         "force-winograd vs no-winograd) for every conv. These arms are always "
                         "measured; the three tables above hide them because they print only the "
                         "winning arm per conv and Winograd is usually already the default.")
    a = ap.parse_args()

    if not Path(a.db).exists():
        raise SystemExit(f"no results database at {a.db} -- run full_sweep.py first")
    st = ResultStore(a.db)
    runs = list(st._conn.execute(
        "SELECT run_id, started, device, shape_family, harness, clock_valid, notes"
        " FROM runs ORDER BY started DESC"))
    if not runs:
        raise SystemExit(f"{a.db} has no runs")
    if a.runs:
        for r in runs:
            when = datetime.datetime.fromtimestamp(r["started"]).strftime("%Y-%m-%d %H:%M")
            print(f"{r['run_id']}  {when}  {r['device']}  {r['shape_family'] or '?':<8} "
                  f"{r['harness'] or '?':<14} {r['notes'] or ''}")
        return

    if a.run:
        ids = [x.strip() for x in a.run.split(",")]
        known = {r["run_id"] for r in runs}
        missing = [i for i in ids if i not in known]
        if missing:
            raise SystemExit(f"no run(s) {missing}; try --runs")
        run = next(r for r in runs if r["run_id"] == ids[0])
        run_ids = ids
    else:
        # default to the most recent sweep: a variance_probe run has no arms to recommend between
        run = next((r for r in runs if r["harness"] == "full_sweep"), runs[0])
        run_ids = [run["run_id"]]

    n = BS.load_noise_floors(a.floors)
    print(f"run {run['run_id']}  device {run['device']}  family {run['shape_family']}  "
          f"harness {run['harness']}")
    print(f"noise floors: {n} configurations loaded"
          + ("" if n else f" -- {a.floors} missing, falling back to a flat "
                          f"{BS.NOISE_FLOOR_PCT:.0f}% for every conv; run variance_probe.py"))
    if run["clock_valid"] == 0:
        print("clock: THROTTLED during this run -- comparisons hold, absolute times are inflated")
    if len(run_ids) > 1:
        print(f"merging {len(run_ids)} runs: {', '.join(run_ids)}")
    print(render(st, run_ids, a.floors, winograd=a.winograd,
                 shape_family=run['shape_family']))


if __name__ == "__main__":
    main()
