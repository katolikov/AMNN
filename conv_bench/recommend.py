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


def render(st, run_ids, floors_path=None):
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
    print(render(st, run_ids, a.floors))


if __name__ == "__main__":
    main()
