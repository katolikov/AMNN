#!/usr/bin/env python3
"""Read back everything a run measured. full_sweep prints one winner per conv; the store holds
every arm.

    python3 conv_bench/show_results.py                    # latest run, every conv, every arm
    python3 conv_bench/show_results.py --conv 96->96      # just the convs whose name matches
    python3 conv_bench/show_results.py --csv > runs.csv   # everything, for a spreadsheet
    python3 conv_bench/show_results.py --runs             # list runs, then --run <id>

Percentages are against that batch's declared baseline, and each is checked against the
configuration's own measured noise floor (noise_floors.json) -- a difference smaller than the
floor is marked (noise) rather than reported as a result.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_store as BS                      # noqa: E402
from bench_store import ResultStore           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(HERE / "results.db"))
    ap.add_argument("--run", default=None, help="run id (default: the most recent)")
    ap.add_argument("--runs", action="store_true", help="list runs and exit")
    ap.add_argument("--conv", default=None, help="only convs whose label contains this")
    ap.add_argument("--csv", action="store_true", help="machine-readable, all rows")
    ap.add_argument("--floors", default=str(HERE / "noise_floors.json"))
    a = ap.parse_args()

    st = ResultStore(a.db)
    BS.load_noise_floors(a.floors)

    runs = list(st._conn.execute(
        "SELECT run_id, started, device, shape_family, harness, clock_start, clock_end,"
        " clock_valid, notes FROM runs ORDER BY started DESC"))
    if not runs:
        raise SystemExit(f"no runs in {a.db}")
    if a.runs:
        import datetime
        for r in runs:
            when = datetime.datetime.fromtimestamp(r["started"]).strftime("%Y-%m-%d %H:%M")
            print(f"{r['run_id']}  {when}  {r['device']}  {r['shape_family'] or '?':<8} "
                  f"{r['harness'] or '?':<14} {r['notes'] or ''}")
        return

    run = next((r for r in runs if r["run_id"] == a.run), runs[0])
    batches = list(st._conn.execute(
        "SELECT batch_id, label, reps, baseline_arm FROM batches WHERE run_id=? ORDER BY created",
        (run["run_id"],)))

    if a.csv:
        print("conv,arm,us,spread,n,mode,env,baseline_arm,baseline_env,deployed_baseline,valid,batch")
        for b in batches:
            for r in st.rows(b["batch_id"]):
                print(f'"{r["conv"]}","{r["arm"]}",{r["us"]:.2f},'
                      f'{r["spread"] if r["spread"] is not None else ""},{r["n"]},{r["mode"]},'
                      f'"{(r["env"] or "").strip()}","{r["baseline_arm"]}",'
                      f'"{(r["baseline_env"] or "").strip()}",{r["deployed"]},{r["valid"]},'
                      f'{r["batch_id"]}')
        return

    print(f"run {run['run_id']}  device {run['device']}  family {run['shape_family']}  "
          f"harness {run['harness']}")
    print(f"clock {run['clock_start']} -> {run['clock_end']} "
          f"({'VALID' if run['clock_valid'] else 'THROTTLED -- absolute times inflated'})\n")

    for b in batches:
        rows = {}
        for r in st.rows(b["batch_id"]):
            rows.setdefault(r["conv"], []).append(r)
        for conv in sorted(rows):
            if a.conv and a.conv not in conv:
                continue
            rs = sorted(rows[conv], key=lambda r: r["us"] if r["valid"] else 9e9)
            base = next((r for r in rs if r["arm"] == b["baseline_arm"]), None)
            fl = max(BS.noise_floor(conv, "buffer"), BS.noise_floor(conv, "image"))
            print(f"{conv}   (baseline `{b['baseline_arm']}`, floor {fl:.0f}%, {b['reps']} reps)")
            for r in rs:
                if not r["valid"]:
                    print(f"    {r['arm']:<24}      INVALID  {r['invalid_why']}")
                    continue
                if base and base["us"] and r["arm"] != b["baseline_arm"]:
                    p = 100 * (r["us"] - base["us"]) / base["us"]
                    tag = f"{p:+6.0f}%" + ("" if abs(p) >= fl else "  (noise)")
                else:
                    tag = "baseline" if r["arm"] == b["baseline_arm"] else ""
                sp = f" +/-{r['spread']:.1f}" if r["spread"] else ""
                print(f"    {r['arm']:<24}{r['us']:>8.1f} us{sp:<8} {tag}")
            print()


if __name__ == "__main__":
    main()
