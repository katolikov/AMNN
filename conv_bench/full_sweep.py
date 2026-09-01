#!/usr/bin/env python3
"""The full conv sweep, rebuilt on the new harness.

The old suite was 21 hand-written sections, each with its own timing path, its own baseline
convention, and its own table. That is what let section 5 quietly measure against a
Winograd-disabled baseline while section 17 measured against the deployed one, and let the same
conv read 33us in one section and 99us in another.

Here every strategy -- a kernel, a memory mode, NCHW, im2col, implicit GEMM, LDS, split-K -- is
just an ARM: a (mode, env) pair. One code path measures them all, against one declared baseline,
with per-conv attribution from the kernel shape tags. The whole strategy space becomes a single
table per conv, and comparisons are the store's job, not prose.

    python3 conv_bench/full_sweep.py <serial> [--reps 3]

Resumable: state is written after every rep. An interruption costs the rep in flight, not the run.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "conv_probe_bundle"
sys.path.insert(0, str(BUNDLE)); sys.path.insert(0, str(HERE))
import run_report as R                                      # noqa: E402
from probe_perconv import per_conv, label_of, PROBE_MODELS  # noqa: E402
from clock_guard import GpuTelemetry, Watchdog              # noqa: E402
import bench_store as BS                                    # noqa: E402
from bench_store import ResultStore                         # noqa: E402
import make_bundle as M                                     # noqa: E402


def is_stride2(tag: str) -> bool:
    m = re.match(r"b\d+ci\d+hi(\d+)wi\d+co\d+ho(\d+)", tag)
    return bool(m) and int(m.group(2)) * 2 <= int(m.group(1)) + 1


def arms():
    """Every strategy in the investigation, as (name, mode, env).

    The env flags are the same ones the old sections used; what changes is that they are now
    measured on one code path against one baseline instead of in 21 separate tables."""
    out = [("buffer default", 68, ""), ("image default", 132, "")]
    for v in M.VARIANTS:
        spec = "MNN_CONV_SPEC=1 " if v in M.SPEC_ONLY else ""
        short = v.replace("conv_2d_", "")
        out.append((f"buf {short}", 68, f"{spec}MNN_CONV_FORCE={v} "))
        out.append((f"img {short}", 132, f"{spec}MNN_CONV_FORCE={v} "))
    # algorithm-level strategies (old sections 7-9, 14-16, 19)
    out += [
        ("buf force-winograd", 68, "MNN_FORCE_WINOGRAD=1 "),
        ("buf no-winograd",    68, "MNN_NO_WINOGRAD=1 "),
        ("buf LDS",            68, "MNN_CONV_LDS=1 "),
        ("buf LDS w2",         68, "MNN_CONV_LDS=w2 "),
        ("buf splitK=2",       68, "MNN_CONV_SPLITK=2 "),
        ("buf splitK=4",       68, "MNN_CONV_SPLITK=4 "),
        ("buf NCHW",           68, "MNN_CONV_NCHW=1 "),
        ("buf im2col+GEMM",    68, "MNN_CONV_IMGEMM=1 MNN_NO_WINOGRAD=1 "),
        ("buf implicit GEMM",  68, "MNN_CONV_IGEMM=1 MNN_NO_WINOGRAD=1 "),
        ("buf constant-w",     68, "MNN_CONV_CONSTW=1 "),
        ("buf HARD",           68, "MNN_CONV_HARD=1 "),
    ]
    return out


def kernel_of(name):
    return None if name.endswith("default") or " " not in name else name.split(" ", 1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("serial", nargs="?", default="R3CY905E04M")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--settle-c", type=float, default=42.0)
    ap.add_argument("--db", default=str(HERE / "results.db"))
    a = ap.parse_args()

    d = R.Dev(a.serial)
    tel = GpuTelemetry(a.serial)
    d.shell(f"mkdir -p {R.DEV}/tdir"); d.shell(f"mkdir -p {R.TUNE}")
    for f in sorted((BUNDLE / "bin").iterdir()):
        d.push(f, f"{R.DEV}/")
    d.shell(f"chmod +x {R.DEV}/ModuleBasic.out")
    for model, _ in PROBE_MODELS:
        d.push(BUNDLE / "models" / model, f"{R.DEV}/{model}")

    A = arms()
    s1only = set(M.STRIDE1_ONLY)
    pin = tel.pin()
    print(f"=== full sweep: {len(A)} arms x {len(PROBE_MODELS)} probe models x {a.reps} reps ===")
    print(f"clock: {'PINNED at ' + pin['governor'] if pin['pinned'] else 'governor ' + pin['governor'] + ' (not pinned; batches certified after the fact)'}")
    print(f"nominal {tel.nominal/1000:.0f} MHz, GPU {tel.temperature_c()} C\n", flush=True)

    state = HERE / "full_sweep_state.json"
    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    clocks: list[dict] = []
    done = 0
    if state.exists():
        blob = json.loads(state.read_text())
        if blob.get("arms") == [n for n, _, _ in A]:
            done = blob["done_reps"]; clocks = blob.get("clocks", [])
            for c, per in blob["samples"].items():
                for arm, v in per.items():
                    samples[c][arm] = v
            print(f"  resuming after {done} rep(s)\n", flush=True)

    def save(n):
        state.write_text(json.dumps(dict(arms=[x[0] for x in A], done_reps=n, clocks=clocks,
                                         samples={c: dict(v) for c, v in samples.items()}), indent=0))

    t0 = time.time(); launches = 0
    for rep in range(done, a.reps):
        s = tel.settle(target_c=a.settle_c, max_wait=90)
        order = A[rep % len(A):] + A[:rep % len(A)]     # rotate so drift favours nobody
        with Watchdog(tel) as w:
            for name, mode, env in order:
                for model, shape in PROBE_MODELS:
                    out, _ = R.run_model(d, model, shape, 120, env=env, mode=mode)
                    launches += 1
                    k = kernel_of(name)
                    for tag, us in per_conv(out).items():
                        if k and f"conv_2d_{k}" in s1only and is_stride2(tag):
                            continue
                        samples[label_of(tag)][name].append(us)
        clocks.append(w.result)
        save(rep + 1)
        print(f"  rep {rep+1}/{a.reps}: {launches} launches, {time.time()-t0:.0f}s, "
              f"settle {s.get('waited',0):.0f}s @{s.get('temp')}C, "
              f"clock {w.result.get('mean_mhz')} MHz "
              f"({'valid' if w.result.get('valid') else 'THROTTLED'}), "
              f"GPU {w.result.get('temp_end')}C", flush=True)

    # ------------------------------------------------------------------ record + report
    BS.load_noise_floors(HERE / "noise_floors.json")
    st = ResultStore(a.db)
    st.begin_run(device=a.serial, shape_family="reduced", harness="full_sweep",
                 clock_start=min((c.get("mean_mhz") or 0) * 1000 for c in clocks) if clocks else None,
                 notes=f"{len(A)} arms, {a.reps} reps")
    valid_run = all(c.get("valid") for c in clocks)
    with st.batch(section="full", label=f"full sweep ({len(A)} arms)", reps=a.reps) as b:
        b.baseline("buffer default")
        for conv in sorted(samples):
            for arm, vals in samples[conv].items():
                if vals:
                    b.record(conv, arm, 0, mode=dict((n, m) for n, m, _ in A)[arm],
                             env=dict((n, e) for n, _, e in A)[arm], samples=vals)
        bid = b.batch_id
    st.finish_run(clock_end=(min((c.get("mean_mhz") or 0) for c in clocks) * 1000) if clocks else None)

    print(f"\n{'conv':<18}{'deployed':>9}{'best arm':<20}{'best':>8}{'gain':>7}{'floor':>7}  verdict")
    print("-" * 78)
    wins = 0
    for conv in sorted(samples):
        rows = {r["arm"]: r["us"] for r in st.rows(bid, conv) if r["valid"]}
        if "buffer default" not in rows:
            continue
        base = rows["buffer default"]
        best = min(rows, key=lambda x: rows[x])
        gain = 100 * (rows[best] - base) / base
        fl = max(BS.noise_floor(conv, "buffer"), BS.noise_floor(conv, "image"))
        sig = abs(gain) >= fl
        wins += sig
        print(f"{conv:<18}{base:>9.1f}{best:<20}{rows[best]:>8.1f}{gain:>6.0f}%{fl:>6.0f}%  "
              f"{'USE IT' if sig else 'keep default'}")
    print(f"\n{wins}/{len(samples)} convs have a win clearing their measured noise floor")
    print(f"run clock validity: {'VALID' if valid_run else 'SOME REPS THROTTLED'}  "
          f"({', '.join(str(c.get('mean_mhz')) for c in clocks)} MHz per rep)")
    print(f"{launches} launches in {time.time()-t0:.0f}s -> batch {bid}")


if __name__ == "__main__":
    main()
