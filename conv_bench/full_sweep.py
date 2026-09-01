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

ONE BATCH PER PROBE MODEL. The first version of this measured all 43 arms across all 5 models as a
single "interleaved" batch: 215 launches taking ~2h, over which the GPU fell 980 -> 899 MHz. Arm 1
was timed on a cool device and arm 43 on a hot one, so its ranking mixed thermal state with
strategy -- and its verdicts contradicted three shorter runs that agreed with each other. Rotating
arm order across reps cannot fix that; only a shorter batch can. Each model is now its own batch
(~43 arms x reps launches, a few minutes warm), and ResultStore refuses a batch that runs long.

Every conv lives in exactly one probe model, so no conv is split across batches.

Resumable: state is written after every model. An interruption costs the model in flight.
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
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip cache warm-up (only safe when every arm has been run before)")
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
    samples_all: dict[str, dict[str, list[float]]] = {}
    clocks: list[dict] = []
    batch_ids: list = []
    done = 0
    if state.exists():
        blob = json.loads(state.read_text())
        if blob.get("arms") == [n for n, _, _ in A]:
            done = blob.get("done_models", 0); clocks = blob.get("clocks", [])
            samples_all = blob.get("samples", {})
            print(f"  resuming after {done} model(s)\n", flush=True)

    def save(n):
        state.write_text(json.dumps(dict(arms=[x[0] for x in A], done_models=n, clocks=clocks,
                                         samples=samples_all), indent=0))

    BS.load_noise_floors(HERE / "noise_floors.json")
    st = ResultStore(a.db)
    st.begin_run(device=a.serial, shape_family="reduced", harness="full_sweep",
                 notes=f"{len(A)} arms, {a.reps} reps, one batch per probe model")

    # ---- warm the tuning cache BEFORE measuring anything ----------------------------------
    # Compilation is not measurement. A cold (arm, model) pair costs 12-50s of shader build, so a
    # 43-arm batch on a cold cache takes ~13 min, heats the device, drops the clock below tolerance
    # and is then correctly rejected -- meaning the first run on a device could never complete.
    # Touching every pair once first moves that cost outside the measurement window: the batches
    # afterwards run warm (~22s each) and hold their clock. Costs nothing on an already-warm cache.
    if not a.no_warmup:
        need = [(m, sh, n, mo, e) for m, sh in PROBE_MODELS for n, mo, e in A]
        print(f"  warming {len(need)} (arm, model) pairs -- one-time shader compilation, "
              f"not measured", flush=True)
        wt0 = time.time()
        for i, (m, sh, n, mo, e) in enumerate(need, 1):
            R.run_model(d, m, sh, 2, env=e, mode=mo)      # 2 loops: build the cache, time nothing
            if i % 43 == 0:
                print(f"    {i}/{len(need)} ({time.time()-wt0:.0f}s)", flush=True)
        print(f"  warm-up done in {(time.time()-wt0)/60:.0f} min\n", flush=True)

    t0 = time.time(); launches = 0
    for mi, (model, shape) in enumerate(PROBE_MODELS):
        if mi < done:
            continue
        s_ = tel.settle(target_c=a.settle_c, max_wait=90)
        acc: dict = {}
        tm0 = time.time()          # measurement window start, for the batch duration guard
        with Watchdog(tel) as w:
            for rep in range(a.reps):
                order = A[rep % len(A):] + A[:rep % len(A)]   # rotate: drift favours nobody
                for name, mode, env in order:
                    out, _ = R.run_model(d, model, shape, 120, env=env, mode=mode)
                    launches += 1
                    k = kernel_of(name)
                    for tag, us in per_conv(out).items():
                        if k and f"conv_2d_{k}" in s1only and is_stride2(tag):
                            continue
                        acc.setdefault((label_of(tag), name), []).append(us)
        clocks.append(dict(model=model, **w.result))
        # One batch for this model. If it ran long the store raises rather than recording a
        # comparison whose arms saw different clocks.
        # The Watchdog measured what the clock actually did across this batch; hand that to the
        # store so a long-but-stable batch is accepted and a short-but-throttled one is not.
        with st.batch(section="full", label=f"{model} ({len(A)} arms)", reps=a.reps,
                      started=tm0, clock_valid=w.result.get("valid")) as b:
            b.baseline("buffer default")
            for (conv, arm), samples in acc.items():
                b.record(conv, arm, 0, samples=samples,
                         mode=dict((n, m) for n, m, _ in A)[arm],
                         env=dict((n, e) for n, _, e in A)[arm])
            batch_ids.append((model, b.batch_id))
        for (conv, arm), samples in acc.items():
            samples_all.setdefault(conv, {})[arm] = samples
        save(mi + 1)
        print(f"  {model:<14} {launches} launches, {time.time()-t0:.0f}s, "
              f"settle {s_.get('waited',0):.0f}s @{s_.get('temp')}C, "
              f"clock {w.result.get('mean_mhz')} MHz "
              f"({'valid' if w.result.get('valid') else 'THROTTLED'})", flush=True)

    # ------------------------------------------------------------------ report
    st.finish_run()
    valid_run = all(c.get("valid") for c in clocks)
    print(f"\n{'conv':<18}{'deployed':>9}  {'best arm':<22}{'best':>8}{'gain':>7}{'floor':>7}  verdict")
    print("-" * 82)
    wins = 0; total = 0
    for model, bid in batch_ids:
        rows: dict = {}
        for r in st.rows(bid):
            if r["valid"]:
                rows.setdefault(r["conv"], {})[r["arm"]] = r["us"]
        for conv in sorted(rows):
            r = rows[conv]
            if "buffer default" not in r:
                continue
            total += 1
            base = r["buffer default"]; best = min(r, key=lambda x: r[x])
            gain = 100 * (r[best] - base) / base
            fl = max(BS.noise_floor(conv, "buffer"), BS.noise_floor(conv, "image"))
            sig = abs(gain) >= fl; wins += sig
            print(f"{conv:<18}{base:>9.1f}  {best:<22}{r[best]:>8.1f}{gain:>6.0f}%{fl:>6.0f}%  "
                  f"{'USE IT' if sig else 'keep default'}")
    print(f"\n{wins}/{total} convs have a win clearing their measured noise floor")
    print("clock per batch: " + ", ".join(
        f"{c['model'].replace('.mnn','')} {c.get('mean_mhz')}MHz"
        f"{'' if c.get('valid') else ' THROTTLED'}" for c in clocks))
    if not valid_run:
        print("  ^ a throttled batch keeps its internal comparisons (arms interleaved inside it) "
              "but its absolute microseconds are inflated.")
    print(f"{launches} launches in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
