#!/usr/bin/env python3
"""Measure ACROSS-BATCH variance per configuration, so the noise floor stops being a guess.

The suite has always used one global 6% noise floor, derived from control arms on the stride-1
cores. It does not hold everywhere: 34->32@48x64 in image mode read 71.3us in one batch and 46.2us
in two later ones -- a 54% swing on a run whose clock held 980MHz start to finish and whose
correctness gate passed. No existing check sees that, because every existing check looks INSIDE a
batch (reps are interleaved, so within-batch spread is small even when the batch as a whole is off).

So this measures the other axis: run the same batch end-to-end R times and look at how much the
batch MEDIAN moves. That is the quantity that decides whether a reported difference is real, and it
is what was silently assumed to be 6%.

Cheap only because of the content-addressed cache (0.6s/launch instead of 3.5s) -- repeating every
batch five times was not affordable before.

    python3 conv_bench/variance_probe.py <serial> [--repeats 5] [--reps 3]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "conv_probe_bundle"
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(HERE))
import run_report as R                       # noqa: E402
from probe_perconv import per_conv, label_of, PROBE_MODELS  # noqa: E402
import block_fixture
from bench_store import ResultStore          # noqa: E402

def configs():
    """The five probe models, which between them hold all 13 convs of the reduced shape set.

    This used to read the stride-1 cores from the manifest plus a set of single-conv stride-2
    models listed in a scratch directory. That path existed only on the machine the models were
    generated on, and it was guarded by .exists(), so anywhere else this measured five cores and
    silently reported nothing about the eight stride-2 convs. Using the probe models instead
    covers all 13, needs nothing outside the bundle, and matches how the rest of the harness
    attributes per-conv times."""
    return list(PROBE_MODELS)



def default_serial():
    """The attached device, when there is exactly one.

    These scripts used to default to the serial of the machine they were written on, so on any
    other setup the first run targeted a device that does not exist. Auto-detect instead, and say
    plainly what to pass when the answer is ambiguous."""
    import subprocess
    out = subprocess.run("adb devices", shell=True, text=True, capture_output=True).stdout
    devs = [l.split()[0] for l in out.splitlines()[1:] if "\tdevice" in l]
    if len(devs) == 1:
        return devs[0]
    if not devs:
        raise SystemExit("no device attached (check `adb devices`)")
    raise SystemExit(f"several devices attached; pass one as the first argument: {', '.join(devs)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("serial", nargs="?", default=None,
                    help="adb serial; auto-detected when one device is attached")
    ap.add_argument("--repeats", type=int, default=5, help="how many times to re-run each batch")
    ap.add_argument("--reps", type=int, default=3, help="interleaved reps inside one batch")
    ap.add_argument("--cool", type=int, default=10, help="seconds between batches")
    ap.add_argument("--db", default=str(HERE / "results.db"))
    a = ap.parse_args()
    a.serial = a.serial or default_serial()

    d = R.Dev(a.serial)
    R.push_binaries(d, BUNDLE / "bin", BUNDLE / "manifest.json")

    st = ResultStore(a.db)
    s0l, _ = R.sample_clock(d, configs()[0][0], configs()[0][1])
    s0 = statistics.median(s0l) if s0l else 0
    st.begin_run(device=a.serial, shape_family=block_fixture.SHAPE_FAMILY, harness="variance_probe",
                 clock_start=s0, notes=f"across-batch variance, {a.repeats} repeats")

    cfgs = configs()
    print(f"=== across-batch variance: {len(cfgs)} probe models (13 convs) x 2 modes "
          f"x {a.repeats} batches ({a.reps} reps each) ===")
    print(f"(clock at start: {s0} MHz)")
    print(f"(first pass compiles shaders for each new configuration -- a cold model can take "
          f"minutes before its first line appears)\n", flush=True)
    print(f"{'conv':<22}{'mode':<8}{'median':>9}{'min':>8}{'max':>8}{'spread':>9}   verdict")

    t0 = time.time()
    per_batch: dict[tuple, list] = {}
    total_batches = a.repeats * len(cfgs)
    done_batches = 0
    # Progress, because this used to print the header and then nothing until every model was
    # finished. On a cold cache that is half an hour of silence, which is indistinguishable from
    # the harness being broken -- and the first thing anyone checks is whether output is going to
    # logcat instead of stdout.
    for rep in range(a.repeats):
        for model, shape in cfgs:
            R.cooldown(d, a.cool)
            d.push(BUNDLE / "models" / model, f"{R.DEV}/{model}")
            acc: dict = {}
            for mode, arm in ((68, "buffer"), (132, "image")):
                for _ in range(a.reps):
                    out, _ = R.run_model(d, model, shape, 120, mode=mode)
                    for tag, us in per_conv(out).items():
                        acc.setdefault((label_of(tag), arm), []).append(us)
            with st.batch(section="variance", label=f"{model} rep{rep}", reps=a.reps) as b:
                b.baseline("buffer")
                for (conv, arm), samples in acc.items():
                    row = b.record(conv, arm, 0, env="",
                                   mode=68 if arm == "buffer" else 132, samples=samples)
                    per_batch.setdefault((conv, arm), []).append(row["us"])
            done_batches += 1
            print(f"   [{done_batches}/{total_batches}] {model} rep{rep+1}/{a.repeats}  "
                  f"{len(acc)//2} convs, {time.time()-t0:.0f}s elapsed", flush=True)

    s1l, _ = R.sample_clock(d, cfgs[0][0], cfgs[0][1])
    s1 = statistics.median(s1l) if s1l else 0
    valid = st.finish_run(clock_end=s1)

    noise = {}
    for (conv, arm), vals in per_batch.items():
        vals = [v for v in vals if v]
        if len(vals) < 2:
            continue
        med = statistics.median(vals)
        spread = 100.0 * (max(vals) - min(vals)) / med if med else 0.0
        noise[f"{conv}|{arm}"] = dict(median=med, lo=min(vals), hi=max(vals),
                                      spread_pct=spread, n=len(vals))
        verdict = ("stable" if spread <= 6 else
                   "NOISY -- a 6% claim here is meaningless" if spread <= 20 else
                   "*** UNSTABLE -- do not trust single-batch results ***")
        print(f"{conv:<22}{arm:<8}{med:>9.1f}{min(vals):>8.1f}{max(vals):>8.1f}"
              f"{spread:>8.0f}%   {verdict}")

    out = HERE / "noise_floors.json"
    out.write_text(json.dumps(dict(device=a.serial, repeats=a.repeats, reps=a.reps,
                                   clock_start=s0, clock_end=s1, clock_valid=valid,
                                   floors=noise), indent=2))
    print(f"\n(clock {s0} -> {s1} MHz, {'VALID' if valid else 'THROTTLED'};"
          f" {time.time()-t0:.0f}s total)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
