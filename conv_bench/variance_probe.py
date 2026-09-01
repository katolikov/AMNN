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
from bench_store import ResultStore          # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-sam-Documents-projects-MNN--claude-worktrees-"
               "convolution-benchmark-reduced-shapes-00e84c/4d41964f-a31f-4e23-b852-189d900ce502/"
               "scratchpad")


def configs():
    """The 13 convs of the reduced shape set: 5 stride-1 cores + 8 stride-2 singles."""
    man = json.loads((BUNDLE / "manifest.json").read_text())
    out = []
    for c in man["cores"]:
        out.append(dict(conv=c["label"], model=c["model"], shape=c["shape"], depth=c["depth"]))
    s2 = SCRATCH / "s2_manifest.json"
    if s2.exists():
        for c in json.loads(s2.read_text()):
            out.append(dict(conv=c["label"], model=c["model"], shape=c["shape"], depth=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("serial", nargs="?", default="R3CY905E04M")
    ap.add_argument("--repeats", type=int, default=5, help="how many times to re-run each batch")
    ap.add_argument("--reps", type=int, default=3, help="interleaved reps inside one batch")
    ap.add_argument("--cool", type=int, default=10, help="seconds between batches")
    ap.add_argument("--db", default=str(HERE / "results.db"))
    a = ap.parse_args()

    d = R.Dev(a.serial)
    d.shell(f"mkdir -p {R.DEV}/tdir"); d.shell(f"mkdir -p {R.TUNE}")
    for f in sorted((BUNDLE / "bin").iterdir()):
        d.push(f, f"{R.DEV}/")
    d.shell(f"chmod +x {R.DEV}/ModuleBasic.out")

    st = ResultStore(a.db)
    s0l, _ = R.sample_clock(d, configs()[0]["model"], configs()[0]["shape"])
    s0 = statistics.median(s0l) if s0l else 0
    st.begin_run(device=a.serial, shape_family="reduced", harness="variance_probe",
                 clock_start=s0, notes=f"across-batch variance, {a.repeats} repeats")

    cfgs = configs()
    print(f"=== across-batch variance: {len(cfgs)} convs x 2 modes x {a.repeats} batches "
          f"({a.reps} reps each) ===")
    print(f"(clock at start: {s0} MHz)\n")
    print(f"{'conv':<22}{'mode':<8}{'median':>9}{'min':>8}{'max':>8}{'spread':>9}   verdict")

    t0 = time.time()
    per_batch: dict[tuple, list] = {}
    for rep in range(a.repeats):
        for c in cfgs:
            R.cooldown(d, a.cool)
            d.push(BUNDLE / "models" / c["model"], f"{R.DEV}/{c['model']}")
            with st.batch(section="variance", label=f"{c['conv']} rep{rep}", reps=a.reps) as b:
                b.baseline("buffer")
                for mode, arm in ((68, "buffer"), (132, "image")):
                    samples = [R.conv_all_us(
                        R.run_model(d, c["model"], c["shape"], 120, mode=mode)[0], c["depth"])
                        for _ in range(a.reps)]
                    row = b.record(c["conv"], arm, 0, env="", mode=mode, samples=samples)
                    per_batch.setdefault((c["conv"], arm), []).append(row["us"])

    s1l, _ = R.sample_clock(d, cfgs[0]["model"], cfgs[0]["shape"])
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
