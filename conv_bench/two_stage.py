#!/usr/bin/env python3
"""Two-stage arm selection, and the experiment that decides whether it is safe.

Today every kernel variant is measured at full repeats on every shape. Across this whole
investigation about four have ever won, so most of that time buys nothing. The proposal: screen
all arms at ONE repeat, keep the top K, then re-measure only those at full repeats.

That is only sound if the cheap screen never discards the arm the expensive stage would have
ranked first. This script measures that rather than assuming it: every arm is run at `--reps`
repeats, the median is the ground truth, and each individual repeat is then replayed as an
independent one-repeat screen. The reported number is how often the screen's top-K contained the
ground-truth winner.

    python3 conv_bench/two_stage.py <serial> --reps 5 --topk 3

Two details that would otherwise corrupt the ranking:
  * A stride-1-only kernel forced on a stride-2 conv is silently ignored, and the arm then measures
    the DEFAULT. Left in, it looks like a kernel that ties the default; ranked, it can displace a
    real contender. Those pairs are excluded per conv.
  * Forcing a kernel applies to every conv in a probe model, so one launch yields one arm's time
    for every conv in that model -- which is what makes screening 32 arms affordable at all.
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
import run_report as R                                    # noqa: E402
from probe_perconv import per_conv, label_of, PROBE_MODELS  # noqa: E402
import make_bundle as M                                   # noqa: E402


def is_stride2(tag: str) -> bool:
    """hi/ho tell us the stride without needing the model definition."""
    m = re.match(r"b\d+ci\d+hi(\d+)wi\d+co\d+ho(\d+)", tag)
    return bool(m) and int(m.group(2)) * 2 <= int(m.group(1)) + 1


# Every kernel that has won ANY shape in this investigation, plus the two defaults. The other
# nine variants of M.VARIANTS have never placed first on any conv, in either memory mode, at either
# shape scale -- and each one costs a full cold compile per probe model (12-50s) to re-confirm that.
# M.VARIANTS remains the exhaustive list for a from-scratch sweep on new hardware; this is the
# working set for deciding the current model.
CANDIDATES = ["conv_2d_c4h1w1", "conv_2d_c4h1w2", "conv_2d_c4h4w1",
              "conv_2d_c8h1w1", "conv_2d_c8h2w1", "conv_2d_c8h4w1"]


def arms():
    """(name, mode, env) for every candidate: both memory modes, plain and per-kernel."""
    out = [("buffer default", 68, ""), ("image default", 132, "")]
    for v in CANDIDATES:
        spec = "MNN_CONV_SPEC=1 " if v in M.SPEC_ONLY else ""
        short = v.replace("conv_2d_", "")
        out.append((f"buf {short}", 68, f"{spec}MNN_CONV_FORCE={v} "))
        out.append((f"img {short}", 132, f"{spec}MNN_CONV_FORCE={v} "))
    return out


def kernel_of(arm_name: str) -> str | None:
    if arm_name.endswith("default"):
        return None
    return "conv_2d_" + arm_name.split(" ", 1)[1]



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
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--cool", type=int, default=0, help="seconds between arms")
    a = ap.parse_args()
    a.serial = a.serial or default_serial()

    d = R.Dev(a.serial)
    d.shell(f"mkdir -p {R.DEV}/tdir"); d.shell(f"mkdir -p {R.TUNE}")
    for f in sorted((BUNDLE / "bin").iterdir()):
        d.push(f, f"{R.DEV}/")
    d.shell(f"chmod +x {R.DEV}/ModuleBasic.out")
    for model, _ in PROBE_MODELS:
        d.push(BUNDLE / "models" / model, f"{R.DEV}/{model}")

    A = arms()
    stride1_only = set(M.STRIDE1_ONLY)
    print(f"=== two-stage validation: {len(A)} arms x {len(PROBE_MODELS)} models "
          f"x {a.reps} reps ===\n", flush=True)

    # samples[conv][arm] = [us per rep]. Persisted after EVERY rep: the first rep pays cold-cache
    # cost for every (arm, model) pair -- 12-50s each depending on how many distinct conv shapes
    # the model holds -- so an interrupted run that kept its results only in memory threw away an
    # hour of work. Now an interruption costs at most the rep in flight.
    state = HERE / "two_stage_state.json"
    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    done_reps = 0
    if state.exists():
        blob = json.loads(state.read_text())
        if blob.get("arms") == [n for n, _, _ in A]:
            done_reps = blob["done_reps"]
            for conv, per in blob["samples"].items():
                for arm, vals in per.items():
                    samples[conv][arm] = vals
            print(f"  resuming: {done_reps} rep(s) already recorded in {state.name}\n", flush=True)
        else:
            print("  (existing state has a different arm set; starting fresh)\n", flush=True)

    def save(done):
        state.write_text(json.dumps(
            dict(arms=[n for n, _, _ in A], done_reps=done,
                 samples={c: dict(v) for c, v in samples.items()}), indent=0))

    t0 = time.time()
    launches = 0
    for rep in range(done_reps, a.reps):
        # rotate arm order every rep so drift does not favour whoever goes first
        order = A[rep % len(A):] + A[:rep % len(A)]
        for name, mode, env in order:
            if a.cool:
                R.cooldown(d, a.cool)
            for model, shape in PROBE_MODELS:
                out, _ = R.run_model(d, model, shape, 120, env=env, mode=mode)
                launches += 1
                for tag, us in per_conv(out).items():
                    k = kernel_of(name)
                    if k and k in stride1_only and is_stride2(tag):
                        continue          # silently ignored by MNN; would measure the default
                    samples[label_of(tag)][name].append(us)
        save(rep + 1)
        print(f"  rep {rep+1}/{a.reps} done ({launches} launches, {time.time()-t0:.0f}s) "
              f"-> saved", flush=True)

    # ---------------------------------------------------------------- analysis
    print(f"\n{'conv':<18}{'ground-truth winner':<22}{'us':>7}   "
          f"screen top-{a.topk} contained it")
    hits = total = 0
    report = {}
    for conv in sorted(samples):
        nrep = max((len(v) for v in samples[conv].values()), default=0)
        per_arm = {n: v for n, v in samples[conv].items() if len(v) == nrep and nrep}
        if not per_arm:
            continue
        truth = {n: statistics.median(v) for n, v in per_arm.items()}
        winner = min(truth, key=lambda n: truth[n])
        ok = 0
        for r in range(nrep):
            screen = {n: v[r] for n, v in per_arm.items()}
            topk = sorted(screen, key=lambda n: screen[n])[:a.topk]
            if winner in topk:
                ok += 1
        hits += ok; total += nrep
        report[conv] = dict(winner=winner, us=truth[winner], hit_rate=ok / nrep,
                            ranking=sorted(truth, key=lambda n: truth[n])[:5])
        flag = "" if ok == nrep else "   <-- screen would have dropped the winner"
        print(f"{conv:<18}{winner:<22}{truth[winner]:>7.1f}   {ok}/{nrep}{flag}")

    print(f"\noverall: the {a.topk}-arm screen retained the true winner in {hits}/{total} trials "
          f"({100*hits/total:.0f}%)")
    print(f"{launches} launches in {time.time()-t0:.0f}s")
    full = len(A) * len(PROBE_MODELS) * a.reps
    staged = len(A) * len(PROBE_MODELS) + (a.topk + 1) * len(PROBE_MODELS) * a.reps
    print(f"cost model: full {full} launches vs staged {staged} -> {full/staged:.1f}x fewer")
    (HERE / "two_stage_validation.json").write_text(json.dumps(
        dict(reps=a.reps, topk=a.topk, hit_rate=hits / total, convs=report), indent=2))


if __name__ == "__main__":
    main()
