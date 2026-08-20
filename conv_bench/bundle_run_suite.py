#!/usr/bin/env python3
"""MAIN SCRIPT — set the clocks on one Android device, run every conv strategy, write a report.

    python3 run_suite.py <ADB-SERIAL>                     # pin everything to max, full suite
    python3 run_suite.py <ADB-SERIAL> --gpu min           # slow GPU, fast memory (the interesting case)
    python3 run_suite.py <ADB-SERIAL> --gpu-sweep 980,600,300
    python3 run_suite.py <ADB-SERIAL> --quick             # ~4 min per clock point instead of ~10
    python3 run_suite.py --list                           # show attached devices

Clock specs are `max`, `min`, a number in MHz (snapped to the nearest supported step), or `none`
to leave a domain alone. Defaults: --gpu max --mif max --int max.

Pinning needs root. Without it the script still runs everything, records the clock the governor
actually used, and says so in the report — the numbers are then only comparable within one run.

Output: ONE combined report (plain-English verdict first), plus the detailed per-clock reports
and a .json of every number next to it.
"""
import argparse, datetime, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import run_report                      # inside the bundle
except ImportError:
    import bundle_run_report as run_report  # inside the MNN repo
try:
    from clocks import Clocks
except ImportError:
    from bundle_clocks import Clocks

OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def devices():
    o = subprocess.run("adb devices", shell=True, text=True, capture_output=True).stdout
    return [l.split()[0] for l in o.splitlines()[1:] if "\tdevice" in l]


def pct(v, base):
    if not v or not base: return "n/a"
    return f"{100*(v-base)/base:+.0f}%"


def plain_name(k):
    """kernel id -> what it actually does, for readers who don't know the naming scheme"""
    if k == "MNN default": return "MNN's own automatic choice"
    if k == "LDS": return "shared-memory (LDS) tiled kernel"
    n = k.replace("conv_2d_", "")
    m = re.match(r"c(\d+)h(\d+)w(\d+)(_\w+)?$", n)
    if not m: return n
    c, h, w, sfx = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    extra = " with split accumulators" if sfx == "_pa" else ""
    return (f"`{n}` — each GPU thread computes {c} output channels x {h} row(s) x {w} column(s)"
            f"{extra}")


def preflight(serial, skip=False):
    """Run the measurement-integrity audit BEFORE any timing, and gate the suite on it.

    run_report.py masks cells that preflight proved are not measuring what their column header
    claims. Without a preflight_result.json it cannot mask anything, so a confounded or
    never-engaged arm would be published as a plain number. That file is a per-device, per-build
    artifact -- it is deliberately NOT in git, and is regenerated here.

    A GLOBAL failure aborts (the run would be meaningless). CELL-scoped failures are fine: those
    cells render as `invalid` and everything else stands.
    """
    pf_py = HERE / "preflight.py"
    if skip or not pf_py.exists():
        if not pf_py.exists():
            say("> ⚠️ preflight.py not found — cells cannot be validated and NOTHING will be "
                "masked. Any confounded or non-engaging arm will print as a normal number.")
        return True
    say(f"running preflight (integrity audit, ~1 min) ...")
    r = subprocess.run([sys.executable, str(pf_py), serial], text=True)
    if r.returncode == 2:
        say("")
        say("**PREFLIGHT FAILED (global).** The suite would not measure what it claims, so it has "
            "not been started. Fix the failures above, or re-run with --skip-preflight to "
            "override (the report will then be unvalidated).")
        return False
    return True


def run_one(serial, label, quick, cooldown, outdir):
    """Run the full probe once at the current clock setting."""
    p = outdir / f"detail_{label}.md"
    run_report.reset()
    argv = ["--serial", serial, "-o", str(p)]
    if quick: argv.append("--quick")
    if cooldown is not None: argv += ["--cooldown", str(cooldown)]
    return run_report.main(argv)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("serial", nargs="?", help="adb serial of the device (see --list)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="do NOT run the integrity audit first. The report is then unvalidated: "
                         "no cell can be marked invalid, so a confounded or non-engaging arm "
                         "prints as a normal number. Not recommended.")
    ap.add_argument("--list", action="store_true", help="list attached devices and exit")
    ap.add_argument("--gpu", default="max", help="GPU clock: max | min | MHz | none")
    ap.add_argument("--mif", default="max", help="DRAM-bus (MIF) clock: max | min | MHz | none")
    ap.add_argument("--int", dest="cpu_int", default="max",
                    help="interconnect (INT) clock: max | min | MHz | none")
    ap.add_argument("--gpu-sweep", default=None,
                    help="comma-separated GPU clocks to repeat the whole suite at, e.g. 980,600,300")
    ap.add_argument("--quick", action="store_true", help="fewer repeats (~4 min per clock point)")
    ap.add_argument("--cooldown", type=int, default=None, help="GPU idle seconds between sections")
    ap.add_argument("-o", "--out", default=None, help="combined report path")
    a = ap.parse_args()

    devs = devices()
    if a.list:
        print("attached devices:")
        for x in devs: print("  " + x)
        if not devs: print("  (none)")
        return
    serial = a.serial or (devs[0] if len(devs) == 1 else None)
    if not serial:
        print(f"Give the device adb serial as the first argument. Attached: {devs or '(none)'}")
        sys.exit(1)
    if serial not in devs:
        print(f"Device '{serial}' is not attached. Attached: {devs or '(none)'}"); sys.exit(1)

    if not preflight(serial, skip=a.skip_preflight):
        sys.exit(2)

    outdir = Path(a.out).parent if a.out else HERE
    out_path = Path(a.out) if a.out else HERE / f"suite_{serial}.md"
    outdir.mkdir(parents=True, exist_ok=True)

    clk = Clocks(serial)
    print("=== clock domains ===");  print(clk.describe());  print()
    points = [p.strip() for p in a.gpu_sweep.split(",")] if a.gpu_sweep else [a.gpu]

    results, pinned_info = {}, {}
    try:
        for gp in points:
            label = f"gpu{gp}"
            print(f"\n=== clock point: GPU={gp}  MIF={a.mif}  INT={a.cpu_int} ===")
            pinned_info[label] = clk.pin_all(gpu=gp, mif=a.mif, cpu_int=a.cpu_int)
            for k, v in pinned_info[label].items():
                print(f"   {k}: requested {v['requested']} -> "
                      f"{v['achieved_mhz']} MHz ({v['note']})")
            _, data = run_one(serial, label, a.quick, a.cooldown, outdir)
            results[label] = data
    finally:
        clk.restore()
        print("\n(clocks restored)")

    # ------------------------------------------------------------ combined report
    any_pin = any(v.get("pinned") for pt in pinned_info.values() for v in pt.values())
    first = results[next(iter(results))]
    hw = first.get("hw", {})

    say("# Conv strategy suite — results")
    say(f"\n_device `{serial}` — {hw.get('name','?')}, "
        f"{hw.get('max_compute_units','?')} compute units — "
        f"{datetime.datetime.now():%Y-%m-%d %H:%M}_\n")

    # ---- plain-English verdict
    say("## What this says, in one paragraph\n")
    lines = []
    for label, data in results.items():
        for key, w in (data.get("winner") or {}).items():
            base, best, bus = w["default_us"], w["best"], w["best_us"]
            if best == "MNN default" or bus >= base * 0.97:
                lines.append(f"- **{w['label']}** (at {label.replace('gpu','GPU ')}): MNN's own kernel "
                             f"choice is already the best at **{base:.0f} µs**. Nothing we tried beat "
                             f"it by more than 3%.")
            else:
                lines.append(f"- **{w['label']}** (at {label.replace('gpu','GPU ')}): the fastest "
                             f"option is {plain_name(best)}, at **{bus:.0f} µs** — "
                             f"**{pct(bus, base)} vs MNN's default** ({base:.0f} µs). Worth forcing "
                             f"with `MNN_CONV_SPEC=1 MNN_CONV_FORCE={best}`.")
    for b, v in (first.get("blocks") or {}).items():
        if v["plain_us"] and v["prelu_fused_us"]:
            lines.append(f"- **{b}**: fusing PReLU into the conv saves **{pct(v['prelu_fused_us'], v['plain_us'])}** "
                         f"({v['plain_us']:.0f} → {v['prelu_fused_us']:.0f} µs). This is free — just convert "
                         f"the model with `MNN_FUSE_CONV_PRELU=1`.")
    c = first.get("concurrency")
    if c:
        lines.append(f"- **Two models at once** finish in {c['ratio']:.2f}× the time of one. "
                     + ("Below 1.6× means the GPU has spare capacity — running independent branches "
                        "on two threads/sessions should pay off." if c["ratio"] < 1.6 else
                        "Above 1.6× means the GPU is already saturated — concurrency will not help."))
    bad = [k for k, v in (first.get("correctness") or {}).items() if not (v > 0.99)]
    if bad:
        lines.append(f"- ⚠️ **These kernels produced wrong output: {', '.join(bad)}** — ignore their timings.")
    for l in lines: say(l)
    say("")

    # ---- clocks
    say("## Clocks the numbers were taken at\n")
    if not any_pin:
        say("> ⚠️ **Clocks could NOT be pinned** (no root on this device: `adb root` is refused and\n"
            "> there is no `su`). Everything below ran under the governor. The report still records\n"
            "> the clock measured under load, but a requested `--gpu min` did **not** take effect —\n"
            "> so a low-clock comparison needs a rooted or engineering-build device.\n")
    say("| clock point | GPU | MIF (DRAM bus) | INT (interconnect) | measured GPU under load |")
    say("|---|---|---|---|---|")
    for label, pt in pinned_info.items():
        def cell(k):
            v = pt.get(k)
            if not v: return "not present"
            return (f"**{v['achieved_mhz']} MHz** (pinned)" if v["pinned"]
                    else f"{v['achieved_mhz'] or '?'} MHz ({v['note']})")
        d = results.get(label, {})
        say(f"| {label} | {cell('gpu')} | {cell('mif')} | {cell('int')} | "
            f"{d.get('clock_start_mhz','?')} → {d.get('clock_end_mhz','?')} MHz |")
    say("\n(The last column is start → end of the run. A big drop means the device throttled and\n"
        "absolute numbers drifted; the A/B comparisons inside each section are still valid because\n"
        "their arms are interleaved.)\n")

    # ---- per-strategy table
    say("## Every strategy, per shape (per-conv µs, lower is better)\n")
    for label, data in results.items():
        say(f"**{label}**\n")
        vr = data.get("variants") or {}
        if not vr: say("(no data)\n"); continue
        keys = list(next(iter(vr.values())).keys())
        say("| shape | " + " | ".join(k.replace("conv_2d_", "") for k in keys) + " |")
        say("|" + "---|" * (len(keys) + 1))
        for shape, row in vr.items():
            say(f"| {shape} | " + " | ".join(f"{row[k]:.0f}" if row.get(k) else "-" for k in keys) + " |")
        say("")

    # ---- cross-clock comparison (the point of --gpu-sweep)
    if len(results) > 1:
        say("## Does the best strategy change with the GPU clock?\n")
        shapes = sorted({s for d in results.values() for s in (d.get("variants") or {})})
        say("| shape | " + " | ".join(results) + " |")
        say("|" + "---|" * (len(results) + 1))
        flip = False
        for sh in shapes:
            cells, winners = [], []
            for label, d in results.items():
                row = (d.get("variants") or {}).get(sh, {})
                cand = {k: v for k, v in row.items() if v}
                if not cand: cells.append("-"); continue
                b = min(cand, key=lambda k: cand[k])
                winners.append(b); cells.append(f"**{b.replace('conv_2d_','')}** {cand[b]:.0f}µs")
            if len(set(winners)) > 1: flip = True
            say(f"| {sh} | " + " | ".join(cells) + " |")
        say("")
        say("> **The winner changes with the clock** — pick the kernel for the clock you actually "
            "ship at.\n" if flip else
            "> The same kernel wins at every clock tested — the choice is clock-independent here.\n")

    # ---- pointers
    say("## Detail\n")
    for label in results:
        say(f"- `detail_{label}.md` — full 12-section report at that clock "
            f"(+ `detail_{label}.json` with every raw number)")
    say(f"\nRe-run: `python3 {Path(__file__).name} {serial} --gpu <spec> --mif max --int max`\n")

    out_path.write_text("\n".join(OUT))
    (out_path.with_suffix(".json")).write_text(json.dumps(
        {"serial": serial, "clocks": pinned_info, "results": results}, indent=2))
    print(f"\n=== wrote {out_path} (+ .json, + per-clock detail reports) ===")


if __name__ == "__main__":
    main()
