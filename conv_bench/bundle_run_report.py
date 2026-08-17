#!/usr/bin/env python3
"""Conv-strategy probe — STANDALONE. Requires only python3 (stdlib) + adb on PATH.

Runs every convolution strategy/implementation against one Android device and writes a markdown
report. Goal: find the BEST configuration ON THIS DEVICE (not to match any previous device).

    python3 run_report.py --list                       # show attached devices
    python3 run_report.py --serial R3CY905E04M -o report.md
    python3 run_report.py --serial XXXX --quick        # ~4 min instead of ~10

Clock: this script does NOT change the GPU clock (pin it yourself beforehand if you want).
It DOES sample /sys/kernel/gpu/gpu_clock under load and records what clock the numbers were
taken at, because the clock materially affects which strategy wins.

Everything needed (models, android binaries) ships in this bundle -- no MNN repo, no MNNConvert,
no numpy required.
"""
import argparse, json, math, os, re, subprocess, statistics, sys, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN, MODELS, REFD = HERE / "bin", HERE / "models", HERE / "ref"
DEV = "/data/local/tmp/convprobe"
R = []
D = {}   # machine-readable results, written next to the report as .json


def reset():
    """Allow a driver script (run_suite.py) to call main() more than once."""
    R.clear(); D.clear()


def say(s=""):
    print(s, flush=True); R.append(s)


class Dev:
    def __init__(self, serial): self.s = serial
    def sh(self, cmd, timeout=600):
        return subprocess.run(f"adb -s {self.s} {cmd}", shell=True, text=True,
                              capture_output=True, timeout=timeout)
    def shell(self, cmd, timeout=600): return self.sh(f"shell '{cmd}'", timeout=timeout).stdout
    def push(self, local, remote): return self.sh(f"push \"{local}\" {remote}")
    def prop(self, p): return self.shell(f"getprop {p}").strip()


def list_devices():
    out = subprocess.run("adb devices", shell=True, text=True, capture_output=True).stdout
    return [l.split()[0] for l in out.splitlines()[1:] if "\tdevice" in l]


# --------------------------------------------------------------- device run helpers
def run_model(d, model, shape, loops, env="", cache="p.bin", pull=False, timeout=600):
    """Push model+input.json, run ModuleBasic, return (stdout, output_floats_or_None)."""
    inj = HERE / "_input.json"
    inj.write_text(json.dumps({"inputs": [{"name": "input", "shape": shape}],
                               "outputs": ["output"], "shapeMutable": False}))
    d.push(inj, f"{DEV}/tdir/input.json")
    d.push(MODELS / model, f"{DEV}/{model}")
    if pull:
        d.shell(f"rm -rf {DEV}/output"); d.shell(f"mkdir -p {DEV}/output")
    out = d.shell(f"cd {DEV} && {env} LD_LIBRARY_PATH=. ./ModuleBasic.out {model} tdir 0 3 "
                  f"{loops} 68 2 {cache} 2>&1", timeout=timeout)
    vals = None
    if pull:
        p = HERE / "_out.txt"
        if p.exists(): p.unlink()
        d.sh(f"pull {DEV}/output/0_0.txt \"{p}\"")
        if p.exists():
            try: vals = [float(x) for x in p.read_text().split()]
            except Exception: vals = None
    return out, vals


def conv_us(out, depth=1):
    c = [int(t) for t in re.findall(r"conv time = (\d+) us", out)][3:]
    return (statistics.median(c) / depth) if c else 0.0


def conv_paths(out):
    """Which conv implementation MNN actually ran, from the profiler summary line.
    -> {'ori': us, 'wino': us, '1x1': us, 'gemm1': us, 'gemm2': us, 'other': us} (medians)."""
    keys = ("gemm2", "gemm1", "1x1", "ori", "wino", "other")
    acc = {k: [] for k in keys}
    for m in re.finditer(r"conv time = \d+ us \(gemm2:(\d+) us, gemm1:(\d+) us, 1x1:(\d+) us, "
                         r"ori:(\d+) us, wino: (\d+) us, other: (\d+) us\)", out):
        for k, v in zip(keys, m.groups()):
            acc[k].append(int(v))
    return {k: (statistics.median(v[3:]) if len(v) > 3 else (statistics.median(v) if v else 0))
            for k, v in acc.items()}


def per_conv_us(out, tags):
    """Median GPU time for EACH conv in a heterogeneous chain.

    MNN names every conv kernel with its shape, e.g.
        kernel time = 155 us ConvBuf2D-ori-b1ci18hi288wi384co16ho144wo192kh3kw3-total:...
    so a "ci18hi288wi384co16" tag identifies one conv unambiguously. Averaging a 2-conv head
    the way conv_us() does would mix two different shapes into one meaningless number.
    """
    acc = {t: [] for t in tags}
    for m in re.finditer(r"kernel time = (\d+)\s+us (ConvBuf2D-\S+)", out):
        us, name = int(m.group(1)), m.group(2)
        for t in tags:
            if t in name:
                acc[t].append(us); break
    return {t: (statistics.median(v[3:]) if len(v) > 3 else 0.0) for t, v in acc.items()}


def total_us(out):
    t = [int(x) for x in re.findall(r"total kernel time = (\d+)  us", out)][3:]
    return statistics.median(t) if t else 0.0


def med(fn, reps):
    v = [x for x in (fn(i) for i in range(reps)) if x]
    return statistics.median(v) if v else 0.0


def cosine(a, b):
    if not a or not b: return float("nan")
    n = min(len(a), len(b))
    sa = sb = sab = 0.0
    for i in range(n):
        x, y = a[i], b[i]; sa += x * x; sb += y * y; sab += x * y
    return sab / (math.sqrt(sa) * math.sqrt(sb) + 1e-12)


def pct(v, base):
    if not v or not base: return "n/a"
    return f"{100*(v-base)/base:+.0f}%"


def sample_clock(d, model, shape, seconds=6):
    """Sample GPU clock while a long run is in flight. Returns (samples_mhz, freq_table).

    NOTE: kills the helper run by PID -- `pkill -f ModuleBasic.out` would match (and kill) the
    very shell issuing it, leaving the background run alive to hog the GPU.
    """
    tbl = d.shell("cat /sys/kernel/gpu/gpu_freq_table").strip()
    inj = HERE / "_input.json"
    inj.write_text(json.dumps({"inputs": [{"name": "input", "shape": shape}],
                               "outputs": ["output"], "shapeMutable": False}))
    d.push(inj, f"{DEV}/tdir/input.json"); d.push(MODELS / model, f"{DEV}/{model}")
    n = int(seconds / 0.3)
    out = d.shell(f"cd {DEV} && LD_LIBRARY_PATH=. ./ModuleBasic.out {model} tdir 0 3 20000 68 2 "
                  f"clk.bin >/dev/null 2>&1 & "
                  f"BGPID=$!; echo PID=$BGPID; sleep 1; for i in $(seq 1 {n}); do "
                  f"printf \"%s \" $(cat /sys/kernel/gpu/gpu_clock 2>/dev/null); sleep 0.3; done; "
                  f"kill -9 $BGPID 2>/dev/null; wait $BGPID 2>/dev/null; true", timeout=180)
    m = re.search(r"PID=(\d+)", out)
    if m:                                   # belt-and-braces: make sure the helper is gone
        d.shell(f"kill -9 {m.group(1)} 2>/dev/null; true")
    body = out.split("PID=", 1)[-1]
    body = body.split(None, 1)[1] if len(body.split(None, 1)) > 1 else ""
    # 0 = GPU power-gated (idle), not a downclock -> drop
    s = [v // 1000 for v in (int(x) for x in body.split() if x.isdigit()) if v > 0]
    return s, tbl


def cooldown(d, seconds):
    """Let the GPU cool so a later section isn't measured on a throttled clock."""
    if seconds <= 0: return
    print(f"   (cooldown {seconds}s)", flush=True)
    d.shell(f"sleep {seconds}", timeout=seconds + 60)


def interleaved(fns, reps):
    """Run labelled measurements round-robin (a,b,a,b,...) so thermal drift hits all arms
    equally. fns = {label: callable(i)->float}. Returns {label: median}."""
    acc = {k: [] for k in fns}
    for i in range(reps):
        for k, f in fns.items():
            v = f(i)
            if v: acc[k].append(v)
    return {k: (statistics.median(v) if v else 0.0) for k, v in acc.items()}


# --------------------------------------------------------------- sections
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", "-s", help="adb serial (see --list)")
    ap.add_argument("--list", action="store_true", help="list attached devices and exit")
    ap.add_argument("--quick", action="store_true", help="fewer repeats/tiles (~4 min)")
    ap.add_argument("-o", "--out", default=None, help="report path (default report_<serial>.md)")
    ap.add_argument("--cooldown", type=int, default=None,
                    help="seconds of GPU idle between heavy sections (default 20, 5 with --quick); "
                         "this device throttles under prolonged load, so do not set 0 unless you "
                         "have pinned the clock")
    a = ap.parse_args(argv)

    devs = list_devices()
    if a.list:
        print("attached devices:"); [print("  " + x) for x in devs] or print("  (none)"); return
    serial = a.serial or (devs[0] if len(devs) == 1 else None)
    if not serial:
        print(f"Specify --serial. Attached: {devs or '(none)'}"); sys.exit(1)
    d = Dev(serial)
    reps = 1 if a.quick else 3
    cool_s = a.cooldown if a.cooldown is not None else (5 if a.quick else 20)
    man = json.loads((HERE / "manifest.json").read_text())
    cores = man["cores"]           # [{key,label,C,H,W}]
    variants = man["variants"]     # [kernel names]
    spec_only = set(man["spec_only"])
    stride1_only = set(man.get("stride1_only", []))
    hard_capable = [v for v in man.get("hard_capable", []) if v in variants]
    tiles = man["lds_tiles"][:3] if a.quick else man["lds_tiles"]
    out_path = Path(a.out) if a.out else HERE / f"report_{serial}.md"

    # ---- stage
    print(f"staging to {serial} ...")
    d.shell(f"mkdir -p {DEV}/tdir")
    for f in sorted(BIN.iterdir()):
        d.push(f, f"{DEV}/")
    d.shell("svc power stayon usb")

    say("# Conv strategy report")
    say(f"\n_device `{serial}` — {d.prop('ro.product.model')} / soc `{d.prop('ro.soc.model')}` — "
        f"generated {datetime.datetime.now():%Y-%m-%d %H:%M}{' (quick)' if a.quick else ''}_\n")
    say("> Goal: find the **best conv configuration on this device**. All timings are per-conv GPU\n"
        "> kernel time (median of repeats) under sustained load, OpenCL buffer mode, fp16.\n")

    summary = {}

    # ---- 1. hardware
    say("## 1. Hardware\n")
    out, _ = run_model(d, cores[0]["model"], cores[0]["shape"], 1, env="MNN_DUMP_CL_EXT=1", cache="hw.bin")
    hw, exts, shuffle = {}, "", None
    for ln in out.splitlines():
        if ln.startswith("[HWINFO]"):
            kv = ln.replace("[HWINFO]", "").strip()
            if kv.startswith("name="): hw["name"] = kv[5:].strip()
            for part in kv.split():
                if "=" in part: hw.setdefault(*part.split("=", 1))
        elif ln.startswith("[CL_EXT]"):
            if "has_subgroup_shuffle" in ln: shuffle = ln.strip().endswith("=1")
            elif not exts: exts = ln.replace("[CL_EXT]", "").strip()
    for k in ("name", "max_compute_units", "max_clock_mhz", "local_mem_bytes",
              "max_work_group_size", "pref_vec_half", "native_vec_half", "driver_version"):
        if k in hw: say(f"- **{k}**: `{hw[k]}`")
    say(f"- **subgroup_shuffle**: `{shuffle}`")
    D["hw"] = dict(hw); D["hw"]["subgroup_shuffle"] = shuffle
    if exts: say(f"\n<details><summary>CL extensions</summary>\n\n```\n{exts}\n```\n</details>\n")

    # ---- 2. clock at the START (re-checked at the end; see §12)
    say("## 2. GPU clock at start of run\n")
    s, tbl = sample_clock(d, cores[0]["model"], cores[0]["shape"])
    clk_start = max(s) if s else 0
    if s:
        mx, mn = max(s), min(s)
        pin = sum(1 for v in s if v == mx)
        say(f"- sampled under sustained load: **{pin}/{len(s)} samples @ {mx} MHz**, min {mn} MHz")
        say(f"- min/max clock settings: `{d.shell('cat /sys/kernel/gpu/gpu_min_clock').strip()}` / "
            f"`{d.shell('cat /sys/kernel/gpu/gpu_max_clock').strip()}`")
        say(f"- freq table: `{tbl}`")
        D["clock_start"] = {"max_mhz": mx, "min_mhz": mn, "pinned_samples": pin, "n": len(s)}
        say(f"\n> This device **throttles under prolonged load** (observed dropping from 980 to "
            "~400-550 MHz after minutes of continuous GPU work). Mitigations used here: every A/B\n"
            "> comparison is **interleaved** (arms alternate, so drift hits both equally) and there are\n"
            "> **cooldown pauses** between heavy sections. §15 re-measures the clock at the end — if it\n"
            "> dropped, sections are not comparable across time and should be re-run individually.\n")
    else:
        say("- could not read `/sys/kernel/gpu/gpu_clock` (timings unvalidated)\n")

    # ---- 3. subgroup
    say("## 3. Subgroup capability\n")
    out, _ = run_model(d, cores[0]["model"], cores[0]["shape"], 1, env="MNN_SUBGROUP_PROBE=1", cache="sg.bin")
    sg = {}
    for ln in out.splitlines():
        if "[SGPROBE]" in ln and "build:" in ln:
            say(f"- `{ln.strip()}`")
            if "broadcast" in ln: sg["bcast"] = "OK" in ln
            if "shuffle" in ln: sg["shuffle"] = "OK" in ln
    if sg.get("shuffle"):
        say("\n> 🔥 **`sub_group_shuffle` COMPILES ON THIS DEVICE.** Register-level halo sharing /\n"
            "> implicit-GEMM conv is implementable here in OpenCL — it was blocked on the reference\n"
            "> device. This is the highest-value thing to build next.\n")
    else:
        say("\n> No `sub_group_shuffle` — register-level halo sharing not available in OpenCL here.\n")

    # ---- 4. variants -> the winner
    say("## 4. Kernel strategies (per-conv µs, lower is better)\n")
    say("| shape | " + " | ".join(["MNN default"] + [v.replace("conv_2d_", "") for v in variants] + ["LDS"]) + " |")
    say("|" + "---|" * (len(variants) + 3))
    allrows = {}
    for c in cores:
        m, shp, dep = c["model"], c["shape"], c["depth"]
        # INTERLEAVED (a,b,c,...,a,b,c,...) not arm-by-arm: this device throttles, and measuring
        # 13 arms sequentially would systematically penalise whichever runs last.
        fns = {"MNN default": lambda i: conv_us(
            run_model(d, m, shp, 120, cache=f"st{c['key']}{i}.bin")[0], dep)}
        for v in variants:
            env = ("MNN_CONV_SPEC=1 " if v in spec_only else "") + f"MNN_CONV_FORCE={v}"
            fns[v] = (lambda i, e=env, v=v: conv_us(
                run_model(d, m, shp, 120, env=e, cache=f"f{c['key']}{v[-5:]}{i}.bin")[0], dep))
        fns["LDS"] = lambda i: conv_us(
            run_model(d, m, shp, 120, env="MNN_CONV_LDS=1", cache=f"l{c['key']}{i}.bin")[0], dep)
        cooldown(d, cool_s)
        row = interleaved(fns, reps)
        allrows[c["key"]] = row
        D.setdefault("variants", {})[c["key"]] = dict(row)
        say(f"| {c['label']} | " + " | ".join(f"{row[k]:.1f}" if row.get(k) else "-"
                                              for k in ["MNN default"] + variants + ["LDS"]) + " |")
    say("")
    for c in cores:
        row = allrows[c["key"]]; base = row["MNN default"]
        cand = {k: v for k, v in row.items() if v and k != "MNN default"}
        if not cand: continue
        best = min(cand, key=lambda k: cand[k])
        summary[c["key"]] = (base, best, cand[best])
        D.setdefault("winner", {})[c["key"]] = {"default_us": base, "best": best, "best_us": cand[best], "label": c["label"]}
        verdict = f"**{pct(cand[best], base)} vs MNN default**" if cand[best] < base else \
                  f"(MNN default already best; closest custom {best} {pct(cand[best], base)})"
        say(f"- **{c['label']}** → default `{base:.1f}µs`, best strategy **`{best}` `{cand[best]:.1f}µs`** {verdict}")
    say("")

    # ---- 5. shape hardcoding
    say("## 5. Shape hardcoding (MNN_CONV_HARD=1)\n")
    if not hard_capable:
        say("_(this bundle has no kernels that read the HC_* constants)_\n")
    else:
        say("> `MNN_CONV_HARD=1` passes the shape (in/out H,W, channel blocks, stride, pad, dilation)\n"
            "> as **compile-time constants**, so index arithmetic folds and halo bounds collapse. Costs\n"
            "> one cached program build per distinct shape. Only the kernels below read them.\n")
        cols = ["MNN default"] + [x for v in hard_capable for x in (v, v + "+HARD")]
        say("| shape | " + " | ".join(c.replace("conv_2d_", "") for c in cols) + " | best |")
        say("|" + "---|" * (len(cols) + 2))
        for c in cores:
            cooldown(d, cool_s)
            m, shp, dep = c["model"], c["shape"], c["depth"]
            fns = {"MNN default": lambda i, m=m, shp=shp, c=c: conv_us(
                run_model(d, m, shp, 120, cache=f"hd{c['key']}{i}.bin")[0], dep)}
            for v in hard_capable:
                base = ("MNN_CONV_SPEC=1 " if v in spec_only else "") + f"MNN_CONV_FORCE={v}"
                fns[v] = (lambda i, e=base, v=v, m=m, shp=shp, c=c: conv_us(
                    run_model(d, m, shp, 120, env=e, cache=f"hp{c['key']}{v[-5:]}{i}.bin")[0], dep))
                fns[v + "+HARD"] = (lambda i, e="MNN_CONV_HARD=1 " + base, v=v, m=m, shp=shp, c=c: conv_us(
                    run_model(d, m, shp, 120, env=e, cache=f"hh{c['key']}{v[-5:]}{i}.bin")[0], dep))
            row = interleaved(fns, reps)
            cand = {k: t for k, t in row.items() if t}
            best = min(cand, key=lambda k: cand[k]) if cand else "-"
            D.setdefault("hardcoding", {})[c["key"]] = dict(row)
            say(f"| {c['label']} | " + " | ".join(f"{row[k]:.0f}" if row.get(k) else "-" for k in cols)
                + f" | **{best.replace('conv_2d_','')}** |")
        say("\n> If a `+HARD` column beats its plain twin, ship that kernel with the flag on. On the\n"
            "> reference device this was worth an extra ~11 percentage points on the main core.\n")

    # ---- 6. stride-2 head pairs (per conv)
    say("## 6. Stride-2 head pairs (per-conv µs, lower is better)\n")
    heads = man.get("heads", [])
    if not heads:
        say("_(no head blocks in this bundle)_\n")
    else:
        say("> These are 3x3 **stride-2** convs. The LDS, im2col and fused2 kernels are stride-1\n"
            "> only, so they cannot run here — only the direct variants apply. Each conv in the\n"
            "> pair is reported separately (they have different shapes).\n")
        head_vars = ["MNN default"] + [v for v in variants if v not in stride1_only]
        if stride1_only:
            say("> Skipped here (stride-1 only): "
                + ", ".join(sorted(v.replace("conv_2d_", "") for v in stride1_only)) + "\n")
        for h in heads:
            say(f"\n**{h['label']}**\n")
            tags = [c["tag"] for c in h["convs"]]
            fns = {"MNN default": lambda i, h=h: per_conv_us(
                run_model(d, h["model"], h["shape"], 60, cache=f"h{h['key']}{i}.bin")[0], tags)}
            for v in head_vars[1:]:
                env = ("MNN_CONV_SPEC=1 " if v in spec_only else "") + f"MNN_CONV_FORCE={v}"
                fns[v] = (lambda i, e=env, v=v, h=h: per_conv_us(
                    run_model(d, h["model"], h["shape"], 60, env=e,
                              cache=f"h{h['key']}{v[-5:]}{i}.bin")[0], tags))
            # interleaved by hand: interleaved() medians floats, these are dicts
            acc = {k: {t: [] for t in tags} for k in fns}
            for i in range(reps):
                for k, f in fns.items():
                    r = f(i)
                    for t in tags:
                        if r.get(t): acc[k][t].append(r[t])
            row = {k: {t: (statistics.median(v) if v else 0.0) for t, v in d2.items()}
                   for k, d2 in acc.items()}
            say("| conv | " + " | ".join(k.replace("conv_2d_", "") for k in head_vars) + " | best |")
            say("|" + "---|" * (len(head_vars) + 2))
            for c in h["convs"]:
                t = c["tag"]
                cand = {k: row[k][t] for k in head_vars if row[k][t]}
                best = min(cand, key=lambda k: cand[k]) if cand else "-"
                say(f"| {c['label']} | " +
                    " | ".join(f"{row[k][t]:.0f}" if row[k][t] else "-" for k in head_vars) +
                    f" | **{best.replace('conv_2d_','')}** |")
            D.setdefault("heads", {})[h["key"]] = {
                "label": h["label"],
                "convs": {c["label"]: {k: row[k][c["tag"]] for k in head_vars} for c in h["convs"]}}
        say("")

    # ---- 7. LDS tiles
    say("## 7. LDS tile sweep\n")
    say("| shape | " + " | ".join(tiles) + " | best LDS | vs MNN default |")
    say("|" + "---|" * (len(tiles) + 3))
    for c in cores:
        m, shp, dep = c["model"], c["shape"], c["depth"]; row = {}
        for t in tiles:
            row[t] = med(lambda i, t=t: conv_us(run_model(d, m, shp, 120,
                         env=f"MNN_CONV_LDS=1 MNN_LDS_TILE={t}", cache=f"t{c['key']}{t}{i}.bin")[0], dep),
                         1 if a.quick else 2)
        bt = min(row, key=lambda t: row[t] if row[t] else 9e9)
        say(f"| {c['label']} | " + " | ".join(f"{row[t]:.0f}" for t in tiles) +
            f" | **{bt} = {row[bt]:.0f}** | {pct(row[bt], allrows[c['key']]['MNN default'])} |")
    say("")

    # ---- 6. im2col + GEMM
    say("## 8. im2col + GEMM (and implicit-GEMM headroom)\n")
    say("| shape | im2col | GEMM | total | MNN default | explicit verdict | **GEMM vs default** |")
    say("|---|---|---|---|---|---|---|")
    for c in cores:
        cooldown(d, cool_s)
        # interleaved with a fresh default measurement so all three share the same thermal state
        r = interleaved({
            "im2col": lambda i, c=c: conv_us(run_model(d, c["im2col_model"], c["shape"], 200,
                      env="MNN_NO_WINOGRAD=1 MNN_CONV_IM2COL=1", cache=f"ic{c['key']}{i}.bin")[0]),
            "gemm": lambda i, c=c: conv_us(run_model(d, c["gemm_model"], c["gemm_shape"], 200,
                    cache=f"gp{c['key']}{i}.bin")[0]),
            "default": lambda i, c=c: conv_us(run_model(d, c["model"], c["shape"], 120,
                       cache=f"dr{c['key']}{i}.bin")[0], c["depth"]),
        }, reps)
        im, gm, base = r["im2col"], r["gemm"], r["default"]
        D.setdefault("im2col", {})[c["key"]] = {"im2col_us": im, "gemm_us": gm, "default_us": base}
        say(f"| {c['label']} | {im:.0f} | {gm:.0f} | {im+gm:.0f} | {base:.0f} | "
            f"{'faster' if im+gm < base else 'slower'} {pct(im+gm, base)} | **{pct(gm, base)}** |")
    say("\n> Last column is the lever: if the **GEMM reduce alone beats the default**, an *implicit*\n"
        "> GEMM (gather columns in registers, never materialize im2col) is worth building — and is\n"
        "> straightforward if `sub_group_shuffle` is available (§3).\n")

    # ---- 7. Winograd: is it even selected, and does disabling it help?
    say("## 9. Winograd vs direct\n")
    say("| shape | path MNN chose | default | Winograd OFF | verdict |")
    say("|---|---|---|---|---|")
    for c in cores:
        cooldown(d, cool_s)
        m, shp, dep = c["model"], c["shape"], c["depth"]
        o_def, _ = run_model(d, m, shp, 120, cache=f"wg{c['key']}p.bin")
        paths = conv_paths(o_def)
        chose = max(paths, key=lambda k: paths[k]) if any(paths.values()) else "?"
        r = interleaved({
            "on": lambda i, m=m, shp=shp, c=c: conv_us(
                run_model(d, m, shp, 120, cache=f"wa{c['key']}{i}.bin")[0], dep),
            "off": lambda i, m=m, shp=shp, c=c: conv_us(
                run_model(d, m, shp, 120, env="MNN_NO_WINOGRAD=1",
                          cache=f"wb{c['key']}{i}.bin")[0], dep),
        }, reps)
        on, off = r["on"], r["off"]
        D.setdefault("winograd", {})[c["key"]] = {"path": chose, "default_us": on, "no_wino_us": off}
        if chose != "wino":
            verdict = f"Winograd NOT used here (MNN picked `{chose}`) — switch is a no-op"
        else:
            verdict = (f"Winograd wins {pct(off, on)} if disabled" if off > on
                       else f"**disabling Winograd is {pct(off, on)} FASTER**")
        say(f"| {c['label']} | `{chose}` | {on:.0f} | {off:.0f} | {verdict} |")
    say("\n> Winograd trades multiply-adds for extra transform passes. It is contraindicated at a\n"
        "> high GPU clock for these small-channel shapes, but is the top candidate to flip when the\n"
        "> GPU clock is LOW and memory is fast (arithmetic becomes the scarce resource).\n")

    # ---- 8. fused megakernel
    say("## 10. Fused 2-layer megakernel\n")
    say("| shape | fused (2 layers) | 2x single conv | verdict |")
    say("|---|---|---|---|")
    for c in cores:
        cooldown(d, cool_s)
        r = interleaved({
            "one": lambda i, c=c: conv_us(run_model(d, c["single_model"], c["shape"], 200,
                   cache=f"s1{c['key']}{i}.bin")[0]),
            "fused": lambda i, c=c: conv_us(run_model(d, c["single_model"], c["shape"], 200,
                     env="MNN_CONV_FUSED2=1", cache=f"f2{c['key']}{i}.bin")[0]),
        }, reps)
        one, fu = r["one"], r["fused"]
        D.setdefault("fused2", {})[c["key"]] = {"fused_us": fu, "two_single_us": 2*one}
        say(f"| {c['label']} | {fu:.0f} | {2*one:.0f} | {'FASTER' if fu < 2*one else 'SLOWER'} {pct(fu, 2*one)} |")
    say("")

    # ---- 8. real blocks
    say("## 11. Real model blocks (deployment numbers)\n")
    say("| block | plain | PReLU-fused | saving |")
    say("|---|---|---|---|")
    for b in man["blocks"]:
        cooldown(d, cool_s)
        r = interleaved({
            "plain": lambda i, b=b: total_us(run_model(d, b["model"], b["shape"], 120,
                     cache=f"{b['key']}a{i}.bin")[0]),
            "fused": lambda i, b=b: total_us(run_model(d, b["fused_model"], b["shape"], 120,
                     cache=f"{b['key']}b{i}.bin")[0]),
        }, 2 if a.quick else 3)
        t0, t1 = r["plain"], r["fused"]
        warn = "" if t1 < t0 * 0.995 else "  ⚠️ no saving — is PReLU fusion supported by this libMNN_CL?"
        D.setdefault("blocks", {})[b["key"]] = {"plain_us": t0, "prelu_fused_us": t1}
        say(f"| {b['key']} | {t0:.0f} | {t1:.0f} | **{pct(t1, t0)}**{warn} |")
    say("")

    # ---- 9. concurrency
    say("## 12. Concurrency (2 independent streams)\n")
    MIN = re.compile(r"min= ([\d.]+) ms")
    m0, shp0 = cores[0]["model"], cores[0]["shape"]
    def solo():
        o, _ = run_model(d, m0, shp0, 200, cache="ca.bin"); mm = MIN.search(o)
        return float(mm.group(1)) if mm else None
    def duo():
        c = (f"adb -s {serial} shell 'cd {DEV} && LD_LIBRARY_PATH=. ./ModuleBasic.out {m0} tdir "
             f"0 3 200 68 2 %s 2>&1'")
        p1 = subprocess.Popen(c % "ca.bin", shell=True, text=True, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(c % "cb.bin", shell=True, text=True, stdout=subprocess.PIPE)
        oa, ob = p1.communicate()[0], p2.communicate()[0]
        ma, mb = MIN.search(oa), MIN.search(ob)
        return max(float(ma.group(1)), float(mb.group(1))) if (ma and mb) else None
    cooldown(d, cool_s)
    rc = interleaved({"solo": lambda i: solo(), "duo": lambda i: duo()}, reps)
    s1, s2 = rc["solo"], rc["duo"]
    if s1 and s2:
        D["concurrency"] = {"solo_ms": s1, "duo_ms": s2, "ratio": s2/s1}
        say(f"- solo **{s1:.2f} ms**, both-done **{s2:.2f} ms** → **{s2/s1:.2f}x**")
        say(f"- {'SPARE CAPACITY — running independent branches concurrently should pay off' if s2 < 1.6*s1 else 'SATURATED — concurrency will not help'}")
        say("- (wall-clock incl. CPU/submission overhead; treat as a signal)\n")
    else:
        say("- probe failed\n")

    # ---- 10. correctness
    say("## 13. Correctness (custom kernels vs MNN default output)\n")
    cc = man["correctness"]
    d.push(REFD / "cc_input.txt", f"{DEV}/tdir/input.txt")
    _, base_out = run_model(d, cc["model"], cc["shape"], 1, cache="k0.bin", pull=True)
    say("| kernel | cosine | verdict |")
    say("|---|---|---|")
    gates = [("LDS", "MNN_CONV_LDS=1")] + [
        (v.replace("conv_2d_", ""), f"MNN_CONV_SPEC=1 MNN_CONV_FORCE={v}") for v in sorted(spec_only)]
    # NB: one cache file PER GATE. Reusing ONE autotune cache file across different forced kernels
    # makes some runs emit no output at all (empty pull -> cosine nan -> a correct kernel is
    # reported as FAIL). Always give each forced kernel its own cache file.
    for gi, (label, env) in enumerate(gates):
        _, v = run_model(d, cc["model"], cc["shape"], 1, env=env, cache=f"k1_{gi}.bin", pull=True)
        c = cosine(base_out, v)
        D.setdefault("correctness", {})[label] = c
        say(f"| {label} | {c:.6f} | {'PASS' if c > 0.99 else 'FAIL — do not trust its timing'} |")
    ref2 = [float(x) for x in (REFD / "fused2_ref.txt").read_text().split()]
    _, v = run_model(d, cc["model"], cc["shape"], 1, env="MNN_CONV_FUSED2=1", cache="k2.bin", pull=True)
    c = cosine(ref2, v)
    say(f"| fused2 (vs conv² reference) | {c:.6f} | {'PASS' if c > 0.99 else 'FAIL'} |")
    d.shell(f"rm -f {DEV}/tdir/input.txt")
    say("")

    # ---- 11. what to do on this device
    say("## 14. Recommendations for THIS device\n")
    recs = []
    for c in cores:
        if c["key"] in summary:
            base, best, bt = summary[c["key"]]
            if bt < base * 0.97:
                recs.append(f"✅ **{c['label']}: use `{best}`** — {pct(bt, base)} vs MNN's default "
                            f"({base:.0f}→{bt:.0f}µs). Wire it in via the autotuner or force it.")
            else:
                recs.append(f"➖ **{c['label']}**: MNN's default ({base:.0f}µs) is already best; no "
                            f"custom kernel beat it (closest `{best}` {bt:.0f}µs).")
    if sg.get("shuffle"):
        recs.append("🔥 **Build the implicit-GEMM conv** — `sub_group_shuffle` works here, so the "
                    "3×3 column gather can happen in registers (no LDS, no barriers). Combine with "
                    "the §6 GEMM headroom number to size the expected win.")
    cu = int(hw.get("max_compute_units", 0) or 0)
    if cu >= 12:
        recs.append(f"📈 **{cu} compute units** — more occupancy headroom than the 8-CU reference; "
                    "wider/deeper tiles (c8h8w1) and fusion deserve a re-test here, they lost on "
                    "occupancy before.")
    recs.append("⚙️ **PReLU fusion** (§11) is free and applies to every conv — keep it on.")
    for r in recs: say(f"- {r}")

    # ---- 12. clock re-check: did the device throttle over the run?
    say("\n## 15. GPU clock at END of run (thermal validity check)\n")
    cooldown(d, cool_s)
    s2c, _ = sample_clock(d, cores[0]["model"], cores[0]["shape"])
    clk_end = max(s2c) if s2c else 0
    D["clock_end_mhz"] = clk_end; D["clock_start_mhz"] = clk_start
    say(f"- start of run: **{clk_start} MHz**  →  end of run: **{clk_end} MHz**"
        + (f" (samples {s2c})" if s2c else ""))
    if clk_start and clk_end:
        drop = 100 * (clk_start - clk_end) / clk_start
        if drop <= 5:
            say(f"- **VALID** — clock held within {drop:.0f}%; numbers across sections are comparable.")
        else:
            say(f"- ⚠️ **THROTTLED — clock fell {drop:.0f}% during the run.** Within-section A/B "
                "comparisons are still valid (arms are interleaved), but **absolute numbers in later "
                "sections are inflated** and must not be compared against earlier sections. Re-run "
                "with a longer `--cooldown`, or pin the clock, or run sections separately.")
    say("\n---\n_Generated by the standalone conv-probe bundle. Send this file back with the "
        "device model + the clock you pinned for analysis._")

    out_path.write_text("\n".join(R))
    out_path.with_suffix(".json").write_text(json.dumps(D, indent=2))
    print(f"\n=== wrote {out_path} (+ .json) ===")
    return out_path, dict(D)


if __name__ == "__main__":
    main()
