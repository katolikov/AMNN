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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", "-s", help="adb serial (see --list)")
    ap.add_argument("--list", action="store_true", help="list attached devices and exit")
    ap.add_argument("--quick", action="store_true", help="fewer repeats/tiles (~4 min)")
    ap.add_argument("-o", "--out", default=None, help="report path (default report_<serial>.md)")
    ap.add_argument("--cooldown", type=int, default=None,
                    help="seconds of GPU idle between heavy sections (default 20, 5 with --quick); "
                         "this device throttles under prolonged load, so do not set 0 unless you "
                         "have pinned the clock")
    a = ap.parse_args()

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
        say(f"\n> This device **throttles under prolonged load** (observed dropping from 980 to "
            "~400-550 MHz after minutes of continuous GPU work). Mitigations used here: every A/B\n"
            "> comparison is **interleaved** (arms alternate, so drift hits both equally) and there are\n"
            "> **cooldown pauses** between heavy sections. §12 re-measures the clock at the end — if it\n"
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
        row = {"MNN default": med(lambda i: conv_us(run_model(d, m, shp, 120, cache=f"st{c['key']}{i}.bin")[0], dep), reps)}
        for v in variants:
            env = ("MNN_CONV_SPEC=1 " if v in spec_only else "") + f"MNN_CONV_FORCE={v}"
            row[v] = med(lambda i, e=env, v=v: conv_us(
                run_model(d, m, shp, 120, env=e, cache=f"f{c['key']}{v[-5:]}{i}.bin")[0], dep), reps)
        row["LDS"] = med(lambda i: conv_us(run_model(d, m, shp, 120, env="MNN_CONV_LDS=1",
                                                     cache=f"l{c['key']}{i}.bin")[0], dep), reps)
        allrows[c["key"]] = row
        say(f"| {c['label']} | " + " | ".join(f"{row[k]:.1f}" if row.get(k) else "-"
                                              for k in ["MNN default"] + variants + ["LDS"]) + " |")
    say("")
    for c in cores:
        row = allrows[c["key"]]; base = row["MNN default"]
        cand = {k: v for k, v in row.items() if v and k != "MNN default"}
        if not cand: continue
        best = min(cand, key=lambda k: cand[k])
        summary[c["key"]] = (base, best, cand[best])
        verdict = f"**{pct(cand[best], base)} vs MNN default**" if cand[best] < base else \
                  f"(MNN default already best; closest custom {best} {pct(cand[best], base)})"
        say(f"- **{c['label']}** → default `{base:.1f}µs`, best strategy **`{best}` `{cand[best]:.1f}µs`** {verdict}")
    say("")

    # ---- 5. LDS tiles
    say("## 5. LDS tile sweep\n")
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
    say("## 6. im2col + GEMM (and implicit-GEMM headroom)\n")
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
        say(f"| {c['label']} | {im:.0f} | {gm:.0f} | {im+gm:.0f} | {base:.0f} | "
            f"{'faster' if im+gm < base else 'slower'} {pct(im+gm, base)} | **{pct(gm, base)}** |")
    say("\n> Last column is the lever: if the **GEMM reduce alone beats the default**, an *implicit*\n"
        "> GEMM (gather columns in registers, never materialize im2col) is worth building — and is\n"
        "> straightforward if `sub_group_shuffle` is available (§3).\n")

    # ---- 7. fused megakernel
    say("## 7. Fused 2-layer megakernel\n")
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
        say(f"| {c['label']} | {fu:.0f} | {2*one:.0f} | {'FASTER' if fu < 2*one else 'SLOWER'} {pct(fu, 2*one)} |")
    say("")

    # ---- 8. real blocks
    say("## 8. Real model blocks (deployment numbers)\n")
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
        say(f"| {b['key']} | {t0:.0f} | {t1:.0f} | **{pct(t1, t0)}**{warn} |")
    say("")

    # ---- 9. concurrency
    say("## 9. Concurrency (2 independent streams)\n")
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
        say(f"- solo **{s1:.2f} ms**, both-done **{s2:.2f} ms** → **{s2/s1:.2f}x**")
        say(f"- {'SPARE CAPACITY — running independent branches concurrently should pay off' if s2 < 1.6*s1 else 'SATURATED — concurrency will not help'}")
        say("- (wall-clock incl. CPU/submission overhead; treat as a signal)\n")
    else:
        say("- probe failed\n")

    # ---- 10. correctness
    say("## 10. Correctness (custom kernels vs MNN default output)\n")
    cc = man["correctness"]
    d.push(REFD / "cc_input.txt", f"{DEV}/tdir/input.txt")
    _, base_out = run_model(d, cc["model"], cc["shape"], 1, cache="k0.bin", pull=True)
    say("| kernel | cosine | verdict |")
    say("|---|---|---|")
    for label, env in [("c8h8w1", "MNN_CONV_SPEC=1 MNN_CONV_FORCE=conv_2d_c8h8w1"),
                       ("c8h4w1_pa", "MNN_CONV_SPEC=1 MNN_CONV_FORCE=conv_2d_c8h4w1_pa"),
                       ("LDS", "MNN_CONV_LDS=1")]:
        _, v = run_model(d, cc["model"], cc["shape"], 1, env=env, cache="k1.bin", pull=True)
        c = cosine(base_out, v)
        say(f"| {label} | {c:.6f} | {'PASS' if c > 0.99 else 'FAIL — do not trust its timing'} |")
    ref2 = [float(x) for x in (REFD / "fused2_ref.txt").read_text().split()]
    _, v = run_model(d, cc["model"], cc["shape"], 1, env="MNN_CONV_FUSED2=1", cache="k2.bin", pull=True)
    c = cosine(ref2, v)
    say(f"| fused2 (vs conv² reference) | {c:.6f} | {'PASS' if c > 0.99 else 'FAIL'} |")
    d.shell(f"rm -f {DEV}/tdir/input.txt")
    say("")

    # ---- 11. what to do on this device
    say("## 11. Recommendations for THIS device\n")
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
    recs.append("⚙️ **PReLU fusion** (§8) is free and applies to every conv — keep it on.")
    for r in recs: say(f"- {r}")

    # ---- 12. clock re-check: did the device throttle over the run?
    say("\n## 12. GPU clock at END of run (thermal validity check)\n")
    cooldown(d, cool_s)
    s2c, _ = sample_clock(d, cores[0]["model"], cores[0]["shape"])
    clk_end = max(s2c) if s2c else 0
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
    print(f"\n=== wrote {out_path} ===")


if __name__ == "__main__":
    main()
