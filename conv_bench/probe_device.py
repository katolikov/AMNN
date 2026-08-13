#!/usr/bin/env python3
"""ONE-SHOT: run every conv strategy/implementation we tested, on any device, -> markdown report.

NOTE: for runs on OTHER machines/devices prefer the portable bundle
(`python3 conv_bench/make_bundle.py` -> conv_probe_bundle/run_report.py). That version is
standalone (no MNNConvert/numpy) AND has the thermal safeguards this one lacks: interleaved A/B
comparisons, cooldowns between sections, and an end-of-run clock re-check. This device throttles
from 980 MHz to ~400-600 MHz under prolonged load, which silently inflates later sections.

Runs the full matrix (HW caps, clock stability, subgroup capability, all kernel variants, LDS tile
sweep, im2col+GEMM, fused 2-layer megakernel, block baselines + PReLU fusion, concurrency, and
correctness gates), compares every number against the Xclipse-960 reference, and writes a
self-contained report you can hand back for analysis.

    python3 conv_bench/probe_device.py [adb_serial] [--quick] [-o report.md]

Requires the `opencl-conv-specialize` build (its libMNN_CL carries the env switches):
  build_android_profile/{ModuleBasic.out, OFF/arm64-v8a/libMNN.so, .../libMNN_CL.so,
                         express/OFF/arm64-v8a/libMNN_Express.so}   and build_host/MNNConvert
Env switches used: MNN_DUMP_CL_EXT, MNN_SUBGROUP_PROBE, MNN_CONV_SPEC, MNN_CONV_FORCE,
                   MNN_CONV_LDS, MNN_LDS_TILE, MNN_CONV_IM2COL, MNN_CONV_FUSED2, MNN_NO_WINOGRAD
Runtime: ~8-12 min (--quick: ~4 min).
"""
import json, os, re, subprocess, sys, statistics, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
import numpy as np
from gen_conv import make_conv
from mk_chain import make_chain
from bench import convert, LIBS, MODULE

DEV = "/data/local/tmp/mnnopt"
WORK = REPO / "conv_bench" / "probework"; (WORK / "tdir").mkdir(parents=True, exist_ok=True)
SERIAL = None
QUICK = "--quick" in sys.argv
REPS = 1 if QUICK else 3
CORES = [(32, 72, 96), (48, 36, 48)]          # (C, H, W) homogeneous cores of Block1 / Block2
VARIANTS = ["conv_2d_c4h1w1", "conv_2d_c4h1w2", "conv_2d_c4h4w1", "conv_2d_c4h1w4",
            "conv_2d_c8h2w1", "conv_2d_c8h4w1", "conv_2d_c8h1w4", "conv_2d_c8h8w1",
            "conv_2d_c8h4w1_pa", "conv_2d_c8h1w1", "conv_2d_c4h8w1"]
SPEC_ONLY = {"conv_2d_c8h8w1", "conv_2d_c8h4w1_pa", "conv_2d_c8h1w1", "conv_2d_c4h8w1"}   # need MNN_CONV_SPEC to be in the candidate list
LDS_TILES = ["16x4", "48x4", "16x12"] if QUICK else ["16x4", "8x4", "24x4", "48x4", "16x2", "16x12"]

# ---- Xclipse 960 reference (measured; see FINDINGS §H) ----
REF = {
    "hw": {"name": "Samsung Xclipse 960", "cu": 8, "clk": 980, "lds": 65536,
           "shuffle": False, "max_wg": 1024},
    "clock_pinned": "980 MHz, 23/25 samples (4% worst dip); table 226-980",
    "subgroup": {"broadcast": "OK", "shuffle": "FAIL (no sub_group_shuffle in Clspv)"},
    # per-conv us, from this same script on Xclipse 960
    "variants": {
        (32, 72, 96): {"conv_2d_c4h1w1": 211.8, "conv_2d_c4h1w2": 207.5, "conv_2d_c8h4w1": 120.3,
                       "conv_2d_c8h8w1": 151.8, "conv_2d_c8h4w1_pa": 132.5, "LDS": 214.0},
        (48, 36, 48): {"conv_2d_c4h1w1": 130.3, "conv_2d_c4h1w2": 101.7, "conv_2d_c8h4w1": 115.7,
                       "conv_2d_c8h8w1": 146.2, "conv_2d_c8h4w1_pa": 105.3, "LDS": 142.8},
    },
    "im2col": {(32, 72, 96): 94, (48, 36, 48): 37},
    "gemm":   {(32, 72, 96): 139, (48, 36, 48): 80},
    "direct": {(32, 72, 96): 119.2, (48, 36, 48): 101.7},
    "fused2": {(32, 72, 96): 1570, (48, 36, 48): 1245},
    "blocks": {"Block1": (1540, 1405), "Block2": (1143, 1054)},   # (unfused, prelu-fused) total kernel us
    "concurrency": "1.04-1.34x (spare capacity)",
}

R = []           # report lines
def say(s=""):
    print(s, flush=True); R.append(s)


def adb(cmd):
    s = f"-s {SERIAL} " if SERIAL else ""
    return subprocess.run(f"adb {s}{cmd}", shell=True, text=True, capture_output=True)


def pick_serial():
    global SERIAL
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        SERIAL = args[0]; return
    out = subprocess.run("adb devices", shell=True, text=True, capture_output=True).stdout
    devs = [l.split()[0] for l in out.splitlines()[1:] if "\tdevice" in l]
    if not devs:
        print("No adb device found. Plug in, then `adb kill-server && adb start-server`."); sys.exit(1)
    SERIAL = devs[0]


def stage():
    adb(f"shell mkdir -p {DEV}/tdir")
    for L in LIBS + [MODULE]:
        adb(f"push {L} {DEV}/")
    adb("shell svc power stayon usb")


def run(model, in_shape, loops, env="", cache="p.bin", pull_out=False):
    (WORK / "tdir" / "input.json").write_text(json.dumps({
        "inputs": [{"name": "input", "shape": in_shape}], "outputs": ["output"],
        "shapeMutable": False}))
    base = os.path.basename(model)
    adb(f"push {model} {DEV}/{base}")
    adb(f"push {WORK/'tdir'/'input.json'} {DEV}/tdir/input.json")
    if (WORK / "tdir" / "input.txt").exists():
        adb(f"push {WORK/'tdir'/'input.txt'} {DEV}/tdir/input.txt")
    if pull_out:
        adb(f"shell rm -rf {DEV}/output")
        adb(f"shell mkdir -p {DEV}/output")
    r = adb(f"shell 'cd {DEV} && {env} LD_LIBRARY_PATH=. ./ModuleBasic.out {base} tdir 0 3 "
            f"{loops} 68 2 {cache} 2>&1'")
    out = r.stdout + r.stderr
    arr = None
    if pull_out:
        p = WORK / "o.txt"
        if p.exists(): p.unlink()
        adb(f"pull {DEV}/output/0_0.txt {p}")
        if p.exists():
            try: arr = np.loadtxt(p, dtype=np.float32)
            except Exception: arr = None
    return out, arr


def conv_us(out, depth=1):
    c = [int(t) for t in re.findall(r"conv time = (\d+) us", out)][3:]
    return (statistics.median(c) / depth) if c else 0.0


def med_runs(fn, n=None):
    vals = [v for v in (fn(i) for i in range(n or REPS)) if v]
    return statistics.median(vals) if vals else 0.0


def cos_of(a, b):
    if a is None or b is None: return float("nan")
    n = min(a.size, b.size); x, y = a.reshape(-1)[:n], b.reshape(-1)[:n]
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def d(v, ref):
    """format delta vs reference"""
    if not v or not ref: return "n/a"
    return f"{100*(v-ref)/ref:+.0f}%"


# ---------------------------------------------------------------- sections
def sec_hw():
    say("## 1. Hardware & capabilities\n")
    m = adb("shell getprop ro.product.model").stdout.strip()
    soc = adb("shell getprop ro.soc.model").stdout.strip()
    plat = adb("shell getprop ro.board.platform").stdout.strip()
    say(f"- adb serial: `{SERIAL}`  |  model: **{m}**  |  soc: `{soc}`  |  platform: `{plat}`\n")
    mnn = str(WORK / "core_32.mnn")
    out, _ = run(mnn, [1, 32, 72, 96], 1, env="MNN_DUMP_CL_EXT=1", cache="hw.bin")
    hw, exts, shuffle = {}, "", None
    for ln in out.splitlines():
        if ln.startswith("[HWINFO]"):
            kv = ln.replace("[HWINFO]", "").strip()
            for part in kv.split():
                if "=" in part:
                    k, v = part.split("=", 1); hw.setdefault(k, v)
            if "name=" in kv: hw["name"] = kv.split("name=", 1)[1].strip()
        if ln.startswith("[CL_EXT]"):
            if "has_subgroup_shuffle" in ln: shuffle = ln.strip().endswith("=1")
            elif not exts: exts = ln.replace("[CL_EXT]", "").strip()
    say("| property | THIS DEVICE | Xclipse 960 ref |")
    say("|---|---|---|")
    say(f"| GPU name | {hw.get('name','?')} | {REF['hw']['name']} |")
    say(f"| **compute units** | **{hw.get('max_compute_units','?')}** | {REF['hw']['cu']} |")
    say(f"| max clock MHz | {hw.get('max_clock_mhz','?')} | {REF['hw']['clk']} |")
    say(f"| local mem (LDS) | {hw.get('local_mem_bytes','?')} | {REF['hw']['lds']} |")
    say(f"| max workgroup | {hw.get('max_work_group_size','?')} | {REF['hw']['max_wg']} |")
    say(f"| pref/native vec half | {hw.get('pref_vec_half','?')}/{hw.get('native_vec_half','?')} | 8/2 |")
    say(f"| **subgroup_shuffle** | **{shuffle}** | False (blocked) |")
    say(f"| driver | {hw.get('driver_version','?')} | 25.x Samsung |")
    say(f"\n<details><summary>CL extensions</summary>\n\n```\n{exts}\n```\n</details>\n")
    return hw, shuffle


def sec_clock():
    say("## 2. Clock stability (validates every timing below)\n")
    tbl = adb("shell cat /sys/kernel/gpu/gpu_freq_table").stdout.strip()
    mnn = os.path.basename(str(WORK / "core_32.mnn"))
    adb(f"push {WORK/'core_32.mnn'} {DEV}/{mnn}")          # self-contained: don't rely on §1
    (WORK / "tdir" / "input.json").write_text(json.dumps({
        "inputs": [{"name": "input", "shape": [1, 32, 72, 96]}], "outputs": ["output"],
        "shapeMutable": False}))
    adb(f"push {WORK/'tdir'/'input.json'} {DEV}/tdir/input.json")
    # loops must outlast the ~6s sampling window, else the run ends and the GPU power-gates to 0
    r = adb(f"shell 'cd {DEV} && (LD_LIBRARY_PATH=. ./ModuleBasic.out {mnn} tdir 0 3 4000 68 2 "
            f"clk.bin >/dev/null 2>&1 &) ; sleep 1; for i in $(seq 1 20); do "
            f"printf \"%s \" $(cat /sys/kernel/gpu/gpu_clock); sleep 0.3; done; "
            f"pkill -f ModuleBasic.out >/dev/null 2>&1; true'")
    # drop 0 readings: GPU is power-gated when idle (run not yet started / already finished),
    # which is not a downclock and would corrupt the "worst dip" stat.
    samples = [v for v in (int(x) // 1000 for x in r.stdout.split() if x.isdigit()) if v > 0]
    if samples:
        mx = max(samples); pin = sum(1 for s in samples if s == mx)
        say(f"- sampled during sustained load: **{pin}/{len(samples)} at {mx} MHz** "
            f"(min {min(samples)}, worst dip {100*(mx-min(samples))/mx:.0f}%)")
        say(f"- freq table: `{tbl}`")
        say(f"- Xclipse 960 ref: {REF['clock_pinned']}")
        say(f"- **{'STABLE — timings below are trustworthy' if pin >= 0.8*len(samples) else 'UNSTABLE — treat timings with caution!'}**\n")
    else:
        say("- (could not read /sys/kernel/gpu/gpu_clock; timings unvalidated)\n")


def sec_subgroup():
    say("## 3. Subgroup capability (gates the shuffle/implicit-GEMM lever)\n")
    out, _ = run(str(WORK / "core_32.mnn"), [1, 32, 72, 96], 1,
                 env="MNN_SUBGROUP_PROBE=1", cache="sg.bin")
    res = {}
    for ln in out.splitlines():
        if "[SGPROBE]" in ln and "build:" in ln:
            say(f"- `{ln.strip()}`")
            if "broadcast" in ln: res["broadcast"] = "OK" in ln
            if "shuffle" in ln: res["shuffle"] = "OK" in ln
    say(f"\n- Xclipse 960 ref: broadcast {REF['subgroup']['broadcast']}, shuffle {REF['subgroup']['shuffle']}")
    if res.get("shuffle"):
        say("\n> 🔥 **`sub_group_shuffle` COMPILES HERE** — the register-level halo-gather / implicit-GEMM\n"
            "> conv that was impossible on the Xclipse 960 is implementable on this device **in OpenCL**.\n")
    say("")
    return res


def sec_variants(models):
    say("## 4. Kernel variants (per-conv us, sustained 6-deep chain)\n")
    say("| shape | " + " | ".join(v.replace("conv_2d_", "") for v in VARIANTS) + " | LDS |")
    say("|" + "---|" * (len(VARIANTS) + 2))
    got = {}
    for (C, H, W) in CORES:
        mnn = models[(C, H, W)]; shp = [1, C, H, W]; row = {}
        for v in VARIANTS:
            env = ("MNN_CONV_SPEC=1 " if v in SPEC_ONLY else "") + f"MNN_CONV_FORCE={v}"
            row[v] = med_runs(lambda i, v=v, e=env: conv_us(
                run(mnn, shp, 120, env=e, cache=f"f{C}{v[-5:]}{i}.bin")[0], 6))
        row["LDS"] = med_runs(lambda i: conv_us(
            run(mnn, shp, 120, env="MNN_CONV_LDS=1", cache=f"l{C}{i}.bin")[0], 6))
        got[(C, H, W)] = row
        say(f"| {C}->{C}@{H}x{W} | " + " | ".join(f"{row[v]:.1f}" for v in VARIANTS) +
            f" | {row['LDS']:.1f} |")
    say("\n**vs Xclipse 960 reference:**\n")
    say("| shape | variant | this | ref | delta |")
    say("|---|---|---|---|---|")
    for k, row in got.items():
        for v, rv in REF["variants"][k].items():
            if v in row: say(f"| {k[0]}->{k[0]}@{k[1]}x{k[2]} | {v.replace('conv_2d_','')} | "
                             f"{row[v]:.1f} | {rv} | {d(row[v], rv)} |")
    for k, row in got.items():
        best = min((v for v in row if row[v]), key=lambda v: row[v])
        say(f"\n- **{k[0]}->{k[0]}@{k[1]}x{k[2]} winner: `{best}` ({row[best]:.1f}us)** "
            f"(Xclipse winner: c8h4w1 / c4h1w2)")
    say("")
    return got


def sec_lds_sweep(models):
    say("## 5. LDS tile sweep (does any launch config rescue LDS?)\n")
    say("| shape | " + " | ".join(LDS_TILES) + " | best | vs best non-LDS |")
    say("|" + "---|" * (len(LDS_TILES) + 3))
    for (C, H, W) in CORES:
        mnn = models[(C, H, W)]; shp = [1, C, H, W]; row = {}
        for t in LDS_TILES:
            row[t] = med_runs(lambda i, t=t: conv_us(
                run(mnn, shp, 120, env=f"MNN_CONV_LDS=1 MNN_LDS_TILE={t}",
                    cache=f"s{C}{t}{i}.bin")[0], 6), n=1 if QUICK else 2)
        bt = min(row, key=lambda t: row[t] if row[t] else 9e9)
        ref = REF["direct"][(C, H, W)]
        say(f"| {C}->{C}@{H}x{W} | " + " | ".join(f"{row[t]:.0f}" for t in LDS_TILES) +
            f" | **{bt} = {row[bt]:.0f}us** | {d(row[bt], ref)} |")
    say("")


def sec_im2col_gemm():
    say("## 6. im2col + GEMM (explicit) — and the implicit-GEMM headroom\n")
    say("| shape | im2col us | GEMM us | total | direct | verdict | GEMM-vs-direct |")
    say("|---|---|---|---|---|---|---|")
    for (C, H, W) in CORES:
        # im2col kernel (conv C -> C*9 hijacked), winograd bypassed
        p = WORK / f"ic_{C}"; make_conv(str(p) + ".onnx", 1, C, C * 9, H, W, 3, 3, 1, 1, 1, 1, "none")
        convert(str(p) + ".onnx", str(p) + ".mnn")
        im = med_runs(lambda i: conv_us(run(str(p) + ".mnn", [1, C, H, W], 200,
                      env="MNN_NO_WINOGRAD=1 MNN_CONV_IM2COL=1", cache=f"ic{C}{i}.bin")[0]))
        # GEMM half = 1x1 conv at im2col dims
        q = WORK / f"gp_{C}"; make_conv(str(q) + ".onnx", 1, C * 9, C, H, W, 1, 1, 1, 0, 1, 1, "none")
        convert(str(q) + ".onnx", str(q) + ".mnn")
        gm = med_runs(lambda i: conv_us(run(str(q) + ".mnn", [1, C * 9, H, W], 200,
                      cache=f"gp{C}{i}.bin")[0]))
        dr = REF["direct"][(C, H, W)]
        tot = im + gm
        say(f"| {C}->{C}@{H}x{W} | {im:.0f} (ref {REF['im2col'][(C,H,W)]}) | {gm:.0f} "
            f"(ref {REF['gemm'][(C,H,W)]}) | {tot:.0f} | {dr:.0f} | "
            f"{'FASTER' if tot < dr else 'slower'} {d(tot, dr)} | **{d(gm, dr)}** |")
    say("\n> The last column is the key number: if the **GEMM reduce alone beats direct**, an\n"
        "> *implicit* GEMM (gather in registers, no global im2col tensor) is worth building —\n"
        "> especially if subgroup_shuffle compiles (§3) or coopmat exists.\n")


def sec_fused2(models):
    say("## 7. Fused 2-layer megakernel (cross-layer fusion, intermediate in LDS)\n")
    say("| shape | fused 2-layer | 2x direct | verdict | ref (Xclipse) |")
    say("|---|---|---|---|---|")
    for (C, H, W) in CORES:
        mnn = str(WORK / f"single_{C}.mnn")
        one = med_runs(lambda i: conv_us(run(mnn, [1, C, H, W], 200, cache=f"d1{C}{i}.bin")[0]))
        fu = med_runs(lambda i: conv_us(run(mnn, [1, C, H, W], 200, env="MNN_CONV_FUSED2=1",
                                            cache=f"f2{C}{i}.bin")[0]))
        base = 2 * one
        say(f"| {C}->{C}@{H}x{W} | {fu:.0f} | {base:.0f} | "
            f"{'FASTER' if fu < base else 'SLOWER'} {d(fu, base)} | {REF['fused2'][(C,H,W)]}us "
            f"(6.5x slower) |")
    say("")


def sec_blocks():
    say("## 8. Real block baselines + PReLU fusion (deployment numbers)\n")
    try:
        from block_fixture import load_blocks, build_onnx
        blocks = load_blocks()
    except Exception as e:
        say(f"- (skipped: {e})\n"); return
    say("| block | unfused | PReLU-fused | saving | ref |")
    say("|---|---|---|---|---|")
    for name, convs in blocks.items():
        c0 = convs[0]; shp = [1, c0["cin"], c0["H"], c0["W"]]; res = {}
        for fuse in (False, True):
            p = WORK / f"{name}{'_f' if fuse else ''}"
            build_onnx(str(p) + ".onnx", convs)
            convert(str(p) + ".onnx", str(p) + ".mnn", fp16=False, fuse_prelu=fuse)
            o, _ = run(str(p) + ".mnn", shp, 120, cache=f"{name}{int(fuse)}.bin")
            t = [int(x) for x in re.findall(r"total kernel time = (\d+)  us", o)][3:]
            res[fuse] = statistics.median(t) if t else 0
        r0, r1 = REF["blocks"].get(name, (0, 0))
        say(f"| {name} | {res[False]:.0f} | {res[True]:.0f} | {d(res[True], res[False])} | "
            f"{r0}/{r1} |")
    say("")


def sec_concurrency():
    say("## 9. Concurrency (2 independent streams = spare GPU capacity?)\n")
    mnn = os.path.basename(str(WORK / "core_32.mnn"))
    MIN = re.compile(r"min= ([\d.]+) ms")
    def solo():
        o, _ = run(str(WORK / "core_32.mnn"), [1, 32, 72, 96], 200, cache="ca.bin")
        m = MIN.search(o); return float(m.group(1)) if m else None
    def duo():
        s = f"-s {SERIAL} " if SERIAL else ""
        cmd = (f"adb {s}shell 'cd {DEV} && LD_LIBRARY_PATH=. ./ModuleBasic.out {mnn} tdir 0 3 200 "
               f"68 2 %s 2>&1'")
        p1 = subprocess.Popen(cmd % "ca.bin", shell=True, text=True, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd % "cb.bin", shell=True, text=True, stdout=subprocess.PIPE)
        oa, ob = p1.communicate()[0], p2.communicate()[0]
        ma, mb = MIN.search(oa), MIN.search(ob)
        if not (ma and mb): return None
        return max(float(ma.group(1)), float(mb.group(1)))
    s = med_runs(lambda i: solo()); c = med_runs(lambda i: duo())
    if s and c:
        say(f"- solo **{s:.2f} ms**, both-done **{c:.2f} ms** → **ratio {c/s:.2f}x** "
            f"({'SPARE CAPACITY' if c < 1.6*s else 'SATURATED'})")
        say(f"- Xclipse 960 ref: {REF['concurrency']}")
        say("- (wall-clock; profiling build inflates CPU overhead — treat as a signal, not a number)\n")
    else:
        say("- (concurrency probe failed)\n")


def sec_correct(models):
    say("## 10. Correctness gates (every custom kernel vs the stock path)\n")
    C, H, W = 32, 72, 96
    p = WORK / "cc"; make_conv(str(p) + ".onnx", 1, C, C, H, W, 3, 3, 1, 1, 1, 1, "none")
    convert(str(p) + ".onnx", str(p) + ".mnn")
    rng = np.random.default_rng(17); x = rng.standard_normal([1, C, H, W]).astype(np.float32)
    np.savetxt(WORK / "tdir" / "input.txt", x.reshape(-1), fmt="%.6f")
    _, cpu = run(str(p) + ".mnn", [1, C, H, W], 1, cache="cc0.bin", pull_out=True)  # fwd 3 stock
    say("| kernel | cosine vs stock OpenCL | pass |")
    say("|---|---|---|")
    for label, env in [("c8h8w1", "MNN_CONV_SPEC=1 MNN_CONV_FORCE=conv_2d_c8h8w1"),
                       ("c8h4w1_pa", "MNN_CONV_SPEC=1 MNN_CONV_FORCE=conv_2d_c8h4w1_pa"),
                       ("LDS", "MNN_CONV_LDS=1")]:
        _, a = run(str(p) + ".mnn", [1, C, H, W], 1, env=env, cache="cc1.bin", pull_out=True)
        c = cos_of(cpu, a)
        say(f"| {label} | {c:.6f} | {'PASS' if c > 0.99 else 'CHECK'} |")
    # fused2 computes conv^2(x), so it must be checked against a numpy conv-of-conv reference,
    # NOT against the stock single conv (that comparison is mathematically meaningless).
    try:
        import onnx
        from onnx import numpy_helper
        m = onnx.load(str(p) + ".onnx")
        inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
        Wt = [v for v in inits.values() if v.ndim == 4][0]
        bt = [v for v in inits.values() if v.ndim == 1][0]
        def c3(xx):
            Ci, Hh, Ww = xx.shape; xp = np.pad(xx, ((0, 0), (1, 1), (1, 1)))
            y = np.tile(bt[:, None, None], (1, Hh, Ww)).astype(np.float64)
            for kh in range(3):
                for kw in range(3):
                    y += np.einsum('oc,chw->ohw', Wt[:, :, kh, kw].astype(np.float64),
                                   xp[:, kh:kh + Hh, kw:kw + Ww].astype(np.float64))
            return y
        ref2 = c3(c3(x[0])).reshape(-1)
        _, a = run(str(p) + ".mnn", [1, C, H, W], 1, env="MNN_CONV_FUSED2=1",
                   cache="cc2.bin", pull_out=True)
        c = cos_of(ref2.astype(np.float32), a)
        say(f"| fused2 (vs numpy conv^2) | {c:.6f} | {'PASS' if c > 0.99 else 'CHECK'} |")
    except Exception as e:
        say(f"| fused2 | (check failed: {e}) | - |")
    (WORK / "tdir" / "input.txt").unlink(missing_ok=True)
    say("")


def sec_flags(hw, shuffle, variants):
    say("## 11. AUTO-FLAGS — what differs from Xclipse 960 (read this first)\n")
    flags = []
    cu = int(hw.get("max_compute_units", 0) or 0)
    if cu > REF["hw"]["cu"]:
        flags.append(f"🔥 **{cu} compute units vs {REF['hw']['cu']}** ({cu/REF['hw']['cu']:.1f}x) — "
                     "more occupancy headroom: bigger-tile variants (c8h8w1) and fusion may stop "
                     "regressing; re-test the levers that lost purely on occupancy.")
    elif cu and cu < REF["hw"]["cu"]:
        flags.append(f"⚠️ fewer CUs ({cu}) — expect the occupancy wall to be *worse*.")
    if shuffle:
        flags.append("🔥 **subgroup_shuffle available** — build the implicit-GEMM conv "
                     "(register-level halo gather, no LDS/barriers). This was THE blocked lever.")
    for k, row in variants.items():
        if not row: continue
        best = min((v for v in row if row[v]), key=lambda v: row[v])
        refwin = "conv_2d_c8h4w1" if k == (32, 72, 96) else "conv_2d_c4h1w2"
        if best != refwin:
            flags.append(f"📌 {k[0]}->{k[0]}@{k[1]}x{k[2]}: winner is **{best}**, not {refwin} — "
                         "the occupancy sweet spot moved; re-run the blocking search.")
        rv = REF["variants"][k].get(best if best in REF["variants"][k] else refwin)
        if rv and row.get(refwin) and row[refwin] < 0.8 * rv:
            flags.append(f"📌 {k[0]}->{k[0]}@{k[1]}x{k[2]}: baseline is {d(row[refwin], rv)} vs ref — "
                         "device is materially faster; absolute targets shift.")
    if not flags:
        flags.append("No material differences from the Xclipse 960 — the same conclusions apply "
                     "(direct autotuned path is at its occupancy ceiling; levers live outside the kernel).")
    for f in flags: say(f"- {f}")
    say("")


def main():
    pick_serial()
    out_path = Path(sys.argv[sys.argv.index("-o") + 1]) if "-o" in sys.argv else \
        REPO / "conv_bench" / f"device_report_{SERIAL}.md"
    say(f"# Conv optimization — device report")
    say(f"\n_generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `conv_bench/probe_device.py`"
        f"{' (--quick)' if QUICK else ''}; reference = Samsung Xclipse 960 / Exynos 2600_\n")
    stage()
    # build the models everything reuses
    models = {}
    for (C, H, W) in CORES:
        a = WORK / f"core_{C}"; make_chain(str(a) + ".onnx", 1, C, H, W, depth=6, act="prelu", k=3, stride=1)
        convert(str(a) + ".onnx", str(a) + ".mnn", fp16=False, fuse_prelu=True)
        models[(C, H, W)] = str(a) + ".mnn"
        b = WORK / f"single_{C}"; make_conv(str(b) + ".onnx", 1, C, C, H, W, 3, 3, 1, 1, 1, 1, "none")
        convert(str(b) + ".onnx", str(b) + ".mnn")
    steps = [("hw", None), ("clock", None), ("subgroup", None), ("variants", None), ("lds", None),
             ("im2col", None), ("fused2", None), ("blocks", None), ("conc", None), ("correct", None)]
    hw, shuffle, variants = {}, None, {}
    for name, _ in steps:
        try:
            if name == "hw":        hw, shuffle = sec_hw()
            elif name == "clock":   sec_clock()
            elif name == "subgroup":
                sg = sec_subgroup(); shuffle = sg.get("shuffle", shuffle)
            elif name == "variants": variants = sec_variants(models)
            elif name == "lds":     sec_lds_sweep(models)
            elif name == "im2col":  sec_im2col_gemm()
            elif name == "fused2":  sec_fused2(models)
            elif name == "blocks":  sec_blocks()
            elif name == "conc":    sec_concurrency()
            elif name == "correct": sec_correct(models)
        except Exception as e:
            say(f"\n> ⚠️ section `{name}` failed: `{type(e).__name__}: {e}`\n")
    try: sec_flags(hw, shuffle, variants)
    except Exception as e: say(f"> flags failed: {e}")
    say("---\n_Context for whoever reads this: full prior findings in "
        "`conv_bench/OPTIMIZATION_HANDOFF.md`; experiment log in `FINDINGS.md` §H._")
    out_path.write_text("\n".join(R))
    print(f"\n=== wrote {out_path} ===")


if __name__ == "__main__":
    main()
