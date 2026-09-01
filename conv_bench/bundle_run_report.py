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
import argparse, hashlib, json, math, os, re, subprocess, statistics, sys, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN, MODELS, REFD = HERE / "bin", HERE / "models", HERE / "ref"
DEV = "/data/local/tmp/convprobe"
# Persistent, content-addressed autotune caches. See cache_name() for why this exists.
TUNE = DEV + "/tune"
R = []

# Set by main()'s sec() marker; consulted by say(), run_model(), cooldown(), sample_clock() and D.
_SKIP = {"on": False}


class ResultDict(dict):
    """The .json payload, but writes are DROPPED while a section is skipped.

    A skipped section still executes its Python -- run_model() returns empty, the parsers return
    0.0, and the section then stores those zeros. That produced a .json full of `0.0` entries
    indistinguishable from real measurements, and under --resume the skipped sections would
    OVERWRITE the very data resume had just loaded. Guarding the container is the only place that
    catches every write path without touching 20 section bodies.
    """
    def __setitem__(self, k, v):
        if _SKIP["on"]: return
        super().__setitem__(k, v)

    def setdefault(self, k, default=None):
        # the caller usually subscripts the RESULT, e.g. D.setdefault("blocks", {})[key] = row --
        # so while skipping, hand back a throwaway dict that is not attached to D.
        if _SKIP["on"]:
            return type(default)() if isinstance(default, (dict, list)) else default
        return super().setdefault(k, default)

    def update(self, *a, **kw):
        if _SKIP["on"]: return
        super().update(*a, **kw)

    def __missing__(self, k):
        # Sections commonly write then immediately read back: `D["hw"] = {...}` followed by
        # `D["hw"]["x"] = y`. With the write dropped, that read would KeyError and kill the run.
        # While skipping, hand back a throwaway dict so the chained write lands nowhere.
        if _SKIP["on"]: return {}
        raise KeyError(k)


D = ResultDict()   # machine-readable results, written next to the report as .json

# Every arm that FORCES a specific direct conv kernel must disable Winograd first. MNN picks the
# Winograd path before MNN_CONV_FORCE is ever consulted, so without this the forced arm silently
# runs the SAME Winograd kernels as the baseline and reports the no-flag value -- which reads as
# "this kernel is exactly as fast as the default" instead of "the flag did nothing" (trap 4).
# This became load-bearing once the §19 gate started admitting Winograd on the mid-size cores.
NOWG = "MNN_NO_WINOGRAD=1 "


def reset():
    """Allow a driver script (run_suite.py) to call main() more than once."""
    R.clear(); D.clear()


def say(s=""):
    if _SKIP["on"]: return
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


# --------------------------------------------------------------- autotune cache
def push_binaries(d, bin_dir, manifest=None):
    """Push the bundle's binaries, and DROP the tuning cache if they changed.

    The cache holds programs compiled by a specific libMNN_CL.so. Every script pushes the bundle's
    binaries on each run, but nothing used to invalidate the cache, so rebuilding the library and
    re-running silently reused programs built by the previous one -- the run succeeds, the numbers
    look reasonable, and you are measuring a mix of two builds. Clearing it by hand worked only
    while someone remembered to.

    The fingerprint covers the binaries and the manifest, so a rebuilt library and a rebuilt set of
    models both invalidate it. Returns True when the cache was dropped."""
    import hashlib
    h = hashlib.sha1()
    for f in sorted(Path(bin_dir).iterdir()):
        h.update(f.name.encode()); h.update(f.read_bytes())
    if manifest and Path(manifest).exists():
        h.update(Path(manifest).read_bytes())
    fp = h.hexdigest()[:16]

    d.shell(f"mkdir -p {DEV}/tdir; mkdir -p {TUNE}")
    seen = d.shell(f"cat {DEV}/.binaries 2>/dev/null").strip()
    dropped = False
    if seen != fp:
        n = d.shell(f"ls {TUNE}/*.bin 2>/dev/null | wc -l").strip()
        d.shell(f"rm -f {TUNE}/*.bin")
        dropped = True
        if seen:
            print(f"   binaries changed ({seen} -> {fp}); dropped {n} stale cache entries",
                  flush=True)
    for f in sorted(Path(bin_dir).iterdir()):
        d.push(f, f"{DEV}/")
    d.shell(f"chmod +x {DEV}/ModuleBasic.out")
    d.shell(f"echo {fp} > {DEV}/.binaries")
    return dropped


def cache_name(model, shape, env, mode, ftype):
    """Content-addressed name for MNN's autotune cache: hash of everything that can change which
    kernel is selected.

    Measured cost of one harness call: 3.5s with a cold cache, 0.6s with a warm one -- 85% of every
    measurement was cache-cold startup, and the GPU work itself is ~0.2s. The harness used to pass a
    unique filename per arm AND PER REPEAT (cache=f"...{i}.bin"), so every one of the ~1400 launches
    in a sweep paid the cold price and left the cache behind unused (918 MB of write-once files).

    That per-repeat uniqueness was guarding a real bug: sharing ONE cache across DIFFERENT forced
    kernels makes some runs emit no output, and a correct kernel then reports as a correctness
    failure. But the guard was too broad. The cache only has to be unique per CONFIGURATION -- and
    `env` is in the key here, so two different MNN_CONV_FORCE values still get different files. The
    repeat index is not a configuration, so it is not in the key, and repeats 2..N run warm.

    `loops` is deliberately NOT in the key: it does not affect kernel selection, so a 2-loop
    correctness run warms the cache for the 120-loop timing run of the same configuration.
    Env tokens are sorted so "A=1 B=2" and "B=2 A=1" share one cache."""
    key = json.dumps([model, list(shape), " ".join(sorted(env.split())), mode, ftype])
    return f"tune/{hashlib.sha1(key.encode()).hexdigest()[:16]}.bin"


# --------------------------------------------------------------- device run helpers
def run_model(d, model, shape, loops, env="", cache="p.bin", pull=False, timeout=600,
              mode=68, ftype=3, isolate_cache=False):
    """Push model+input.json, run ModuleBasic, return (stdout, output_floats_or_None)."""
    if _SKIP["on"]:
        return "", None          # section not selected: no device work at all
    inj = HERE / "_input.json"
    inj.write_text(json.dumps({"inputs": [{"name": "input", "shape": shape}],
                               "outputs": ["output"], "shapeMutable": False}))
    d.push(inj, f"{DEV}/tdir/input.json")
    d.push(MODELS / model, f"{DEV}/{model}")
    # Callers historically passed a unique per-repeat filename; that is what made every launch
    # cold. Their name is now ignored unless isolate_cache=True, and the cache is addressed by
    # configuration instead. Pass isolate_cache=True only when a test needs a guaranteed-cold run.
    if not isolate_cache:
        cache = cache_name(model, shape, env, mode, ftype)
    if pull:
        d.shell(f"rm -rf {DEV}/output"); d.shell(f"mkdir -p {DEV}/output")
    out = d.shell(f"cd {DEV} && {env} LD_LIBRARY_PATH=. ./ModuleBasic.out {model} tdir 0 {ftype} "
                  f"{loops} {mode} 2 {cache} 2>&1", timeout=timeout)
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
    if not out: return 0.0
    c = [int(t) for t in re.findall(r"conv time = (\d+) us", out)][3:]
    return (statistics.median(c) / depth) if c else 0.0


def conv_all_us(out, depth=1):
    if not out: return 0.0
    """Sum of EVERY conv-related kernel per loop -- winograd transforms, layout conversions and
    im2col included. MUST be used instead of conv_us() whenever an arm can change which conv
    IMPLEMENTATION runs (MNN_FORCE_WINOGRAD / MNN_NO_WINOGRAD / MNN_CONV_LDS / MNN_CONV_SPLITK /
    MNN_CONV_NCHW / MNN_CONV_IMGEMM). MNN's `conv time` counter omits the Winograd rearrange
    kernels, which are launched TWICE per conv -- scoring a Winograd arm with it credits the
    batchgemm half only, and turned a +15% regression into an apparent -45% win (FINDINGS §H.27)."""
    acc = {}
    for m in re.finditer(r"kernel time = (\d+)\s+us (\S+)", out):
        us, name = int(m.group(1)), m.group(2)
        if "onv" not in name:
            continue
        acc.setdefault(name.split("-total")[0], []).append(us)
    loops = len(re.findall(r"total kernel time = \d+  us", out))
    if not loops:
        return 0.0
    tot = sum(statistics.median(v[3:] if len(v) > 6 else v) * len(v) for v in acc.values())
    return tot / loops / depth


def conv_kernel_only_us(out, depth=1):
    """conv_all_us minus the NCHW layout-conversion kernels (profiler tags gemm2-0 / gemm2-2).
    Used for the NCHW sections: those conversions belong at the boundary of a conv BLOCK in any
    real deployment, not on every conv, so charging them per-conv would measure a design nobody
    would ship. The excluded amount is always reported alongside."""
    acc, cvt = {}, []
    for m in re.finditer(r"kernel time = (\d+)\s+us (\S+)", out):
        us, name = int(m.group(1)), m.group(2)
        if "onv" not in name:
            continue
        if "gemm2-0" in name or "gemm2-2" in name:
            cvt.append(us); continue
        acc.setdefault(name.split("-total")[0], []).append(us)
    loops = len(re.findall(r"total kernel time = \d+  us", out))
    if not loops:
        return 0.0, 0.0
    tot = sum(statistics.median(v[3:] if len(v) > 6 else v) * len(v) for v in acc.values())
    return tot / loops / depth, (sum(cvt) / loops if cvt else 0.0)


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
    if not out: return 0.0
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
    if _SKIP["on"]:
        return [], ""      # skipped section: do not run a 20000-loop job just to sample the clock
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
    # Final sweep. The PID kill above has been observed to MISS, leaving a 20000-loop job pinned to
    # the GPU after the report exits -- which then throttles or contends with whatever runs next,
    # silently. Nothing else of ours is on the device at this point (sample_clock is only called
    # between sections), so a name-based kill is safe here and is the only reliable guarantee.
    d.shell("pkill -9 -f ModuleBasic.out 2>/dev/null; true")
    body = out.split("PID=", 1)[-1]
    body = body.split(None, 1)[1] if len(body.split(None, 1)) > 1 else ""
    # 0 = GPU power-gated (idle), not a downclock -> drop
    s = [v // 1000 for v in (int(x) for x in body.split() if x.isdigit()) if v > 0]
    return s, tbl


def cooldown(d, seconds):
    """Let the GPU cool so a later section isn't measured on a throttled clock."""
    # A skipped section performs no measurement, so it has nothing to cool for. Without this,
    # --sections still slept the full wall-clock of every section it skipped: `--sections 3` ran
    # for over 10 minutes doing nothing but 60s sleeps, which defeats the entire point.
    if _SKIP["on"]: return
    if seconds <= 0: return
    print(f"   (cooldown {seconds}s)", flush=True)
    d.shell(f"sleep {seconds}", timeout=seconds + 60)


def interleaved(fns, reps):
    """Run labelled measurements round-robin so thermal drift hits all arms equally.
    fns = {label: callable(i)->float}. Returns {label: median}.

    The arm ORDER IS ROTATED every rep. Without that, the first arm in the dict is measured first in
    every rep and gets a systematic advantage whenever the device is still warming: a run of
    section 9 on a cool device read default=119 / NO_WINOGRAD=268 for two arms that are the SAME
    code path (verified 120.0 vs 119.0 cooled). Interleaving alone protects against drift ACROSS
    reps, not WITHIN one."""
    acc = {k: [] for k in fns}
    keys = list(fns)
    for i in range(reps):
        r = i % len(keys)
        for k in keys[r:] + keys[:r]:
            v = fns[k](i)
            if v: acc[k].append(v)
    return {k: (statistics.median(v) if v else 0.0) for k, v in acc.items()}


# --------------------------------------------------------------- sections
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", "-s", help="adb serial (see --list)")
    ap.add_argument("--list", action="store_true", help="list attached devices and exit")
    ap.add_argument("--quick", action="store_true", help="fewer repeats/tiles (~4 min)")
    ap.add_argument("-o", "--out", default=None, help="report path (default report_<serial>.md)")
    ap.add_argument("--sections", default=None,
                    help="only run these sections, e.g. --sections 15,17 (default: all). Sections "
                         "not selected are skipped entirely -- no device work, no output. Use with "
                         "--resume to reuse data already collected by an earlier run.")
    ap.add_argument("--resume-from", default=None,
                    help="explicit comma-separated json file(s) to resume from (default: every "
                         "prior result json in the bundle dir, oldest first)")
    ap.add_argument("--resume", action="store_true",
                    help="load numbers from a previous run's .json so skipped sections still have "
                         "their data available for cross-section references and §20.")
    ap.add_argument("--list-sections", action="store_true", help="print the section list and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="report which sections would run/skip, touch no device, exit")
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
    WANT = None if not a.sections else {x.strip() for x in a.sections.split(",")}
    SECTIONS = {1:"Hardware", 2:"GPU clock at start", 3:"Subgroup capability",
                4:"Kernel strategies", 5:"Shape hardcoding", 6:"Stride-2 head pairs",
                7:"LDS tile sweep", 8:"im2col + GEMM", 9:"Winograd vs direct",
                10:"Fused 2-layer megakernel", 11:"Real model blocks", 12:"Concurrency",
                13:"Correctness", 14:"LDS-at-constant-blocking + split-K", 15:"NCHW layout",
                16:"im2col+GEMM in NCHW", 17:"Memory mode buffer vs IMAGE",
                18:"Winograd batchgemm tile", 19:"Winograd selection gate",
                20:"Recommendations", 21:"GPU clock at END (thermal validity)"}
    if a.list_sections:
        print("sections:")
        for k, v in SECTIONS.items(): print(f"  {k:>2}  {v}")
        return
    if a.dry_run:
        print(f"selection: {a.sections or 'ALL'}")
        for k, v in SECTIONS.items():
            mark = "RUN " if (WANT is None or str(k) in WANT) else "skip"
            print(f"  [{mark}] {k:>2}  {v}")
        print("\n(dry run: no device was touched)")
        return

    STATE = {"sec": "0", "skip": False}
    def sec(n):
        """Mark the start of section n. Everything after this is skipped unless n is selected."""
        STATE["sec"] = str(n)
        STATE["skip"] = (WANT is not None and str(n) not in WANT)
        _SKIP["on"] = STATE["skip"]
        if STATE["skip"]:
            print(f"   (section {n}: skipped)", flush=True)
        return not STATE["skip"]

    # --resume: reuse an earlier run's numbers so a partial re-run still has cross-section data
    # (§7 and §20 read §4's baseline; §20 reads almost everything).
    d = Dev(serial)
    reps = 1 if a.quick else 3
    # 20s was not enough (a 3.5h run fell 980 -> 747 MHz, invalidating every cross-section
    # absolute); 60s made a full run ~9h. 30s is the compromise -- verify against §21's end-of-run
    # clock rather than assuming it holds.
    cool_s = a.cooldown if a.cooldown is not None else (5 if a.quick else 30)
    man = json.loads((HERE / "manifest.json").read_text())
    # The bundle carries pre-converted .mnn files, so which shape family it was built from is
    # NOT recoverable from the numbers. Print it, and fail loudly on a bundle predating the field.
    SHAPE_FAMILY = man.get("shape_family")
    if not SHAPE_FAMILY:
        print("!! manifest has no 'shape_family' -- rebuild the bundle with make_bundle.py")
        SHAPE_FAMILY = "unknown"
    # ---- section selection ------------------------------------------------------------------
    # The report is one long linear function, so instead of re-indenting 20 blocks (and risking a
    # silent behaviour change in a 5h run) an unselected section is neutralised at the EDGES: say()
    # suppresses output and run_model() returns immediately without touching the device. The
    # section's Python still executes, but it does no work and emits nothing.
    if a.resume:
        # Chunked runs each write their own <out>.json, so resume must MERGE every prior result,
        # not read one file. It globbed only detail_*.json before -- which our --out runs never
        # produce, so resume silently restored nothing (and would happily have loaded a stray
        # hand-made test file). Oldest first, so newer sections win on key collisions.
        cands = [p for p in sorted(HERE.glob("*.json"), key=lambda x: x.stat().st_mtime)
                 if p.name not in ("manifest.json", "_input.json", "preflight_result.json")]
        if a.resume_from:
            cands = [Path(x) for x in a.resume_from.split(",")]
        n = 0
        for c in cands:
            try:
                D.update(json.loads(c.read_text())); n += 1
            except Exception as e:
                print(f"   (--resume: skipping {c.name}: {e})")
        if n:
            print(f"   (resumed {len(D)} keys from {n} file(s): "
                  f"{', '.join(c.name for c in cands)})")
        else:
            print("   (--resume: no previous result json found; nothing to resume)")

    cores = man["cores"]           # [{key,label,C,H,W}]
    # Cells that preflight proved are not measuring what their column header claims (a flag that a
    # path guard rejected, or a comparison confounded by algorithm). They are rendered `invalid`
    # rather than printed as numbers -- a plausible number in the wrong column is the failure mode
    # this whole audit exists to stop.
    INVALID = {}
    _pf = HERE / "preflight_result.json"
    if not _pf.exists(): _pf = HERE.parent / "preflight_result.json"
    if _pf.exists():
        for _c in json.loads(_pf.read_text()).get("invalid_cells", []):
            for _sec in str(_c["section"]).split(","):
                INVALID.setdefault(_sec.strip(), {})[_c["shape"]] = _c["why"]
        _known = {c["key"] for c in man["cores"]} | {h["key"] for h in man.get("heads", [])} \
                 | {b["key"] for b in man.get("blocks", [])}
        _cells = {k for v in INVALID.values() for k in v}
        _orphan = _cells - _known
        print(f"   (preflight: {sum(len(v) for v in INVALID.values())} invalid cell(s) will be masked)")
        if _orphan:
            # Silent no-op guard: this exact mismatch (preflight said "core32@72x96", the manifest
            # says "32_72x96") printed "1 cell will be masked" and then masked nothing, so a
            # confounded number was published as if valid.
            print(f"   !! WARNING: {len(_orphan)} preflight cell(s) match NO case in the manifest "
                  f"and will mask NOTHING: {sorted(_orphan)}")
            print(f"      known keys: {sorted(_known)}")
    else:
        print("   (no preflight_result.json -- cells cannot be validated; run preflight.py first)")
    def invalid(sec, key):
        """sec may be a single section ("17") or several ("4,7,14") -- preflight stores each
        section separately, so a multi-section lookup must check each one, not the joined string."""
        for one in str(sec).split(","):
            why = INVALID.get(one.strip(), {}).get(key)
            if why: return why
        return None
    variants = man["variants"]     # [kernel names]
    spec_only = set(man["spec_only"])
    stride1_only = set(man.get("stride1_only", []))
    hard_capable = list(man.get("hard_capable", []))
    tiles = man["lds_tiles"][:3] if a.quick else man["lds_tiles"]
    out_path = Path(a.out) if a.out else HERE / f"report_{serial}.md"

    # ---- stage
    print(f"staging to {serial} ...")
    d.shell(f"mkdir -p {DEV}/tdir")
    # Persistent across runs on purpose: a warm cache is the difference between a 3.5s and
    # a 0.6s launch, and it stays valid as long as the .so and the models do not change.
    d.shell(f"mkdir -p {TUNE}")
    for f in sorted(BIN.iterdir()):
        d.push(f, f"{DEV}/")
    d.shell("svc power stayon usb")

    say("# Conv strategy report")
    say(f"\n_device `{serial}` — {d.prop('ro.product.model')} / soc `{d.prop('ro.soc.model')}` — "
        f"generated {datetime.datetime.now():%Y-%m-%d %H:%M}{' (quick)' if a.quick else ''}_\n")
    D["shape_family"] = SHAPE_FAMILY
    say(f"> **Shape family: `{SHAPE_FAMILY}`** — "
        + ("every conv's input spatial size is the original model's divided by 3 "
           "(`[1,C,H,W]` -> `[1,C,H/3,W/3]`); channels, kernel, stride and pad are unchanged, "
           "so MACs are exactly 1/9 of the original set. Results here must NOT be assumed to "
           "carry over from the full-size runs and vice versa.\n"
           if SHAPE_FAMILY == "reduced" else
           f"shapes as recorded in the manifest.\n"))
    say("> Goal: find the **best conv configuration on this device**. All timings are per-conv GPU\n"
        "> kernel time (median of repeats) under sustained load, OpenCL buffer mode, fp16.\n")

    summary = {}

    # ---- 1. hardware
    sec(1)
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
              "max_work_group_size", "pref_vec_half", "native_vec_half", "driver_version",
              "image2d_max_width", "image2d_max_height"):
        if k in hw: say(f"- **{k}**: `{hw[k]}`")
    say(f"- **subgroup_shuffle**: `{shuffle}`")
    D["hw"] = dict(hw); D["hw"]["subgroup_shuffle"] = shuffle
    # numeric, because §17 gates every image-mode shape against these (a tensor over the limit can
    # REBOOT the device, not merely fail to allocate)
    for k in ("image2d_max_width", "image2d_max_height"):
        try: D["hw"][k] = int(hw.get(k, 0))
        except ValueError: D["hw"][k] = 0
    if exts: say(f"\n<details><summary>CL extensions</summary>\n\n```\n{exts}\n```\n</details>\n")

    # ---- 2. clock at the START (re-checked at the end; see §12)
    sec(2)
    say("## 2. GPU clock at start of run\n")
    s, tbl = sample_clock(d, cores[0]["model"], cores[0]["shape"])
    clk_start = int(statistics.median(s)) if s else 0
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
    sec(3)
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
    sec(4)
    say("## 4. Kernel strategies (per-conv µs, lower is better)\n")
    say("> **MNN default** is whatever MNN picks for this shape, which after the §19 gate may be\n"
        "> Winograd. Every custom-kernel column runs with `MNN_NO_WINOGRAD=1` so the forced kernel\n"
        "> actually engages -- otherwise it silently re-measures the default. All columns use\n"
        "> `conv_all_us` (every conv kernel, transforms included), so a Winograd default and a\n"
        "> direct custom kernel are charged on the same basis.\n")
    say("> **Baseline parity:** `MNN default` is measured with `MNN_NO_WINOGRAD=1`, the same\n"
        "> constraint every variant arm carries, so this table compares KERNELS. `(free choice)` is\n"
        "> MNN picking any algorithm including Winograd -- compare that against §19, not against\n"
        "> the variants.\n")
    say("| shape | " + " | ".join(["MNN default", "(free choice)"]
                                  + [v.replace("conv_2d_", "") for v in variants] + ["LDS"]) + " |")
    say("|" + "---|" * (len(variants) + 4))
    allrows = {}
    for c in cores:
        m, shp, dep = c["model"], c["shape"], c["depth"]
        # INTERLEAVED (a,b,c,...,a,b,c,...) not arm-by-arm: this device throttles, and measuring
        # 13 arms sequentially would systematically penalise whichever runs last.
        # conv_all_us, NOT conv_us: the baseline may be Winograd (§19), and conv_us omits the two
        # rearrange kernels -- on 48->48@36x48 it reads 34.5 against a true 67.0, which would make
        # the default look ~2x better than it is and bury every custom kernel (§H.27).
        # BASELINE PARITY: every variant arm carries MNN_NO_WINOGRAD=1, so the baseline must too.
        # Without it, on a Winograd shape (48->48@36x48, 96->96@18x24) the "MNN default" column was
        # Winograd while all 15 variant columns were direct convolution -- "nothing beat the
        # default" then meant "Winograd beats direct", which is a §19 question, not a kernel one.
        # Free-choice default is reported separately so the Winograd advantage is not lost.
        fns = {"MNN default": lambda i: conv_all_us(
            run_model(d, m, shp, 120, env=NOWG, cache=f"st{c['key']}{i}.bin")[0], dep)}
        fns["(free choice)"] = lambda i: conv_all_us(
            run_model(d, m, shp, 120, cache=f"fc{c['key']}{i}.bin")[0], dep)
        for v in variants:
            env = NOWG + ("MNN_CONV_SPEC=1 " if v in spec_only else "") + f"MNN_CONV_FORCE={v}"
            fns[v] = (lambda i, e=env, v=v: conv_all_us(
                run_model(d, m, shp, 120, env=e, cache=f"f{c['key']}{v[-5:]}{i}.bin")[0], dep))
        fns["LDS"] = lambda i: conv_all_us(
            run_model(d, m, shp, 120, env=NOWG + "MNN_CONV_LDS=1", cache=f"l{c['key']}{i}.bin")[0], dep)
        cooldown(d, cool_s)
        row = interleaved(fns, reps)
        allrows[c["key"]] = row
        D.setdefault("variants", {})[c["key"]] = dict(row)
        def _cell(k, c=c, row=row):
            if k == "LDS" and invalid("4,7,14", c["key"]):
                return "**invalid**"   # flag rejected by a path guard: this would be the default
            return f"{row[k]:.1f}" if row.get(k) else "-"
        say(f"| {c['label']} | " + " | ".join(
            _cell(k) for k in ["MNN default", "(free choice)"] + variants + ["LDS"]) + " |")
    say("")
    for c in cores:
        row = allrows[c["key"]]; base = row["MNN default"]
        cand = {k: v for k, v in row.items()
                if v and k not in ("MNN default", "(free choice)")}
        if not cand: continue
        best = min(cand, key=lambda k: cand[k])
        summary[c["key"]] = {"label": c["label"], "kind": "stride-1 core",
                             "base": base, "best": best, "best_us": cand[best]}
        D.setdefault("winner", {})[c["key"]] = {"default_us": base, "best": best, "best_us": cand[best], "label": c["label"]}
        verdict = f"**{pct(cand[best], base)} vs MNN default**" if cand[best] < base else \
                  f"(MNN default already best; closest custom {best} {pct(cand[best], base)})"
        say(f"- **{c['label']}** → default `{base:.1f}µs`, best strategy **`{best}` `{cand[best]:.1f}µs`** {verdict}")
    say("")

    # ---- 5. shape hardcoding
    sec(5)
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
            # every arm direct (NOWG) so HARD is isolated against a like-for-like baseline
            fns = {"MNN default": lambda i, m=m, shp=shp, c=c: conv_all_us(
                run_model(d, m, shp, 120, env=NOWG, cache=f"hd{c['key']}{i}.bin")[0], dep)}
            for v in hard_capable:
                base = NOWG + ("MNN_CONV_SPEC=1 " if v in spec_only else "") + f"MNN_CONV_FORCE={v}"
                fns[v] = (lambda i, e=base, v=v, m=m, shp=shp, c=c: conv_all_us(
                    run_model(d, m, shp, 120, env=e, cache=f"hp{c['key']}{v[-5:]}{i}.bin")[0], dep))
                fns[v + "+HARD"] = (lambda i, e="MNN_CONV_HARD=1 " + base, v=v, m=m, shp=shp, c=c: conv_all_us(
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
    sec(6)
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
                # feed §20. These winners used to be printed here and thrown away, so the
                # recommendations silently covered stride-1 cores only.
                # exclude the baseline from the CANDIDATE set: otherwise "best" can come back as
                # "MNN default" and §20 prints the nonsense "no custom kernel beat it (closest
                # `MNN default`)". §4 already does this; the head path did not.
                custom = {k: v for k, v in cand.items() if k != "MNN default"}
                if custom and row["MNN default"][t]:
                    cbest = min(custom, key=lambda k: custom[k])
                    summary[f"{h['key']}:{t}"] = {
                        "label": c["label"], "kind": "stride-2 head",
                        "base": row["MNN default"][t], "best": cbest, "best_us": custom[cbest]}
            D.setdefault("heads", {})[h["key"]] = {
                "label": h["label"],
                "convs": {c["label"]: {k: row[k][c["tag"]] for k in head_vars} for c in h["convs"]}}
        say("")

    # ---- 7. LDS tiles
    sec(7)
    say("## 7. LDS tile sweep\n")
    say("> Shapes where a path guard rejects `MNN_CONV_LDS` (no tile divides the output) are shown\n"
        "> as `invalid`: the flag is accepted on the command line but never applied, so the arm\n"
        "> silently re-measures the default (preflight §B2).\n")
    say("| shape | " + " | ".join(tiles) + " | best LDS | vs MNN default |")
    say("|" + "---|" * (len(tiles) + 3))
    for c in cores:
        m, shp, dep = c["model"], c["shape"], c["depth"]; row = {}
        for t in tiles:
            row[t] = med(lambda i, t=t: conv_all_us(run_model(d, m, shp, 120,
                         env=NOWG + f"MNN_CONV_LDS=1 MNN_LDS_TILE={t}", cache=f"t{c['key']}{t}{i}.bin")[0], dep),
                         1 if a.quick else 2)
        if invalid("4,7,14", c["key"]):
            # every tile was rejected by the path guard, so every number here is the default
            say(f"| {c['label']} | " + " | ".join("—" for _ in tiles)
                + f" | **invalid** | {invalid('4,7,14', c['key'])} |")
            continue
        bt = min(row, key=lambda t: row[t] if row[t] else 9e9)
        say(f"| {c['label']} | " + " | ".join(f"{row[t]:.0f}" for t in tiles) +
            f" | **{bt} = {row[bt]:.0f}** | {pct(row[bt], allrows[c['key']]['MNN default'])} |")
    say("")

    # ---- 6. im2col + GEMM
    sec(8)
    say("## 8. im2col + GEMM (and implicit-GEMM headroom)\n")
    say("| shape | im2col | GEMM | total | MNN default | explicit verdict | **GEMM vs default** |")
    say("|---|---|---|---|---|---|---|")
    for c in cores:
        cooldown(d, cool_s)
        # interleaved with a fresh default measurement so all three share the same thermal state
        r = interleaved({
            "im2col": lambda i, c=c: conv_all_us(run_model(d, c["im2col_model"], c["shape"], 200,
                      env=NOWG + "MNN_CONV_IM2COL=1", cache=f"ic{c['key']}{i}.bin")[0]),
            "gemm": lambda i, c=c: conv_all_us(run_model(d, c["gemm_model"], c["gemm_shape"], 200,
                    env=NOWG, cache=f"gp{c['key']}{i}.bin")[0]),
            "default": lambda i, c=c: conv_all_us(run_model(d, c["model"], c["shape"], 120,
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
    sec(9)
    say("## 9. Winograd vs direct\n")
    say("> Measured with **conv_all_us** (every conv kernel, transforms included). MNN's `conv\n"
        "> time` counter omits the Winograd rearrange passes and would flatter Winograd by ~2x\n"
        "> (FINDINGS §H.27).\n")
    say("| shape | path MNN chose | default | Winograd OFF | Winograd FORCED | verdict |")
    say("|---|---|---|---|---|---|")
    for c in cores:
        cooldown(d, cool_s)
        m, shp, dep = c["model"], c["shape"], c["depth"]
        o_def, _ = run_model(d, m, shp, 120, cache=f"wg{c['key']}p.bin")
        paths = conv_paths(o_def)
        chose = max(paths, key=lambda k: paths[k]) if any(paths.values()) else "?"
        r = interleaved({
            # conv_all_us, NOT conv_us: this arm changes the conv implementation (§H.27)
            "on": lambda i, m=m, shp=shp, c=c: conv_all_us(
                run_model(d, m, shp, 120, cache=f"wa{c['key']}{i}.bin")[0], dep),
            "off": lambda i, m=m, shp=shp, c=c: conv_all_us(
                run_model(d, m, shp, 120, env="MNN_NO_WINOGRAD=1",
                          cache=f"wb{c['key']}{i}.bin")[0], dep),
            "forced": lambda i, m=m, shp=shp, c=c: conv_all_us(
                run_model(d, m, shp, 120, env="MNN_FORCE_WINOGRAD=1",
                          cache=f"wf{c['key']}{i}.bin")[0], dep),
        }, reps)
        on, off, forced = r["on"], r["off"], r["forced"]
        D.setdefault("winograd", {})[c["key"]] = {"path": chose, "default_us": on,
                                                  "no_wino_us": off, "forced_wino_us": forced}
        if chose != "wino":
            verdict = (f"**forcing Winograd is {pct(forced, on)}**" if forced and forced < on * 0.97
                       else f"MNN picked `{chose}`; forcing Winograd {pct(forced, on)}")
        else:
            verdict = (f"Winograd wins {pct(off, on)} if disabled" if off > on
                       else f"**disabling Winograd is {pct(off, on)} FASTER**")
        say(f"| {c['label']} | `{chose}` | {on:.0f} | {off:.0f} | {forced:.0f} | {verdict} |")
    say("\n> Winograd trades multiply-adds for extra transform passes, so it wins at SMALL spatial\n"
        "> size with ENOUGH channels and loses at large spatial / few channels. On the reference\n"
        "> device MNN's selection gate (`in_w < out_c`) is mis-calibrated: forcing it on wins\n"
        "> -13% at 48->48@36x48 and -25% at 32->32@36x48, while losing at 72x96 and above\n"
        "> (FINDINGS §H.28/§H.29). It is a per-conv decision, never a per-model one.\n")

    # ---- 8. fused megakernel
    sec(10)
    say("## 10. Fused 2-layer megakernel (NC4HW4 and NCHW)\n")
    say("> Both fused kernels apply the SAME weights twice, so they fit in one conv Execution and\n"
        "> are checked against the numpy conv(conv(x)) reference in §13. Each is compared with 2x\n"
        "> the single conv IN ITS OWN LAYOUT, so fusion is the only variable. All arms run with\n"
        "> MNN_NO_WINOGRAD=1: at higher channel counts MNN selects Winograd before any MNN_CONV_*\n"
        "> flag is read, which would silently make every arm measure the same kernel.\n")
    say("| shape | 1 conv | fused2 | 2x single | verdict | NCHW 1 conv | NCHW fused2 | verdict |")
    say("|---|---|---|---|---|---|---|")
    NWG = "MNN_NO_WINOGRAD=1 "
    for c in cores:
        cooldown(d, cool_s)
        # the fused tile must divide both spatial dims
        ft = next((x for x in (6, 8, 4, 2) if c["W"] % x == 0 and c["H"] % x == 0), 0)
        arms = {
            "one": lambda i, c=c: conv_all_us(run_model(d, c["single_model"], c["shape"], 200,
                   env=NWG, cache=f"s1{c['key']}{i}.bin")[0]),
            "fused": lambda i, c=c: conv_all_us(run_model(d, c["single_model"], c["shape"], 200,
                     env=NWG + "MNN_CONV_FUSED2=1", cache=f"f2{c['key']}{i}.bin")[0]),
            "none": lambda i, c=c: conv_kernel_only_us(run_model(d, c["single_model"], c["shape"],
                    200, env=NWG + "MNN_CONV_NCHW=1", cache=f"n1{c['key']}{i}.bin")[0])[0],
        }
        if ft:
            arms["nfused"] = lambda i, c=c, ft=ft: conv_kernel_only_us(run_model(
                d, c["single_model"], c["shape"], 200,
                env=NWG + f"MNN_CONV_NCHW_FUSE2=1 MNN_FUSE_TILE={ft}",
                cache=f"nf{c['key']}{i}.bin")[0])[0]
        r = interleaved(arms, reps)
        one, fu, n1 = r["one"], r["fused"], r["none"]
        nf = r.get("nfused", 0.0)
        D.setdefault("fused2", {})[c["key"]] = {"fused_us": fu, "two_single_us": 2 * one,
                                                "nchw_one_us": n1, "nchw_fused_us": nf}
        v1 = f"{'FASTER' if fu < 2*one else 'SLOWER'} {pct(fu, 2*one)}"
        v2 = (f"{'FASTER' if nf < 2*n1 else 'SLOWER'} {pct(nf, 2*n1)}" if nf and n1
              else "n/a (tile)")
        say(f"| {c['label']} | {one:.0f} | {fu:.0f} | {2*one:.0f} | {v1} "
            f"| {n1:.0f} | {nf:.0f} | {v2} |")
    say("\n> **Why fusion costs arithmetic:** a T x T output tile needs conv1 over a (T+2)^2 halo,\n"
        "> so conv1 is recomputed ((T+2)/T)^2 times -- 1.78x at T=6. It buys back only the\n"
        "> intermediate write+re-read, which §14's depth sweep shows is nearly free on this class of\n"
        "> device. Fusion therefore wins only where inter-layer traffic is genuinely expensive.\n")

    # ---- 8. real blocks
    sec(11)
    say("## 11. Real model blocks (deployment numbers)\n")
    say("| block | plain | PReLU-fused | saving |")
    say("|---|---|---|---|")
    for b in man["blocks"]:
        cooldown(d, cool_s)
        # Bare-conv blocks have no PReLU to fold: run the single arm rather than scoring a
        # byte-identical model against itself and flagging the tautological "no saving".
        if not b.get("has_prelu", True):
            t0 = med(lambda i, b=b: total_us(run_model(d, b["model"], b["shape"], 120,
                     cache=f"{b['key']}a{i}.bin")[0]), 2 if a.quick else 3)
            D.setdefault("blocks", {})[b["key"]] = {"plain_us": t0, "prelu_fused_us": None}
            say(f"| {b['key']} | {t0:.0f} | _n/a (no PReLU)_ | — |")
            continue
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
    sec(12)
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
    sec(13)
    say("## 13. Correctness (custom kernels vs MNN default output)\n")
    # Two fixtures. The large one is the historical shape; the small one has an output plane
    # the same size as what the suite now times, so a kernel whose halo/remainder handling only
    # breaks on a short or narrow output cannot pass on the large fixture and then silently
    # corrupt every timed shape. A kernel is trusted only if it passes BOTH.
    fixtures = [("cc", man["correctness"], "cc_input.txt", "fused2_ref.txt")]
    cc2 = man.get("correctness2") or {}
    if cc2.get("model"):
        fixtures.append(("cc2", cc2, cc2.get("input", "cc2_input.txt"),
                         cc2.get("fused2_ref", "fused2_ref2.txt")))
    gates = [("LDS", "MNN_CONV_LDS=1")] + [
        (v.replace("conv_2d_", ""), f"MNN_CONV_SPEC=1 MNN_CONV_FORCE={v}") for v in sorted(spec_only)]
    say("| kernel | " + " | ".join(f"cosine ({n} {c['shape'][1]}@{c['shape'][2]}x{c['shape'][3]})"
                                   for n, c, _, _ in fixtures) + " | verdict |")
    say("|---|" + "---|" * (len(fixtures) + 1))
    cosines, fused2 = {}, {}
    for fname, cc, inp, ref in fixtures:
        d.push(REFD / inp, f"{DEV}/tdir/input.txt")
        _, base_out = run_model(d, cc["model"], cc["shape"], 1, cache=f"k0_{fname}.bin", pull=True)
        # NB: one cache file PER GATE. Reusing ONE autotune cache file across different forced
        # kernels makes some runs emit no output at all (empty pull -> cosine nan -> a correct
        # kernel is reported as FAIL). Always give each forced kernel its own cache file.
        for gi, (label, env) in enumerate(gates):
            _, v = run_model(d, cc["model"], cc["shape"], 1, env=env,
                             cache=f"k1_{fname}_{gi}.bin", pull=True)
            cosines.setdefault(label, {})[fname] = cosine(base_out, v)
        refv = [float(x) for x in (REFD / ref).read_text().split()]
        _, v = run_model(d, cc["model"], cc["shape"], 1, env="MNN_CONV_FUSED2=1",
                         cache=f"k2_{fname}.bin", pull=True)
        fused2[fname] = cosine(refv, v)
        d.shell(f"rm -f {DEV}/tdir/input.txt")
    for label, _e in gates:
        vals = cosines.get(label, {})
        # The stored scalar is the WORST fixture: everything downstream reads D["correctness"]
        # as one number per kernel and gates timings on > 0.99, so storing the best would let a
        # kernel that fails only the small fixture keep its timings.
        worst = min(vals.values()) if vals else float("nan")
        D.setdefault("correctness", {})[label] = worst
        say(f"| {label} | " + " | ".join(f"{vals.get(n, float('nan')):.6f}" for n, _, _, _ in fixtures)
            + f" | {'PASS' if worst > 0.99 else 'FAIL — do not trust its timing'} |")
    fw = min(fused2.values()) if fused2 else float("nan")
    say(f"| fused2 (vs conv² reference) | "
        + " | ".join(f"{fused2.get(n, float('nan')):.6f}" for n, _, _, _ in fixtures)
        + f" | {'PASS' if fw > 0.99 else 'FAIL'} |")
    say("")

    # ---- Session-B strategies (all env-gated, all default OFF)
    sec(14)
    say("## 14. LDS-at-constant-blocking and split-K\n")
    say("> Two strategies that were previously closed by reasoning alone and are now measured.\n"
        "> `MNN_CONV_LDS=w2` is conv_2d_c4h1w2 with the input staged in __local -- its comparison\n"
        "> partner is conv_2d_c4h1w2 itself, so LDS is the only difference (FINDINGS §H.30).\n"
        "> `MNN_CONV_SPLITK=n` splits the Cin reduction across n workgroups plus a reduce pass,\n"
        "> attacking occupancy starvation directly (§H.31). Metric: conv_all_us.\n")
    say("| shape | MNN default | c4h1w2 | c4h1w2+LDS | splitK=2 | splitK=4 |")
    say("|---|---|---|---|---|---|")
    for c in cores:
        cooldown(d, cool_s)
        m, shp, dep, k = c["model"], c["shape"], c["depth"], c["key"]
        # the w2 LDS kernel needs out_w % (2*TILE_W) == 0 and out_h % TILE_H == 0
        tile = None
        for tw, th in ((16, 4), (8, 4), (4, 2)):
            if c["W"] % (2 * tw) == 0 and c["H"] % th == 0:
                tile = f"{tw}x{th}"; break
        fns = {
            "def": lambda i, m=m, shp=shp, dep=dep, k=k: conv_all_us(
                run_model(d, m, shp, 60, cache=f"sb_d{k}{i}.bin")[0], dep),
            "w2base": lambda i, m=m, shp=shp, dep=dep, k=k: conv_all_us(
                run_model(d, m, shp, 60, env="MNN_CONV_SPEC=1 MNN_CONV_FORCE=conv_2d_c4h1w2",
                          cache=f"sb_b{k}{i}.bin")[0], dep),
        }
        # ConvBufExecution gates split-K on `inputChannelBlocks >= splitK`. Below that the flag is
        # silently ignored and the arm re-measures the DEFAULT -- it would print as a real number
        # equal to the baseline, i.e. "split-K has no effect", which is not what was measured.
        # C=8 has 2 channel blocks, so splitK=4 must not be run there at all.
        in_blocks = (c["C"] + 3) // 4
        for f in (2, 4):
            if in_blocks >= f:
                fns[f"sk{f}"] = (lambda i, m=m, shp=shp, dep=dep, k=k, f=f: conv_all_us(
                    run_model(d, m, shp, 60, env=f"MNN_CONV_SPLITK={f}",
                              cache=f"sb_{f}{k}{i}.bin")[0], dep))
        if tile:
            fns["lds"] = lambda i, m=m, shp=shp, dep=dep, k=k, tl=tile: conv_all_us(
                run_model(d, m, shp, 60, env=f"MNN_CONV_LDS=w2 MNN_LDS_TILE={tl}",
                          cache=f"sb_l{k}{i}.bin")[0], dep)
        r = interleaved(fns, reps)
        D.setdefault("session_b", {})[k] = r
        lds_s = ("**invalid**" if invalid("4,7,14", k)
                 else (f"{r['lds']:.0f}" if tile else "n/a (tile)"))
        sk = {f: (f"{r[f'sk{f}']:.0f}" if f"sk{f}" in r else f"n/a (Cin<{4*f})") for f in (2, 4)}
        say(f"| {c['label']} | {r['def']:.0f} | {r['w2base']:.0f} | {lds_s} | "
            f"{sk[2]} | {sk[4]} |")
    say("\n> **How to read this table on YOUR device** (the reference numbers are in FINDINGS\n"
        "> §H.30/§H.31 and are NOT repeated here, because the whole point is to re-decide):\n"
        "> - `c4h1w2+LDS` vs `c4h1w2` isolates LDS at constant blocking. LDS pays when the input\n"
        ">   working set stops fitting in L2, and costs barriers regardless — expect it to help\n"
        ">   more at large spatial size and on parts with less L2 per CU.\n"
        "> - `splitK` multiplies thread count without changing the output shape, so it only helps\n"
        ">   when the conv is output-starved; more CUs make that more likely. It always adds a\n"
        ">   write+re-read of the partials, which is what it has to pay for.\n"
        "> - Compare each against **MNN default**, not against each other. If the winner here is\n"
        ">   not MNN default, that is a real finding for this device.\n"
        "> Absolute values are only comparable within this section (arms are interleaved); check\n"
        "> the thermal validity section at the end before trusting cross-section comparisons.\n")

    # ---- NCHW layout
    sec(15)
    say("\n## 15. NCHW layout instead of NC4HW4 (conv-kernel time only)\n")
    say("> **Metric caveat, read before using these numbers.** The NCHW path needs an\n"
        "> NC4HW4->NCHW conversion before the conv and an NCHW->NC4HW4 conversion after it. In a\n"
        "> real deployment those fold into custom ops at the boundaries of a conv BLOCK -- paid\n"
        "> once per block, not once per conv -- so they are EXCLUDED from the conv column and\n"
        "> reported separately. Any adoption decision must budget the conversion column and\n"
        "> confirm that fold is actually possible for the graph in question (FINDINGS §H.36/§H.39).\n")
    say("| shape | NC4HW4 | NCHW | NCHW+HARD | best vs NC4HW4 | conversions (excluded) |")
    say("|---|---|---|---|---|---|")
    for c in cores + [{"label": h["label"], "model": h["model"], "shape": h["shape"],
                       "depth": 1, "key": h["key"], "H": 0, "W": 0} for h in man.get("heads", [])]:
        cooldown(d, cool_s)
        m, shp, dep, k = c["model"], c["shape"], c["depth"], c["key"]
        base = conv_kernel_only_us(run_model(d, m, shp, 60, cache=f"nc_d{k}.bin")[0], dep)[0]
        n_us, n_cvt = conv_kernel_only_us(
            run_model(d, m, shp, 60, env="MNN_CONV_NCHW=1", cache=f"nc_n{k}.bin")[0], dep)
        h_us, _ = conv_kernel_only_us(
            run_model(d, m, shp, 60, env="MNN_CONV_NCHW=1 MNN_CONV_HARD=1",
                      cache=f"nc_h{k}.bin")[0], dep)
        D.setdefault("nchw", {})[k] = {"nc4hw4": base, "nchw": n_us, "nchw_hard": h_us,
                                       "conversions": n_cvt}
        best = min(x for x in (n_us, h_us) if x) if (n_us or h_us) else 0.0
        say(f"| {c['label']} | {base:.0f} | {n_us:.0f} | {h_us:.0f} | "
            f"{pct(best, base)} | {n_cvt:.0f} |")
    say("\n> **Mechanism, so the table can be read on any device.** NC4HW4 pads channels to a\n"
        "> multiple of 4 — C=18 costs what C=20 costs, C=34 what C=36 costs — and NCHW has no such\n"
        "> cliff, so NCHW gains most where channel counts are NOT multiples of 4. Against that,\n"
        "> NC4HW4's float4 load serves 4 channels of the reduction at once while NCHW reads one\n"
        "> channel plane at a time, which costs NCHW more load instructions per MAC — and more so\n"
        "> at stride 2, where the input plane is 4x the output. Whichever effect dominates on your\n"
        "> part decides the sign. Reference-device outcome: FINDINGS §H.36 (stride 1) / §H.39\n"
        "> (stride 2) / §H.40 (hardcoding).\n")

    # ---- im2col + GEMM in NCHW
    sec(16)
    say("\n## 16. im2col + GEMM in NCHW\n")
    say("> **Trap:** at C>=64 MNN selects the Winograd path before any MNN_CONV_* flag is read, so\n"
        "> every arm would silently measure the same kernel. All arms here therefore run with\n"
        "> MNN_NO_WINOGRAD=1. im2col is counted as conv work (it is part of the algorithm); only\n"
        "> the final NC4HW4 repack is excluded, matching §15 (FINDINGS §H.38).\n")
    say("| shape | direct (NC4HW4) | im2col+GEMM | implicit GEMM (NC4HW4) | implicit GEMM (NCHW) | verdict |")
    say("|---|---|---|---|---|---|")
    NW = "MNN_NO_WINOGRAD=1 "
    for c in cores:
        cooldown(d, cool_s)
        m, shp, dep, k = c["model"], c["shape"], c["depth"], c["key"]
        r = interleaved({
            "dir": lambda i, m=m, shp=shp, dep=dep, k=k: conv_kernel_only_us(
                run_model(d, m, shp, 60, env=NW, cache=f"ig_d{k}{i}.bin")[0], dep)[0],
            "gem": lambda i, m=m, shp=shp, dep=dep, k=k: conv_kernel_only_us(
                run_model(d, m, shp, 60, env=NW + "MNN_CONV_IMGEMM=1",
                          cache=f"ig_g{k}{i}.bin")[0], dep)[0],
            # implicit GEMM: gathers the columns on the fly, never materialising the matrix.
            # The NC4HW4 variant uses no conversion kernels at all.
            "imp4": lambda i, m=m, shp=shp, dep=dep, k=k: conv_kernel_only_us(
                run_model(d, m, shp, 60, env=NW + "MNN_CONV_IGEMM=1",
                          cache=f"ig_i{k}{i}.bin")[0], dep)[0],
            "impn": lambda i, m=m, shp=shp, dep=dep, k=k: conv_kernel_only_us(
                run_model(d, m, shp, 60, env=NW + "MNN_CONV_IGEMM=nchw",
                          cache=f"ig_n{k}{i}.bin")[0], dep)[0],
        }, reps)
        D.setdefault("imgemm", {})[k] = r
        best = min(x for x in (r["gem"], r["imp4"], r["impn"]) if x) if any(
            (r["gem"], r["imp4"], r["impn"])) else 0.0
        v = ("**a GEMM wins**" if best and best < r["dir"] * 0.97
             else f"direct wins ({pct(best, r['dir'])})")
        say(f"| {c['label']} | {r['dir']:.0f} | {r['gem']:.0f} | {r['imp4']:.0f} | "
            f"{r['impn']:.0f} | {v} |")
    say("\n> **Mechanism.** im2col materialises each input element 9 times and then STREAMS that\n"
        "> matrix through the ALUs, so it inflates inner-loop bytes-per-MAC ~9x — this is not the\n"
        "> same as a one-off buffer round-trip, which on some parts is free. For the GEMM to win it\n"
        "> must out-tile the direct conv, but both are capped by the same register budget, so a\n"
        "> device where LARGER register tiles pay off (see §4: does c4h4w4 stop regressing?) is\n"
        "> where this could flip. This GEMM is deliberately simple — no LDS tiling of the column\n"
        "> matrix — so treat a loss here as a bound, not a proof. Reference outcome: §H.38.\n\n")

    # ---- image (texture) memory mode vs buffer
    # These last three sections DECIDE things (memory mode, GEMM tiles, the Winograd gate) and run
    # at the hottest point of the report. A 5s cooldown is not enough there: on the reference device
    # §17 read image mode at 115us where a cooled interleaved run reads 67.9us -- a 70% inflation
    # that silently flipped the recommendation. Give them a real cooldown.
    late_cool = max(cool_s, 30)
    sec(17)
    say("\n## 17. Memory mode: buffer vs IMAGE (texture)\n")
    say("> `gpuMode` 68 = `MNN_GPU_MEMORY_BUFFER|WIDE`, 132 = `MNN_GPU_MEMORY_IMAGE|WIDE`. These are\n"
        "> two SEPARATE, fully-implemented backends (`execution/buffer/` vs `execution/image/`), both\n"
        "> compiled into the same libMNN_CL.so -- the choice is purely runtime.\n>\n"
        "> **The comparison is confounded unless you control for the Winograd gate.** The two\n"
        "> backends select Winograd differently:\n>\n"
        "> * buffer `ConvBufWinograd::valid`: `ic>=32 && oc>=32 && in_w < out_c`\n"
        "> * image  `ConvWinograd::valid`:    `ic>=32 && oc>=32`   (no width clause)\n>\n"
        "> so on a shape like 48->48@36x48 (`48 < 48` false) buffer runs a DIRECT conv while image\n"
        "> runs Winograd, and a naive buffer-vs-image delta is really an algorithm delta. The\n"
        "> `buffer forced-wino` column is the matched-algorithm control (FINDINGS §H.51).\n>\n"
        "> IMAGE MODE CAN REBOOT THE DEVICE on tensors larger than the 2D image limit reported in\n"
        "> section 1. MNN maps an NC4HW4 tensor to an image of `W*ceil(C/4)` x `N*H`; every shape\n"
        "> below is checked against the limit before it is run.\n")
    maxw, maxh = D.get("hw", {}).get("image2d_max_width", 0), D.get("hw", {}).get("image2d_max_height", 0)
    say("| shape | image dims | buffer default | buffer forced-wino | **image** | image vs default | "
        "image vs best buffer | buf direct | img direct | **memory-mode only** | vs CPU |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    D["memory_mode"] = {}
    for c in cores:
        m, shp, dep, key = c["model"], c["shape"], c["depth"], c["key"]
        iw, ih = shp[3] * ((shp[1] + 3) // 4), shp[0] * shp[2]
        if maxw and (iw > maxw or ih > maxh):
            say(f"| {c['label']} | {iw}x{ih} | — | — | **SKIPPED** | exceeds {maxw}x{maxh} "
                f"| — | — | — | — | — |")
            D["memory_mode"][key] = {"skipped": "exceeds image2d limit", "image_dims": [iw, ih]}
            continue
        conf = invalid("17f", key)   # gates disagree: free-choice columns are algorithm-confounded
        cooldown(d, late_cool)
        # correctness gate: the CPU backend is the only independent ground truth. Comparing image
        # against the buffer GPU arm cannot tell "image is wrong" from "buffer does something extra"
        # -- that mistake nearly recorded a wrong result twice (FINDINGS §H.47).
        # The CPU arm must run the UNFUSED twin: CPU does not implement leakyReluSlope, so CPU on a
        # fused model returns the un-activated result and would report a false MISMATCH at ~0.45.
        _, cpu_v = run_model(d, c.get("unfused_model", m), shp, 2, cache=f"mmcpu{key}.bin",
                             pull=True, ftype=0)
        _, img_v = run_model(d, m, shp, 2, cache=f"mmimg{key}.bin", pull=True, mode=132)
        cos = cosine(cpu_v, img_v)
        # The free-choice arms (bd/im) answer "what do I get by flipping gpuMode" -- but the two
        # backends have DIFFERENT Winograd gates (ConvBufWinograd::valid has an in_w term,
        # ConvWinograd::valid does not), so on a shape where they disagree that comparison is
        # algorithm-vs-algorithm, not memory mode. bdd/imd pin BOTH backends to the direct path via
        # MNN_NO_WINOGRAD (now honoured by the image backend too) to isolate the memory mode.
        r = interleaved({
            "bd": lambda i, m=m, shp=shp, key=key: conv_all_us(
                run_model(d, m, shp, 120, cache=f"mmbd{key}{i}.bin")[0], dep),
            "bw": lambda i, m=m, shp=shp, key=key: conv_all_us(
                run_model(d, m, shp, 120, env="MNN_FORCE_WINOGRAD=1",
                          cache=f"mmbw{key}{i}.bin")[0], dep),
            "im": lambda i, m=m, shp=shp, key=key: conv_all_us(
                run_model(d, m, shp, 120, cache=f"mmim{key}{i}.bin", mode=132)[0], dep),
            "bdd": lambda i, m=m, shp=shp, key=key: conv_all_us(
                run_model(d, m, shp, 120, env=NOWG, cache=f"mmbdd{key}{i}.bin")[0], dep),
            "imd": lambda i, m=m, shp=shp, key=key: conv_all_us(
                run_model(d, m, shp, 120, env=NOWG, cache=f"mmimd{key}{i}.bin", mode=132)[0], dep),
        }, reps)
        bd, bw, im = r["bd"], r["bw"], r["im"]
        bdd, imd = r["bdd"], r["imd"]
        best_buf = min(x for x in (bd, bw) if x) if (bd or bw) else 0
        D["memory_mode"][key] = {"confounded_free_choice": conf, "buffer_default_us": bd, "buffer_forced_wino_us": bw,
                                 "image_us": im, "buffer_direct_us": bdd, "image_direct_us": imd,
                                 "image_dims": [iw, ih], "cos_vs_cpu": cos}
        flag = "ok" if cos > 0.999 else f"**{cos:.4f} MISMATCH**"
        if conf:
            # free-choice columns compare algorithms here, so they are not rendered as numbers;
            # the pinned pair below is a valid memory-mode comparison and IS rendered.
            say(f"| {c['label']} | {iw}x{ih} | _conf._ | _conf._ | _confounded_ | — | — | "
                f"{bdd:.0f} | {imd:.0f} | **{pct(imd, bdd)}** | {flag} |")
        else:
            say(f"| {c['label']} | {iw}x{ih} | {bd:.0f} | {bw:.0f} | **{im:.0f}** | {pct(im, bd)} | "
                f"{pct(im, best_buf)} | {bdd:.0f} | {imd:.0f} | **{pct(imd, bdd)}** | {flag} |")
    say("\n> **How to read the last two columns.** `image vs default` is what you would actually get\n"
        "> by flipping gpuMode. `image vs best buffer` is the part attributable to the memory mode\n"
        "> itself, once buffer is allowed the same algorithm. On the reference device those differ\n"
        "> by ~10 points on 48->48@36x48 (-33% vs -23%): about a third of the headline win is just\n"
        "> the Winograd gate, reachable in buffer mode with MNN_FORCE_WINOGRAD=1.\n>\n"
        "> **A `vs CPU` mismatch is a correctness failure, not a slow arm.** The image backend\n"
        "> ignored `Convolution2DCommon.leakyReluSlope` until FINDINGS §H.51 ported the fusion into\n"
        "> `ConvWinograd` + `winogradTransformDest*.cl`; before that it silently dropped the fused\n"
        "> PReLU and looked fast because it was doing less work. If this column mismatches on a\n"
        "> PReLU-fused model, that port is missing from the build under test.\n")

    # ---- XgemmBatched tile selection on tall-skinny GEMM
    sec(18)
    say("\n## 18. Winograd batchgemm tile selection (tall-skinny GEMM)\n")
    say("> MNN packs the Winograd GEMM to `M_pack x N_pack x K_pack` and picks CLBlast-style tile\n"
        "> parameters in `getGemmParams`. `isCandidateValid` requires **NWG to divide N exactly**,\n"
        "> and every stock candidate pairs a small NWG only with a small MWG. So when N_pack is an\n"
        "> **odd multiple of 16** (48, 80, 112 ...) every NWG in {32,64,128} is rejected and the GEMM\n"
        "> silently falls back to a 16x16 tile with **16 threads per workgroup** -- measured 1.83x\n"
        "> off at equal FLOPs. Extra candidates (NWG=16 paired with large MWG, and NWG=48) fix it;\n"
        "> they are ordinary tuner candidates, so a slower one is simply never selected.\n"
        "> `MNN_GEMM_NO_TALLSKINNY=1` removes them again for A/B. Reference outcome: FINDINGS §H.52.\n")
    say("| shape | M x N x K | tile MWG x NWG | threads/group | stock gemm | fixed gemm | delta |")
    say("|---|---|---|---|---|---|---|")
    D["gemm_tile"] = {}
    for c in cores:
        cooldown(d, late_cool)
        m, shp, dep, key = c["model"], c["shape"], c["depth"], c["key"]
        # instrument the selection rather than infer it: nothing in the profiler identifies which
        # XgemmBatched configuration ran (FINDINGS trap 4)
        o, _ = run_model(d, m, shp, 3, env="MNN_GEMM_DEBUG=1 MNN_FORCE_WINOGRAD=1",
                         cache=f"gt{key}.bin")
        gm = None
        for mt in re.finditer(r"\[GEMM\] M=(\d+) N=(\d+) K=(\d+) batch=(\d+) -> (?:tuned|cached) \| "
                              r"MWG=(\d+) NWG=(\d+) KWG=\d+ MDIMC=(\d+) NDIMC=(\d+)", o):
            gm = mt
        if gm is None:
            say(f"| {c['label']} | (no batchgemm on this path) | — | — | — | — | — |")
            continue
        M, N, K, B, MWG, NWG, MDIMC, NDIMC = (int(x) for x in gm.groups())
        thr = MDIMC * NDIMC
        r = interleaved({
            "stock": lambda i, m=m, shp=shp, key=key: conv_all_us(
                run_model(d, m, shp, 120, env="MNN_GEMM_NO_TALLSKINNY=1 MNN_FORCE_WINOGRAD=1",
                          cache=f"gs{key}{i}.bin")[0], dep),
            "fixed": lambda i, m=m, shp=shp, key=key: conv_all_us(
                run_model(d, m, shp, 120, env="MNN_FORCE_WINOGRAD=1",
                          cache=f"gf{key}{i}.bin")[0], dep),
        }, reps)
        st, fx = r["stock"], r["fixed"]
        D["gemm_tile"][key] = {"M": M, "N": N, "K": K, "batch": B, "MWG": MWG, "NWG": NWG,
                               "threads_per_group": thr, "stock_us": st, "fixed_us": fx}
        warn = " ⚠️ starved" if thr <= 32 else ""
        say(f"| {c['label']} | {M}x{N}x{K} (b{B}) | {MWG}x{NWG} | {thr}{warn} | {st:.0f} | "
            f"{fx:.0f} | {pct(fx, st)} |")
    say("\n> **How to read it.** A `threads/group` of 16-32 means the tile selector was cornered by\n"
        "> the divisibility rule, and this shape is a candidate for the fix. N here is the padded\n"
        "> output-channel count (`ROUND_UP(out_channels, mAlignN)`), so it is the CONV's channel\n"
        "> count that decides: any conv whose padded output channels land on an odd multiple of 16\n"
        "> hits this. The delta column only moves on shapes that were starved -- elsewhere the\n"
        "> extra candidates lose the tuner's own benchmark and change nothing.\n")

    # ---- Winograd selection gate
    sec(19)
    say("\n## 19. Winograd selection gate\n")
    say("> MNN admits Winograd only when `ic>=32 && oc>=32 && in_w < out_c` (or both >=64). The\n"
        "> width clause has the right idea -- the transform cost scales with SPATIAL size and the\n"
        "> arithmetic saving with CHANNELS -- but is calibrated so tightly it refuses shapes where\n"
        "> Winograd wins by 20-34%. This build uses **`in_w <= 2 * out_c`**, fit on-device with zero\n"
        "> missed wins and zero admitted losses over 14 shapes (FINDINGS §H.53).\n>\n"
        "> The coefficient is a fit on ONE device and depends on the GEMM tile fix of §18: before\n"
        "> that fix the right answer was ~1.5, and `48->48@72x96` flipped from +11.9% to -19.0%.\n"
        "> **Re-fit both together on new hardware.** `MNN_WINOGRAD_GATE_OLD=1` restores the stock\n"
        "> clause; the `forced` column below is the upper bound the gate is trying to reach.\n")
    say("| shape | in_w | out_c | old gate | NEW gate | forced wino | gate verdict |")
    say("|---|---|---|---|---|---|---|")
    D["wino_gate"] = {}
    for c in cores:
        cooldown(d, late_cool)
        m, shp, dep, key = c["model"], c["shape"], c["depth"], c["key"]
        in_w, out_c = shp[3], shp[1]
        r = interleaved({
            "old": lambda i, m=m, shp=shp, key=key: conv_all_us(
                run_model(d, m, shp, 120, env="MNN_WINOGRAD_GATE_OLD=1",
                          cache=f"wo{key}{i}.bin")[0], dep),
            "new": lambda i, m=m, shp=shp, key=key: conv_all_us(
                run_model(d, m, shp, 120, cache=f"wn{key}{i}.bin")[0], dep),
            "forced": lambda i, m=m, shp=shp, key=key: conv_all_us(
                run_model(d, m, shp, 120, env="MNN_FORCE_WINOGRAD=1",
                          cache=f"wfz{key}{i}.bin")[0], dep),
        }, reps)
        old_, new_, forced_ = r["old"], r["new"], r["forced"]
        D["wino_gate"][key] = {"in_w": in_w, "out_c": out_c, "old_gate_us": old_,
                               "new_gate_us": new_, "forced_us": forced_}
        if new_ < old_ * 0.97:
            v = f"**new gate captures {pct(new_, old_)}**"
        elif forced_ < new_ * 0.97:
            v = f"⚠️ still leaving {pct(forced_, new_)} on the table — re-fit the coefficient"
        else:
            v = "gate agrees with the clock"
        say(f"| {c['label']} | {in_w} | {out_c} | {old_:.0f} | **{new_:.0f}** | {forced_:.0f} | {v} |")
    say("\n> **If the `forced` column is materially below `NEW gate` on some shape, the coefficient\n"
        "> is wrong for this device** -- Winograd would win there and the gate is refusing it. Raise\n"
        "> K until that stops, then check no shape where `forced` is ABOVE `old gate` got admitted.\n")

    # ---- 11. what to do on this device
    sec(20)
    say("## 20. Recommendations for THIS device\n")
    say("> Per-conv advice covers **both** conv families measured here: the stride-1 cores (§4) and\n"
        "> the stride-2 heads (§6). A claim counts only if it clears the **6% noise floor** measured\n"
        "> from this suite's own control arms — 3% is inside run-to-run spread on this device.\n")
    recs = []
    NOISE = 0.94   # 6% noise floor; controls in §9 spread +/-6%, mostly +/-2%

    # Thermal gate. §17-§19 run at the hottest point of the report, and a throttled reading there
    # is not a slow configuration -- it is a void measurement. Sample the sustained clock here and
    # refuse to give late-section advice if the device is not at nominal.
    s20, _ = sample_clock(d, cores[0]["model"], cores[0]["shape"])
    nominal20 = int(hw.get("max_clock_mhz", 0) or 0)
    sustained = int(statistics.median(s20)) if s20 else 0
    thermal_ok = (not nominal20) or (not sustained) or sustained >= nominal20 * 0.9
    if not thermal_ok:
        say(f"> ⚠️ **The device is throttling ({sustained} MHz sustained vs {nominal20} MHz "
            "nominal).** The memory-mode / GEMM-tile / Winograd-gate advice below is derived from\n"
            "> §17-§19, which run at the end of this report, so those readings are inflated and the\n"
            "> recommendations are SUPPRESSED. Re-run those three sections on a cooled device\n"
            "> (`--cooldown 60`) before acting on them. The per-kernel advice from §4/§6 was\n"
            "> measured early and is unaffected.\n")
    D["recommend_thermal_ok"] = thermal_ok
    D["recommend_sustained_mhz"] = sustained
    for kind in ("stride-1 core", "stride-2 head"):
        rows = [v for v in summary.values() if v["kind"] == kind]
        if not rows:
            continue
        recs.append(f"**{kind}s**")
        for r in rows:
            base, best, bt = r["base"], r["best"], r["best_us"]
            if not base:
                continue
            if bt < base * NOISE:
                recs.append(f"  ✅ **{r['label']}: use `{best}`** — {pct(bt, base)} vs MNN's default "
                            f"({base:.0f}→{bt:.0f}µs). Wire it in via the autotuner or force it.")
            elif bt < base:
                recs.append(f"  ➖ **{r['label']}**: `{best}` is {pct(bt, base)} — **inside the 6% "
                            f"noise floor**, not a result. Treat the default ({base:.0f}µs) as best.")
            else:
                recs.append(f"  ➖ **{r['label']}**: MNN's default ({base:.0f}µs) is best; no custom "
                            f"kernel beat it (closest `{best}` {bt:.0f}µs).")

    # --- levers that are NOT per-kernel, and are usually larger than any of the above
    mm = D.get("memory_mode", {}) if thermal_ok else {}
    img_wins = [(k, v) for k, v in mm.items() if v.get("image_us") and v.get("buffer_default_us")
                and v["image_us"] < v["buffer_default_us"] * NOISE]
    if img_wins:
        recs.append("**Memory mode**")
        for k, v in img_wins:
            lab = summary.get(k, {}).get("label", k)
            recs.append(f"  🖼️ **{lab}: run this model in IMAGE mode** (`gpuMode=132`) — "
                        f"{pct(v['image_us'], v['buffer_default_us'])} vs buffer "
                        f"({v['buffer_default_us']:.0f}→{v['image_us']:.0f}µs). gpuMode is per-"
                        "Interpreter, so a split pipeline can choose per model. Check §17 first: it "
                        "is a shape-gated win, not a global one.")
    starved = [(k, v) for k, v in (D.get("gemm_tile", {}) if thermal_ok else {}).items()
               if v.get("threads_per_group", 99) <= 32]
    if starved:
        recs.append("**GEMM tile selection**")
        for k, v in starved:
            lab = summary.get(k, {}).get("label", k)
            recs.append(f"  ⚙️ **{lab}**: the Winograd batchgemm runs {v['MWG']}x{v['NWG']} with only "
                        f"{v['threads_per_group']} threads/group (N={v['N']}). If N is an odd multiple "
                        "of 16 this is the §18 starvation case — check a tile candidate with NWG "
                        "dividing N exists.")
    gate_gap = [(k, v) for k, v in (D.get("wino_gate", {}) if thermal_ok else {}).items()
                if v.get("forced_us") and v.get("new_gate_us")
                and v["forced_us"] < v["new_gate_us"] * NOISE]
    if gate_gap:
        recs.append("**Winograd gate**")
        for k, v in gate_gap:
            lab = summary.get(k, {}).get("label", k)
            recs.append(f"  📐 **{lab}**: forcing Winograd is {pct(v['forced_us'], v['new_gate_us'])} "
                        "below what the gate selects — the `in_w <= 2*out_c` coefficient is wrong for "
                        "this device. Re-fit it (§19), together with the §18 GEMM tiles.")
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
    sec(21)
    say("\n## 21. GPU clock at END of run (thermal validity check)\n")
    cooldown(d, cool_s)
    s2c, _ = sample_clock(d, cores[0]["model"], cores[0]["shape"])
    clk_end = int(statistics.median(s2c)) if s2c else 0
    nominal = int(hw.get("max_clock_mhz", 0) or 0)
    throttled = ([x for x in s2c if nominal and x < nominal * 0.9] if s2c else [])
    D["clock_end_mhz"] = clk_end; D["clock_start_mhz"] = clk_start
    D["clock_end_min_mhz"] = min(s2c) if s2c else 0
    D["clock_throttled_fraction"] = (len(throttled) / len(s2c)) if s2c else 0.0
    say(f"- start of run: **{clk_start} MHz** (median)  →  end of run: **{clk_end} MHz** (median)"
        + (f", min {min(s2c)} MHz, {100*len(throttled)/len(s2c):.0f}% of samples below 90% of "
           f"nominal" if s2c else ""))
    say(f"- (raw end samples {s2c})" if s2c else "")
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
