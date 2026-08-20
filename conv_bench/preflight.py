#!/usr/bin/env python3
"""preflight.py -- measurement-INTEGRITY audit for the conv suite. Assertions only, no timing.

Every defect found in the suite so far (silently-dropped MNN_CONV_FORCE, confounded Winograd
baselines, inert flags, parser corruption, a kernel returning garbage) is an integrity bug, not a
timing bug. None of them need performance numbers to detect, so this runs at loops=2 and is immune
to the thermal drift that invalidated the 3.5h suite.

    python3 preflight.py [SERIAL] [--quick]

Exit 0 = suite results can be trusted structurally. Non-zero = do NOT spend 3.5h.
"""
import json, re, subprocess, sys, os
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
import make_bundle as M
from block_fixture import load_blocks

MODELS = REPO / "conv_bench" / "conv_probe_bundle" / "models"
BUILD  = REPO / "build_android_profile"
LIBS   = [BUILD / "OFF/arm64-v8a/libMNN.so",
          BUILD / "express/OFF/arm64-v8a/libMNN_Express.so",
          BUILD / "source/backend/opencl/OFF/arm64-v8a/libMNN_CL.so"]
MODULE = BUILD / "ModuleBasic.out"
DEV    = "/data/local/tmp/preflight"
TMP    = Path("/tmp/_preflight"); TMP.mkdir(exist_ok=True)

SEL = re.compile(r"\[CONV_SELECT\] b(\d+)ci(\d+)hi(\d+)wi(\d+)co(\d+)ho(\d+)wo(\d+)kh(\d+)kw(\d+) "
                 r"selected=(\S+) force=(\S+) force_ok=(-?\d+) ncand=(\d+) cands=(\S*)")
KT  = re.compile(r"kernel time = (\d+)\s+us (\S+)")
PATH= re.compile(r"\[CONV_PATH\] \S+ (.*)")
TOT = re.compile(r"total kernel time = (\d+)  us")
WARN= re.compile(r'WARN: [^\n]*blobcache for saving!')

FAIL, WARNS, OK = [], [], []
# A failure is either GLOBAL (the run is meaningless -- wrong binary, unparseable output) or
# CELL-scoped (one shape x one section is invalid while everything else stands). Collapsing both
# into "NOT TRUSTWORTHY" blocks a 3.5h run that is 99% sound, and a gate that cries wolf gets
# ignored -- which would cost more than it saves.
INVALID_CELLS = []
def fail(sec, msg, cell=None):
    FAIL.append((sec, msg))
    if cell: INVALID_CELLS.append({"section": cell[0], "shape": cell[1], "why": cell[2]})
    print(f"   \033[31mFAIL\033[0m {msg}", flush=True)
def warn(sec, msg): WARNS.append((sec, msg)); print(f"   \033[33mWARN\033[0m {msg}", flush=True)
def ok(msg):        OK.append(msg);           print(f"   ok   {msg}", flush=True)

_serial = ""
def adb(cmd, timeout=300):
    return subprocess.run(f"adb {_serial} {cmd}", shell=True, text=True,
                          capture_output=True, timeout=timeout).stdout

_pushed = set()
# gpuMode bits (include/MNN/MNNForwardType.h): TUNING_NONE=1<<0, WIDE=1<<2,
# MEMORY_BUFFER=1<<6, MEMORY_IMAGE=1<<7. The suite uses 68 = BUFFER|WIDE, i.e. FULL autotuning on
# every launch. Integrity checks do not care whether the local work size is optimal, only which
# kernel/algorithm was chosen -- and the algorithm is decided in onCreate(), before any tuning.
# 65/129 = BUFFER|NONE / IMAGE|NONE removes the autotune cost entirely.
# NOTE: with tuning off, `selected=` is NOT the production choice (that is decided by benchmarking);
# preflight therefore reports availability/algorithm/engagement, never "which variant MNN picks".
BUF, IMG = 64 | 1, 128 | 1
_launches = 0
def run(model, shape, env="", mode=BUF, ftype=3, loops=2, cache=None, pull=False):
    """Run ModuleBasic once. Models pushed once each; cache shared per (model,mode) so nothing
    re-tunes between checks. Process launches are the unit of cost here (each pays full shader
    compilation -- the ANGLE blob cache is unwritable on this device), so minimise launches."""
    global _launches
    _launches += 1
    if model not in _pushed:
        adb(f"push {MODELS/model} {DEV}/{model}"); _pushed.add(model)
    inj = TMP / "input.json"
    inj.write_text(json.dumps({"inputs": [{"name": "input", "shape": shape}],
                               "outputs": ["output"], "shapeMutable": False}))
    adb(f"push {inj} {DEV}/tdir/input.json")
    if pull:
        # MUST be single-quoted: `adb shell "a && b"` lets the HOST shell eat the &&, so mkdir ran
        # locally and the device dir never existed -> every pull returned nothing -> cosine=nan,
        # which reported as "WRONG OUTPUT" on 11 correct cases.
        adb(f"shell 'rm -rf {DEV}/output && mkdir -p {DEV}/output'")
    cache = cache or f"pf_{model.replace('.mnn','')}_{mode}.bin"   # shared across checks
    out = adb(f"shell 'cd {DEV} && {env}LD_LIBRARY_PATH=. ./ModuleBasic.out {model} tdir 0 "
              f"{ftype} {loops} {mode} 2 {cache} 2>&1'")
    vals = None
    if pull:
        p = TMP / "out.txt"
        if p.exists(): p.unlink()
        adb(f"pull {DEV}/output/0_0.txt {p}")
        if p.exists():
            try: vals = [float(x) for x in p.read_text().split()]
            except Exception: vals = None
    return WARN.sub('', out), vals

def kernels(out):
    return [(n, int(t)) for t, n in KT.findall(out)]

def algo(out, nconv=None):
    """Which convolution algorithm ran.

    BUFFER backend encodes it in the event name (ConvBuf2D-ori / -gemmN / Conv-winograd-*).
    IMAGE backend does NOT: every conv is just `ConvolutionN`, algorithm-free. There, the only
    available signal is dispatches-per-conv -- image Winograd runs source-transform + batched gemm
    + dest-transform (3) where direct runs 1. That is INFERENCE, not observation; it is labelled
    `winograd?`/`direct?` so it can never be mistaken for a measured fact.
    """
    names = " ".join(n for n, _ in kernels(out)).lower()
    if "winograd" in names: return "winograd"
    if "convbuf2d-gemm" in names: return "gemm"
    if "conv1x1" in names:  return "conv1x1"
    if "convbuf2d" in names: return "direct"
    conv_ev = [n for n, _ in kernels(out) if n.lower().startswith("convolution")]
    if conv_ev and nconv:
        per = per_window(out, "convolution") / max(nconv, 1)
        if per >= 2.5: return "winograd?"
        if per >= 0.5: return "direct?"
    return "unknown"


def per_window(out, prefix):
    """Mean count of dispatches whose name starts with `prefix`, PER INFERENCE (not summed over
    every profiling window -- that made C2's `47 dispatches >= 6 convs` vacuous)."""
    wins, cur = [], 0
    for ln in out.splitlines():
        m = KT.search(ln)
        if m:
            if m.group(2).lower().startswith(prefix): cur += 1
        elif "total kernel time" in ln:
            wins.append(cur); cur = 0
    wins = [w for w in wins if w] or ([cur] if cur else [])
    if not wins: return 0
    # MODE, not mean: a truncated first/last profiling window drags the mean below the true
    # per-inference count (2 convs read as 1.9) and tripped a spurious "dispatches missing".
    return max(set(wins), key=wins.count)

def sel_lines(out):
    return [{"ci":int(m[1]),"hi":int(m[2]),"wi":int(m[3]),"co":int(m[4]),
             "selected":m[9],"force":m[10],"force_ok":int(m[11]),
             "ncand":int(m[12]),"cands":m[13].split(",") if m[13] else []}
            for m in SEL.findall(out)]

def cosine(a, b):
    if not a or not b: return float("nan")
    n=min(len(a),len(b)); sa=sb=sab=0.0
    for i in range(n):
        x,y=a[i],b[i]; sa+=x*x; sb+=y*y; sab+=x*y
    return sab/((sa**0.5)*(sb**0.5)+1e-12)

# ---- gate predictions (pure python mirrors of the two C++ gates) -------------
def buf_gate(ci, co, in_w, stride, k=3):
    if stride != 1 or k != 3: return False
    return (ci>=32 and co>=32 and in_w <= 2*co) or (ci>=64 and co>=64)

def img_gate(ci, co, ow, oh, stride, k=3):
    if stride != 1 or k not in (3,5): return False
    if ci>=32 and co>=32: return True          # NOTE: no in_w term -- differs from buffer
    return (co*oh*ow)/(ci*k) <= 5

# ---- case list ---------------------------------------------------------------
def cases():
    cs=[]
    for C,H,W in M.CORES:
        # `mkey` MUST match make_bundle's manifest key ("{C}_{H}x{W}") -- the report looks cells up
        # by that, and a mismatch makes masking silently do nothing (it printed "1 cell will be
        # masked" and then masked none). `key` stays human-readable for this tool's own output.
        cs.append(dict(kind="core", key=f"core{C}@{H}x{W}", mkey=f"{C}_{H}x{W}",
                       model=f"core_{C}.mnn",
                       shape=[1,C,H,W], cpu_model=f"core_{C}_unfused.mnn",
                       convs=[dict(cin=C,cout=C,H=H,W=W,stride=1)]*6))
    for k,cv in M.HEADS:
        cs.append(dict(kind="head", key=k, mkey=k, model=f"{k}.mnn",
                       shape=[1,cv[0]["cin"],cv[0]["H"],cv[0]["W"]],
                       cpu_model=f"{k}_unfused.mnn",
                       convs=[dict(cin=c["cin"],cout=c["cout"],H=c["H"],W=c["W"],
                                   stride=c["stride"]) for c in cv]))
    for name,cv in load_blocks().items():
        cs.append(dict(kind="block", key=name, mkey=name, model=f"{name}.mnn",
                       shape=[1,cv[0]["cin"],cv[0]["H"],cv[0]["W"]], cpu_model=f"{name}.mnn",
                       convs=[dict(cin=c["cin"],cout=c["cout"],H=c["H"],W=c["W"],
                                   stride=c["stride"]) for c in cv]))
    return cs

def out_hw(c):
    return ((c["H"]+2-3)//c["stride"]+1, (c["W"]+2-3)//c["stride"]+1)

# ============================== sections =====================================
def section_A(CS, results):
    """A (algorithm/selection) + C (parser/dispatch integrity) share ONE launch per case."""
    print("\n== A+C. Implementation actually run, and parser integrity (one launch per case) ==",
          flush=True)
    for c in CS:
        out,_ = run(c["model"], c["shape"], env="MNN_CONV_SELECT=1 ", loops=6)
        _check_C(c, out, results)
        if not out.strip():
            fail("A", f"{c['key']}: no output from device"); continue
        sl  = sel_lines(out)
        a   = algo(out, len(c["convs"]))
        # convs that printed CONV_SELECT took the DIRECT path; the rest took another algorithm
        direct = {(s["ci"],s["co"],s["hi"],s["wi"]) for s in sl}
        pred_w = [buf_gate(x["cin"],x["cout"],x["W"],x["stride"]) for x in c["convs"]]
        n_pred = sum(pred_w)
        n_obs  = sum(1 for x in c["convs"]
                     if (x["cin"],x["cout"],x["H"],x["W"]) not in direct)
        results.setdefault(c["key"],{}).update(algo=a, n_direct=len(direct),
                                               n_pred_wino=n_pred, n_obs_nondirect=n_obs,
                                               selected=[s["selected"] for s in sl])
        status = "ok  " if n_obs == n_pred else "FAIL"
        line = (f"{c['key']:16} algo={a:9} direct_convs={len(direct)}/{len(c['convs'])} "
                f"predicted_wino={n_pred} observed_non_direct={n_obs}")
        if n_obs == n_pred: ok(line)
        else: fail("A3", line + "  <- gate prediction != observed")

def section_A2(CS, results):
    print("\n== A2. Variant availability (does MNN_CONV_FORCE engage?) ==")
    print("   (a variant absent from `cands` is silently dropped -> the arm re-measures the default)")
    for c in CS:
        if c["kind"] == "block": continue
        avail=set()
        for env in ("MNN_CONV_SELECT=1 MNN_NO_WINOGRAD=1 ",
                    "MNN_CONV_SELECT=1 MNN_NO_WINOGRAD=1 MNN_CONV_SPEC=1 "):
            out,_ = run(c["model"], c["shape"], env=env)
            for s in sel_lines(out): avail.update(s["cands"])
        # The report EXCLUDES stride1_only kernels from head tables (bundle_run_report.py) because
        # they hardcode stride 1; they are not expected to exist on a stride-2 conv. Only count a
        # variant as missing if the suite would actually try to measure it here.
        expected = [v for v in M.VARIANTS
                    if not (c["kind"] == "head" and v in set(M.STRIDE1_ONLY))]
        miss=[v for v in expected if v not in avail]
        results.setdefault(c["key"],{})["unavailable"]=miss
        if miss:
            fail("A2", f"{c['key']:16} {len(expected)-len(miss)}/{len(expected)} measured-variants available; "
                       f"UNSUPPORTED: {', '.join(x.replace('conv_2d_','') for x in miss)}")
        else:
            ok(f"{c['key']:16} all {len(expected)} measured-variants available"
               + (f" ({len(M.VARIANTS)-len(expected)} stride1-only correctly skipped)"
                  if len(expected)!=len(M.VARIANTS) else ""))

def section_A4(CS, results):
    print("\n== A4. Winograd gate parity: buffer (68) vs image (132) ==")
    print("   (the two gates are different C++ functions; image has no in_w term)")
    for c in CS:
        if c["kind"] != "core": continue
        ob,_ = run(c["model"], c["shape"], mode=BUF)
        oi,_ = run(c["model"], c["shape"], mode=IMG)
        n = len(c["convs"]); ab, ai = algo(ob, n), algo(oi, n)
        cv=c["convs"][0]; oh,ow = out_hw(cv)
        pb = buf_gate(cv["cin"],cv["cout"],cv["W"],cv["stride"])
        pi = img_gate(cv["cin"],cv["cout"],ow,oh,cv["stride"])
        results.setdefault(c["key"],{}).update(algo_buffer=ab, algo_image=ai,
                                               pred_buf_wino=pb, pred_img_wino=pi)
        if ab.rstrip("?") != ai.rstrip("?"):
            fail("A4", f"{c['key']:16} buffer={ab:9} image={ai:9} "
                       f"(gates predict buf_wino={pb} img_wino={pi}) "
                       f"-> image-vs-buffer here is CONFOUNDED by algorithm",
                 cell=("17f", c["mkey"],
                       f"gates disagree (buffer={ab}, image={ai}): the FREE-CHOICE columns compare "
                       f"algorithms, not memory mode. The pinned buf/img direct columns are valid."))
        else:
            ok(f"{c['key']:16} buffer={ab:9} image={ai:9} same algorithm")

def section_B(CS, results):
    print("\n== B. Arm integrity ==")
    c = next(x for x in CS if x["kind"]=="core")
    out,_ = run(c["model"], c["shape"],
                env="MNN_CONV_SELECT=1 MNN_NO_WINOGRAD=1 MNN_CONV_FORCE=conv_2d_NOT_A_REAL_KERNEL ")
    sl = sel_lines(out)
    if sl and all(s["force_ok"]==0 for s in sl):
        ok(f"B1 canary: impossible kernel correctly reports force_ok=0 ({c['key']})")
    else:
        fail("B1", f"canary did not report force_ok=0 -> force_ok itself is unreliable ({sl[:1]})")
    # Flag engagement is now a FACT, not an inference: [CONV_PATH] reports requested/applied per
    # flag. A flag that is requested but not applied was silently rejected by a path guard, and any
    # "no effect" number measured under it is really "never ran" (FINDINGS §H.55).
    FLAGS = {"MNN_CONV_SPLITK=2":"splitk", "MNN_CONV_NCHW=1":"nchw",
             "MNN_CONV_IGEMM=1":"igemm", "MNN_CONV_IMGEMM=1":"imgemm", "MNN_CONV_CONSTW=1":"constw"}
    for case in [x for x in CS if x["kind"] == "core"]:
        for flag, field in FLAGS.items():
            o,_ = run(case["model"], case["shape"],
                      env=f"MNN_CONV_SELECT=1 MNN_NO_WINOGRAD=1 {flag} ")
            m = PATH.search(o)
            if not m:
                fail("B2", f"{case['key']:16} {flag:20} no [CONV_PATH] line (old libMNN_CL.so?)")
                continue
            kv = dict(x.split("=") for x in m.group(1).split())
            rq, ap = (kv.get(field, "0/0").split("/") + ["0"])[:2]
            results.setdefault("flags", {}).setdefault(case["key"], {})[flag] = f"{rq}/{ap}"
            if rq == "1" and ap == "0":
                fail("B2", f"{case['key']:16} {flag:20} REQUESTED but NOT APPLIED "
                           f"-> any number measured under this flag is the DEFAULT, not the flag",
                     cell=("7,14" if field=="lds" else "4,8", case["mkey"],
                           f"{flag} rejected by a path guard; the arm measures the default"))
            elif rq == "1":
                ok(f"{case['key']:16} {flag:20} requested and APPLIED")
    # LDS is tile-dependent: the suite sweeps LDS_TILES, so the real question is whether ANY tile
    # it will try actually applies on this shape -- not whether the default 16x4 does.
    for case in [x for x in CS if x["kind"] == "core"]:
        good = []
        for tile in M.LDS_TILES:
            o,_ = run(case["model"], case["shape"],
                      env=f"MNN_CONV_SELECT=1 MNN_NO_WINOGRAD=1 MNN_CONV_LDS=1 MNN_LDS_TILE={tile} ",
                      cache=f"pf_lds_{case['key'].replace('@','_').replace('x','_')}_{tile}.bin")
            m = PATH.search(o)
            if m and dict(x.split("=") for x in m.group(1).split()).get("lds") == "1/1":
                good.append(tile)
        results.setdefault("lds_tiles", {})[case["key"]] = good
        if good:
            ok(f"{case['key']:16} MNN_CONV_LDS applies with {len(good)}/{len(M.LDS_TILES)} tiles: "
               f"{','.join(good)}")
        else:
            fail("B2", f"{case['key']:16} MNN_CONV_LDS applies with NO tile in LDS_TILES "
                       f"-> every LDS arm on this shape silently measures the default",
                 cell=("7,14", case["mkey"], "no LDS tile divides this output shape"))

    # MNN_CONV_HARD only toggles compile-time defines; report it as requested (it has no
    # applied-state to observe), so it is never silently counted as a measured strategy.
    o,_ = run(CS[0]["model"], CS[0]["shape"],
              env="MNN_CONV_SELECT=1 MNN_NO_WINOGRAD=1 MNN_CONV_HARD=1 ")
    m = PATH.search(o)
    if m and dict(x.split("=") for x in m.group(1).split()).get("hard") == "1":
        ok("MNN_CONV_HARD=1        reaches the conv (compile-time defines; no applied-state exists)")
    else:
        fail("B2", "MNN_CONV_HARD=1 not seen by the conv at all")


def _check_C(c, out, results):
    """Parser / dispatch integrity, from the SAME output section A already collected."""
    ks = kernels(out)
    if not ks:
        fail("C", f"{c['key']}: no kernel lines parsed"); return
    win, cur, bad = [], [], []
    for ln in out.splitlines():
        m = KT.search(ln)
        if m: cur.append(int(m.group(1)))
        elif "total kernel time" in ln and cur:
            # Defensive: a line can contain the phrase yet not match (device output is not a
            # guaranteed format, and warnings splice into lines without a newline). A parser must
            # report an unparseable line, never crash on it.
            mt = TOT.search(ln)
            if mt is None:
                bad.append(ln[:90])
            else:
                win.append((sum(cur), int(mt.group(1))))
            cur = []
    if win:
        s_, t_ = win[-1]
        d = abs(s_ - t_) / max(t_, 1) * 100
        if d < 2:
            ok(f"C1 {c['key']:16} per-kernel sum {s_} ~= total {t_} ({d:.1f}%)")
        else:
            fail("C1", f"{c['key']:16} per-kernel sum {s_} != total {t_} ({d:.1f}%) "
                       f"-> parser dropping/double-counting dispatches")
    # per INFERENCE. Summing across every profiling window made this vacuous ("47 >= 6").
    if bad:
        fail("C1", f"{c['key']:16} {len(bad)} unparseable 'total kernel time' line(s): {bad[0]!r}")
    nconv = per_window(out, "convbuf2d") + per_window(out, "conv-winograd") + per_window(out, "convolution")
    want  = len(c["convs"])
    results.setdefault(c["key"], {})["conv_dispatch_per_inference"] = nconv
    if nconv < want - 0.01:
        fail("C2", f"{c['key']:16} {nconv:.1f} conv dispatches/inference < {want} convs "
                   f"-> dispatches missing from the profile")
    else:
        ratio = nconv / max(want, 1)
        ok(f"C2 {c['key']:16} {nconv:.1f} conv dispatches/inference for {want} convs "
           f"({ratio:.1f}/conv)")


def section_D(CS, results):
    print("\n== D. Correctness vs CPU (cosine, 2 loops) ==")
    for c in CS:
        if not c["cpu_model"]:
            warn("D", f"{c['key']:16} no unfused twin -> CPU reference impossible "
                      f"(CPU ignores leakyReluSlope; comparing would report a false MISMATCH)")
            continue
        _, g = run(c["model"], c["shape"], pull=True)
        _, cpu = run(c["cpu_model"], c["shape"], ftype=0, pull=True)
        cs = cosine(cpu, g)
        results.setdefault(c["key"],{})["cosine"]=cs
        if cs != cs or cs < 0.99:
            fail("D", f"{c['key']:16} cosine={cs:.6f} vs CPU  <- WRONG OUTPUT")
        else:
            ok(f"{c['key']:16} cosine={cs:.6f}")

def main():
    global _serial
    args=[a for a in sys.argv[1:] if not a.startswith("--")]
    if args: _serial = f"-s {args[0]}"
    quick = "--quick" in sys.argv
    print(f"PREFLIGHT  (models: {MODELS})")
    adb(f"shell mkdir -p {DEV}/tdir")
    for L in LIBS+[MODULE]: adb(f"push {L} {DEV}/")
    adb(f"shell chmod +x {DEV}/ModuleBasic.out")
    CS = cases(); results={}
    print(f"cases: {sum(1 for c in CS if c['kind']=='core')} cores, "
          f"{sum(1 for c in CS if c['kind']=='head')} heads, "
          f"{sum(1 for c in CS if c['kind']=='block')} blocks")
    section_A(CS, results)
    section_A2(CS, results)
    section_A4(CS, results)
    section_B(CS, results)
    if not quick: section_D(CS, results)
    print(f"\n  device launches: {_launches}")
    glob = [f for f in FAIL if not any(
        c["why"] and (f[1].split()[0] in c["shape"] or c["shape"] in f[1]) for c in INVALID_CELLS)]
    glob = [f for f in FAIL if f[0] not in ("A4", "B2")]
    print("\n================ VERDICT ================")
    print(f"  ok: {len(OK)}   WARN: {len(WARNS)}   FAIL: {len(FAIL)} "
          f"({len(glob)} global, {len(INVALID_CELLS)} cell-scoped)")
    if glob:
        print("\n  BLOCKING (the whole run would be meaningless):")
        for s_, m in glob: print(f"    [{s_}] {m}")
    if INVALID_CELLS:
        print("\n  INVALID CELLS (everything else stands; these must render as 'invalid'):")
        for c in INVALID_CELLS:
            print(f"    section {c['section']:5} {c['shape']:16} {c['why']}")
    Path(REPO/"conv_bench"/"preflight_result.json").write_text(
        json.dumps({"fail":FAIL,"warn":WARNS,"results":results,
                    "invalid_cells":INVALID_CELLS,"global_fail":glob}, indent=2, default=str))
    print(f"\n  -> conv_bench/preflight_result.json")
    if glob:
        print("  VERDICT: BLOCKED -- fix the global failures before running the suite")
    elif INVALID_CELLS:
        print(f"  VERDICT: RUNNABLE -- suite is structurally sound; {len(INVALID_CELLS)} cell(s) "
              f"will be rendered 'invalid' and must not be read as measurements")
    else:
        print("  VERDICT: CLEAN")
    return 2 if glob else 0

if __name__ == "__main__":
    sys.exit(main())
