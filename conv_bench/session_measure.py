#!/usr/bin/env python3
"""Small, direct measurement harness for one-off kernel/strategy experiments.

Deliberately NOT the full report suite: it pushes the CURRENT build, then runs a handful of
labelled arms interleaved and cooled, so a single hypothesis can be closed in a couple of minutes.

    python3 conv_bench/session_measure.py --push            # stage the freshly built libs
    python3 conv_bench/session_measure.py --arms ...        # (used from the driver scripts below)

Everything here assumes the conv-probe bundle is already staged on the device
(/data/local/tmp/convprobe) -- it only refreshes the binaries.

Traps this encodes (FINDINGS §H.24, §H.9 trap list):
  * an input.json matching the model is pushed before EVERY run (a stale one segfaults MNN
    before it prints anything, and looks exactly like a compiler crash);
  * every arm gets its own autotune cache file (sharing one makes some runs emit no output);
  * arms are interleaved and cooled, because the device throttles ~2.75x under sustained load.
"""
import argparse, json, math, re, statistics, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build_android_profile"
DEV = "/data/local/tmp/convprobe"
LIBS = [BUILD / "OFF/arm64-v8a/libMNN.so",
        BUILD / "express/OFF/arm64-v8a/libMNN_Express.so",
        BUILD / "source/backend/opencl/OFF/arm64-v8a/libMNN_CL.so"]
MODULE = BUILD / "ModuleBasic.out"
SERIAL = None
TMP = Path("/private/tmp/claude-501/-Users-sam-Documents-projects-MNN/"
           "46bcca62-befd-4ebe-afd6-d2e80817fca5/scratchpad")


def sh(cmd, timeout=900):
    s = f"adb -s {SERIAL} " if SERIAL else "adb "
    return subprocess.run(s + cmd, shell=True, text=True, capture_output=True,
                          timeout=timeout).stdout


def shell(cmd, timeout=900):
    return sh(f"shell '{cmd}'", timeout=timeout)


def push_build():
    """Strip + push the current build. Strip matters: unstripped libs are ~135 MB."""
    strip = None
    for root in (Path.home() / "Library/Android/sdk/ndk", Path.home() / "Android/Sdk/ndk"):
        c = sorted(root.glob("*/toolchains/llvm/prebuilt/*/bin/llvm-strip"), reverse=True)
        if c:
            strip = str(c[0]); break
    TMP.mkdir(parents=True, exist_ok=True)
    for f in LIBS + [MODULE]:
        d = TMP / f.name
        subprocess.run(f'cp "{f}" "{d}"', shell=True)
        if strip:
            subprocess.run(f'"{strip}" -S -x "{d}"', shell=True, capture_output=True)
        sh(f'push "{d}" {DEV}/{f.name}')
        print(f"   pushed {f.name} {d.stat().st_size/1e6:.1f} MB", flush=True)
    shell(f"chmod 755 {DEV}/ModuleBasic.out")


def run(model, shape, loops=120, env="", cache="p.bin", pull=False, timeout=900):
    """Push a matching input.json, run ModuleBasic, return (stdout, output floats or None)."""
    inj = TMP / "_input.json"
    inj.parent.mkdir(parents=True, exist_ok=True)
    inj.write_text(json.dumps({"inputs": [{"name": "input", "shape": shape}],
                               "outputs": ["output"], "shapeMutable": False}))
    sh(f'push "{inj}" {DEV}/tdir/input.json')
    if pull:
        shell(f"rm -rf {DEV}/output && mkdir -p {DEV}/output")
    out = shell(f"cd {DEV} && {env} LD_LIBRARY_PATH=. ./ModuleBasic.out {model} tdir 0 3 "
                f"{loops} 68 2 {cache} 2>&1", timeout=timeout)
    vals = None
    if pull:
        p = TMP / "_out.txt"
        if p.exists(): p.unlink()
        sh(f'pull {DEV}/output/0_0.txt "{p}"')
        if p.exists():
            try: vals = [float(x) for x in p.read_text().split()]
            except Exception: vals = None
    return out, vals


def conv_us(out, depth=1):
    """MNN's `conv time` counter (excludes the NCHW<->NC4HW4 rasters). First 3 loops dropped.

    *** DO NOT use this to compare against the Winograd path. *** This counter does NOT include
    the `Conv-winograd-rearrange` transform kernels, which are launched TWICE per conv and cost
    roughly as much as the batchgemm they feed. Scoring a Winograd arm with it credits the
    batchgemm half only and can turn a +15% regression into an apparent -45% win (this happened;
    see FINDINGS §H.27). For any arm that may change which conv IMPLEMENTATION runs, use
    conv_all_us() or total_us() instead.
    """
    c = [int(t) for t in re.findall(r"conv time = (\d+) us", out)][3:]
    return (statistics.median(c) / depth) if c else 0.0


def conv_all_us(out, depth=1):
    """Sum of EVERY conv-related kernel per loop -- winograd rearrange + batchgemm included.

    This is the implementation-agnostic conv cost, and the only safe metric when comparing conv
    paths against each other. Matches on the kernel-name events, so it counts each launch."""
    acc = {}
    for m in re.finditer(r"kernel time = (\d+)\s+us (\S+)", out):
        us, name = int(m.group(1)), m.group(2)
        if "onv" not in name:
            continue
        acc.setdefault(name.split("-total")[0], []).append(us)
    tot = sum(statistics.median(v[3:] if len(v) > 6 else v) * len(v) for v in acc.values())
    loops = len(re.findall(r"total kernel time = \d+  us", out))
    return (tot / loops / depth) if loops else 0.0


def total_us(out):
    t = [int(x) for x in re.findall(r"total kernel time = (\d+)  us", out)][3:]
    return statistics.median(t) if t else 0.0


def wall_ms(out):
    m = re.search(r"min\s*=\s*([\d.]+)\s*ms", out)
    return float(m.group(1)) if m else 0.0


def per_conv_us(out, tags):
    """Median GPU time per conv in a heterogeneous chain, matched on MNN's shape tag."""
    acc = {t: [] for t in tags}
    for m in re.finditer(r"kernel time = (\d+)\s+us (ConvBuf2D\S*|Conv\S*)", out):
        us, name = int(m.group(1)), m.group(2)
        for t in tags:
            if t in name:
                acc[t].append(us); break
    return {t: (statistics.median(v[3:]) if len(v) > 3 else 0.0) for t, v in acc.items()}


def cosine(a, b):
    if not a or not b: return float("nan")
    n = min(len(a), len(b))
    sa = sb = sab = 0.0
    for i in range(n):
        x, y = a[i], b[i]; sa += x * x; sb += y * y; sab += x * y
    return sab / (math.sqrt(sa) * math.sqrt(sb) + 1e-12)


def cool(seconds):
    if seconds > 0:
        print(f"   (cool {seconds}s)", flush=True)
        shell(f"sleep {seconds}", timeout=seconds + 60)


def interleaved(arms, reps=3, cool_s=8, verbose=True):
    """arms = {label: callable(rep)->float}. Round-robin so thermal drift hits every arm equally.
    Returns {label: (median, [raw...])} -- raw values are printed because a median spanning the
    throttle point is meaningless and you must be able to see that."""
    acc = {k: [] for k in arms}
    for i in range(reps):
        for k, f in arms.items():
            v = f(i)
            if v: acc[k].append(v)
            if verbose: print(f"   rep{i} {k:<34} {v:8.1f}", flush=True)
        cool(cool_s)
    return {k: ((statistics.median(v) if v else 0.0), v) for k, v in acc.items()}


def report(res, base_label=None):
    base = res[base_label][0] if base_label and res.get(base_label) else None
    print()
    for k, (m, raw) in res.items():
        d = f"  {100*(m-base)/base:+6.1f}%" if base else ""
        print(f"  {k:<34} {m:8.1f}{d}   raw={[round(x,1) for x in raw]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", "-s", default=None)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    global SERIAL
    SERIAL = a.serial or (subprocess.run("adb devices", shell=True, text=True,
                                         capture_output=True).stdout.splitlines()[1].split()[0])
    if a.push:
        push_build()


if __name__ == "__main__":
    main()
