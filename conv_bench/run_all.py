#!/usr/bin/env python3
"""ONE COMMAND: build everything, push it to the phone, measure every conv strategy, write a report.

    python3 conv_bench/run_all.py                    # everything, ~25 min the first time
    python3 conv_bench/run_all.py --quick            # ~10 min
    python3 conv_bench/run_all.py -s <ADB-SERIAL>    # pick the device
    python3 conv_bench/run_all.py --skip-build       # reuse what is already built
    python3 conv_bench/run_all.py --rebuild          # force a fresh build

What it does, in order:
  1. preflight   — adb device, python deps, NDK, cmake; tells you exactly what is missing
  2. host build  — MNNConvert (converts the test models)
  3. android build — libMNN / libMNN_CL / libMNN_Express / ModuleBasic.out
                     (MNN_GPU_TIME_PROFILE=ON is mandatory: without it every timing is 0)
  4. models      — generates + converts every test model, writes a manifest
  5. push + run  — stages the libs on the device and measures every strategy
  6. report      — writes report_<serial>.md next to a .json of every raw number

Clocks are NOT touched. Pin them yourself first if you want a specific one; the report samples
and records the clock every number was taken at, and re-checks it at the end to catch throttling.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CB = REPO / "conv_bench"
sys.path.insert(0, str(CB))

HOST_BUILD = REPO / "build_host"
ANDROID_BUILD = REPO / "build_android_profile"
BUNDLE = CB / "conv_probe_bundle"
NEEDED_LIBS = ["libMNN.so", "libMNN_CL.so", "libMNN_Express.so", "ModuleBasic.out"]


def ok(m):   print(f"  \033[32m✓\033[0m {m}", flush=True)
def bad(m):  print(f"  \033[31m✗\033[0m {m}", flush=True)
def step(m): print(f"\n\033[1m== {m}\033[0m", flush=True)


def sh(cmd, cwd=None, log=None):
    """Run a shell command, streaming to a log file. Returns True on success."""
    print(f"  $ {cmd}", flush=True)
    with open(log, "a") if log else subprocess.DEVNULL as f:
        r = subprocess.run(cmd, shell=True, cwd=cwd,
                           stdout=f if log else subprocess.DEVNULL,
                           stderr=subprocess.STDOUT if log else subprocess.DEVNULL)
    return r.returncode == 0


def preflight(serial):
    """Check everything BEFORE spending 20 minutes on a build. Returns (ok, serial, ndk)."""
    step("1. Preflight")
    fail = []

    if shutil.which("adb"):
        ok("adb found")
    else:
        bad("adb not on PATH — install platform-tools"); fail.append("adb")

    devs = []
    if shutil.which("adb"):
        out = subprocess.run("adb devices", shell=True, text=True, capture_output=True).stdout
        devs = [l.split()[0] for l in out.splitlines()[1:] if "\tdevice" in l]
    if serial and serial not in devs:
        bad(f"device '{serial}' not attached (attached: {devs or 'none'})"); fail.append("device")
    elif not serial and len(devs) == 1:
        serial = devs[0]; ok(f"device {serial}")
    elif not serial and len(devs) > 1:
        bad(f"several devices attached, pass -s <serial>: {devs}"); fail.append("device")
    elif not serial:
        bad("no device attached / authorized (`adb devices` is empty)"); fail.append("device")
    else:
        ok(f"device {serial}")

    for mod in ("numpy", "onnx"):
        try:
            __import__(mod); ok(f"python module {mod}")
        except ImportError:
            bad(f"python module {mod} missing — pip3 install {mod}"); fail.append(mod)

    def is_ndk(p):
        return Path(p, "build/cmake/android.toolchain.cmake").exists()

    ndk = os.environ.get("ANDROID_NDK") or os.environ.get("ANDROID_NDK_HOME") or ""
    if ndk and is_ndk(ndk):
        ok(f"NDK {ndk}")
    elif ndk:
        bad(f"ANDROID_NDK={ndk} has no build/cmake/android.toolchain.cmake"); fail.append("ndk")
    else:
        # only real NDK directories -- the sdk/ndk folder also collects .dmg/.zip downloads
        guess = [p for root in (Path.home() / "Library/Android/sdk/ndk",
                                Path.home() / "Android/Sdk/ndk")
                 for p in sorted(root.glob("*"), reverse=True) if p.is_dir() and is_ndk(p)]
        if guess:
            ndk = str(guess[0]); ok(f"NDK auto-detected: {ndk}")
        else:
            bad("no NDK — set ANDROID_NDK=/path/to/ndk/<version>"); fail.append("ndk")

    if shutil.which("cmake"): ok("cmake found")
    else: bad("cmake not on PATH"); fail.append("cmake")

    if fail:
        print(f"\n  Fix the above and re-run. Missing: {', '.join(fail)}")
        return False, serial, ndk
    return True, serial, ndk


def build_host(rebuild, log):
    step("2. Host build (MNNConvert)")
    exe = HOST_BUILD / "MNNConvert"
    if exe.exists() and not rebuild:
        ok(f"already built: {exe}"); return True
    HOST_BUILD.mkdir(exist_ok=True)
    if not sh("cmake .. -DMNN_BUILD_CONVERTER=ON -DCMAKE_BUILD_TYPE=Release",
              cwd=HOST_BUILD, log=log) or \
       not sh(f"make -j{os.cpu_count()} MNNConvert", cwd=HOST_BUILD, log=log):
        bad(f"host build failed — see {log}"); return False
    ok("MNNConvert built"); return True


def build_android(rebuild, ndk, log):
    step("3. Android build (libMNN_CL + ModuleBasic.out)")
    have = all((ANDROID_BUILD / p).exists() for p in
               ("OFF/arm64-v8a/libMNN.so",
                "source/backend/opencl/OFF/arm64-v8a/libMNN_CL.so",
                "ModuleBasic.out"))
    if have and not rebuild:
        ok("already built — rebuilding only what changed")
    ANDROID_BUILD.mkdir(exist_ok=True)
    if not (ANDROID_BUILD / "CMakeCache.txt").exists() or rebuild:
        cfg = (f"cmake .. -DCMAKE_TOOLCHAIN_FILE={ndk}/build/cmake/android.toolchain.cmake "
               "-DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 "
               "-DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON -DMNN_ARM82=ON "
               "-DMNN_GPU_TIME_PROFILE=ON -DMNN_BUILD_TOOLS=ON -DMNN_SEP_BUILD=ON "
               "-DMNN_LOW_MEMORY=ON")
        if not sh(cfg, cwd=ANDROID_BUILD, log=log):
            bad(f"cmake configure failed — see {log}"); return False
    # MNN_GPU_TIME_PROFILE is what makes per-kernel timings exist at all
    cache = (ANDROID_BUILD / "CMakeCache.txt").read_text()
    if "MNN_GPU_TIME_PROFILE:BOOL=ON" not in cache:
        bad("this build tree has MNN_GPU_TIME_PROFILE=OFF — every timing would be 0. "
            "Re-run with --rebuild."); return False
    if not sh(f"make -j{os.cpu_count()}", cwd=ANDROID_BUILD, log=log):
        bad(f"android build failed — see {log}"); return False
    ok("android libs built"); return True


def build_models(quick):
    step("4. Test models")
    if (BUNDLE / "manifest.json").exists() and (BUNDLE / "bin" / "libMNN_CL.so").exists():
        # always refresh: the libs may have just been rebuilt
        pass
    import make_bundle
    make_bundle.main()
    ok(f"models + binaries staged in {BUNDLE}")
    return True


def measure(serial, quick, cooldown, out):
    step("5-6. Measure on device + write report")
    sys.path.insert(0, str(BUNDLE))
    spec = importlib.util.spec_from_file_location("run_report", BUNDLE / "run_report.py")
    rr = importlib.util.module_from_spec(spec)
    sys.modules["run_report"] = rr
    spec.loader.exec_module(rr)
    argv = ["--serial", serial, "-o", str(out)]
    if quick: argv.append("--quick")
    if cooldown is not None: argv += ["--cooldown", str(cooldown)]
    rr.main(argv)
    return True


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("-s", "--serial", help="adb serial (auto if only one device)")
    ap.add_argument("--quick", action="store_true", help="fewer repeats (~10 min total)")
    ap.add_argument("--skip-build", action="store_true", help="reuse what is already built")
    ap.add_argument("--rebuild", action="store_true", help="reconfigure and rebuild from scratch")
    ap.add_argument("--cooldown", type=int, default=None,
                    help="GPU idle seconds between heavy sections (default 20, 5 with --quick)")
    ap.add_argument("-o", "--out", default=None, help="report path")
    a = ap.parse_args()

    log = REPO / "conv_bench" / "run_all_build.log"
    if log.exists(): log.unlink()

    good, serial, ndk = preflight(a.serial)
    if not good: sys.exit(1)

    if not a.skip_build:
        if not build_host(a.rebuild, log): sys.exit(1)
        if not build_android(a.rebuild, ndk, log): sys.exit(1)
    else:
        step("2-3. Build skipped (--skip-build)")

    if not build_models(a.quick): sys.exit(1)

    out = Path(a.out) if a.out else CB / f"report_{serial}.md"
    measure(serial, a.quick, a.cooldown, out)

    print(f"\n\033[1m== done ==\033[0m\n  report: {out}\n  raw numbers: {out.with_suffix('.json')}")
    print("  build log: " + str(log))


if __name__ == "__main__":
    main()
