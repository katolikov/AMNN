#!/usr/bin/env python3
"""Build a SELF-CONTAINED conv-probe bundle that runs on any machine with python3 + adb.

Pre-converts every model here (needs this repo's MNNConvert + onnx/numpy), precomputes the
correctness references, copies+strips the arm64 binaries, and drops in the standalone runner.
The resulting directory/tarball needs NO MNN repo, NO MNNConvert, NO numpy on the target machine.

    python3 conv_bench/make_bundle.py            # -> conv_bench/conv_probe_bundle[.tar.gz]
"""
import json, os, shutil, subprocess, sys, tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
import numpy as np
import onnx
from onnx import numpy_helper
from gen_conv import make_conv
from mk_chain import make_chain
from bench import convert, LIBS, MODULE
from block_fixture import load_blocks, build_onnx

OUT = REPO / "conv_bench" / "conv_probe_bundle"
CORES = [(32, 72, 96), (48, 36, 48), (96, 18, 24)]   # homogeneous 6-deep stride-1 cores

# Stride-2 "head" pairs: two 3x3 s2 convs that halve the spatial size twice. Each conv has a
# different shape, so these are reported PER CONV (matched by the shape tag MNN puts in the
# kernel name), not as one averaged number.
HEADS = [
    ("head18", [dict(cin=18, cout=16, H=288, W=384, stride=2, pad=1, prelu=1),
                dict(cin=16, cout=32, H=144, W=192, stride=2, pad=1, prelu=1)]),
    ("head34", [dict(cin=34, cout=32, H=144, W=192, stride=2, pad=1, prelu=1),
                dict(cin=32, cout=48, H=72,  W=96,  stride=2, pad=1, prelu=1)]),
    ("head64", [dict(cin=64, cout=64, H=72,  W=96,  stride=2, pad=1, prelu=1),
                dict(cin=64, cout=96, H=36,  W=48,  stride=2, pad=1, prelu=1)]),
]
CC = (32, 24, 48)                            # correctness shape: %16/%4 (LDS) and %6/%6 (fused2) ok
VARIANTS = ["conv_2d_c4h1w1", "conv_2d_c4h1w2", "conv_2d_c4h4w1", "conv_2d_c4h1w4",
            "conv_2d_c8h2w1", "conv_2d_c8h4w1", "conv_2d_c8h1w4", "conv_2d_c8h8w1",
            "conv_2d_c8h4w1_pa", "conv_2d_c8h1w1", "conv_2d_c4h8w1"]
SPEC_ONLY = ["conv_2d_c8h8w1", "conv_2d_c8h4w1_pa", "conv_2d_c8h1w1", "conv_2d_c4h8w1"]
LDS_TILES = ["16x4", "48x4", "16x12", "8x4", "24x4", "16x2"]


def find_strip():
    """llvm-strip that can handle arm64 ELF. The host `strip` on macOS CANNOT, and silently
    leaves the libs unstripped (135 MB instead of 6 MB, re-pushed to the device every run)."""
    ndk = os.environ.get("ANDROID_NDK") or os.environ.get("ANDROID_NDK_HOME") or ""
    cands = []
    if ndk:
        cands += list(Path(ndk).glob("toolchains/llvm/prebuilt/*/bin/llvm-strip"))
    for root in (Path.home() / "Library/Android/sdk/ndk", Path.home() / "Android/Sdk/ndk"):
        cands += sorted(root.glob("*/toolchains/llvm/prebuilt/*/bin/llvm-strip"), reverse=True)
    for c in cands:
        if c.exists():
            return str(c)
    return shutil.which("llvm-strip")


STRIP = None


def strip_to(src: Path, dst: Path):
    """Copy, stripping symbols (cuts each .so by ~20x -> much faster adb push)."""
    global STRIP
    if STRIP is None:
        STRIP = find_strip() or ""
        if not STRIP:
            print("   (no llvm-strip found — libs stay unstripped, pushes will be slow)")
    shutil.copy2(src, dst)
    if STRIP:
        subprocess.run(f"\"{STRIP}\" -S -x \"{dst}\"", shell=True, capture_output=True)


def conv3x3(x, W, b):
    Ci, H, Wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    y = np.tile(b[:, None, None], (1, H, Wd)).astype(np.float64)
    for kh in range(3):
        for kw in range(3):
            y += np.einsum('oc,chw->ohw', W[:, :, kh, kw].astype(np.float64),
                           xp[:, kh:kh + H, kw:kw + Wd].astype(np.float64))
    return y


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    for sub in ("bin", "models", "ref"):
        (OUT / sub).mkdir(parents=True)
    tmp = REPO / "conv_bench" / "_bundle_tmp"; tmp.mkdir(exist_ok=True)

    # ---------- binaries ----------
    print("== binaries ==")
    for f in LIBS + [MODULE]:
        f = Path(f)
        strip_to(f, OUT / "bin" / f.name)
        print(f"   {f.name}  {(OUT/'bin'/f.name).stat().st_size/1e6:.1f} MB")

    man = {"variants": VARIANTS, "spec_only": SPEC_ONLY, "lds_tiles": LDS_TILES,
           "cores": [], "heads": [], "blocks": [], "correctness": {}}

    # ---------- core models ----------
    print("== models ==")
    for (C, H, W) in CORES:
        key = f"{C}_{H}x{W}"
        # 6-deep chain +PReLU (fused) -> the sustained-load timing vehicle
        chain = tmp / f"core_{C}.onnx"; make_chain(str(chain), 1, C, H, W, depth=6, act="prelu", k=3, stride=1)
        convert(str(chain), str(OUT / "models" / f"core_{C}.mnn"), fp16=False, fuse_prelu=True)
        # single conv (direct baseline + fused2 vehicle)
        one = tmp / f"single_{C}.onnx"; make_conv(str(one), 1, C, C, H, W, 3, 3, 1, 1, 1, 1, "none")
        convert(str(one), str(OUT / "models" / f"single_{C}.mnn"))
        # im2col hijack: conv C -> C*9
        ic = tmp / f"ic_{C}.onnx"; make_conv(str(ic), 1, C, C * 9, H, W, 3, 3, 1, 1, 1, 1, "none")
        convert(str(ic), str(OUT / "models" / f"ic_{C}.mnn"))
        # GEMM proxy: 1x1 conv C*9 -> C
        gp = tmp / f"gp_{C}.onnx"; make_conv(str(gp), 1, C * 9, C, H, W, 1, 1, 1, 0, 1, 1, "none")
        convert(str(gp), str(OUT / "models" / f"gp_{C}.mnn"))
        man["cores"].append({
            "key": key, "label": f"{C}->{C}@{H}x{W}", "C": C, "H": H, "W": W,
            "model": f"core_{C}.mnn", "shape": [1, C, H, W], "depth": 6,
            "single_model": f"single_{C}.mnn",
            "im2col_model": f"ic_{C}.mnn",
            "gemm_model": f"gp_{C}.mnn", "gemm_shape": [1, C * 9, H, W]})
        print(f"   core {key}")

    # ---------- stride-2 head pairs ----------
    for key, convs in HEADS:
        p_onnx = tmp / f"{key}.onnx"
        build_onnx(str(p_onnx), convs)
        convert(str(p_onnx), str(OUT / "models" / f"{key}.mnn"), fp16=False, fuse_prelu=True)
        c0 = convs[0]
        man["heads"].append({
            "key": key,
            "label": " -> ".join(f"{c['cin']}->{c['cout']}@{c['H']}x{c['W']} s{c['stride']}"
                                 for c in convs),
            "model": f"{key}.mnn", "shape": [1, c0["cin"], c0["H"], c0["W"]],
            "convs": [{"label": f"{c['cin']}->{c['cout']}@{c['H']}x{c['W']} s{c['stride']}",
                       # MNN encodes the shape in the kernel name: ...b1ci18hi288wi384co16...
                       "tag": f"ci{c['cin']}hi{c['H']}wi{c['W']}co{c['cout']}"} for c in convs]})
        print(f"   head {key}: {man['heads'][-1]['label']}")

    # ---------- real blocks ----------
    try:
        for name, convs in load_blocks().items():
            c0 = convs[0]
            p = tmp / f"{name}.onnx"; build_onnx(str(p), convs)
            convert(str(p), str(OUT / "models" / f"{name}.mnn"), fp16=False, fuse_prelu=False)
            convert(str(p), str(OUT / "models" / f"{name}_fused.mnn"), fp16=False, fuse_prelu=True)
            man["blocks"].append({"key": name, "model": f"{name}.mnn",
                                  "fused_model": f"{name}_fused.mnn",
                                  "shape": [1, c0["cin"], c0["H"], c0["W"]]})
            print(f"   block {name}")
    except Exception as e:
        print(f"   (blocks skipped: {e})")

    # ---------- correctness model + references ----------
    C, H, W = CC
    cc = tmp / "cc.onnx"; make_conv(str(cc), 1, C, C, H, W, 3, 3, 1, 1, 1, 1, "none")
    convert(str(cc), str(OUT / "models" / "cc.mnn"))
    m = onnx.load(str(cc)); inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    Wt = [v for v in inits.values() if v.ndim == 4][0]
    bt = [v for v in inits.values() if v.ndim == 1][0]
    rng = np.random.default_rng(4242)
    x = rng.standard_normal([1, C, H, W]).astype(np.float32)
    np.savetxt(OUT / "ref" / "cc_input.txt", x.reshape(-1), fmt="%.6f")
    np.savetxt(OUT / "ref" / "fused2_ref.txt", conv3x3(conv3x3(x[0], Wt, bt), Wt, bt).reshape(-1), fmt="%.6f")
    man["correctness"] = {"model": "cc.mnn", "shape": [1, C, H, W]}
    print(f"   correctness cc {C}@{H}x{W} + conv^2 reference")

    (OUT / "manifest.json").write_text(json.dumps(man, indent=2))
    # run_report is required; the clock-pinning driver is optional (a repo that pins clocks with
    # its own tooling simply does not carry it).
    for src, dst, required in (("bundle_run_report.py", "run_report.py", True),
                               ("bundle_run_suite.py", "run_suite.py", False),
                               ("bundle_clocks.py", "clocks.py", False)):
        s = REPO / "conv_bench" / src
        if not s.exists():
            if required:
                raise SystemExit(f"missing required file: {s}")
            continue
        shutil.copy2(s, OUT / dst)
        os.chmod(OUT / dst, 0o755)

    (OUT / "README.md").write_text(f"""# Conv-strategy probe — self-contained bundle

Sets the clocks on one Android device, runs every convolution strategy we have implemented, and
writes a report that tells you **which one is fastest on that device**.

## Requirements
* `python3` (standard library only — no numpy, no onnx)
* `adb` on PATH, device connected and authorized (`adb devices` shows it)
* Nothing else. Models are pre-converted; the arm64 MNN libs are in `bin/`.

## Use — one command

```bash
python3 run_suite.py --list                  # show attached devices, copy the serial
python3 run_suite.py <ADB-SERIAL>            # pin all clocks to max, run everything (~10 min)
python3 run_suite.py <ADB-SERIAL> --quick    # ~4 min
```

Pick the clocks you want to measure at:

```bash
# slow GPU, fast memory — the case where the ranking is expected to change
python3 run_suite.py <ADB-SERIAL> --gpu min --mif max --int max

# repeat the whole suite at several GPU clocks and compare
python3 run_suite.py <ADB-SERIAL> --gpu-sweep 980,600,300
```

Clock specs: `max`, `min`, a number in MHz (snapped to the nearest supported step), or `none` to
leave that domain alone. Defaults are `--gpu max --mif max --int max`.

**Pinning needs root.** On a production build (`adb root` refused, no `su`) nothing can be pinned;
the suite still runs, records the clock the governor actually used, and says so at the top of the
report. A `--gpu min` request on such a device does NOT take effect — use a rooted or
engineering-build phone for low-clock numbers.

## Output
* `suite_<serial>.md` — **read this one**. Plain-English verdict first, then the clocks the numbers
  were taken at, every strategy per shape, and (with `--gpu-sweep`) whether the winner changes with
  the clock.
* `detail_gpu<spec>.md` — the full 12-section report at each clock point.
* `.json` next to each — every raw number, for plotting or diffing between devices.

## What it measures
1. Hardware (compute units, clock, LDS, vector widths, `subgroup_shuffle`)
2. GPU clock under load, start and end (validates the timings / detects throttling)
3. Subgroup broadcast/shuffle compile test
4. All {len(VARIANTS)} kernel strategies + LDS vs **MNN's own default** → picks the winner
5. LDS tile/workgroup sweep
6. im2col + GEMM, incl. the GEMM-vs-default headroom (implicit-GEMM lever)
7. Fused 2-layer megakernel
8. Real model blocks ± PReLU fusion (deployment numbers)
9. Concurrency (2 independent streams)
10. Correctness of every custom kernel (bad kernels get flagged so timings aren't trusted)
11. Recommendations for that device

## Notes
* Everything runs in `/data/local/tmp/convprobe` on the device.
* `run_report.py` is the single-clock runner; `run_suite.py` drives it and handles clocks. You can
  still call `run_report.py --serial <SERIAL>` directly if you pinned the clocks yourself.
* The custom strategies exist only in the bundled `libMNN_CL.so` (built from branch
  `opencl-conv-specialize`). A stock MNN build has none of them — use `conv_probe_source.patch`
  (shipped alongside this bundle) if you want to land these changes in your own MNN tree.
""")

    # ---------- tarball ----------
    tar = REPO / "conv_bench" / "conv_probe_bundle.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(OUT, arcname="conv_probe_bundle")
    shutil.rmtree(tmp, ignore_errors=True)
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\n== bundle: {OUT}  ({total/1e6:.0f} MB)   tarball: {tar} ({tar.stat().st_size/1e6:.0f} MB) ==")


if __name__ == "__main__":
    main()
