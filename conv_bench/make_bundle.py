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
CORES = [(32, 72, 96), (48, 36, 48)]        # homogeneous cores of the real blocks
CC = (32, 24, 48)                            # correctness shape: %16/%4 (LDS) and %6/%6 (fused2) ok
VARIANTS = ["conv_2d_c4h1w1", "conv_2d_c4h1w2", "conv_2d_c4h4w1", "conv_2d_c4h1w4",
            "conv_2d_c8h2w1", "conv_2d_c8h4w1", "conv_2d_c8h1w4", "conv_2d_c8h8w1",
            "conv_2d_c8h4w1_pa"]
SPEC_ONLY = ["conv_2d_c8h8w1", "conv_2d_c8h4w1_pa"]
LDS_TILES = ["16x4", "48x4", "16x12", "8x4", "24x4", "16x2"]


def strip_to(src: Path, dst: Path):
    """Copy, stripping symbols if a strip tool is available (cuts bundle size a lot)."""
    shutil.copy2(src, dst)
    for tool in ("llvm-strip", "strip"):
        if shutil.which(tool):
            r = subprocess.run(f"{tool} -S -x \"{dst}\"", shell=True, capture_output=True)
            if r.returncode == 0:
                return
    # stripping is optional; unstripped just means a bigger bundle


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
           "cores": [], "blocks": [], "correctness": {}}

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
    shutil.copy2(REPO / "conv_bench" / "bundle_run_report.py", OUT / "run_report.py")
    os.chmod(OUT / "run_report.py", 0o755)

    (OUT / "README.md").write_text(f"""# Conv-strategy probe — self-contained bundle

Measures every convolution strategy we have implemented against **one Android device** and writes a
markdown report. Finds the **best configuration on that device**.

## Requirements
* `python3` (standard library only — no numpy, no onnx)
* `adb` on PATH, device connected and authorized (`adb devices` shows it)
* Nothing else. Models are pre-converted; the arm64 MNN libs are in `bin/`.

## Use
```bash
python3 run_report.py --list                      # show attached devices
python3 run_report.py --serial <SERIAL> -o report.md
python3 run_report.py --serial <SERIAL> --quick    # ~4 min instead of ~10
```
Then send back `report.md` (plus the clock you pinned, if any).

## GPU clock
The script does **not** change the clock — pin it yourself first if you want a specific one, e.g.
(needs root):
```bash
adb -s <SERIAL> shell 'su -c "echo 980000 > /sys/kernel/gpu/gpu_min_clock"'
adb -s <SERIAL> shell 'su -c "echo 980000 > /sys/kernel/gpu/gpu_max_clock"'
```
Without root those nodes are not writable (they are `system`-owned); the script instead uses a
long sustained run, which drives the governor to its top rail, and it **samples
`/sys/kernel/gpu/gpu_clock` under load** so §2 of the report records the clock every number was
taken at. The clock changes the compute/memory balance, so it can change which strategy wins —
if you re-pin, re-run.

## What it measures
1. Hardware (compute units, clock, LDS, vector widths, `subgroup_shuffle`)
2. GPU clock under load (validates the timings)
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
