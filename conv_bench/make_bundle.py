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
import block_fixture
from block_fixture import load_blocks, build_onnx

OUT = REPO / "conv_bench" / "conv_probe_bundle"
# Homogeneous 6-deep stride-1 cores. Every kernel-strategy section of the report iterates over
# CORES (and HEADS); a conv that appears in neither list gets NO strategy coverage at all, which is
# why the small-channel convs of Block3/Block4 were unmeasured until they were added here.
# 8@288x384 and 16@144x192 are 63.7 MMAC/conv -- identical to 32@72x96, so they extend the existing
# MAC-matched comparison rather than introducing a new workload class.
# Cores and heads are the ORIGINAL sizes put through the same divisor the blocks use, so a family
# cannot describe two different models. They were previously hardcoded per family, which is exactly
# how CONV_BENCH_SHAPES=full came to build full-size blocks alongside 1/3-size cores and heads.
_CORES_FULL = [(32, 72, 96), (48, 36, 48), (96, 18, 24), (8, 288, 384), (16, 144, 192)]
CORES = [(c, block_fixture.scale(h, f"core{c} H"), block_fixture.scale(w, f"core{c} W"))
         for c, h, w in _CORES_FULL]

# Stride-2 "head" pairs: two 3x3 s2 convs that halve the spatial size twice. Each conv has a
# different shape, so these are reported PER CONV (matched by the shape tag MNN puts in the kernel
# name), not as one averaged number. head1's cin=1 is the interesting case: NC4HW4 pads it to 4
# channels, a 4x input-read tax, where the other heads only pay 6-11% (cin=18/34).
_HEADS_FULL = [
    ("head18", [(18, 16, 288, 384), (16, 32, 144, 192)]),
    ("head34", [(34, 32, 144, 192), (32, 48, 72, 96)]),
    ("head64", [(64, 64, 72, 96),   (64, 96, 36, 48)]),
    ("head1",  [(1, 8, 576, 768),   (8, 16, 288, 384)]),
]
HEADS = [(k, [dict(cin=ci, cout=co, stride=2, pad=1, prelu=1,
                   H=block_fixture.scale(h, f"{k} H"), W=block_fixture.scale(w, f"{k} W"))
              for ci, co, h, w in pair])
         for k, pair in _HEADS_FULL]

CC = (32, 24, 48)                            # correctness shape: %16/%4 (LDS) and %6/%6 (fused2) ok
# Second correctness shape in the REDUCED regime. The first one (24x48) is nine times the
# area of anything the suite now times, so a kernel that is only wrong on a short/narrow
# output -- a partial last w-block, a halo bound that only bites when out_w < the tile --
# would pass it and still corrupt every timed shape. 12x24: %4 w-blocks, %6/%6 for fused2,
# and out_h=12 is small enough to exercise the h-remainder paths.
CC2 = (48, 12, 24)
VARIANTS = ["conv_2d_c4h1w1", "conv_2d_c4h1w2", "conv_2d_c4h4w1", "conv_2d_c4h1w4",
            "conv_2d_c8h2w1", "conv_2d_c8h4w1", "conv_2d_c8h1w4", "conv_2d_c8h8w1",
            "conv_2d_c8h4w1_pa", "conv_2d_c8h1w1", "conv_2d_c4h8w1",
            # 2-D register tiles (ILP-M/HNTMP geometry). Kept in the suite even where they lose on
            # the reference device: the winner is device- and clock-dependent, and c4h4w2 already
            # beats MNN's default by 8.5% on one core (FINDINGS §H.21).
            "conv_2d_c4h4w2", "conv_2d_c4h2w2", "conv_2d_c4h2w4", "conv_2d_c4h4w4"]
SPEC_ONLY = ["conv_2d_c8h8w1", "conv_2d_c8h4w1_pa", "conv_2d_c8h1w1", "conv_2d_c4h8w1",
             "conv_2d_c4h4w2", "conv_2d_c4h2w2", "conv_2d_c4h2w4", "conv_2d_c4h4w4"]
# these hardcode stride 1; forcing them on a stride-2 conv silently measures the default instead
STRIDE1_ONLY = ["conv_2d_c4h4w2", "conv_2d_c4h2w2", "conv_2d_c4h2w4", "conv_2d_c4h4w4"]
# kernels that read the HC_* compile-time constants, i.e. respond to MNN_CONV_HARD=1.
# Keep in sync with the .cl: a kernel listed here but not using the macros just measures twice.
HARD_CAPABLE = ["conv_2d_c4h4w2", "conv_2d_c4h2w2", "conv_2d_c4h2w4", "conv_2d_c4h4w4",
                # shape-specialised copies of the stock kernels (conv_2d_hc_buf.cl). Only
                # meaningful WITH MNN_CONV_HARD -- without it they are their originals.
                "conv_2d_c4h1w1_hc", "conv_2d_c4h1w2_hc", "conv_2d_c4h1w4_hc",
                "conv_2d_c4h4w1_hc", "conv_2d_c8h4w1_hc", "conv_2d_c8h2w1_hc", "conv_2d_c8h1w4_hc"]
SPEC_ONLY += HARD_CAPABLE   # every one of them needs MNN_CONV_SPEC to enter the candidate list
# useLDS requires out_w % TILE_W == 0 and out_h % TILE_H == 0. NONE of the original tiles divide
# 24x18 (core96), so MNN_CONV_LDS was accepted and then silently rejected there -- the arm
# re-measured the default and LDS was unmeasurable at that shape (preflight §B2). 24x2 and 8x6
# both divide 24x18 exactly.
# At the reduced shapes the old tile list stops dividing: core96's output is 6x8, so only
# "8x6" of the original eight qualifies (TILE_W must divide out_w=8, TILE_H divide out_h=6),
# and core48's is 12x16. A tile that does not divide is ACCEPTED and then silently rejected
# inside the kernel selector, so the arm re-measures the default and LDS reads as "no effect"
# at exactly the shapes where it would matter most (preflight B2). Small tiles added below.
LDS_TILES = ["16x4", "48x4", "16x12", "8x4", "24x4", "16x2", "24x2", "8x6",
             "8x2", "8x3", "4x2", "4x3", "4x6", "16x6", "16x3"]
# LDS staging modes (env MNN_CONV_LDS=<mode>). "1" = the original 1-output/thread kernel;
# "w2" = c4h1w2 blocking + LDS, which isolates LDS at constant register blocking (FINDINGS §H.30).
# w2 needs out_w % (2*TILE_W) == 0, so its tile must be chosen per shape.
LDS_MODES = ["1", "w2"]
# Split-K over the input-channel reduction (env MNN_CONV_SPLITK=<n>, FINDINGS §H.31). Loses to the
# autotuned default here, but wins 11.7% at constant blocking on the most output-starved shape --
# on a device with more CUs, or a model with smaller outputs, the balance can flip.
SPLITK_FACTORS = [2, 4, 8]
# Env flags that change which conv IMPLEMENTATION runs. Anything measured across these MUST use
# conv_all_us()/total kernel time, never MNN's `conv time` counter (FINDINGS §H.27).
IMPL_SWITCHING_ENVS = ["MNN_FORCE_WINOGRAD", "MNN_CONV_LDS", "MNN_CONV_SPLITK", "MNN_NO_WINOGRAD",
                       "MNN_CONV_NCHW", "MNN_CONV_IMGEMM", "MNN_CONV_IGEMM"]
# Plain-NCHW conv path (env MNN_CONV_NCHW=1, FINDINGS §H.36). Loses on this model's shapes but wins
# at C>=48 with large spatial, and has no channel-padding cliff -- re-decide on new hardware/shapes.
# Its profiler events split: gemm2-0 = layout in, gemm2-2 = layout out, ori-* = the conv itself.
NCHW_MODE = True
# im2col + GEMM in NCHW (env MNN_CONV_IMGEMM=1, FINDINGS §H.38). Falsified here (+83..120%), kept in
# the suite because the loss is bandwidth-per-MAC driven and a device with more L2 per CU, or a
# shape with much larger N, could plausibly flip it. NOTE: at C>=64 MNN takes the Winograd path
# before any MNN_CONV_* flag is read -- pair it with MNN_NO_WINOGRAD=1 or every arm measures the same thing.
IMGEMM_MODE = True
# Implicit GEMM (env MNN_CONV_IGEMM=1 for NC4HW4, =nchw for NCHW; FINDINGS §H.46). Gathers the
# im2col columns on the fly, so nothing is materialised. Falsified here (+98..300%) because it is
# LDS-throughput bound -- 9 LDS accesses per 8 mad instructions at the 8-float4 register budget.
# Kept in the suite: a device that tolerates LARGER register tiles (check whether c4h4w4 still
# regresses in section 4) would amortise that traffic and could flip it.
IGEMM_MODES = ["1", "nchw"]


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

    man = {"variants": VARIANTS, "spec_only": SPEC_ONLY, "stride1_only": STRIDE1_ONLY,
           "hard_capable": HARD_CAPABLE,
           "lds_tiles": LDS_TILES,
           "lds_modes": LDS_MODES,
           "splitk_factors": SPLITK_FACTORS,
           "impl_switching_envs": IMPL_SWITCHING_ENVS,
           "igemm_modes": IGEMM_MODES,
           # Which shape family this bundle was BUILT from. The report prints it at the top:
           # the bundle carries pre-converted .mnn files, so a bundle built for one family and
           # read as the other is undetectable from the numbers alone.
           "shape_family": block_fixture.SHAPE_FAMILY,
           "cores": [], "heads": [], "blocks": [], "correctness": {}, "correctness2": {}}

    # ---------- core models ----------
    print("== models ==")
    for (C, H, W) in CORES:
        key = f"{C}_{H}x{W}"
        # 6-deep chain +PReLU (fused) -> the sustained-load timing vehicle
        chain = tmp / f"core_{C}.onnx"; make_chain(str(chain), 1, C, H, W, depth=6, act="prelu", k=3, stride=1)
        convert(str(chain), str(OUT / "models" / f"core_{C}.mnn"), fp16=False, fuse_prelu=True)
        # UNFUSED twin of the same chain. Needed as CPU ground truth for the image-mode section:
        # CPU ignores Convolution2DCommon.leakyReluSlope, so running the FUSED model on CPU yields
        # the un-activated answer (cosine ~0.45-0.56 against a correct GPU arm) and would report a
        # false MISMATCH. Gate = CPU on this model vs GPU on the fused one (FINDINGS §H.51).
        convert(str(chain), str(OUT / "models" / f"core_{C}_unfused.mnn"), fp16=False, fuse_prelu=False)
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
            "unfused_model": f"core_{C}_unfused.mnn",
            "im2col_model": f"ic_{C}.mnn",
            "gemm_model": f"gp_{C}.mnn", "gemm_shape": [1, C * 9, H, W]})
        print(f"   core {key}")

    # ---------- stride-2 head pairs ----------
    for key, convs in HEADS:
        p_onnx = tmp / f"{key}.onnx"
        build_onnx(str(p_onnx), convs)
        convert(str(p_onnx), str(OUT / "models" / f"{key}.mnn"), fp16=False, fuse_prelu=True)
        # UNFUSED twin, same reason as the cores: CPU does not implement leakyReluSlope, so a fused
        # model on CPU returns the un-activated result and any cosine against it is a FALSE
        # mismatch. Without this twin the heads have no independent ground truth at all and their
        # correctness is simply unchecked (preflight §D).
        convert(str(p_onnx), str(OUT / "models" / f"{key}_unfused.mnn"), fp16=False, fuse_prelu=False)
        c0 = convs[0]
        man["heads"].append({
            "key": key,
            "label": " -> ".join(f"{c['cin']}->{c['cout']}@{c['H']}x{c['W']} s{c['stride']}"
                                 for c in convs),
            "model": f"{key}.mnn", "unfused_model": f"{key}_unfused.mnn",
            "shape": [1, c0["cin"], c0["H"], c0["W"]],
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
            # Only emit the fused model where there is a PReLU to fold. For a bare-conv block the
            # fused convert is a no-op, and the report would score plain-vs-plain as "no saving".
            has_prelu = any(c["prelu"] for c in convs)
            if has_prelu:
                convert(str(p), str(OUT / "models" / f"{name}_fused.mnn"), fp16=False, fuse_prelu=True)
            man["blocks"].append({"key": name, "model": f"{name}.mnn",
                                  "fused_model": (f"{name}_fused.mnn" if has_prelu else None),
                                  "has_prelu": has_prelu,
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

    # Second correctness model, in the reduced regime. Same construction, smaller output, so a
    # kernel that only breaks on a short/narrow output plane cannot pass unnoticed.
    C2, H2, W2 = CC2
    cc2 = tmp / "cc2.onnx"; make_conv(str(cc2), 1, C2, C2, H2, W2, 3, 3, 1, 1, 1, 1, "none")
    convert(str(cc2), str(OUT / "models" / "cc2.mnn"))
    m2 = onnx.load(str(cc2)); i2 = {t.name: numpy_helper.to_array(t) for t in m2.graph.initializer}
    Wt2 = [v for v in i2.values() if v.ndim == 4][0]
    bt2 = [v for v in i2.values() if v.ndim == 1][0]
    x2 = np.random.default_rng(1717).standard_normal([1, C2, H2, W2]).astype(np.float32)
    np.savetxt(OUT / "ref" / "cc2_input.txt", x2.reshape(-1), fmt="%.6f")
    np.savetxt(OUT / "ref" / "fused2_ref2.txt",
               conv3x3(conv3x3(x2[0], Wt2, bt2), Wt2, bt2).reshape(-1), fmt="%.6f")
    man["correctness2"] = {"model": "cc2.mnn", "shape": [1, C2, H2, W2],
                           "input": "cc2_input.txt", "fused2_ref": "fused2_ref2.txt"}
    print(f"   correctness cc2 {C2}@{H2}x{W2} + conv^2 reference")

    (OUT / "manifest.json").write_text(json.dumps(man, indent=2))
    # run_report is required; the clock-pinning driver is optional (a repo that pins clocks with
    # its own tooling simply does not carry it).
    for src, dst, required in (("bundle_run_report.py", "run_report.py", True),
                               ("bundle_run_suite.py", "run_suite.py", False),
                               ("bundle_clocks.py", "clocks.py", False),
                               # the integrity gate must travel WITH the bundle: run_suite.py runs
                               # it before any timing, and without it no cell can be validated
                               ("preflight.py", "preflight.py", False),
                               # the result store travels with the bundle: it is what records
                               # each number's baseline env and refuses cross-batch comparison,
                               # so a bundle without it can still produce the class of confident,
                               # wrong table this store exists to prevent.
                               ("bench_store.py", "bench_store.py", False),
                               ("test_bench_store.py", "test_bench_store.py", False)):
        s = REPO / "conv_bench" / src
        if not s.exists():
            if required:
                raise SystemExit(f"missing required file: {s}")
            continue
        shutil.copy2(s, OUT / dst)
        os.chmod(OUT / dst, 0o755)

    (OUT / "README.md").write_text(f"""# Conv-strategy probe — self-contained bundle

**Before any timing, `run_suite.py` runs `preflight.py` (~1 min) and gates on it.** Preflight is a
measurement-INTEGRITY audit, not a benchmark: it proves each arm actually engages, that comparisons
are not confounded by algorithm choice, that the parser sees every dispatch, and that every case is
numerically correct against CPU. It writes `preflight_result.json`, which `run_report.py` reads to
mark cells `invalid` where an arm does not measure what its column header claims. That file is a
per-device, per-build artifact and is intentionally not version-controlled -- it is regenerated on
each run. A GLOBAL preflight failure aborts the suite; cell-scoped failures only mask those cells.
Override with `--skip-preflight`, which produces an UNVALIDATED report.

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
