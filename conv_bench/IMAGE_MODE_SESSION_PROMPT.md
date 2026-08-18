# Session prompt — audit and port the whole strategy set to OpenCL IMAGE (texture) mode

Self-contained brief for a dedicated session. Everything below is measured on-device unless marked
as analysis. **Read "Traps" before writing any code, and read "Plan first" before writing any
kernel — this session is explicitly designed to plan, ask, and then build.**

## Why this session exists

The entire conv investigation (FINDINGS §B–§H.50, ~20 strategies) ran in OpenCL **buffer** mode. That
was never a considered choice: an early session hit `Alloc Image -40` and a **device reboot** on
tensors wider than the max 2D image size, and buffer mode was mandated globally from that point on.
The failure was real, but it was a *wide-tensor* failure that got over-generalised to everything.

The literature is emphatic that this is the wrong default for mobile GPUs. Adreno's OpenCL
best-practices paper reports local memory reaching only ~half the throughput of the texture cache
for convolution; TMModel (ICS'25) reports **1.48–3.61× on individual kernels** purely from modelling
2D texture memory; TFLite and MNN both pack tensors to 4 channels specifically to match the image
format.

**And a first measurement on this device agrees** (§H.47). Fair comparison — single conv, `act=none`
so the work is identical, both bit-exact against the CPU backend, median of 3, cooled:

| core | buffer | **image** | |
|---|---|---|---|
| 32→32@72×96 | **119.0** | 147.5 | +24.0% |
| **48→48@36×48** | 100.0 | **67.3** | **−32.7%** |
| 96→96@18×24 | 56.0 | 54.5 | −2.8% |

−32.7% is **larger than any kernel win found in the whole investigation** (forced Winograd −13%,
c4h4w2+HARD −18.7%). Image mode runs the conv as a 3-kernel decomposition (`Convolution0/1/2`).

## Your task

1. **Audit** every strategy in FINDINGS §B–§H.50 and classify each as: already exists in image mode /
   portable to image mode / meaningless in image mode / blocked. Produce that table *before* coding.
2. **Port and measure** the ones worth porting, adding each to `conv_bench/bundle_run_report.py` and
   `make_bundle.py` so they re-decide themselves on new hardware.
3. **Re-run the whole comparison** buffer-vs-image per shape, and state plainly which mode should be
   the default for which shape.

## Plan first — this is a requirement, not a suggestion

Before implementing anything, produce and share:
- the audit table above,
- which image-mode kernels MNN already has (`conv_2d.cl`, `winogradTransform.cl`, … — image mode is
  a **separate, fully-implemented backend**, not a flag on the buffer kernels),
- the image-size limits of this device vs every shape in the suite and in the real model,
- your proposed order of work, with the cheapest decisive measurement first.

**Ask questions if anything is unclear.** Specifically worth asking about: whether PReLU-fused models
must keep working (see below), whether the wide-tensor shapes in `pipeline/` are in scope, and
whether an accuracy regression is acceptable anywhere.

## The blocker you must solve first

**Image mode silently DROPS the fused PReLU.** On the PReLU-fused core models the image output
disagrees with buffer (cosine 0.457) while *matching* the CPU backend (0.999995) — because CPU and
the image conv both ignore `leakyReluSlope`, which only the buffer conv implements (§E). The very
first image-mode measurement was therefore an artifact: it was fast because it was doing less work.

The shipped model is converted with `MNN_FUSE_CONV_PRELU=1`, so **image mode is not usable on the
real model until the PReLU fusion is ported to the image conv kernels.** Estimated cost of *not*
porting it (running an unfused PReLU kernel instead) is ~6 µs at the 48-core output size, which
would leave the win around −28% — but that number is an ESTIMATE and has never been measured.

Recommended order: port PReLU fusion to the image conv → re-verify the −32.7% survives → only then
audit the rest.

## What the audit must cover (FINDINGS §B–§H.50)

Group them; do not treat 20 strategies as 20 independent ports.
- **Layout-level:** NC4HW4 vs NCHW (§H.36/§H.39) — note image mode is *inherently* 4-channel-packed,
  so "NCHW in image mode" may be meaningless. Say so if it is.
- **Kernel blocking:** the `conv_2d_c*` family, the 2-D register tiles (§H.21/§H.22), c4h4w2+HARD
  (§H.23/§H.34). Image mode has its own kernel family — the question is whether the *findings*
  transfer (8 accumulators optimal, 16 catastrophic), not whether the code does.
- **Algorithmic:** Winograd (§H.28/§H.29 — the one shipped win), im2col+GEMM (§H.38), implicit GEMM
  (§H.46), fusion (§H.25/§H.44), split-K (§H.31), LDS staging (§H.30).
- **Compile-time:** shape hardcoding (§H.23/§H.34/§H.40), constant weights (§H.49 — note image-mode
  kernels *already* use `__constant`, which may be part of why image mode wins).
- **Already known dead regardless of layout:** LLC placement (§H.35), cross-layer fusion (§H.1) —
  both fail because inter-layer traffic is measured at ~0, which is layout-independent. Do not redo.

## Device facts

Samsung Xclipse 960 (Exynos 2600, RDNA4-derived), 8 CUs, 980 MHz, 64 KB LDS, fp16 packed math.
OpenCL is **ANGLE → clspv → SPIR-V → Vulkan**, not a native driver: no `sub_group_shuffle`, no
cooperative matrix. Measured practical conv ceiling **~3.03 TFLOP/s**, but see §H.27 — that figure
was derived with the broken metric and needs re-deriving.
`gpuMode` values: **buffer = 68** (`MNN_GPU_MEMORY_BUFFER|WIDE`), **image = 132**
(`MNN_GPU_MEMORY_IMAGE|WIDE`). `MNN_GPU_MEMORY_IMAGE = 1<<7`.

## Traps that have already cost real sessions

0. **Metric.** MNN's `conv time` counter EXCLUDES the Winograd transform kernels (launched twice per
   conv). It turned a +15% regression into an apparent −45% win (§H.27). For anything that can
   change which conv implementation runs — and switching buffer↔image always does — use
   `conv_all_us()` / `conv_kernel_only_us()` from `session_measure.py`, never `conv_us()`.
1. **Ground truth must be the CPU backend.** Comparing a new arm against the previous *GPU* arm is
   not enough: it could not distinguish "image is wrong" from "buffer applies an extra correct
   activation". Only `forwardType 0` separated them (§H.47). Every correctness gate in this session
   must be against CPU.
2. **A stale `tdir/input.json` is indistinguishable from a compiler crash** — MNN segfaults before
   printing anything and stdout is block-buffered. Always push an `input.json` matching the model.
3. **The bundle ships its own binaries.** `run_report.py` stages `bin/` from the bundle, silently
   reverting the device to whatever was current when `make_bundle.py` last ran. Order is always
   `make_bundle.py` → `run_report.py` (§H.45).
4. **A flag reading *exactly* the no-flag value did not engage.** Profiler kernel names cannot
   identify which direct-path kernel ran — they all report as `ConvBuf2D-ori-…`. Instrument the gate
   (`MNN_CONV_NCHW_DEBUG=1` prints the emitted build options) rather than inferring.
5. **Image mode can REBOOT the device** on tensors wider than the max 2D image size. Query
   `CL_DEVICE_IMAGE2D_MAX_WIDTH/HEIGHT` first, compute every shape's image dimensions, and test
   smallest-first. The conv_bench cores are ≤768×72 and safe; `pipeline/` models are NOT.
6. **The device throttles ~2.75×** under sustained load. Interleave arms, cool between reps, inspect
   raw per-rep values. A regression check on a hot device is meaningless (198 µs vs 120 cooled).
7. `.cl` authoring: the codegen does not escape `"` (a quote in a comment breaks the generated C++);
   the first `__kernel` sits inside `#ifdef CONV_LOCAL_SIZE`; a macro must precede its first use in
   the file; `DEAL_NON_UNIFORM_DIM2` is backslash-continued. Always re-run
   `python3 opencl_codegen.py .` **from inside `cl/`**, and cmake-reconfigure if you add a new `.cl`.
8. **Four separate strategies this session were nearly falsified by their own first implementation**
   (2-D tile 2.8×, NCHW conv 3.4×, implicit GEMM 1.8×, and forced Winograd via a metric artifact 3×).
   A single bad measurement is not evidence an idea is bad. Before recording a negative, remove the
   known defects: no dynamically-indexed accumulators, no runtime integer division in inner loops,
   vector loads where the layout allows, weights pre-packed for the access pattern.

## Harness

Branch `opencl-conv-specialize` (repo `MNN`) or `feature/opencl-conv-specialize` (repo `AMNN`).
```bash
python3 conv_bench/session_measure.py --push      # stage the current build
python3 conv_bench/make_bundle.py                 # regenerate bundle (do this before the report)
cd conv_bench/conv_probe_bundle && python3 run_report.py --serial <SERIAL> --quick
```
Run one shape in image mode:
```bash
adb shell 'cd /data/local/tmp/convprobe && LD_LIBRARY_PATH=. \
  ./ModuleBasic.out core_48.mnn tdir 0 3 120 132 2 x.bin 2>&1'
```

## Deliverable

The audit table; PReLU fusion working in image mode (or a measured statement of what it costs not
to); every portable strategy measured in both modes; new report sections so the buffer-vs-image
decision re-runs on any device; and a FINDINGS §H.51+ entry stating plainly, per shape, which mode
wins and why. **Negative results are expected and are a perfectly good outcome** — most of this
investigation's value has been in well-mechanised negatives.
