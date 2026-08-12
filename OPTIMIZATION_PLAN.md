# OPTIMIZATION_PLAN.md — MNN OpenCL (Buffer, FP16) Conv+Activation on Galaxy S26

> Living document. Written before heavy work; updated when the plan changes.
> Companion files: `STATUS.md` (timeline/state), `FINDINGS.md` (results).

## 0. Target & constraints (fixed)

- **Backend:** OpenCL **BUFFER memory mode only**. Never Image.
- **Precision:** FP16 (`half` storage *and* compute).
- **No quantization** (no int8/quantized paths — neither propose nor benchmark).
- **Device:** Galaxy **S26** → Exynos **2600** (`erd9965` / S5E9965) → **Samsung Xclipse 960 (AMD RDNA4)**.
- **Ops in scope:** Convolution + ReLU / PReLU / ReLU6. Nothing else.
- **No correctness regression** vs unmodified buffer-mode baseline (fp16 tolerance).
- **Hero shape:** Conv `W=[96,96,3,3]` (Cout=96, Cin=96, k=3×3, group=1, stride 1, pad 1),
  data `X=[N,96,H,W]`, **N≈8** (to confirm from real model; treat N as swept).

## 1. Confirmed environment (Phase 0 — measured, see FINDINGS/STATUS)

- Device attached over adb: `SM-S942B`, platform `erd9965` = **Exynos 2600**, CPU has
  i8mm/sve2/**sme2** (very recent core). adb server needed a restart to see it.
- GPU self-report (device Vulkan driver string, authoritative): **"Samsung Xclipse 960"**,
  Vulkan 1.4, **subgroup size = 64**, maxWorkgroup = 1024, **LDS/shared = 64 KB**,
  fp16 storage+compute = yes, int8 dot = yes, **coopmat = 0** (no WMMA exposed).
- **OpenCL on this device is ANGLE CL-over-Vulkan** (`CLPlatformVk.cpp`,
  `angle_cl_pipeline_cache.bin`) — *not* a native CL driver. Big implication: kernels are
  compiled to SPIR-V and run through Vulkan; `cl_intel_subgroups`/`cl_khr_subgroups`
  semantics, native vector widths, and LWS behavior are Vulkan/ANGLE-mediated, **must be
  measured, not assumed**. (`clinfo` from a shell uid enumerates 0 platforms; MNN's load
  path works — backend 3 ReluTest passes.)
- MNN classifies this GPU as **`RADEON`** (`OpenCLRuntime.cpp:164-168`): sets
  `isSetWorkGroupAttribute=true`; **does not** set a `mGpuLevel`; **subgroup conv/unary
  kernels are gated to `INTEL` only** (`:173`, `MNN_SUPPORT_INTEL_SUBGROUP`) → the
  `*_subgroup_buf.cl` path is **dead on this GPU**. (Hypothesis pre-falsified — see §3.)

### Still to confirm on-device (Phase 0 tail)
- OpenCL-exposed caps as ANGLE reports them: `CL_DEVICE_NAME/VENDOR/VERSION`,
  `MAX_COMPUTE_UNITS`, `PREFERRED/NATIVE_VECTOR_WIDTH_HALF`, `MAX_WORK_GROUP_SIZE`,
  `LOCAL_MEM_SIZE`, extension list (subgroups? fp16?). Will capture during first rebuild
  via a one-shot cap dump.

## 2. Tooling / harness decision

- **Timing engine:** reuse existing `build_android_profile` (arm64, `MNN_OPENCL=ON`,
  `MNN_ARM82=ON`, **`MNN_GPU_TIME_PROFILE=ON`** → `ENABLE_OPENCL_TIME_PROFILER`). This gives
  **true per-kernel GPU time** from CL profiling events (`OpenCLRuntime.cpp:1079`,
  "kernel time = … us <name>" + a conv-type breakdown). Exactly the Phase-1 instrument.
- **Driver:** `ModuleBasic.out model Dir [runMask] [forwardType=3] [loops] [gpuMode] [prec|mem mask] [cache]`.
  - Buffer + WIDE tuning: `gpuMode = MNN_GPU_MEMORY_BUFFER(64) | MNN_GPU_TUNING_WIDE(4) = 68`.
    Other tuning: NONE=65, HEAVY=66, NORMAL=72, FAST=80 (all `|64` for buffer).
  - Precision Low: mask `2` (precision = mask%4, memory = (mask/4)%4).
- **Single-op models:** export tiny ONNX conv(+act) graphs, convert with host
  `build_host/MNNConvert` (present), push `.mnn` + run. Sweep one dim at a time.
- For final wall-clock validation use a **non-profiling** build (profiling serializes the
  queue and perturbs absolute latency, though it's ideal for isolating kernel cost).
- Stability: pin power HIGH, fix tuning level per experiment, manage the LWS tuning cache
  deliberately (persist vs clear), warmup + **median of N** with spread, watch thermal.

## 3. Hypotheses (ranked: payoff × low-risk), each with a falsifier

| # | Hypothesis | Predicted effect | Falsifier / test |
|---|---|---|---|
| H1 | Default LWS / tuning level suboptimal for 96×96×3×3, N=8 | HEAVY/WIDE or hand LWS (mult. of 32/64) beats default | Sweep NONE→FAST→NORMAL→HEAVY→WIDE + manual LWS; if WIDE already optimal & flat, falsified |
| H2 | Activation runs as a **separate** op → extra global round-trip | Fusing ReLU/PReLU/ReLU6 into conv removes a full read+write of the output tensor | First **verify fusion status** in code + profiler op list; if already fused, falsified |
| H3 | Kernel under-uses native `half` width / packed FP16 | wider `half4/8` + right channel pack (C4 vs C8/C16) lifts throughput on RDNA4 | Measure preferred_vector_width_half; A/B vectorization variants; if no gain, mem-bound not ALU |
| H4 | 3×3 conv is compute-bound → **Winograd F(2,3)/F(4,3)** cuts MACs | fewer MACs → faster if compute-bound | Roofline first; if conv is memory-bound, Winograd won't help (falsified) |
| H5 | Insufficient input/weight reuse → global re-reads | LDS tiling of input/weights cuts global traffic | Roofline + bytes model; if already bandwidth-saturated near peak, falsified |
| H6 | A build flag / schedule option changes dispatched kernel for the better | different conv path (gemm1x1 / gemm2 / wino / ori) is faster | Force/observe which conv kernel runs per shape; compare |
| **X** | `*_subgroup_buf.cl` helps | — | **Pre-falsified:** gated to INTEL only; RADEON never dispatches it. Skip. |

## 3b. Direction confirmed by user (2026-06-30)

- **Activation = per-channel PReLU**, ONNX slope `[1,96,1,1]` (96 slopes). → **L1 (PReLU
  fusion) is the priority** (currently an unfused ~219 µs `PReLU0@output` kernel).
- **Safe wins first**: L1 + **hardware-friendly shape/size guidance**; defer L2/L3 kernel
  rewrites to a second pass ("rewrite kernel if needed").
- **Batch N:** user noted possible bias/batch confusion. Bias = `[96]`, so the "8" cannot
  be the bias length → it is the input **batch N=8** (as the task warned). Runtime is
  **perfectly linear in N** (measured), so N is treated as a swept parameter; L1 and
  shape-friendliness are N-independent. Not a blocker.

## 4. Phase gates

- **P0** Orient + reproducible baseline + caps  → *in progress*.
- **P1** Characterize runtime vs shape (sweeps, roofline, per-kernel GPU time) → correlation
  model in FINDINGS. Gate: stable numbers, variance reported.
- **P2** Freeze ranked hypotheses from P1 data.
- **P3** Optimize cheapest-safest first: (1) config/flags/tuning/power, (2) activation
  fusion, (3) layout/vector/LWS, (4) algorithmic (Winograd/GEMM tiling/1×1), (5) `*_buf.cl`
  source edits only. One change at a time, re-measure, decision-log each.
- **P4** Validate: CPU-vs-GPU op correctness (fp16 tol) for every kept change; end-to-end
  conv-stack + hero-shape re-time; thermal recheck.
- **P5** Report: model, ranked levers w/ numbers, best flags+config, kernel diffs+rationale.

## 5. Risk / guardrails

- Image mode kernels (`*_buf.cl` excluded) stay untouched.
- Every kept kernel change gets a CPU-reference correctness check before it counts.
- ANGLE-CL means some "RDNA truths" (wave32) may not surface as CL subgroups — verify each.
- Profiling build for kernel attribution; non-profiling build for headline latency.
