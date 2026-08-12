# Session prompt — Custom Vulkan-compute convolutions on Xclipse 960

Paste everything below as the opening prompt of a new session. It is self-contained.

---

You are a senior GPU performance engineer. A prior effort exhaustively optimized fp16 3×3
convolutions for a fixed model on a **Samsung Xclipse 960** (Exynos 2600, RDNA-derived, **8 compute
units**, 980 MHz, 64 KB LDS, subgroup size 64) through **MNN's OpenCL backend (buffer, fp16)** — and
hit a hard wall. This session changes the substrate: **implement the convolutions as custom Vulkan
compute shaders in buffer mode**, to exploit primitives the OpenCL toolchain doesn't expose. A
device is attached over adb (serial from `adb devices`; it drops off USB — recover with
`adb kill-server && adb start-server`, keep awake with `adb shell svc power stayon usb`). Measure on
it; do not guess.

## Why Vulkan (the whole point)

On this device OpenCL is **ANGLE→Clspv** (CL→SPIR-V→Vulkan), which:
- does **not** expose `sub_group_shuffle` (`cl_khr_subgroup_shuffle` absent), and
- routes everything through MNN's autotuned direct kernels, which already sit at the occupancy sweet
  spot for these shapes.

**Vulkan compute (GLSL/SPIR-V) removes both limits:** it exposes the full `GL_KHR_shader_subgroup`
family — `subgroupShuffle`, `subgroupShuffleUp/Down`, `subgroupBroadcast`, `subgroupQuadSwap` — plus
explicit control of subgroup size (wave32 vs wave64), shared memory, and direct SPIR-V. The specific
idea OpenCL could NOT do and Vulkan CAN: **subgroup-shuffle input-halo sharing** — lanes computing
adjacent output pixels exchange loaded input via register-to-register shuffle (no LDS, no barriers),
cutting the 9× redundant global input reads *without* the occupancy/barrier cost that made the LDS
approach 1.34× slower under OpenCL.

## The diagnosis you're inheriting (verify, don't re-derive)

The target convs are small-channel/large-spatial 3×3 (Cin/Cout 16–48), **latency/occupancy-bound**:
~1 TFLOP/s ≈ 15% of compute peak, ~2% of BW peak — the GPU idles ~85% waiting on memory latency.
The one lever that helps is more concurrent work in flight; but on 8 CUs, and because every
register-adding kernel change kills occupancy, the OpenCL direct path is maxed. Full falsified-ideas
table and numbers: `conv_bench/OPTIMIZATION_HANDOFF.md` and `FINDINGS.md §H`. **OpenCL baseline to
beat: the `32→32 @72×96` core conv runs at ~119 µs/conv (fp16, buffer, sustained).**

## Target conv set (fixed shapes — specialize hard)

All 3×3, group 1, dilation 1, pad 1, batch 1, fp16, each followed by per-channel PReLU:
- **Cores (where the time is):** `6× 32→32 @72×96 s1` and `6× 48→48 @36×48 s1`.
- Heads (stride 2): `18→16 @288×384`, `16→32 @144×192`, `34→32 @144×192`, `32→48 @72×96`.
Data layout in MNN buffers is **NC4HW4** = `[C/4][B][H][W][4]` (channel-block outermost); weights are
reordered NC4HW4. Match it if interoperating with the MNN graph, or use your own layout inside a
self-contained Vulkan path and convert at the boundary.

## Hard constraints (never violate)

- **Vulkan BUFFER storage** (SSBO), fp16 (`VK_KHR_16bit_storage` + `shaderFloat16`). Never image.
- fp16 storage + compute. No quantization.
- Numerically equivalent to the OpenCL fp16 baseline within fp16 tolerance, **verified on device**.
- Empirical: one variable per experiment, median of ≥3 under sustained load, report spread. These
  small convs have ~2× DVFS noise in isolation — measure in a sustained self-chain, thermal-control
  by alternating A/B runs.
- Test on device before every commit. Commit only when asked; work on a dedicated branch.

## What already exists (build on it, don't restart)

- Branch **`vulkan-conv-exploration`** has prior Vulkan probes: `conv_bench/vk_bench.py`,
  `vk_correctness.py`, `vk_sweep.py`, and a hand GEMM shader `conv_bench/vkgemm/gemm_m8n4_fp16.comp`
  + `gemm_probe.cpp`. **Key prior finding to respect:** MNN's *stock* Vulkan backend conv is a
  **naive** direct shader (~0.3 TFLOP/s, ~18× slower than OpenCL) — so do NOT benchmark MNN's stock
  Vulkan conv and conclude Vulkan is slow; the goal is a **custom, well-tiled** compute shader.
- Vulkan device caps already dumped once: subgroup size **64**, maxWG 1024, LDS **64 KB**, fp16 yes,
  **cooperative-matrix NOT exposed** (no WMMA), 8 CUs.
- OpenCL comparison harness + the falsified-idea switches live on branch `opencl-conv-specialize`
  (`MNN_CONV_FORCE`, `MNN_CONV_LDS`, `MNN_DUMP_CL_EXT`, `MNN_SUBGROUP_PROBE`) and
  `conv_bench/probe_device.py` — reuse for apples-to-apples timing vs OpenCL.

## Central question & ranked hypotheses (each falsifiable, measure it)

Can a custom Vulkan compute conv beat MNN's OpenCL direct path (~119 µs on the core) for the fixed
set? Levers, best-first:

1. **Subgroup-shuffle halo sharing** (the reason we're here): one subgroup (wave of 64) covers a row
   /tile of output; each lane loads one input pixel; `subgroupShuffleUp/Down` supplies the ±1 / ±2
   neighbours for the 3×3 taps — 9× fewer global loads, **no LDS, no barrier**, occupancy preserved.
   Falsifier: measure achieved GB/s + TFLOP/s vs OpenCL; if the shuffle overhead ≥ the load savings,
   it won't beat 119 µs.
2. **Explicit wave32** (`VK_EXT_subgroup_size_control`): force subgroup size 32 and re-tune tiles —
   RDNA is native wave32; ANGLE-CL may have forced wave64. Different occupancy/latency profile.
3. **Direct SPIR-V / packed fp16** (`v_pk_fma` via `GL_EXT_shader_16bit_storage` + explicit `f16vec`
   math), and register-tuned output micro-tiles chosen for the 8-CU occupancy curve.
4. **Cut dispatch/submission overhead**: on the OpenCL side ~50% of per-inference wall was CPU/
   submission overhead. In Vulkan, batch the whole block's dispatches into one command buffer, one
   submit, using barriers only where dependencies require — a wall-time win independent of the kernel.
5. **Persistent/cooperative tiling across the homogeneous 6-conv core** kept in shared memory or
   registers (note: OpenCL cross-layer fusion was falsified on a *traffic* basis — <2% — so justify
   any fusion by dispatch-overhead reduction, not memory traffic).
6. **Cooperative matrix (`VK_KHR_cooperative_matrix`) — CHECK FIRST, could be a step change.** The
   Xclipse 960 did NOT expose coopmat, but this is a *newer* Exynos — if it exposes it, an im2col +
   coopmat-tile conv gets hardware matrix throughput. Even at small N=Cout it may beat the direct
   path. Probe `vulkaninfo | grep -i cooperative` (or the Vulkan feature query) on day one.
7. **⭐ HEADLINE: implicit-GEMM conv, and a fused implicit-GEMM megakernel across the chain.** This is
   the strongest lead from the OpenCL work — but calibrate it honestly. MEASURED: a real explicit
   im2col+GEMM was +15% slower than direct (48→48: 37µs im2col + 80µs GEMM = 117 vs 102), BUT the
   **GEMM reduce alone (80µs @ 0.90 TFLOP/s) beat the direct 3×3 (102µs @ 0.70)** for the SAME 35.8M
   MACs — im2col inflating the reduction dim fixes the occupancy starvation. The killer is
   materializing the 9×-wider im2col tensor to global (37µs). **The idea:** gather the 3×3 columns
   on-the-fly via `subgroupShuffle` (register-level, no LDS, no barrier) and matmul them — an
   **implicit GEMM** that never writes im2col to global; then **fuse consecutive convs** into one
   megakernel so intermediates never round-trip either. **Honest caveat the OpenCL work established:**
   the direct conv is ALREADY a fused implicit GEMM at 0.70 TFLOP/s — the 0.90 exists only because
   im2col pre-cleaned the access pattern; a shuffle gather must recover most of that 0.90 *without*
   LDS/barriers (LDS gather was falsified: +9%, L2 already caches the reuse) for this to win. So it's
   a genuine research bet, not a sure thing — but it's the ONE lever that needs exactly what Vulkan
   adds over OpenCL/Clspv (shuffle) + optionally coopmat (#6). **Target: beat 102µs on the
   48→48-class core, then beat the 6-conv chain (612µs direct) with a fused implicit-GEMM megakernel.**
8. **ION / dma_buf zero-copy** (`VK_EXT_external_memory_dma_buf` / AHardwareBuffer import): zero-copy
   the camera Y-plane input, and share branch-output buffers across sessions to kill the cross-context
   copy in the two-session concurrency merge. A wall-time / integration lever, orthogonal to the kernel.

## Method (phased)

P0 reproduce the OpenCL baseline number for the core convs (run `probe_device.py` for the current
device's per-variant table). P1 stand up a minimal custom Vulkan compute path that runs one 3×3 conv
on device and matches the fp16 reference (correctness gate first). P2 implement hypothesis #1
(subgroup shuffle) and measure vs 119 µs. P3 apply wave32 / packed-fp16 / tile tuning one variable at
a time. P4 validate end-to-end + numerical equivalence. Keep an append-only findings log.

## Deliverables

A custom Vulkan-compute conv with measured before/after vs the OpenCL baseline (per core shape), an
on-device correctness gate (fp16 tol), and a clear verdict: **does direct Vulkan access to subgroup
shuffle (and wave32 / explicit tiling) beat MNN's OpenCL direct path for these occupancy-bound
convs, and by how much** — or is the 8-CU occupancy ceiling fundamental regardless of substrate?

Start by: confirming the device + Vulkan capabilities on-device (`vulkaninfo` or a tiny probe) —
specifically **fp16 (shaderFloat16 + 16bit_storage), subgroup size + `subgroupShuffle`, and
`VK_KHR_cooperative_matrix`** (the three primitives that could each change the game and that OpenCL/
Clspv hid). Then reproduce the OpenCL baseline for the core convs (`conv_bench/probe_device.py`), and
stand up the minimal correct Vulkan conv before optimizing.
