# Conv Optimization — Findings & Open Directions (handoff)

> Self-contained summary of an on-device convolution-optimization investigation, written to hand
> to another engineer/LLM to find **further** ways to cut runtime. Everything below was measured on
> real hardware unless flagged as analysis. If you propose an idea already in the "Falsified" table,
> read its row first — it was tried and why it failed is usually instructive.

## TL;DR

- **Target:** a fixed inference model's convolutions on a **Samsung Xclipse 960** GPU (Exynos 2600,
  RDNA-derived, **8 compute units**, 980 MHz, 64 KB LDS), via **MNN**'s OpenCL backend in **buffer
  mode, fp16**. No quantization. Single specific device+model → full custom-kernel freedom allowed.
- **The convs** are small-channel / large-spatial 3×3 (Cin/Cout 16–48), and are **latency /
  occupancy-bound**: they run at ~1 TFLOP/s ≈ **15% of compute peak** and ~**2% of memory-bandwidth
  peak** — saturating *neither*. The GPU is idle ~85% of the time waiting on memory latency.
- **Net result of a long search:** exactly **one shippable kernel-adjacent win (PReLU fusion, −8/9%)**
  plus **one live non-kernel lever (cross-stream concurrency, ~1.3×)**. **Seven** distinct
  kernel/algorithm approaches were falsified on-device, and the single genuinely-new mechanism
  (subgroup shuffle) is **blocked by the OpenCL toolchain**, not the hardware.
- **Central conclusion:** the limiter is the **OpenCL-via-ANGLE-via-Clspv toolchain + a tiny 8-CU
  GPU at its occupancy sweet spot**, NOT a lack of kernel effort. The most promising unexplored
  directions therefore change the *substrate* (Vulkan compute, which exposes subgroup shuffle) or a
  *newer device* (more CUs / shuffle exposed), not the kernel shape.

## Target details

Two independent submodel "blocks", each a linear chain of 3×3 convs, **all followed by per-channel
PReLU**, batch=1, group=1, dilation=1, pad=1:

| block | head convs (stride 2) | homogeneous core (stride 1) |
|---|---|---|
| Block1 | 18→16 @288×384; 16→32 @144×192 | **6× 32→32 @72×96** |
| Block2 | 34→32 @144×192; 32→48 @72×96 | **6× 48→48 @36×48** |

The model runs **two of these blocks in parallel** (independent branches), then a common tail.

**Hard constraints:** OpenCL buffer mode only (image mode can reboot this device on wide tensors);
fp16 storage + compute; no quantization; numerically equivalent to the fp16 baseline (fp16 tol).

## The diagnosis (this is the key to everything)

Measured on the `32→32 @72×96` core conv (the dominant shape), sustained load:

| | value |
|---|---|
| measured per-conv time | **~119 µs** |
| compute-bound floor (@ 6.5 TFLOP/s peak) | ~78 µs |
| bandwidth-bound floor (@ 460 GB/s peak) | ~8 µs |
| achieved compute util | ~15% |
| achieved BW util | ~2% |

119 µs sits far above *both* floors → the kernel saturates neither ALU nor memory bus. It is
**latency-bound**: each thread issues ~360 global loads for ~288 MACs (input re-read 9× across the
3×3 taps), and those loads aren't hidden. The only thing that helps a latency-bound kernel is
**occupancy** (more concurrent waves to hide latency) — and MNN's autotuner already sits at the
occupancy sweet spot for these shapes. With only **8 compute units**, there simply isn't much
machine to fill.

## Falsified on-device (do not re-propose without reading why)

| # | idea | result | why it failed |
|---|---|---|---|
| 1 | NC4HW4→NCHW input layout | minor | the layout raster is one-time; C=1 padding waste rides one stem conv only |
| 2 | Cross-layer fusion ("megakernel", keep intermediates on-chip) | <2% ceiling | not bandwidth-bound; inter-layer traffic is <1 µs/conv (per-conv time is flat vs chain depth). Full-6 also LDS-infeasible |
| 3 | LDS input-halo tiling | **1.34× slower** standalone; **+0.7%/+9.2%** at constant blocking | reuse is already L2-cached; barriers serialize waves + LDS allocation cuts occupancy = the two things latency-bound kernels can't afford |
| 4 | Register-block unroll (opencl_unroll_hint) | +0.0% | compiler already schedules the small loops well; not ILP-limited |
| 5 | space2depth on stride-2 heads | **+12–17% slower** | s2d 4×'s Cin (16–34 already past the crossover) → ~1.78× more dense FLOPs + extra rasters. Only wins for Cin≤4 (a true C=1 stem) |
| 6 | Larger blocking `c8h8w1` (16 accumulators) | **+48.7% slower** | 2× registers → fewer resident waves → worse latency hiding. Generalizes to any 16-acc tile |
| 7 | Partial accumulators `c8h4w1_pa` (break FMA chain for ILP) | **+29.1% slower** | ILP *worked* (beat c8h8w1 by ~20 pts) but the +registers occupancy cost still dominated. Also targets FMA latency, not the actual memory latency |
| — | Subgroup halo-share (shuffle) | **toolchain-blocked** | device has `cl_khr_subgroups` (broadcast/reduce) but ANGLE's **Clspv has NO `sub_group_shuffle`** (`has_subgroup_shuffle=0`). Neighbor-sharing needs shuffle |

**Unifying pattern:** occupancy is the binding constraint on an 8-CU GPU. *Every* kernel idea that
adds registers (bigger tiles, partial sums, prefetch buffers) loses; the only idea that removes
registers (micro register-minimization) can't free a whole wave's worth on an already-lean kernel;
and the one idea that sidesteps registers (subgroup shuffle) isn't exposed by the CL toolchain.

## Banked / live levers

- ✅ **PReLU fusion** (`MNN_FUSE_CONV_PRELU=1` at convert) — **−8/9% per block**, bit-exact, no new
  code. Every conv here has PReLU. **Ship this.**
- ✅ **Cross-stream concurrency** — two independent streams complete in **~1.34× solo, not 2×** →
  real GPU spare capacity (expected on an occupancy-starved GPU). Run the two parallel block
  instances as **two MNN sessions on two threads (two queues)**, NOT as two branches in one graph
  (which serializes on MNN's single queue, measured −3.4%). Caveats: measured on a profiling build
  where ~50% of wall is CPU/submission overhead, so part of the win is overlapping overhead; the
  clean number needs a two-thread + non-profiling harness (not yet built).
- 🔎 **~50% of per-inference wall is CPU/submission/ANGLE overhead** (wall ~2× the GPU-kernel sum).
  Reducing dispatch/flush count is an untouched **wall-time** lever independent of the kernel.

## Hardware facts (Xclipse 960, from `MNN_DUMP_CL_EXT`)

```
name=Samsung Xclipse 960   vendor=Samsung   OpenCL 3.0 (Samsung proprietary driver 25.x)
max_compute_units=8        max_clock=980MHz  max_work_group_size=1024
local_mem=64KB             global_mem=4GB    pref_vec_half=8  native_vec_half=2
extensions include: cl_khr_fp16, cl_khr_subgroups, cl_khr_integer_dot_product,
                    cl_khr_il_program(spir-v), ... but NOT cl_khr_subgroup_shuffle
CL is compiled via ANGLE→Clspv (CL→SPIR-V→Vulkan); NOT a native CL driver.
Vulkan caps (separate query): subgroup size 64, maxWG 1024, LDS 64KB, fp16 yes, coopmat=NO.
```

## Open directions for further optimization (ranked by "changes the substrate", not the kernel)

1. **Vulkan compute shaders instead of OpenCL.** Vulkan/GLSL/SPIR-V exposes `subgroupShuffle`,
   `subgroupBroadcast`, explicit control of subgroup size (wave32 vs wave64), and direct SPIR-V —
   i.e. the primitives Clspv hides. The subgroup halo-share conv (idea "—" above), impossible in
   OpenCL here, **is implementable in Vulkan**. This is the single most promising unexplored lever.
   (Note: a prior test of MNN's *existing* Vulkan backend found it uses a **naive** direct shader
   ~0.3 TFLOP/s, ~18× slower than OpenCL — so the goal is a **custom** Vulkan compute conv, not MNN's
   stock one.) A dedicated session prompt exists: `VULKAN_SESSION_PROMPT.md`.
2. **A newer device.** Re-run `probe_device.py` on newer silicon. Watch for `has_subgroup_shuffle=1`
   (unlocks the subgroup conv in OpenCL too), a higher `max_compute_units` (more occupancy headroom
   → bigger tiles like c8h8w1 might stop regressing), or a different autotuner winner. More CUs
   directly raise the concurrency ceiling.
3. **Attack the ~50% CPU/submission overhead**, not the kernel: batch the block's dispatches into
   one submission / minimize `clFlush`, or use MNN's record/replay queue. Pure wall-time win.
4. **Cross-stream concurrency, properly** (two sessions / two threads / two queues) — the measured
   ~1.3× is real; a clean non-profiling two-thread harness would quantify the deployable number and
   the merge/common-tail cost.
5. **Model-level (needs retraining — biggest ceiling):** widen channels toward the 128–256 sweet
   spot at equal MACs (measured up to 2.5× at equal MACs in the occupancy-starved regime). The convs
   are slow because Cout 16–48 is architecturally deep in the occupancy-starved zone; no kernel fixes
   that, but a fatter-shallower design would. Also: space2depth *does* pay for a true C=1 stem.
6. **im2col + GEMM — IMPLEMENTED & MEASURED end-to-end: falsified as separate passes, but the GEMM
   compute wins.** Real fused im2col kernel (`im2col_3x3s1`, verified cosine=1.0) + real 1×1 GEMM:
   48→48@36×48 = 37µs(im2col)+80µs(GEMM)=**117µs vs 102µs direct (+15%)**; 32→32 = 233µs vs 119µs.
   The separate im2col pass (write 1.5MB @ ~80 GB/s, latency-bound on 8 CUs) costs 37µs — more than
   the GEMM's 22µs compute advantage. **BUT the GEMM compute alone (80µs) beats the direct 3×3 (102µs)**
   because im2col inflates channels out of the occupancy-starved regime. **Honest caveat:** the
   direct 3×3 conv is ALREADY an implicit GEMM (gather+reduce fused, 0.70 TFLOP/s); the explicit
   GEMM reduce hits 0.90 *only because* im2col pre-arranged the data — fuse the gather back in and
   you drift back toward 0.70, not 0.90. So implicit GEMM is an *uncertain* lead, not a proven win;
   capturing the 0.90 without materializing im2col needs a cheaper gather (LDS falsified — L2 already
   caches; subgroup-shuffle is Vulkan-only). The idea's real forms — (a) **implicit GEMM** via Vulkan
   subgroupShuffle, (b) **coopmat GEMM**, (c) more CUs — all need a different substrate/device (see #7
   and §H.14/§H.15 in FINDINGS). Env: MNN_CONV_IM2COL, MNN_NO_WINOGRAD.
7. **Cooperative matrix (WMMA) — NOT available on the Xclipse 960** (Vulkan `coopmat=0`), so untried.
   **Re-check on newer silicon** (`VK_KHR_cooperative_matrix` / a coopmat probe). If exposed, hardware
   matrix tiles are a *major* lever even at modest N; belongs in the Vulkan session.
8. **ION / dma_buf zero-copy buffers — exposed but untested.** `cl_arm_import_memory` +
   `cl_khr_external_memory_dma_buf` are present. Uses: (a) zero-copy the camera Y-plane input (no
   upload); (b) import the branch-output buffer across contexts to AVOID the cross-context copy in the
   two-session concurrency merge. (LLC itself isn't controllable, but is already why LDS gave nothing.)
9. **Long shots probably dead here** (documented so you don't repeat): Winograd (contraindicated for
   K≤128 — transforms cost more than the matmul saved).

### Methodology caveats (for airtight closure)
- All custom variants were measured via `MNN_CONV_FORCE` (bypasses kernel *selection*) and each still
  got the tuner's LWS search — so the autotuner did NOT mask them. EXCEPT the **LDS kernel used a
  fixed hand-picked LWS (16×4)**, not the tuner's search → a per-variant **LWS sweep** (esp. for LDS,
  and a HEAVY vs WIDE exhaustive search generally) is the one untested refinement. Unlikely to flip
  the big regressions (+48.7%, +29%), could matter for near-ties.

## Reproduce / re-test everything

**One command re-runs every strategy on any device and writes a shareable markdown report:**

```bash
python3 conv_bench/probe_device.py <adb-serial> -o report.md     # ~8-12 min (--quick: ~4 min)
```

Sections: (1) HW caps incl. **compute units** + `subgroup_shuffle`; (2) **clock-stability sampling
under load** (the report validates its own timings); (3) subgroup broadcast/shuffle compile-test;
(4) all 9 kernel variants forced + LDS, per core shape; (5) LDS tile/LWS sweep; (6) im2col+GEMM incl.
the **GEMM-vs-direct headroom** column; (7) fused 2-layer megakernel; (8) real Block1/Block2 ± PReLU
fusion; (9) 2-stream concurrency; (10) correctness gates for every custom kernel; (11) **AUTO-FLAGS**
— reads the results and says what changed vs the Xclipse 960 (more CUs → re-test occupancy-limited
levers; shuffle available → build the implicit-GEMM conv; different variant winning → re-run the
blocking search). Every number is auto-compared to the reference baked into the script.

Requires the `opencl-conv-specialize` build (`build_android_profile` + `build_host/MNNConvert`) —
that libMNN_CL carries the env switches (`MNN_DUMP_CL_EXT`, `MNN_SUBGROUP_PROBE`, `MNN_CONV_SPEC`,
`MNN_CONV_FORCE`, `MNN_CONV_LDS`, `MNN_LDS_TILE`, `MNN_CONV_IM2COL`, `MNN_CONV_FUSED2`,
`MNN_NO_WINOGRAD`). Full experiment log with numbers: `FINDINGS.md` §H.
