# STATUS.md — living log

## Current state (top = newest)

**Phase:** P4 — **PReLU fusion IMPLEMENTED + VALIDATED on device.** Opt-in (env
`MNN_FUSE_CONV_PRELU`), winograd-only scope → default/other backends unchanged.
**Result:** total kernel **3336→3130 µs (−~200 µs, ≈−18% on the conv+PReLU pair)**, the
separate `PReLU0` kernel is **gone**, correctness **cosine 0.999996** (CPU-unfused vs
OpenCL-fused). conv_us unchanged → PReLU absorbed for free into the Winograd dest kernel.
**Next (optional):** CPU/other-backend support to move pass into default chain; extend to
non-winograd conv paths; or L2/L3 deeper kernel work.
**Hero baseline:** conv GPU **1025 µs** (+PReLU **+219 µs** unfused = 1244), buffer fp16,
980 MHz, Winograd, ±1%. CPU-vs-OpenCL correctness cosine **0.999997**.
**Blockers:** none (no root → governor still hits 980 MHz max under load → stable timing).

**Delivered safe wins:** (1) **Cout-multiple-of-16/32 rule** (96→97 = +24%; Cin alignment
irrelevant) — see FINDINGS §B/§D. (2) Confirmed **buffer+WIDE+Low (gpuMode 68, mask 2)**
config is optimal; persist tuning cache; `MNN_GPU_TIME_PROFILE=OFF` for production. (3)
Validated correctness harness (`conv_bench/correctness.py`).

**Key reframe:** hero conv already on MNN's best path (Winograd + ReLU/ReLU6 fused + WIDE
optimal). H1 (tuning) + H2 (ReLU fusion) **falsified**; subgroup path **dead** (INTEL-gated).
Surviving levers: **L1 PReLU fusion (−219 µs, designed)**, L2 Winograd F(4,4) (UNIT hardcoded
=2, needs new transform kernels), L3 batchgemm efficiency (5.3→6.5 TFLOP/s headroom).

**Harness:** `build_android_profile` (arm64, OpenCL+ARM82+`MNN_GPU_TIME_PROFILE=ON`) +
`ModuleBasic.out`. Buffer+WIDE = gpuMode `68`; precision Low = mask `2`. Driver:
`conv_bench/bench.py` (gen ONNX → host MNNConvert → push → run → parse). Results in
`conv_bench/results.jsonl`. Metric = steady-state `conv_us` median.

---

## Timeline (append-only)

### 2026-06-30 — P0 orientation
- adb showed no device; `adb kill-server && start-server` recovered it → `SM-S942B` (USB).
- Identified SoC: prop `ro.board.platform = erd9965` = **Exynos 2600** (S5E9965); CPU
  reports sme2/sve2/i8mm. **This is the S26-class target, not S25** — corrected a stale
  memory note that said Exynos2500/Xclipse940.
- Device Vulkan driver self-reports **"Samsung Xclipse 960"** (from existing on-device
  `dl3dv_warm.log`): subgroup=64, maxWG=1024, LDS=64KB, fp16=1, int8dot=1, coopmat=0.
- OpenCL caps via on-device `clinfo`: enumerates **0 platforms** from a shell uid (Samsung
  ICD `libOpenCL_samsung.so` = ANGLE). Not a real blocker — MNN's own load path works.
- **Discovery: OpenCL = ANGLE CL-over-Vulkan** (`CLPlatformVk.cpp`,
  `angle_cl_pipeline_cache.bin` in run output). Strategy must treat CL as Vulkan-mediated.
- Found existing arm64 build `build_android_profile` with `MNN_OPENCL=ON MNN_ARM82=ON
  MNN_GPU_TIME_PROFILE=ON MNN_BUILD_TEST=ON`. Pushed `libMNN.so/libMNN_Express.so/
  libMNN_CL.so/run_test.out` to `/data/local/tmp/mnnopt`.
- **Validated OpenCL path:** `run_test.out ReluTest 3 2 1` → "√√√ all ReluTest tests passed"
  (backend 3 = OpenCL, precision Low). Confirms buffer-capable fp16 OpenCL runs from shell.
- Read `OpenCLRuntime.cpp`: GPU classed **RADEON**; `isSetWorkGroupAttribute=true`; no
  `mGpuLevel` set for RADEON; **subgroup kernels INTEL-gated** → dead here.
- Wrote `OPTIMIZATION_PLAN.md`, this file, and `FINDINGS.md` skeleton.

#### Open items
- Capture OpenCL-exposed caps (preferred_vector_width_half, CU count, extensions) — planned
  one-shot dump during first rebuild.
- Confirm N (batch) of hero conv with the user once the real model is available.

### 2026-06-30 — P1 characterization (device = today's Xclipse 960)
- Built `conv_bench/` harness (gen_conv.py + bench.py). Hero conv → **Winograd** path,
  conv GPU **1023–1028 µs** @ 980 MHz, ±1% stable. Wall 5.5–11 ms (noisy, ignored).
- Sweeps (see FINDINGS §B): N linear (128 µs/batch); channels & spatial → eff TFLOP/s
  *rises* with size (compute/occupancy-bound, not BW-bound), asymptote ≈6.5 nominal; hero
  C96 at 5.3 (≈20% headroom). Only 3×3 gets Winograd; 1×1→gemm2 (eff 0.65), 5×5→ori (1.5).
- **Activation fusion:** ReLU/ReLU6 fused (no delta); **PReLU NOT fused** → separate
  `PReLU0@output` kernel ~219 µs. Tuning: WIDE≈HEAVY≈optimal, NONE/FAST ~12% worse.
- Wrote FINDINGS §B (model + roofline) and §C (ranked levers). H1+H2 falsified; subgroup dead.

### 2026-06-30 — Safe wins + PReLU fusion scoped
- **Hardware-friendly shapes** (FINDINGS §B): channel cliff at **Cout 96→97 = +24%**
  (tile=16 output ch); **Cin alignment ~irrelevant** (96→97 = +2%); spatial mild even/odd;
  larger aligned Cout most MAC-efficient. Rule: keep Cout mult-of-16/32, pad Cin freely.
- **Config (FINDINGS §D):** buffer+WIDE+Low (gpuMode 68, mask 2) measured optimal; persist
  tuning cache; profiling OFF for production.
- **Correctness harness** (`conv_bench/correctness.py`): CPU fp32 vs OpenCL buffer fp16 on
  hero conv(+prelu) = cosine 0.999997, rel ~0.3%. PReLU neg-branch exercised + matches.
- **PReLU fusion designed (FINDINGS §E):** unfused 219 µs is memory-bound (~460 GB/s ≈ peak)
  → fusion saves ~all of it (−18% on hero). Needs schema field + converter pass + per-channel
  slope in conv kernels (no runtime backend-gated fuse hook exists → cross-backend landmine).
  Staged plan written. **Awaiting go-ahead before this surgery.**

### 2026-06-30 — PReLU fusion implemented (L1)
Patch (opt-in, winograd-only, no cross-backend landmine):
- `schema/default/CaffeOp.fbs`: + optional `leakyReluSlope:[float]` on Convolution2DCommon;
  regen `schema/current/*_generated.h` via `schema/generate.sh`.
- `winogradTransform_buf.cl` `winoTransDstBuf2_3_1`: + `#ifdef PRELU` slope arg + branch
  `res = fmax(res,0)+slope*fmin(res,0)` at all 4 output positions; regen `*_mnn_cl.cpp`.
- `ConvBufWinograd.{hpp,cpp}`: upload per-channel slope buffer (mirrors bias), add `-DPRELU`
  + slope setArg when `common->leakyReluSlope` present.
- `MergePReluToConvolution.cpp` (new postconvert pass) + registered in `PostConverter.cpp`
  **only when `getenv("MNN_FUSE_CONV_PRELU")`**. Fuses per-channel (or scalar) PReLU into
  3×3 s1 d1 g1 convs with in/outCount≥64 — exactly the convs guaranteed to hit ConvBufWinograd
  (channel-≥64 valid() branch, spatial-independent). All other convs keep their separate PReLU.
- Correctness reference = CPU unfused model vs OpenCL fused model (must match within fp16 tol).
- Rebuilding host MNNConvert + android libs (schema touches core → wide recompile). Then validate.

### 2026-06-30 — PReLU fusion VALIDATED on device ✅
- Builds: host MNNConvert + android libs/ModuleBasic rebuilt (schema → wide recompile),
  all exit 0. New converter pass needed a **cmake reconfigure** (GLOB) — done.
- Correctness (`validate_fusion.py correctness`): CPU-unfused vs OpenCL-FUSED
  **cosine 0.999996**, neg_frac 0.50 (PReLU − branch exercised), separate PReLU kernel
  absent in fused run.
- Perf (hero, 3 runs): total kernel **3336/3312/3299 → 3130/3108/3122 µs** (−~200 µs);
  conv_us unchanged (~1020). Separate `PReLU0` kernel eliminated. ≈ **−18% on conv+PReLU**.
- Patch is opt-in + winograd-scoped → zero default/cross-backend regression.

### 2026-06-30 — PReLU fusion extended to ALL buffer conv paths + pushed
- Discovery: ConvBufExecution + conv_2d_buf.cl/gemm_buf.cl ALREADY have full PReLU support
  (mPrelu/mSlope, slope_ptr, select(out*slope,...)), reachable only via internal-only
  `ExtraConvolution2DPrelu` Extra op (no public producer). Fed it from leakyReluSlope instead.
- Relaxed MergePReluToConvolution to fuse any dense (group=1, non-quant) conv. Winograd path
  handles 3x3>=64; ConvBufExecution handles 1x1/general/gemm. Both apply the slope.
- Sweep 11/11 PASS (3x3/1x1/5x5, varying ch, scalar leaky, gemm-1x1), varying per-ch slopes,
  cos>=0.99998. Perf 1x1 [8,256,32,32]: total kernel 1818->1674us (-144us), PReLU0 gone.
- Branch `opencl-buffer-prelu-conv-fusion` pushed to personal fork (2 commits).
- **Next:** deeper kernel work — L2 Winograd F(4x4,3x3) (~2.25x fewer matmul MACs on hero).

### 2026-06-30 — Deeper kernel work assessed: F(4,4) contraindicated (measured)
- Profiled hero per-kernel: Raster0 528us (ONE-TIME input pack, amortized in real net) +
  transforms ~138us + batchgemm ~262us. NOT matmul-dominated.
- batchgemm efficiency only 2.31 TFLOP/s (N=K=96 too small) — the real structural limit.
- F(4,4) cost model: transforms +172us, matmul -115us => +58us REGRESSION for K=96.
  F(4,4) only wins for K>=256. Hero is balanced, no >=2x lever. See FINDINGS §F.
- Decision: do NOT implement F(4,4) (data says it regresses hero). Remaining headroom is
  small-N/K GEMM tiling (<=10%, uncertain) — recommend only if real model has K>=256 3x3 convs.

### 2026-08-10 — Session A: restricted-set specialization (branch opencl-conv-specialize)
Focus shift: stop optimizing the hero; specialize for the REAL model's restricted op set.
- **Constraints (user-confirmed):** single specific model, no generality needed → batch=1,
  always 3x3 g1 d1, stride only 1/2, spatial FIXED (compile-time constants OK). Input is a
  single-channel Y-plane. NC4HW4-at-input theory DEFLATED: the NCHW->NC4HW4 Raster is one-time,
  waste rides on one stem conv only — minor. Real bottleneck = OCCUPANCY STARVATION: measured
  small_C32 (1,32,144,192) 3x3 = 378us but compute-floor ~78us / BW-floor ~8us => ~21% of
  compute peak, ~2% BW peak. ~4.8x gap on every small-channel direct-path ("ori") conv.
- **Real model structure (conv_bench/model_convs_updated.csv):** two INDEPENDENT submodels,
  each a linear conv chain (all 3x3 +per-channel PReLU), fed by upstream ops (not raw input);
  no relation between them; Block2 runs first in the full model.
    Block1: c1 18->16 @288x384 s2; c2 16->32 @144x192 s2; **c3-c8 = 6x 32->32 @72x96 s1**.
    Block2: c10 34->32 @144x192 s2; c11 32->48 @72x96 s2; **c12-c16 = 5x 48->48 @36x48 s1**.
  The homogeneous cores (6x 32->32@72x96, 5x 48->48@36x48) are the fused-layer megakernel targets.
- **Plan (ranked):** (1) macro-specialized register-blocked 3x3 kernel (hardcoded shapes via -D);
  (2) fused-layer megakernel for the consecutive-conv cores (keep intermediates in LDS; SOTA
  cross-layer tiling ~2x precedent) — best fit for occupancy-bound + fixed-shape + known-seq;
  (3) confirm ori-vs-gemm path; (4) per-shape LWS; (5) PReLU fusion already shipped, applies.
- **Env note:** repo HEAD had drifted to feature/dualrangehist-opencl-op between sessions;
  user chose to continue on opencl-conv-specialize (prelu-fusion base; staged Jun-30 libs match).
- **Baseline DONE** (conv_bench/block_fixture.py, see FINDINGS §H): Block1 1540us / Block2 1143us
  sustained. Cores (6x 32->32@72x96 = 711us; 6x 48->48@36x48 = 632us) dominate at 0.68-1.08
  TFLOP/s = occupancy-starved. Path=ori. Boundary rasters (521us) are a fixture artifact (raw-NCHW
  input), not deployment cost — real blocks get NC4HW4 upstream. True Block1 compute ~1088us.
- **Safe win banked — PReLU fusion** (MNN_FUSE_CONV_PRELU=1 at convert): Block1 -135us (-9%),
  Block2 -89us (-8%); PReLU0 kernels gone. Shipped/device-validated, applies to all convs here.
- **Cross-layer fusion FALSIFIED empirically** (FINDINGS §H.1, core_depth_sweep.py): per-conv
  flat ~119.5us across depths 1/2/4/6 (0.6% spread), total linear => inter-layer traffic <1us/conv
  => fusion ceiling <2%. Full-6 also LDS-infeasible. Bottleneck = per-conv LOAD LATENCY, not traffic.
- **LDS input-halo tiling IMPLEMENTED (FINDINGS §H.3): correct but 1.78x SLOWER — falsified.**
  conv_2d_3x3s1_lds + env-gated MNN_CONV_LDS path (off by default). Bit-exact to stock
  (cosine 1.0), vs CPU 0.999997. Perf 213.5us vs stock 120us/conv. Removing 9x redundant global
  reads didn't help => L2 already caches the reuse; lever is register-blocking (stock already does),
  not LDS. Test: conv_bench/lds_test.py. Rebuilt libMNN_CL (needed cmake reconfigure vs branch drift).
- **Levers status:** PReLU fusion = BANKED win (-8/9%). Cross-layer fusion = falsified (<2%).
  Naive LDS = falsified (slower). Cores (~1 TFLOP/s) look near-ceiling for register-blocked direct.
- **Register-block unroll spec = NO-OP (FINDINGS §H.4):** MNN_CONV_SPEC adds opencl_unroll_hint
  on c4h4w1/c8h4w1 ic loop; 32->32 +0.0%, 48->48 -0.7% (noise), bit-exact. Compiler already
  schedules well. THIRD kernel approach to fail on cores (fusion, LDS, unroll).
- **VERDICT (FINDINGS §H.5): cores are at practical ceiling** for direct register-blocked on
  Xclipse-960/ANGLE — no kernel restructuring beats stock. Shippable value = PReLU fusion (-8/9%).
- **space2depth on stride-2 heads = FALSIFIED (FINDINGS §H.6):** correct (cosine 0.99998) but
  SLOWER — Block1 +12.7%, Block2 +17.4%. s2d 4x's Cin (16-34 already past crossover) => ~1.78x more
  dense FLOPs + added rasters. Only wins for tiny Cin<=4 (the C=1 stem, which is upstream, not here).
- **Ceiling stress-tested (user pushback, both refuted rigorously):**
  (1) Larger blocking variant c8h8w1 (16 acc) = +48.7% SLOWER, thermal-controlled (FINDINGS §H.8) —
      occupancy-limited; register cliff generalizes to all 16-acc variants. Tuner sweet spot is real.
  (2) LDS isolation at constant blocking (c4h1w1 vs same+LDS) = +0.7% / +9.2% (FINDINGS §H.9) —
      LDS captures nothing L2 doesn't; barriers+occupancy pure overhead; base-independent (answers
      "build LDS on c4h1w2" without building it). Added MNN_CONV_FORCE=<name> force switch + winner print.
- **FINAL VERDICT (FINDINGS §H.7):** 7 levers tried on the real blocks. BANKED: PReLU fusion
  (-8/9%, ship via MNN_FUSE_CONV_PRELU=1). FALSIFIED on-device: NC4HW4-input, cross-layer fusion,
  LDS tiling, register-unroll, space2depth-heads, larger-blocking(c8h8w1), LDS-at-constant-blocking.
  Convs (cores+heads) at practical ceiling for MNN's autotuned direct path on Xclipse-960/ANGLE.
  Only remaining idea: space2depth on the true C=1 stem (upstream, not in these blocks).
- Branch opencl-conv-specialize, 6 commits, NOT pushed. env flags MNN_CONV_LDS/MNN_CONV_SPEC default off.
