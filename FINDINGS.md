# FINDINGS.md — results

> Filled in as data lands. Sections: device profile, runtime-vs-shape model, ranked levers
> (before/after), recommended build flags + runtime config, kernel patches.

## A. Device / OpenCL profile (Phase 0)

| Property | Value | Source |
|---|---|---|
| Device | Galaxy S26-class, `SM-S942B` | adb `ro.product.model` |
| SoC | Exynos 2600 (`erd9965` / S5E9965) | adb `ro.board.platform` |
| CPU ISA | i8mm, sve2, **sme2**, fp16 | `run_test.out` CPU probe |
| GPU | **Samsung Xclipse 960 (AMD RDNA4)** | device Vulkan driver string |
| Vulkan subgroup size | **64** | device Vulkan driver string |
| Max workgroup | 1024 | device Vulkan driver string |
| LDS / shared mem | **64 KB** | device Vulkan driver string |
| FP16 | storage + compute = yes | device Vulkan driver string |
| coopmat (WMMA) | **0** (not exposed) | device Vulkan driver string |
| OpenCL stack | **ANGLE CL-over-Vulkan** (not native) | `CLPlatformVk.cpp` in run log |
| MNN GPU class | `RADEON` (`isSetWorkGroupAttribute=true`) | `OpenCLRuntime.cpp:164` |
| Subgroup CL kernels | **disabled** (INTEL-gated) | `OpenCLRuntime.cpp:173` |

**OpenCL-exposed caps (preferred_vector_width_half, MAX_COMPUTE_UNITS, extensions):**
_pending — one-shot dump on first rebuild._

**RDNA4 design implications to test (not assume):** wave32 native (LWS multiples of 32, try
64); packed FP16 (2×`half`/lane → ensure kernels use `half2+`); real 64 KB LDS for
input/weight tiling. All mediated by ANGLE→Vulkan, so verify empirically.

## B. Runtime vs shape/size model (Phase 1)

**Instrument:** `build_android_profile` per-kernel GPU time (CL profiling events). Primary
metric = **steady-state conv GPU time** (`conv_us`, median over ~80 loops, warmup dropped).
Very stable (±1%) because the conservative GPU governor pins **980 MHz** (max) under
sustained loops. Wall time is noisy (host/ANGLE overhead) and used only as a sanity cross-check.
All runs: OpenCL **buffer** + **fp16** (gpuMode 68 = BUFFER|WIDE, precision-mask 2).

**Hero baseline:** Conv `X=[8,96,64,64]`, `W=[96,96,3,3]`, s1 p1 → **conv GPU = 1023–1028 µs**.
MNN auto-selects the **Winograd buffer path** (`Conv-winograd-rearrange` +
`Conv-winograd-batchgemm`, F(2,2,3,3): b16/m1024/n96/k96). Internally ≈ **50% winograd
input-transform ("rearrange", launched ~2× as often) + 50% batched GEMM**.

### Sweeps (N8 unless noted, 96ch, 64×64, k3, buffer fp16, 980 MHz)

| dim | points → conv_us (eff TFLOP/s nominal) | reading |
|---|---|---|
| **N** | 1:128 · 2:257 · 4:579 · 8:1028 · 16:2052 | **perfectly linear** (~128 µs/batch); eff flat ≈5.3 |
| **Cin=Cout** | 32:434(1.4) · 64:642(3.8) · 96:1027(5.3) · 128:1623(6.0) · 192:3351(6.5) | eff **rises with C** → compute/occupancy-bound; C32 falls to `ori` path |
| **H=W** | 16:206(1.7) · 32:373(3.6) · 48:639(4.8) · 64:1028(5.3) · 96:2117(5.8) · 128:3907(5.6) | eff **rises with spatial** then plateaus ≈5.7 |
| **kernel** | k1:935 `gemm2`(0.65) · k3:1028 `wino`(5.3) · k5:10063 `ori`(1.5) | **only 3×3 gets Winograd**; 1×1→gemm2 (inefficient), 5×5→direct `ori` (3.5× worse/MAC) |
| **activation** | none:1028 · relu:1024 · relu6:1025 · **prelu:+219µs (separate kernel)** | ReLU/ReLU6 **fused**; PReLU **not** fused |
| **tuning** | NONE:1147 · FAST:1141 · NORMAL:1025 · HEAVY:1018 · WIDE:1023 | WIDE(default)≈HEAVY≈optimal; NONE/FAST ~12% worse |

### Hardware-friendly shape rules (requested) — measured, N8 64×64 k3 buffer fp16

| knob | finding | rule |
|---|---|---|
| **Cout (output channels)** | **96→97 = +24% time** for +2% work (1031→1275 µs); recovers only at next mult-of-16 | **Keep Cout a multiple of 16 (ideally 32).** This is the single biggest shape lever. |
| **Cin (input channels)** | 96→97 ≈ +2% (1031→1052) — barely matters | Cin is the GEMM contraction dim; **pad Cin freely**, alignment ~irrelevant. |
| channel cliff detail | C90-96 flat ≈1010-1031; **C97-100 jump to ~1324**; 112(=7×16) recovers | tile = 16 output channels; an extra near-empty tile costs a full tile. |
| **spatial H×W** | smooth; odd vs even mild (64→65 = +10%); best at even/aligned | prefer **even spatial dims** (Winograd UNIT=2); effect minor vs Cout. |
| larger aligned Cout | Cout128 = 0.353 µs/MMAC (best) vs Cout96 = 0.379 | bigger *aligned* GEMMs are more MAC-efficient (occupancy). |
| **N (batch)** | perfectly linear, no alignment effect | N is free to choose; cost scales 1:1. |

**Actionable for the model designer:** the asymmetry means if a 96-channel conv must grow,
grow **Cin** (cheap) not **Cout** (expensive unless to the next mult-of-16). A Cout of 96 is
already well-aligned (6×16); 80/112/128 are the neighboring sweet spots.

### Roofline verdict
The 3×3 conv is **compute/occupancy-bound, not bandwidth-bound**: effective throughput
*increases monotonically* with channels **and** spatial size toward an asymptote
**≈6.5 TFLOP/s nominal** (≈2.9 TFLOP/s real after Winograd's 2.25× MAC reduction), instead
of plateauing at a low bandwidth ceiling. The **hero C96 (5.3 TFLOP/s) sits ~20% below the
asymptote** → real, if modest, headroom; bigger GEMMs use the GPU better. Linear-in-N and
linear-in-spatial (above the small-size overhead floor) confirm fixed per-launch overhead
dominates only small shapes; the hero is large enough to be near steady-state efficiency.

## C. Ranked optimization levers (Phase 2–3)

Context that reframes the task: **the hero 3×3 conv is already on MNN's most-optimized
buffer path** — Winograd, with ReLU/ReLU6 fused, and WIDE auto-tuning already near-optimal.
So the naive "the conv kernel is slow, rewrite it" premise is **partly falsified**; wins are
narrower and must be found empirically. Ranked candidates:

| Lever | Expected | Risk | Status |
|---|---|---|---|
| **L1 — Fuse PReLU into conv** | −219 µs/conv (the unfused `PReLU0@output` kernel) | low | **✅ DONE & VALIDATED: −~200 µs measured, cosine 0.999996** |
| **L2 — Winograd F(4,4,3,3)** | **−115 µs matmul but +172 µs transforms = +58 µs NET REGRESSION for K=96** | — | **✗ CONTRAINDICATED for hero (measured, §F); only wins for K≥256** |
| **L3 — Close the GEMM efficiency gap** (batchgemm runs at only 2.31 TFLOP/s, N=K=96 too small for big tiles) | ≤~10% on the matmul (~140 µs ceiling), structural | med-high | open; small-N/K GEMM tiling, uncertain payoff |
| H1 — tuning level | ≈0 (WIDE already optimal) | — | **falsified** |
| H2 — fuse ReLU/ReLU6 | 0 (already fused) | — | **falsified** (PReLU survives as L1) |
| subgroup kernels | n/a (INTEL-gated, never runs) | — | **pre-falsified** |

_before→after numbers added as each lever is applied (one change at a time)._

## D. Recommended build flags + runtime config (safe wins — no code change)

**Runtime (the levers that matter, all measured):**
- `forwardType = MNN_FORWARD_OPENCL` (3).
- **`gpuMode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_WIDE` (= 68).** Buffer is mandated
  (and also dodges the image-mode wide-tensor reboot gotcha). WIDE tuning is **measured
  optimal** for the hero (ties HEAVY; NONE/FAST are ~12% slower) — keep the default.
- **Precision = Low** (fp16 storage+compute) — required by the task; mask `2`.
- **Persist the OpenCL tuning cache** (`cache_file` / `rtmgr->setCache`): WIDE tuning is
  slow but one-time; reused runs skip it. Clearing it forces a re-tune (early-loop cost).
- Power = High.
- **Hardware-friendly shapes (model-side, free):** keep **Cout a multiple of 16/32**
  (96 is good); pad Cin freely; prefer even spatial dims. See §B "Hardware-friendly".

**Build (Android arm64):** `MNN_OPENCL=ON`, `MNN_ARM82=ON` (fp16 CPU fallback),
`MNN_USE_LOGCAT=false`, `MNN_SEP_BUILD=ON`. **For production builds turn
`MNN_GPU_TIME_PROFILE=OFF`** — profiling enables `CL_QUEUE_PROFILING` + per-event waits
which serialize the queue and massively inflate wall latency (it's a measurement tool, not
a deploy setting). Keep it ON only for the characterization build used here.

**No-regression note:** OpenCL buffer fp16 vs CPU fp32 on the hero conv(+PReLU) =
cosine 0.999997, rel-err ~0.3% (fp16 noise only). Verified via `conv_bench/correctness.py`.

## E. PReLU fusion (L1) — the top remaining win; design + status

**Measured opportunity:** the per-channel PReLU (`slope=[1,96,1,1]`) runs as a **separate
`PReLU0@output` kernel ≈ 219 µs**, which is **memory-bound**: it reads+writes the whole
[8,96,64,64] fp16 activation (~100 MB) at ~460 GB/s ≈ the device's LPDDR5X peak. So the only
way to remove it is **fusion** (eliminate the round-trip) — and fusion saves ~all 219 µs,
i.e. conv+act 1244 µs → ~1025 µs ≈ **−18%** on the hero. ReLU/ReLU6 are already fused (no win).

**Why it's not a one-liner (and the correctness landmine):** `Convolution2DCommon` carries
only `relu`/`relu6` *bools*; per-channel PReLU needs a *slope vector*, and unlike `Scale`
(linear → folds into weights) PReLU is nonlinear → must be applied post-accumulation in the
conv kernel. MNN has **no runtime backend-gated fusion hook** — fusion is converter-level and
backend-agnostic, so a converter pass that folds PReLU into the conv would produce **wrong
output on any backend that doesn't implement the fused form** (CPU/Metal/…). The existing
ReLU fusion is safe only because *all* backends honor the flag. Activation is applied inline
per kernel (`#ifdef RELU` in `winogradTransform_buf.cl`, `conv_2d_buf.cl`,
`gemm_conv1x1_buf.cl`, `gemm_buf.cl`, …), and the dest kernel already has the output-channel
index — the natural place to index a slope.

**Recommended implementation (staged, each gated on the §D correctness check):**
1. Schema: add optional `leakyReluSlope:[float]` to `Convolution2DCommon` (empty ⇒ no-op;
   flatbuffers fwd/back compatible). Regen `MNN_generated.h`.
2. Converter: new postconvert pass `MergePReluToConvolution` — fold a *single-consumer*,
   per-channel PReLU's slopes onto the conv, delete the PReLU. Mirror `MergeReluToConvolution`.
3. Backends: apply per-channel slope in the conv post-op. **CPU first** (keeps fused models
   portable/correct), then OpenCL: upload a slope buffer, add `-DPRELU` + slope arg to the
   winograd dest kernel and `conv_2d_buf` / `gemm_conv1x1_buf` / `gemm_buf` (cover the paths
   the model actually hits — winograd for the hero 3×3).
4. Validate every step with `conv_bench/correctness.py` (must stay cosine > 0.9999) and
   re-measure `conv_us` (expect the `PReLU0` kernel to vanish, total −≈219 µs).

**Status: ✅ IMPLEMENTED & VALIDATED ON DEVICE (2026-06-30).** Opt-in (env
`MNN_FUSE_CONV_PRELU` at convert time), scoped to winograd-eligible convs (3×3 s1 d1 g1,
in/outCount≥64) so it is provably correct on the OpenCL buffer path and **changes nothing by
default / on other backends**.

**Files changed (the patch):**
- `schema/default/CaffeOp.fbs` → optional `leakyReluSlope:[float]` on `Convolution2DCommon`
  (regen `schema/current/*` via `schema/generate.sh`).
- `source/backend/opencl/execution/cl/winogradTransform_buf.cl` (`winoTransDstBuf2_3_1`):
  `#ifdef PRELU` slope arg + `res = fmax(res,0)+slope*fmin(res,0)` at all 4 outputs
  (regen `winogradTransform_buf_mnn_cl.cpp` via `opencl_codegen.py`).
- `source/backend/opencl/execution/buffer/ConvBufWinograd.{hpp,cpp}`: per-channel slope
  buffer upload (mirrors bias) + `-DPRELU` + slope `setArg` when slopes present.
- `tools/converter/.../postconvert/MergePReluToConvolution.cpp` (new) + gated registration in
  `PostConverter.cpp` (`getenv("MNN_FUSE_CONV_PRELU")`). **Needs cmake reconfigure** (GLOB).

**Measured (hero, buffer fp16, 980 MHz, 3 runs):**
| | conv GPU | total kernel | separate PReLU kernel |
|---|---|---|---|
| unfused | 1019–1058 µs | 3299–3336 µs | **yes (~219 µs)** |
| **fused** | 1017–1025 µs | 3108–3130 µs | **gone** |

→ **−~200 µs per conv+PReLU** (eliminates the activation's global round-trip); conv_us
unchanged (PReLU absorbed for free into the dest kernel). On the conv+PReLU pair
(~1240 → ~1020 µs) that's **≈ −18%**. **Correctness:** CPU-unfused vs OpenCL-fused
cosine **0.999996**, PReLU negative branch exercised (`conv_bench/validate_fusion.py`).

**To productionize (beyond this opt-in PoC):** add CPU (and any other shipped backend)
support for `leakyReluSlope` so the fused model stays correct everywhere, then the pass can
move into the default chain; and extend the slope handling to the non-winograd conv buffer
paths (`conv_2d_buf`, `gemm_conv1x1_buf`) if smaller/odd-shaped 3×3 convs also use PReLU.

## §F — Deeper kernel work: Winograd F(4×4,3×3) is CONTRAINDICATED for the hero (measured)

Per-kernel profile of the hero conv (`X=[8,96,64,64]·W=[96,96,3,3]`, buffer fp16, 980 MHz):

| kernel | time | note |
|---|---|---|
| `Raster0` (input NCHW→NC4HW4) | **528 µs** | **one-time** — appears once even for a 2-conv chain; amortized in a real net, mostly a single-op-benchmark artifact |
| `winograd-rearrange` (src+dst transforms) | ~138 µs | F(2×2,3×3), 4×4 tiles |
| `winograd-batchgemm` ×2 | ~262 µs | b16 m1024 n96 k96 |

So the per-conv cost in a real network is ~**400 µs** (transforms + matmul), **not** matmul-dominated.

**Batchgemm efficiency:** 302 MFLOP / 131 µs = **2.31 TFLOP/s** vs ~6.5 peak — low, because
N=K=96 are too small for efficient tiles. This is the real (structural) inefficiency.

**F(4×4,3×3) cost model** (transforms scale ∝ tile area α²: 16→36 = ×2.25; matmul MACs/output
4→2.25 = ×0.56):
- transforms 138 → **310 µs (+172)**, matmul 262 → **147 µs (−115)** → **net +58 µs REGRESSION**.

→ **F(4,4) is not worth implementing for K≤~128 convs** — the larger transforms cost more than
the matmul saved. It only wins when the matmul dominates (**K ≥ 256**). The hero conv is already
near-balanced across transform/matmul with no single ≥2× lever; UNIT=2 (F(2,2)) is the right
choice for it. **Recommendation depends on the real model:** apply F(4,4) selectively *only* to
its high-channel (K≥256) 3×3 convs, if any; otherwise the remaining headroom is the small-N/K
GEMM efficiency (≤~10%), not algorithmic.

## §G — The real model's worst convs: small-channel / large-rectangular / non-winograd regime

User's actual worst convs (all N=1, 3×3): conv1 in[1,18,288,384]→[16], conv2 in[1,34,144,192]→[32],
conv3 in[1,1,576,768] s2→[8]. **Completely different regime from the hero:** tiny channels
(Cout 8/16/32 << 64 → **no Winograd**, all use the `gemm`/direct path), huge rectangular spatial,
N=1. Baselines (µs, but see variance caveat): conv1 ~460 conv / ~1150 total; conv2 ~424/773;
conv3 ~95/433. Data: `conv_bench/real_regime_sweep.csv`.

**MEASUREMENT CAVEAT (important):** these small convs show **~2× run-to-run variance** (conv1
459↔1191 µs) for the *identical shape, cache and LWS*. It is **not** tuning (same cache varies
460↔950) and not fixable by loop count — it is GPU **DVFS/thermal**: unlike the hero (which
saturated the GPU and pinned 980 MHz at ±1%), these light/bursty kernels let the clock float.
Consequence: **fine spatial-alignment cliffs are below the noise floor here** — do not trust
sub-20% shape effects from an isolated microbench; measure these layers **in the full model**
(sustained load) instead.

**Robust findings (effects well above the 2× noise):**
- **Channel count dominates cost**, ~linear in *groups of 4* (NC4HW4): Cin 1=4→111 µs, 8→206,
  16→371, 32→722, 64→1425; Cout 8→248, 16→~460, 32→~1348, 64→2182. **Cout is the steeper/most
  expensive axis** (Cout 8→64 ≈ 9×).
- **Channels quantize by 4** → **padding Cin/Cout up to the next multiple of 4 is FREE**
  (Cin 17/18/19/20 all cost the same ~460; Cin 1 == Cin 4). Keep channels ≤ a multiple of 4;
  never inflate past it.
- **Do NOT chase Winograd here:** raising Cout to ≥64 to trigger it makes it *worse*
  (Cout16 ~460 → Cout64 ~2182). Small-channel 3×3 is correctly on the direct path.
- **Spatial (W,H): only a mild effect** (≲10–20%, within noise); no sharp width/height
  alignment cliff like the Cout-16 cliff of the compute-bound regime. W (innermost) ≈ H in
  sensitivity here.
- **Input packing (Raster NCHW→NC4HW4) is a large fraction of `total` for these** (conv1
  total 1150 vs conv 460) — amortized mid-model, but conv3 (Cin=1 image stem) always pays it.

## §H — Real model: the two submodel blocks (Session A, branch opencl-conv-specialize)

Real structure (`conv_bench/model_convs_updated.csv`, driver `conv_bench/block_fixture.py`):
two INDEPENDENT linear conv chains (all 3×3 s1/s2 + per-channel PReLU), fed by upstream NC4HW4
tensors (not the raw frame); Block2 executes first in the full model.
- **Block1:** c1 18→16 @288×384 s2; c2 16→32 @144×192 s2; **c3–c8 = 6× 32→32 @72×96 s1**.
- **Block2:** c10 34→32 @144×192 s2; c11 32→48 @72×96 s2; **c12–c17 = 6× 48→48 @36×48 s1**.

**Baseline (sustained load, 61 windows, buffer fp16, gpuMode 68 mask 2):**

| block | total kernel | homogeneous core | core/conv | core TFLOP/s | notes |
|---|---|---|---|---|---|
| Block1 | 1540 µs | 6× 32→32@72×96 = **711 µs** | ~118 µs | **~1.08** | conv1 155, conv2 69 |
| Block2 | 1143 µs | 6× 48→48@36×48 = **632 µs** | ~105 µs | **~0.68** | conv 34→32 139, 32→48 70 |

**Key reads:**
- **Cores dominate** (46% / 55% of block) and are deeply occupancy-starved: 0.68–1.08 TFLOP/s
  vs ~6.5 peak (~10–17%), ~2% of BW peak → neither compute- nor BW-bound = latency/occupancy.
  These are the register-blocking + fused-layer targets. Path confirmed `ori` (direct), not gemm.
- **Boundary rasters are a FIXTURE ARTIFACT, not deployment cost.** Per-window sequence has
  Raster0(117)+Raster1(348) *before* conv1 and Raster0(56, output) at the end — **zero rasters
  between conv layers**. My ONNX feeds raw NCHW so MNN inserts input-conversion; the real blocks
  receive NC4HW4 from upstream → these ~521 µs largely vanish. **True Block1 compute ≈ 1088 µs**
  (conv 939 + PReLU ~149). Do NOT chase Raster1.
- **PReLU fusion (MNN_FUSE_CONV_PRELU=1 at convert):** Block1 1540→**1405** (−135, −9%),
  Block2 1143→**1054** (−89, −8%); separate PReLU0 kernels eliminated, absorbed into ori conv.
  Free (shipped, device-validated). Every conv here has PReLU → applies uniformly. Banked.

**Remaining prize = the occupancy-starved cores** (~708 µs / ~607 µs).

### §H.1 — Cross-layer fusion ceiling: EMPIRICALLY FALSIFIED (<2%), do not build
Depth sweep of the exact core conv (32→32@72×96, 3×3 s1 +PReLU fused), sustained load
(`conv_bench/core_depth_sweep.py`): per-conv time is **flat at ~119.5 µs across depths 1/2/4/6**
(0.6% spread; depth-3 point was a parse glitch), and `total` grows perfectly linearly at
~119.5 µs/conv. Each added conv writes its output + the next reads it — the exact traffic a fused
megakernel removes — yet per-conv cost doesn't change ⇒ inter-layer traffic is <1 µs/conv (matches
the ~1.9 µs/boundary roofline). **Fusion ceiling <2%.** Also: full-6 fusion is LDS-infeasible
(needs (T+2L)² halo tile ×2 ping-pong > 64 KB for usable T). Bottleneck is per-conv LOAD LATENCY
(each thread ~360 global loads for ~288 MACs; input re-read 9× across taps, no reuse), which
fusion doesn't touch and can worsen (barriers, smaller launch). Corroborates prior
"multi-conv vertical fusion FALSIFIED".

### §H.2 — The real lever: single-conv LDS input-halo tiling (+ hardcoded specialization)
MNN's direct `conv_2d_c*` kernels read input from global on EVERY tap (no `__local`); only the
≥64-ch Winograd path uses LDS. So staging the input halo tile in LDS once and reusing it across
the 9 taps + neighboring outputs is untried for these shapes and directly attacks the load-latency
bottleneck (kills 9× redundant global reads; global-latency → LDS-latency). Target 1.3–2× on the cores.

### §H.3 — LDS input-halo tiling: IMPLEMENTED, correct, but SLOWER (naive design falsified)
Kernel `conv_2d_3x3s1_lds` (conv_2d_buf.cl) + env-gated path in ConvBufExecution (MNN_CONV_LDS,
off by default; gated to 3x3 s1 pad1, w%16==0, h%4==0). One workgroup = 16x4 output tile for one
oc-block; input halo staged in __local per ic-block, reused across 9 taps. Test: conv_bench/lds_test.py.
- **Correctness:** bit-exact to stock (OCL-LDS vs OCL-base cosine=1.000000, max_abs=0); vs CPU
  (non-fused) cosine=0.999997. Kernel math verified.
- **Perf: 213.5 us/conv vs stock 120.0 us/conv = 1.78x SLOWER.** Naive LDS (1 output/thread,
  16 barriers/conv) loses to the stock autotuned register-blocked variant (~8 out float4/thread).
- **Why it matters:** removing the 9x redundant global reads did NOT help ⇒ RDNA L2 already serves
  that reuse (confirms not-bandwidth-bound). The lever these convs respond to is WORK-PER-THREAD /
  register blocking — which the stock autotuner already exploits — NOT LDS. Kept as documented
  negative result + reusable custom-kernel harness (buildKernel-in-conv-path pattern).
- **Open:** a register-blocked LDS variant (multi-output/thread) is the only fair remaining LDS
  test; but prior says gains here come from blocking, so expected upside over stock is small.

### §H.4 — Register-block unroll specialization: NO-OP (cores confirmed at ceiling)
Added `#ifdef CONV_SPEC_UNROLL __attribute__((opencl_unroll_hint))` on the ic reduction loop of
c4h4w1/c8h4w1, activated by env MNN_CONV_SPEC (adds -DCONV_SPEC_UNROLL in the autotuner; default off).
Bit-exact (cosine 0.999997). Perf: 32→32@72×96 119.5→119.5 us (+0.0%); 48→48@36×48 100.8→101.5 (noise).
⇒ Loop overhead / ILP is NOT the limiter; compiler already schedules these well. **Third kernel
approach to fail on the cores (fusion, LDS, unroll) — cores are at their practical ceiling for the
direct register-blocked path on Xclipse-960/ANGLE.** Stop core-kernel experiments.

### §H.5 — Verdict for the cores + where value remains
The homogeneous cores (~1 TFLOP/s) will not beat MNN's autotuned register-blocked `ori` kernel via
kernel restructuring on this GPU. Shippable value delivered = **PReLU fusion** (§H, −8/9% per block).
Untested higher-EV lever = **space2depth on the stride-2 head convs** (conv1/conv2 of each block;
prior-validated positive for occupancy-starved low-Cin; cheap model rewrite via optimize_model.py,
no kernel/rebuild). That is the recommended next lever if continuing.

### §H.6 — space2depth on the stride-2 heads: FALSIFIED (slower; Cin past crossover)
Rewrote each block's stride-2 heads to SpaceToDepth(2)+stride-1 (exact self-checked transform,
conv_bench/space2depth_test.py). Correct on device (Block1 cosine 0.999982, Block2 0.999970).
Perf: Block1 1413→1592 us (+12.7%), Block2 1061→1246 us (+17.4%) — SLOWER.
Why: s2d quadruples Cin (18→72, 16→64, 34→136, 32→128); a 3x3-s2 does NOT pack tightly into
2x2-on-4C, so the new head convs do ~1.78x more dense FLOPs (conv1 215us vs orig ~155; conv2 111
vs ~69) PLUS ~114us of added SpaceToDepth rasters. Occupancy gain doesn't repay the FLOP inflation
because these heads' Cin (16-34) is ALREADY PAST the crossover. space2depth only wins for tiny
Cin (<=4) — i.e. the C=1 Y-plane STEM (conv3 [1,1,576,768] s2->8), which is upstream of these
blocks. Recommendation: apply space2depth ONLY to the C=1 stem if that conv is ever in scope.

### §H.8 — Larger blocking variants: FALSIFIED (bigger tile = worse; occupancy confirmed)
Tested whether the autotuner's optimum lies outside its 7 variants by adding `conv_2d_c8h8w1`
(16 accumulators = 8 output-ch × 8 rows/thread, 2× the tile height of the c8h4w1 winner), wired
as a gated candidate (env MNN_CONV_SPEC) + a force switch (MNN_CONV_FORCE=<name>) for clean
per-kernel measurement. Thermal-controlled (3 alternating runs, 0.3% spread):
- `c8h4w1` forced = **119.0 us**; `c8h8w1` forced = **177.0 us (+48.7% SLOWER)**; c8h8w1 correct
  (cosine 0.999997, confirmed it ran). For 48→48@36×48 the tuner picks `c4h1w2` (2 accumulators).
⇒ These latency-bound convs want OCCUPANCY (many small waves), not work-per-thread. Bigger tile =
more registers = fewer resident waves = worse latency hiding. The autotuner's pick is the true
sweet spot, on the SMALL side; going bigger regresses hard. Register pressure is the same for any
16-acc variant (c16h4w1, c8h4w2, …) so they're expected to regress identically — the mechanism
generalizes. **Refutes "the optimum is outside the 7 in the bigger-tile direction."** c8h8w1 kept
as documented negative + reusable force-switch infra; default 7-variant set unchanged.

### §H.9 — LDS isolation at constant blocking: LDS captures NOTHING (airtight close)
Cleanest possible LDS test — hold output blocking constant (1 output float4/thread) and toggle
ONLY LDS: `c4h1w1` (no LDS) vs `conv_2d_3x3s1_lds` (same block + cooperative LDS halo). This is
the best case for LDS (small footprint = room to add LDS). Thermal-controlled (3 alternating runs):
- 32→32@72×96: c4h1w1 211.5us vs LDS 213.0us = **+0.7%** (neutral); c4h1w2 206.7; winner c8h4w1 119.
- 48→48@36×48: c4h1w1 130.5us vs LDS 142.5us = **+9.2%** (regression); winner c4h1w2 100.8.
⇒ LDS delivers zero-to-negative even at constant blocking ⇒ the input reuse LDS would capture is
ALREADY L2-served; LDS's barriers + occupancy cost are pure overhead. Base-independent: a
c4h1w2+LDS would be c4h1w2 + the same overhead (and c4h1w2 no-LDS 100.8us already beats LDS'd
142.5us). The LDS lever is conclusively dead for this regime. (Answers the "build LDS on the
winner c4h1w2" question without needing to build it — the constant-blocking isolation settles it.)
**LWS/tile sweep (closes the autotuner-masking caveat):** parametrized the LDS tile via env
MNN_LDS_TILE=WxH (+ -DTILE_W/-DTILE_H) and swept 11 tiles/core. BEST LDS tile still loses:
32→32@72×96 best 16×4=213.5us vs c8h4w1 119.2 (+79%); 48→48@36×48 best 16×12=130.7us vs c4h1w2
101.7 (+29%). LDS loses across the ENTIRE LWS space → its fixed 16×4 hid no win; the autotuner
did not mask it. (Bigger workgroups help LDS a little via halo amortization but never enough.)

### §H.10 — Concurrency (A1) + fp16-accum (B2): the live lever is cross-stream overlap
- **B2 (fp16 accumulation) already done by the engine:** in Low mode (precisionLevel==2, mask 2)
  OpenCLRuntime injects BOTH -DFLOAT=half AND -DCOMPUTE_FLOAT=half → accumulators are already
  half4. No register-halving lever available (also why c8h8w1's 16 half4 accs still blew occupancy).
- **Intra-model concurrency = NO overlap:** two independent 6-conv branches in one model (one
  session/queue) vs one 12-conv dependent chain, identical work: par 3.15ms vs seq 3.05ms (-3.4%).
  MNN's single command queue serializes; fan-out in one graph does NOT overlap.
- **Two-stream concurrency = real spare capacity:** two independent processes (host-launched, 2
  contexts): worst-to-finish 3.59ms vs solo 2.68ms = **1.34x, not 2x** → ~33% faster for the pair
  vs sequential. CAVEATS: profiling build; ~50% of wall is CPU/submission overhead (2.68ms wall vs
  ~1.5ms GPU kernel), so part of the 1.34x is overlapping that overhead, not necessarily GPU compute;
  scheduler asymmetry (A 2.83 / B 3.59). Clean number needs non-profiling + two-THREAD (one process,
  two sessions/queues) harness.
- **ACTIONABLE:** the model's "same block x2 in parallel" is turnable into wall savings ONLY by
  running the two instances as concurrent sessions/threads (two queues) — NOT as two branches in one
  model. Est ~1.3-1.5x throughput per parallel pair. This is the second positive lever after PReLU
  fusion, and the only one that helps a latency-bound kernel without beating the kernel.
- Also measured: per-inference wall is ~2x the GPU-kernel time → ~50% is CPU/submission/ANGLE
  overhead (partly profiling). Reducing it (non-profiling build, fewer flushes/batched submit) is a
  separate real wall-time lever independent of concurrency.

### §H.11 — Subgroup conv (idea #1): DEAD — no shuffle in the Clspv toolchain
Device exposes `cl_khr_subgroups` (dump via env MNN_DUMP_CL_EXT) → sub_group_broadcast/reduce work.
But CL is compiled through ANGLE's Clspv (CL→SPIR-V); probe (env MNN_SUBGROUP_PROBE, guarded kernels
subgroup_probe_bcast/shuffle in conv_2d_buf.cl): **broadcast OK, shuffle FAIL** ("undeclared
identifier sub_group_shuffle"; no cl_khr_subgroup_shuffle). The halo-sharing subgroup conv needs
lane→neighbor shuffle (broadcast only shares one lane to all → doesn't move input between neighbors),
so it's not implementable here. Diagnostic probes kept (env-gated, default off, guarded kernels).

### §H.12 — Why the remaining single-conv kernel ideas are low-EV
Occupancy is the binding constraint (proven by c8h8w1 +48.7% from register pressure). Every remaining
kernel-implementation idea moves registers the WRONG way: partial accumulators (idea #2, ILP) = 16
accumulators = SAME footprint as the falsified c8h8w1 → same occupancy hit; in-register prefetch (idea
#4) = longer-lived temporaries → same. Only register-MINIMIZATION (idea #3) goes the right way, but
c8h4w1 is already lean and occupancy is wave-quantized (must drop a whole wave's VGPR budget to gain
one). Plus the bottleneck is MEMORY latency, not FMA-chain latency, so ILP ideas target the wrong wall.
⇒ custom single-conv implementation is, for practical purposes, EXHAUSTED on this device (ANGLE/Clspv/
RDNA). Runtime levers that remain live are all OUTSIDE the single kernel: concurrency (§H.10, ~1.3x),
PReLU fusion (banked, §H), and cutting the ~50% CPU/submission wall overhead.

**#2 partial accumulators MEASURED (conv_2d_c8h4w1_pa, forced):** c8h4w1 119.2us vs pa 153.8us =
+29.1% SLOWER, correct (cosine 0.999998). Informative: less bad than c8h8w1 (+48.7%) despite same
~16-accumulator footprint → breaking the FMA chain DID buy ~20pts of ILP (mechanism works), but the
occupancy cost of the doubled accumulator registers still dominates. Confirms occupancy is binding;
adding registers loses regardless of the ILP gained. #1 and #2 now empirically closed; #3 (register-
minimization) is the only untested direction that doesn't add registers but is wave-quantized/low-EV.

### §H.13 — im2col+GEMM (proxy): FALSIFIED for large-spatial, BORDERLINE-PROMISING for small-spatial
MNN has no reachable gemm path for 3×3 (gated to isConv1x1 && Cin>32 && Cout>64), so measured a
proxy = a 1×1 conv at the im2col matrix dims (M=Ho·Wo, N=Cout, K=Cin·9) = the GEMM half without the
im2col build. Median-of-3, stable:
- 32→32@72×96 (M=6912,N=32,K=288): gemm-half **139us** vs direct **119us** → already slower; +im2col
  (~4MB,~9us) makes it worse → **GEMM falsified for this core**.
- 48→48@36×48 (M=1728,N=48,K=432): gemm-half **80us** vs direct **102us** → **22% FASTER**; +im2col
  (~1.5MB,~10-20us) ≈ break-even-to-marginal-win → **GEMM is a genuine open candidate here**.
Mechanism: im2col inflates the reduction dim (Cin 48→K 432), pulling compute OUT of the
occupancy-starved small-channel regime (same as the "more channels helps" model-level lever, without
retraining). The win-or-not hinges on im2col overhead + N=Cout still being small.

**REAL im2col+GEMM IMPLEMENTED & MEASURED (fused im2col kernel `im2col_3x3s1` + env MNN_CONV_IM2COL;
winograd bypass env MNN_NO_WINOGRAD to reach the general path). im2col verified cosine=1.000000:**
- 48→48@36×48: im2col **37us** + GEMM 80us = **117us** vs direct **102us** → **+15% SLOWER**.
- 32→32@72×96: im2col **94us** + GEMM 139us = **233us** vs direct 119us → **+96% SLOWER**.
⇒ **Falsified end-to-end**: the separate im2col pass (write 1.5MB @ ~80 GB/s, itself latency-bound on
8 CUs) costs more (37us) than the GEMM's compute advantage (80 vs 102 = 22us headroom). BUT the GEMM
COMPUTE genuinely beats the direct 3x3 (80 < 102). Deeper analysis + honest correction: see §H.14.
Infra kept env-gated (MNN_CONV_IM2COL, MNN_NO_WINOGRAD).

### §H.14 — GEMM/fusion architecture analysis + CORRECTION of the "implicit GEMM = win" over-claim
An earlier note called implicit GEMM "~80us = real win". That was too optimistic — corrected here:
- **The direct 3x3 conv is ALREADY an implicit GEMM** (gathers the 3x3 columns on-the-fly + reduces),
  achieving 0.70 TFLOP/s. The explicit-GEMM reduce hits 0.90 TFLOP/s for the SAME 35.8M MACs — but
  ONLY because im2col pre-arranged the data into a clean contiguous reduction (no per-tap shift/
  boundary). Fuse the gather back in and you return toward the direct conv's 0.70, NOT 0.90.
- **"Keep im2col layout across the chain" gives no amortization.** The im2col tensor is 9x wider; a
  GEMM output in im2col layout = writing a 9x tensor (same 37us), just relocated from the consumer's
  input to the producer's output. And each GEMM outputs normal [Cout,H,W], so every conv must re-
  im2col its input — in a 6-chain the GEMM path pays im2col 6x: 6x(37+80)=702us vs direct 6x102=612us
  (WORSE). No output/col2im reshape is needed (GEMM outputs normal layout), but that doesn't help.
- **A fused GEMM-arch megakernel** (implicit im2col + on-chip intermediates across layers) is the
  right target but hits BOTH already-falsified walls at once on 8 CUs: single-conv implicit-GEMM-via-
  LDS = the LDS kernel (+9%, L2 already caches the reuse; barriers), and cross-layer fusion (<2%
  traffic ceiling) + halo recompute. The GEMM efficiency edge (~1.3x) < the occupancy cost of holding
  fused state. **Empirical closure — REAL fused megakernel BUILT & MEASURED (conv_2d_3x3s1_fused2,
  env MNN_CONV_FUSED2): 6.5x SLOWER.** One workgroup computes a 6x6 output tile through 2 conv layers,
  input halo (10x10) + intermediate (8x8) staged in LDS, no global intermediate. Correct (computes
  conv^2(x), cosine 0.997 = 2-layer fp16 noise, same precision as direct). Timing:
  32->32@72x96 fused **1570us** vs 2xdirect **240us** (+554%); 48->48@36x48 **1245us** vs **196us**
  (+535%). NOTE: my earlier composition estimate (~+26%) was BADLY too optimistic — the real fused
  kernel is ~6.5x worse because its structure compounds every occupancy penalty at once: 36-thread
  workgroups (< one 64-wave), 10-16KB LDS, inter-phase barriers, ~1.8x redundant halo compute. On
  8 CUs that is catastrophic. Lesson: the occupancy wall is not "a bit worse" for fusion — it is
  dramatically worse. Fused GEMM-arch on OpenCL/8-CU is conclusively dead; it needs Vulkan
  shuffle (no LDS/barriers) + coopmat + more CUs to have any chance.
- **The three ways past the 8-CU occupancy ceiling** (none available on this OpenCL/Clspv device):
  (1) Vulkan + subgroupShuffle (register-level gather → implicit GEMM w/o LDS or barriers);
  (2) cooperative matrix (hardware reduce, sidesteps software occupancy); (3) a GPU with more CUs.
  These are the levers for the Vulkan session / newer device.

### §H.16 — Blocking space now EXHAUSTIVELY bracketed: c8h1w1 + c4h8w1 both FALSIFIED
Question raised: "was a naive vectorized conv using the device's max thread count ever tried?" —
yes, that IS `conv_2d_c4h1w1` (1 output float4/thread, no blocking, max threads): 211.8us vs the
tuner's 119.7 at 32->32 (+77%), 131.3 vs 101.3 at 48->48 (+30%). Raw thread count is not a lever —
the 32->32 launch is already ~55k work-items on 8 CUs (~100x oversubscribed); occupancy is capped by
registers/CU, not by work-items launched. Two blocking points were nonetheless still missing from the
candidate set, so both were built (correct: cosine 1.000000 vs stock, incl. the BLOCK_LEAVE path at
36 rows) and force-measured (median of 3, 6-deep sustained chain):
- `conv_2d_c8h1w1` (2 acc, 8 oc x 1 pixel — highest thread count of any 2-acc variant, input float4
  loaded once and reused across both oc-blocks, zero halo growth): 204.5us @32 (+71% vs c8h4w1),
  146.3us @48 (+44% vs c4h1w2, and 12% WORSE than naive c4h1w1).
- `conv_2d_c4h8w1` (8 acc, 1 oc x 8 rows — same register class as the c8h4w1 winner at DOUBLE its
  weight-load amortization): 194.3us @32 (+62%), 213.0us @48 (+83%). Worse even than c8h8w1, which
  reaches the same amortization with 2x the registers.
**The "kernel is weight-load-bound" hypothesis that motivated c4h8w1 is FALSIFIED.** c4h8w1 and
c8h4w1 have identical accumulator count (8), identical thread count (6912 @32->32) and identical
total loads per inner iteration (12) — 194.3 vs 119.7us. Partial explanation: c4h8w1 doubles the
PER-LANE input loads (8/iter vs 4) to save WAVE-UNIFORM weight loads (4 vs 8), and uniform loads are
near-free (all lanes hit one address). But that model does not fit c8h1w1, which halves per-lane
input traffic vs c4h1w1 and gains only 3%. No counting model (input loads, weight loads, total
loads, accumulators, thread count, amortization) fits all ten measured points.
⇒ The blocking space is now bracketed on EVERY axis: accumulators 1->16, amortization 1x->8x, both
h- and w-direction, both oc-widths. The tuner's pick (c8h4w1 @32, c4h1w2 @48) is the optimum and is
not reachable by reasoning about load counts. Both kernels kept as documented negatives behind
MNN_CONV_SPEC (default off) + MNN_CONV_FORCE. Repro: `conv_bench/weight_amort_test.py` (full 10-point
sweep), `conv_bench/c8h1w1_test.py` (focused + correctness gate).

### §H.17 — Autotuner-fairness audit (closes the "did the tuner change the test?" question)
Read of `ConvBufExecution.cpp` + `OpenCLRunningUtils.cpp::localWS2DDefault`:
- With `MNN_CONV_FORCE=<name>` the candidate list is truncated to length 1, so `min_cost` selects
  that kernel unconditionally — the tuner CANNOT have swapped in a different kernel.
- Every forced variant still gets its own full LWS search: `localWS2DDefault(...)` runs per
  candidate at tune level **Wide** (harness passes gpuMode 68 = MNN_GPU_MEMORY_BUFFER | WIDE),
  which sweeps lws[0], lws[1] over powers of two up to gws / maxWorkItemSizes / maxWorkGroupSize
  and times each with a real enqueue. No divisibility requirement (GWS is ROUND_UP'd), so odd GWS
  like c4h8w1's (768, 9) is not excluded — it gets lws[1] in {1,2,4,8}.
- The LWS cache key is `(kernelName, gws)` and every measurement used a FRESH per-variant cache
  file, so no tuned LWS leaked between variants or between repetitions.
- Residual caveat (unchanged): LWS selection uses a single-shot timing per candidate, so a variant
  can land on a slightly suboptimal LWS from run-to-run noise — a few-percent effect, not the
  30-80% gaps measured. The one real exception remains the LDS kernel (fixed hand-picked LWS),
  which was separately swept over 11 tiles in §H.9.
- Empirical check: the 2026-08-13 re-run reproduced every previously measured variant to within
  ±1% (c4h1w1 211.7 vs 211.8, c4h1w2 207.0 vs 207.5, c8h4w1 119.3 vs 120.3, 48-core likewise).
⇒ The variant measurements are tuner-fair.

### §H.18 — Clock-regime review (INT/MIF max + GPU low) + a FAILED measurement proxy
**Cannot be measured on this device: no root.** `/sys/class/devfreq/23400000.sgpu/{min,max}_freq`
is read-only to shell (`Permission denied`); the MIF and INT domains exist
(`17000010.devfreq_mif`, `17000020.devfreq_int`) and are equally unwritable.
**FAILED proxy (do not repeat):** `conv_bench/clock_regime_test.py` pre-warms the tuner cache,
idles 6 s to let the sgpu governor park, then re-runs cached (no tuning burst) while polling
`cur_freq`, hoping the first inferences execute low-clock and ramp. Measured cold/hot ratios were
1.01-1.06x across six variants with an IDENTICAL ranking — the governor ramps to 980 MHz during
session creation / the first inference, so no low-clock execution is ever observed. The proxy
yields no signal; a rooted device (or a Samsung engineering build) pinning min_freq=max_freq is
required.
**Analytical review.** The regime flips the bottleneck. 32->32@72x96 is 127.4 MFLOP/conv; at
119.7us that is **1.06 TFLOP/s = ~16% of the ~6.5 TF fp16 peak at 980 MHz**. Scale the clock:
461 MHz peak = 3.06 TF (the SAME throughput would need 35% of peak); 226 MHz peak = 1.50 TF (would
need **71% of peak**). So below roughly 400-500 MHz the conv stops being memory-latency-bound and
becomes ALU/issue-bound — and MIF/INT at max pushes the crossover HIGHER still. Every §H verdict
was reached in the latency-bound regime, so the ones whose mechanism was occupancy or memory
traffic are the ones that move:
| strategy | verdict @980MHz | predicted @GPU-low + MIF/INT-max | why |
|---|---|---|---|
| PReLU fusion | -8/9% BANKED | **holds** | removes a whole kernel (cycles AND traffic); regime-independent |
| Winograd F(2,3) | contraindicated K<=128 | **FLIP CANDIDATE #1 — re-test** | the ONLY lever that cuts real MACs (2.25x), paid in memory passes. Verdict was "transforms cost more than the MACs saved" — true only when MACs are free and memory is the wall. Both halves invert here |
| im2col + GEMM | +15% (falsified) | **FLIP CANDIDATE #2 — re-test** | im2col is a pure memory pass (MIF-max makes it cheaper) and the GEMM does the same FLOPs at higher ALU efficiency (0.90 vs 0.70 TF) — exactly what matters when ALU-bound. Only 15% behind at 980 |
| bigger tiles (c8h8w1 +27%, c4h8w1 +62%) | falsified | **penalty shrinks; re-test** | both lost on occupancy / per-lane loads; at low clock latency hiding is easier and instruction efficiency dominates. Optimum shifts toward LARGER tiles |
| cross-stream concurrency | 1.34x, lever #2 | **DEGRADES toward 1.0** | the spare capacity IS the memory stalls; at ~71% ALU occupancy there is little left to overlap |
| cut CPU/submission overhead (~50% of wall) | live lever | **DEGRADES in relative value** | GPU time grows ~4x while CPU cost is fixed -> overhead share falls from ~50% to ~20% |
| LDS tiling | falsified (+0.7/+9.2%) | still negative, less so | saves traffic that is now abundant; costs barriers/occupancy that now matter less |
| space2depth on heads | falsified (+12-17%) | **worse** | buys occupancy by inflating dense FLOPs 1.78x — the exact wrong trade when FLOPs are scarce |
| widen channels at equal MACs (model) | up to 2.5x | **shrinks** | the win was escaping occupancy starvation, which is less punishing at low clock. At GPU-low the model lever inverts to FEWER MACs (separable/pruning), not wider layers |
| hardware-friendly shapes (Cout mult of 16/32) | live | **more important** | wasted lanes = wasted FLOPs, and FLOPs are now the scarce resource |
| custom Vulkan conv (+3.5%/+45%) | top substrate lever | **holds or improves** | its wins are width blocking, half accumulators, NC4HW4 stores, wave32/64 — all instruction-efficiency, which pays more when ALU-bound |
⇒ Headline: at GPU-low the two biggest *falsified* strategies (Winograd, im2col+GEMM) are the two
most likely to flip positive, and the two biggest *live* non-kernel levers (concurrency, CPU
overhead) are the ones that decay. Re-testing needs a rooted device; env switches already exist
(`MNN_NO_WINOGRAD`, `MNN_CONV_IM2COL`, `MNN_CONV_SPEC`+`MNN_CONV_FORCE`).

### §H.19 — Stride-2 head pairs: variant sweep + dedicated stride-2 LDS kernel (FALSIFIED)
New benchmark cases (bundle `heads`): three 2-conv stride-2 pairs, each conv reported SEPARATELY
by matching the shape tag MNN puts in the kernel name (averaging two different shapes would be
meaningless). Plus a new 6-deep stride-1 core 96->96@18x24.
Full 11-variant sweep, per conv (us, Xclipse 960, quick mode):
| conv | MNN default | c8h4w1 | c8h4w1_pa | c4h1w1 | c4h1w2 | c4h1w4 |
|---|---|---|---|---|---|---|
| 18->16@288x384 s2 | 149 | 148 | 148 | 333 | 439 | 712 |
| 16->32@144x192 s2 | 66 | 66 | 72 | 141 | 178 | 214 |
| 34->32@144x192 s2 | 139 | 139 | 151 | 305 | 434 | 475 |
| 32->48@72x96 s2 | 80 | 80 | 71 | 106 | 104 | 110 |
| 64->64@72x96 s2 | 160 | 160 | 157 | 283 | 269 | 248 |
| 64->96@36x48 s2 | 103 | 126 | 108 | 109 | 103 | 165 |
**Finding 1 — w-blocking COLLAPSES at stride 2** (c4h1w4 = 4.8x the default on the first conv; at
stride 1 the same kernel was only ~2x off). Mechanism: with stride 2 a w-blocked thread covers
output cols x..x+3, i.e. input cols 2x-1..2x+7, so the wave's reads become strided with real gaps.
h-blocking does not do this (threads still cover consecutive output columns) and ties the default.
**Finding 2 — nothing beats MNN's own pick by more than ~11%** (c8h4w1_pa: 80->71, 160->157). The
direct-blocking space is saturated for stride 2 exactly as it is for stride 1.

**Dedicated stride-2 LDS kernel (`conv_2d_3x3s2_lds`, MNN_CONV_LDS on a stride-2 conv): BUILT,
CORRECT, FALSIFIED.** Hypothesis: the stride-1 LDS null result does not transfer, because at
stride 2 neighbouring threads read pixels two apart, so each cache line should be half wasted --
a coalescing defect that cooperative contiguous staging would fix. Correct (cosine 1.000000, max
abs diff 0.0 vs the stock path on a [1,32,32,64] s2 fixture) and confirmed to actually run
(41us -> 61us on that fixture, so not a silent fallback). Measured on the real heads:
| conv | default | LDS 16x4 | LDS 32x4 | LDS 16x8 | LDS 8x4 |
|---|---|---|---|---|---|
| 18->16@288x384 s2 | 148 | 318 | 316 | 316 | 366 |
| 16->32@144x192 s2 | 66 | 134 | 132 | 133 | 156 |
| 34->32@144x192 s2 | 139 | 289 | 463 | 643 | 638 |
| 32->48@72x96 s2 | 80 | 123 | 129 | 178 | 258 |
2.0-2.5x SLOWER wherever it runs, across every tile tried. The one cell that looked competitive
(64->64@72x96: LDS 32x4 360 vs default 398) was an artefact twice over: the device was thermally
drifted (that conv is 160us cooled, not 398), AND out_w=48 is not a multiple of TILE_W=32 so the
LDS path silently fell back -- a cooled INTERLEAVED re-run gives default 160 / "LDS" 160, +0%,
i.e. the same kernel measured twice.
**The hypothesis was WRONG, and the reason is instructive:** at stride 2 with 3x3/pad 1, thread x
reads input pixels 2x-1,2x,2x+1 and thread x+1 reads 2x+1,2x+2,2x+3 -- the union across the wave
still covers EVERY input pixel contiguously. The cache lines are fully consumed, just by different
(thread, tap) pairs. There is no cache-line waste to recover, so LDS only adds barriers and
occupancy cost, exactly as at stride 1. The real coalescing loss at stride 2 comes from
W-BLOCKING (Finding 1), not from stride 2 itself.
Kept as a documented negative behind MNN_CONV_LDS (default off), same as the stride-1 variant.

**§H.19b — the 96-channel core: kernel choice stops mattering, and equal MACs run 3.4x faster.**
The new 6-deep core 96->96@18x24 measures **30.0us on ALL THIRTEEN implementations** (29.8-30.8:
MNN default, all 11 forced variants, and LDS). Nothing distinguishes them. Compare at IDENTICAL
work -- 48->48@36x48 and 96->96@18x24 are both exactly 71.7 MFLOP per conv:
| conv | MFLOP | time | achieved | % of ~6.5 TF peak |
|---|---|---|---|---|
| 48->48@36x48 | 71.7 | 101.5us | 0.71 TFLOP/s | 11% |
| 32->32@72x96 | 127.4 | 119.2us | 1.07 TFLOP/s | 16% |
| **96->96@18x24** | **71.7** | **30.0us** | **2.39 TFLOP/s** | **37%** |
Same MACs, **3.4x faster**, and the best utilisation measured anywhere in this investigation. The
flatness across implementations is the tell: at 96 channels the conv has left the occupancy-starved
regime, so blocking choice no longer changes anything -- whereas at 32/48 channels the spread
between the best and worst kernel is ~2x. This is a direct, controlled confirmation of the
model-level lever (§H.7 "widen channels at equal MACs"), now measured on shapes from the real
model rather than synthetic sweeps. Practical reading: **no kernel work on the 32/48-channel cores
can approach what changing their shape to 96 channels at equal MACs would deliver.**
Run was thermally VALID (980 MHz start and end, 0% drift).

### §H.20 — CEILING RECALIBRATION: the practical limit is ~3.0 TFLOP/s, not 6.5
Every prior conclusion in §H was framed against an assumed ~6.5 TF fp16 peak, which made these
convs look like "15% of peak, enormous headroom". That framing was wrong. Measured ceiling sweep
(MNN's own conv, 3x3 s1, fp16 buffer, 200 loops, median of 3):
| shape | MFLOP | time | TFLOP/s | % of observed ceiling |
|---|---|---|---|---|
| 64->64@36x48 | 127.4 | 42.0us | **3.03** | 100% |
| 128->128@18x24 | 127.4 | 42.0us | **3.03** | 100% |
| 192->192@12x16 | 127.4 | 46.0us | 2.77 | 91% |
| 96->96@18x24 | 71.7 | 30.0us | 2.39 | 79% |
| 256->256@9x12 | 127.4 | 57.0us | 2.24 | 74% |
| 384->384@6x8 | 127.4 | 87.0us | 1.46 | 48% |
| **32->32@72x96** | 127.4 | 119.2us | **1.07** | **35%** |
| **48->48@36x48** | 71.7 | 101.5us | **0.71** | **23%** |
Two different shapes land on exactly 3.03 TF, which reads as a real hardware/compiler limit. Cross-
check: 8 CU x 64 lanes x 2 flop x 2 (packed fp16) x 0.98GHz = 2.0 TF, and we EXCEED that, so packed
fp16 (v_pk_fma_f16) is definitely being emitted and the device is better described as ~4 TF peak
with MNN reaching ~76% of it.
**Consequences:** (1) the model's cores run at 23-35% of what this GPU actually delivers, so the
maximum available from ANY kernel change is ~3x, not ~6x; (2) 96->96@18x24 at 79% of ceiling is
nearly done, which is exactly why all 13 implementations tie at 30.0us there; (3) every kernel
strategy failing by <=11% is consistent with the cores being occupancy-starved rather than
badly implemented.

**The sharpest actionable result: widening channels makes the conv faster in ABSOLUTE terms even
while doing more work.**
| conv | MACs | time |
|---|---|---|
| 48->48@36x48 | 35.8M | 101.5us |
| 64->64@36x48 (same spatial, +78% MACs) | 63.7M | **42.0us** |
| 96->96@18x24 (identical MACs to the 48 case) | 35.8M | **30.0us** |
1.78x the arithmetic in 0.41x the time (4.3x efficiency), and at equal MACs 3.4x faster. Corroborated
externally: performance-aware channel pruning on embedded GPUs reports 2x SLOWDOWNS from removing
12% of channels and 3-10x speedups when channel counts match what the library/hardware prefers
(arXiv 2002.08697) - i.e. these cliffs are a known, general mobile-GPU effect, not an MNN quirk.
Also relevant for batch=1 mobile inference: HNTMP / ILP-M Conv reports 2.3x over direct convolution
and 14.6x over im2col on mobile GPUs (arXiv 1909.02765) - the one published algorithm aimed at
exactly this regime, and the only remaining unexplored algorithmic lead.

### §H.7 — FINAL VERDICT (Session A restricted-set specialization)
Levers tried on the real Block1/Block2: PReLU fusion = BANKED (-8/9% per block, shippable, no new
code). Falsified on-device with numbers: NC4HW4-input theory, cross-layer fusion (<2%), LDS tiling
(1.78x slower), register-block unroll (+0%), space2depth-heads (+12-17%). The convs — cores AND
heads — are at their practical ceiling for MNN's autotuned direct path on Xclipse-960/ANGLE; no
kernel restructuring or algebraic rewrite beats it for this shape regime. Ship: convert with
MNN_FUSE_CONV_PRELU=1. Only remaining targeted idea: space2depth on the true C=1 stem (not in these blocks).
