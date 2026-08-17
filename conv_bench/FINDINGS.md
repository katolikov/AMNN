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

### §H.21 — ILP-M / HNTMP evaluated; the 2-D register tile WINS on the main core
Investigated the one published algorithm aimed at our regime (batch=1 mobile-GPU conv).
**What it is:** ILP-M ("Instruction-Level Parallelism Maximizing"), a.k.a. HNTMP (arXiv 1909.02765).
A thread owns ONE output channel plus a 2-D spatial register tile `out_reg[wy][wx]`, so each loaded
filter coefficient is reused across the whole tile with NO barriers. Reported 14.6x over im2col and
2.30x over direct conv on Mali-G76.
**Most of it does not transfer:** its core mechanism is barrier-free filter reuse, and MNN's direct
conv_2d_c* kernels are already barrier-free (only LDS/fused2 use barriers). Its benchmarks are
ResNet conv2-conv5 (64-512 ch at 7x7-56x56); the paper does not cover 16-48 ch or large spatial, and
its home regime is where this device already runs at 74-100% of ceiling (§H.20).
**What DOES transfer:** every MNN variant blocks in h OR w, never BOTH. The 2-D register tile
geometry was untested. Built `conv_2d_c4h4w2` = 4 out-channels x 4 rows x 2 cols = 8 accumulators
(the c8h4w1 winner's register class), stride-1 only, env MNN_CONV_SPEC.

**Implementation matters enormously here — the first attempt was a false negative.** A version that
streamed input rows and dispatched accumulators through a dynamic `for(orow=max(0,r-2); orow<=min(3,r))`
loop with an if/else chain plus ternary column selects measured **308us** (2.6x slower) and looked
like a clean falsification. Fully unrolling it -- every input row, output row and tap expanded
statically, no dynamic accumulator indexing, no selects -- gives **109us on the same shape**. The
unrolling was worth **2.8x**; the dynamic version was almost certainly spilling accumulators to
scratch. Lesson: for register-tiled kernels, never judge a geometry from an implementation that
indexes accumulators dynamically.

**Measured (unrolled, interleaved, cooled; correctness cosine 1.000000 = bit-exact):**
| core | MNN default | c8h4w1 | c4h4w1 | c4h4w2 (2-D, unrolled) |
|---|---|---|---|---|
| **32->32@72x96** | 119.5 | 119.3 | 127.7 | **109.3  (-8.5%)** |
| 48->48@36x48 | 101.3 | 116.5 | 105.2 | 112.2 (+11%) |
| 96->96@18x24 | 30.2 | 30.3 | 30.3 | tie (at 79% of ceiling, everything ties) |
Raw cooled samples on the winning shape: default [119.8, 119.5, 119.3] vs c4h4w2 [109.3, 109.2, ...],
i.e. tight and reproducible. **This is the FIRST kernel in the whole investigation to beat MNN's
autotuned default on the dominant core shape.** It is shape-dependent (wins the large-spatial
32-channel core, loses the smaller 48-channel one), which is exactly what the autotuner exists to
resolve -- adding it to the DEFAULT candidate list would let the tuner pick it where it wins.
**Thermal caveat for anyone reproducing:** this device throttles ~2.75x (119 -> 328us) after
sustained benchmarking. Medians over a batch that straddles the throttle point are meaningless;
interleave the arms AND cool between reps.

### §H.22 — 2-D tile shape sweep: geometry beats register count, and the 16-acc theory holds
Generated the whole 2-D tile family from one template (fully unrolled, no dynamic indexing, stride-1,
env MNN_CONV_SPEC). All bit-exact (cosine 1.000000). Cooled + interleaved on 32->32@72x96, the
dominant core (48->48@36x48 numbers in the same batch were thermally invalid -- its default read
187us against a cooled 101us -- so the 48 figures below come from the earlier cooled run):
| variant | accs | geometry | 32->32@72x96 | vs default |
|---|---|---|---|---|
| MNN default | - | autotuned | 119.7 | - |
| c8h4w1 | 8 | 8ch x 4rows (1-D) | 120.2 | +0.4% |
| **c4h4w2** | **8** | **4ch x 4rows x 2cols (2-D)** | **109.3** | **-8.6%** |
| **c4h2w4** | **8** | **4ch x 2rows x 4cols (2-D)** | **115.5** | **-3.5%** |
| c4h2w2 | 4 | 4ch x 2rows x 2cols | 130.2 | +8.8% |
| c4h4w4 | 16 | 4ch x 4rows x 4cols | 399.3 | **+233.7%** |
**Finding 1 — at EQUAL accumulator count, 2-D beats 1-D.** c4h4w2 (109.3) and c4h2w4 (115.5) both
beat c8h4w1 (120.2) with the same 8 accumulators. So it is the tile GEOMETRY that matters, not the
register budget alone: spreading the tile over two spatial axes gives better input reuse per load
than spreading it over channels or over one axis. TWO different 2-D tiles now beat MNN's autotuned
default; c4h4w2 reproduces at -8.6% (independently measured -8.5% earlier).
**Finding 2 — 4 accumulators is too few** (c4h2w2 +8.8%): the tile stops amortising enough.
**Finding 3 — the 16-accumulator theory SURVIVES.** §H.8 rejected all 16-acc variants by reasoning
("register pressure is the same for any 16-acc variant so they're expected to regress identically"),
without building them. c4h4w4 is exactly that test and regresses +233.7% -- worse than c8h8w1's
+48.7%. The theory-only rejection was correct. **The optimum is 8 accumulators arranged in 2-D.**
On 48->48@36x48 (cooled) c4h4w2 is +11%, i.e. the win is shape-dependent -- which is what the
autotuner exists to resolve. All variants are kept in the standard suite regardless of whether they
win on this device: the winner is device- and clock-dependent (§H.18).

### §H.23 — Shape hardcoding: -18.7% vs MNN's default, the largest kernel win found
Per the "specialise for our exact convs, duplication is fine" brief: every MNN conv kernel takes the
whole shape as RUNTIME arguments (in_hw, out_hw, in_c_blocks, out_c_blocks, batch), so the compiler
cannot fold index arithmetic, cannot bound the channel loop, and must emit a bounds branch per halo
column. Added `MNN_CONV_HARD=1`, which passes -DHC_IN_H/-DHC_IN_W/-DHC_OUT_H/-DHC_OUT_W/-DHC_ICB/
-DHC_OCB/-DHC_BATCH/-DHC_WB/-DHC_HB so every one becomes a compile-time constant. Costs one program
build per distinct shape (cached by MNN's program cache). Correct: cosine 1.000000.
| core | MNN default | c4h4w2 | c4h4w2 + hardcoded shape |
|---|---|---|---|
| **32->32@72x96** | 119.2 | 110.2 (-7.6%) | **96.8 (-18.7%)** |
| 48->48@36x48 | 101.2 | 114.8 (+13.5%) | 101.7 (+0.5%) |
| 96->96@18x24 | 30.3 | 30.3 | 30.3 (already at 79% of ceiling) |
**-18.7% on the dominant core** -- more than double what the 2-D tile geometry alone gives, and the
largest single-kernel win in this whole investigation. On the 48-core it rescues c4h4w2 from +13.5%
back to parity, i.e. constant folding is worth ~13% there too.
**Trap found and isolated:** the first version also put `opencl_unroll_hint` on the now-constant
channel loop. That made it **+13.4% SLOWER than the non-hardcoded kernel** (135.7 vs 110.2) -- fully
unrolling an 8-iteration loop around ~600 lines of generated code blows up I-cache/registers.
Constant folding and loop unrolling are SEPARATE levers and must be measured separately: folding
alone -18.7%, folding+unroll +13.4%. The unroll is now opt-in via -DHC_UNROLL_IC (default off).
**Control:** `default+HARD` measured 119.7 vs 119.7 -- unchanged, because only the four 2-D tile
kernels currently read the HC_* macros. **Applying the same macros to MNN's own stock kernels
(c8h4w1, c4h1w2, ...) is therefore an untested, broadly-applicable lever** and the obvious next step.

### §H.24 — CORRECTED: hardcoding the stock kernels is NOT blocked; the "clspv crash" was my test bug
**The previous version of this section was wrong and is retracted.** It reported that extending the
-18.7% shape-hardcoding win (§H.23) to MNN's 7 stock kernels was blocked by a reproducible ANGLE/clspv
crash. It is not. Every one of those segfaults was a **shape mismatch in the smoke test**, not a
compiler failure.
**Root cause.** The measurement helpers (`run_report.run_model`) rewrite `tdir/input.json` on every
call. After any measurement loop, the staged input shape is whatever that loop ran last — e.g.
`[1,48,36,48]`. The follow-up smoke test then ran `cc.mnn`, which needs `[1,32,24,48]`, against that
stale descriptor. MNN segfaults on the mismatch, before printing anything, and stdout is block-
buffered so the crash presents as a bare `Segmentation fault` with no output whatsoever — exactly
like a compiler crash. Pushing the correct `input.json` makes the identical, unmodified build run
fine (0.31 ms).
**What this invalidates:** the "clspv crashes on insertions/relocations near the stock kernels"
conclusion, the "even a single duplicated kernel crashes" datum, and the §H.25 corollary that
"in-place edits after the macro block are safe while insertions are not". All five of those
"crashes" are explained by the fixture bug. Nothing is known to be wrong with any of the approaches
tried: relocating the macro block, `_hc` duplicate copies, and supplying HC* purely as `-D` build
options are all still viable and **none has actually been measured**.
**Lesson for the harness:** a stale `tdir/input.json` is indistinguishable from a compiler crash.
Any standalone `ModuleBasic.out` invocation must push an `input.json` matching that model first —
do not inherit whatever a previous measurement left behind.
**Still true and worth keeping:** the `fflush(stdout)` added to the build-failure path (a genuine
build failure would otherwise be swallowed the same way), and the three real .cl authoring traps —
the first `__kernel` in conv_2d_buf.cl sits inside `#ifdef CONV_LOCAL_SIZE`; `DEAL_NON_UNIFORM_DIM2`
is a backslash-continued multi-line macro that must not be split; and the codegen does not escape
double quotes, so a `"` in a comment breaks the generated C++.
**Status: OPEN, unblocked, not implemented.** The next attempt should use the `-D` design (host emits
`-DHCINH=in_hw.x` by default and `-DHCINH=72` under MNN_CONV_HARD, with no macro block in the .cl at
all) and must verify with a correctly-shaped `input.json`.

### §H.25 — Hardcoding + conv fusion: DEAD, for three independent reasons
Asked whether the -18.7% shape-hardcoding win (§H.23) could rescue 2-conv fusion, in particular by
letting the intermediate live in REGISTERS instead of the LDS that sank `fused2` (6.5x slower).
**1. Register-resident fusion is capacity-infeasible.** conv2 reduces over ALL input channels, so a
thread owning a TxT tile of conv2's output needs `(T+2)^2 x ICB` float4 of conv1 output:
| conv2 tile | 32ch (ICB 8) | 48ch (ICB 12) | conv1 recompute |
|---|---|---|---|
| 1x1 | 72 float4 = 144 VGPR | 108 = 216 VGPR | **9.00x** |
| 2x2 | 128 = **256 VGPR** | 192 = **384 VGPR** | 4.00x |
| 4x4 | 288 = 576 VGPR | 432 = 864 VGPR | 2.25x |
Against a 256 VGPR/lane hardware maximum. T=2 is already at or past the limit; T=1 fits but
recomputes conv1 **9x** (conv1 and conv2 cost the same, so fusing at T=1 costs 9+1=10 units against
2 unfused). This is arithmetic and hardware capacity, not an estimate: hardcoding cannot unlock
register-resident fusion. The intermediate MUST live in LDS.
**2. Hardcoding the existing LDS fusion changes nothing — measured.** `conv_2d_3x3s1_fused2` now
reads the HC_* macros (it sits after the macro block, so nothing had to move; correct, cosine
0.993444 vs the numpy conv^2 reference either way):
| core | 1 conv | 2x1 conv | fused2 | fused2 + HARD |
|---|---|---|---|---|
| 32->32@72x96 | 120 | 240 | 1569 | **1570** |
| 48->48@36x48 | 100 | 200 | 1245 | **1245** |
Identical to the digit. `fused2` is bound by LDS occupancy and barriers (~10.5 KB per 36-thread
workgroup), not by index arithmetic, so constant folding has nothing to fold that matters.
**3. The prize was never there.** §H.1 measured the cross-layer fusion ceiling at **<2%**:
inter-layer traffic is under 1 us/conv and per-conv time is flat versus chain depth.
⇒ Fusion is closed on all three axes. The only remaining fusion-shaped argument is DISPATCH
reduction for wall-clock (~50% of wall is CPU/submission overhead), which is a different mechanism
and is better attacked with MNN's record/replay queue than with a fused kernel.
**Note:** an earlier version of this section drew a corollary about which .cl edits are "safe" from
the supposed clspv crash. That crash was a test-fixture bug (see the corrected §H.24); the corollary
is withdrawn. The fusion results above are unaffected -- they are direct measurements.

### §H.26 — Shape-specialised copies of the STOCK kernels: implemented, and a new win on the 48-core
§H.24 claimed this was blocked by a clspv crash; that was retracted (it was a stale-input.json test
bug). Implemented properly this time, in a **dedicated new program file** so nothing existing moves:
`conv_2d_hc_buf.cl` holds 7 copies (`conv_2d_c4h1w1_hc` ... `conv_2d_c8h1w4_hc`) of MNN's stock
kernels with every shape value read from a `-D` option; the originals in conv_2d_buf.cl are
byte-identical and untouched. The host picks the program by name suffix (`hcProgramFor`) and emits
each value as the runtime expression by default, a literal under `MNN_CONV_HARD=1`.
All 7 correct: **cosine 1.000000**.
**Measured (cooled, interleaved; raw per-rep values tight):**
| core | MNN default | plain kernel | + hardcoded |
|---|---|---|---|
| 32->32@72x96 | 119.2 | c4h4w1 127.5 | **c4h4w1_hc 113.2** (-11% vs its twin, **-5% vs default**) |
| | | c8h4w1 119.2 | c8h4w1_hc 137.7 (**+16%** -- hardcoding HURTS this one) |
| | | c4h1w2 206.7 | c4h1w2_hc 204.3 (-1%) |
| 48->48@36x48 | 101.0 | c4h1w2 101.3 | **c4h1w2_hc 96.0 (-5% vs default)** |
| | | c4h4w1 105.2 | c4h4w1_hc 133.8 (+33% -- hurts) |
**Reading:** constant folding is **kernel- and shape-dependent, not a free win**. It helps
c4h4w1 on the 32-core (-11%) and c4h1w2 on the 48-core (-5%), and actively hurts c8h4w1 (+16%) and
c4h4w1 on the 48-core (+33%). There is no "hardcode everything" rule -- it has to be measured per
(kernel, shape), which is exactly what the autotuner is for.
**Why it still matters: the 48-core finally has a winner.** Nothing had beaten MNN's default on
48->48@36x48 in this entire investigation (c4h4w2+HARD was +0.5%, every direct variant lost).
`c4h1w2_hc` at 96.0 vs 101.0 is the first real gain there.
**Current best per shape:** 32->32@72x96 = `c4h4w2` + MNN_CONV_HARD (96.8, -18.7%);
48->48@36x48 = `c4h1w2_hc` + MNN_CONV_HARD (96.0, -5%); 96->96@18x24 = anything (at 79% of ceiling).
**Build-system trap found (this is what actually blocked it for two sessions):**
`opencl_codegen.py` writes `opencl_source_map.hpp` **into the current working directory**, not into
`cl/`. Running it from the repo root leaves the real map stale -- harmless for existing programs
(their entries already exist) but fatal for a NEW one, which then fails with `Can't find kernel
source !`. A new `.cl` also needs a **cmake re-configure** so the glob picks up its generated `.cpp`.
Run the codegen from inside `cl/`. Only diagnosable because of the fflush added to the build-failure
path -- without it this too presents as a bare segfault.

### §H.7 — FINAL VERDICT (Session A restricted-set specialization)
Levers tried on the real Block1/Block2: PReLU fusion = BANKED (-8/9% per block, shippable, no new
code). Falsified on-device with numbers: NC4HW4-input theory, cross-layer fusion (<2%), LDS tiling
(1.78x slower), register-block unroll (+0%), space2depth-heads (+12-17%). The convs — cores AND
heads — are at their practical ceiling for MNN's autotuned direct path on Xclipse-960/ANGLE; no
kernel restructuring or algebraic rewrite beats it for this shape regime. Ship: convert with
MNN_FUSE_CONV_PRELU=1. Only remaining targeted idea: space2depth on the true C=1 stem (not in these blocks).

---

# Session B — closing the strategies that were rejected WITHOUT being built

### §H.27 — MEASUREMENT CORRECTION: `conv time` does not count the Winograd transforms
**This invalidates every Winograd-vs-direct comparison made with `conv_us()`, including the first
half of this session and parts of the §H.20 ceiling calibration.**

MNN's profiler prints `conv time = N us (gemm2:.. gemm1:.. 1x1:.. ori:.. wino:.. other:..)`, and
`conv_bench`'s `conv_us()` reads it. That counter includes the Winograd **batchgemm** but **not**
the `Conv-winograd-rearrange` transform kernels — which are launched **twice per conv** and cost
roughly as much as the batchgemm they feed. Scoring a Winograd arm with it therefore credits the
batchgemm half only.

Caught by an end-to-end control: forced Winograd measured **−45%** on `conv time` for the 32-core
but made the whole Block1 graph **+13% slower**. The per-kernel dump settles it — Block1 contains
6 convs of `32→32@72×96`:
| arm | kernels per conv | per-conv total |
|---|---|---|
| default | `ConvBuf2D-ori` 119.0 us | **119.0 us** |
| forced Winograd | `batchgemm` 66.0 + 2 × `rearrange` 40.5 | **147.0 us** |
The rearrange event count confirms the 2x launch rate (x503 events vs x252 batchgemm over 42 loops).

**Harness fix:** `conv_bench/session_measure.py` now carries `conv_all_us()` (sums every
conv-named kernel event per loop) and `conv_us()` documents that it must never be used for an arm
that can change which conv *implementation* runs. `total_us()` (whole-graph kernel time) is the
other safe metric.

**Consequence for §H.20:** the ceiling table was built with `conv time` on shapes where MNN's
heuristic *already selects Winograd* (`64→64@36×48`, `128→128@18×24`, `96→96@18×24` all satisfy the
gate), so those numbers under-count the transforms too. Directly re-measured here:
`96→96@18×24` is **51.0 us**, not the 30.2 us on record — 1.7x. The "3.03 TFLOP/s practical
ceiling" and the "79% of ceiling, everything ties" reading of the 96-core are therefore both
overstated and need re-deriving on `conv_all_us`. **Not re-derived in this session — flagged as open.**

### §H.28 — Forcing Winograd ON: a real win on the 48-core, a regression on the 32-core
`MNN_FORCE_WINOGRAD=1` (new, `ConvBufWinograd::valid`, default off) bypasses the channel/size
heuristic while keeping the hard 3×3 / stride-1 / dilation-1 requirements. This closes the standing
claim that "Winograd is contraindicated for K≤128", which had never been measured on this device —
only the selector's opinion of it had (`MNN_NO_WINOGRAD=1` is a no-op precisely because the
heuristic never selects Winograd for these shapes).

Correct (cosine 0.999979–0.999994 vs default — F(2,3) rounding, not a bug). Cooled, interleaved,
3 reps, metric = `conv_all_us` on the 6-deep chain, cross-checked against whole-graph `total_us`:
| core | default | forced Winograd | per-conv | whole graph |
|---|---|---|---|---|
| 32→32@72×96 | 118.7 | 136.9 | **+15.4%** | +18.8% |
| **48→48@36×48** | 101.0 | **87.8** | **−13.0%** | **−10.3%** |
| 96→96@18×24 | 51.0 | 52.0 | +2.0% (already Winograd) | +0.0% |

**End-to-end on the real blocks** (whole-graph kernel time, PReLU-fused models):
Block1 **+13.1%** (1407 → 1591 us) · Block2 **−3.5%** (1058 → 1021 us). Block1 contains six
32-core convs and regresses; Block2 contains the 48-core and gains. The two agree with the
per-shape table, which is the cross-check that matters.

**Reading:** the Winograd transform cost scales with spatial size while its arithmetic saving
scales with channel count, so the win appears at **small spatial × moderate channels** and the loss
at **large spatial × few channels**. `48→48@36×48` — the shape nothing had ever beaten by more than
5% — is the one core in the set that lands on the winning side, at −13%. This is the largest
verified win on that shape in the investigation, and it needs **no new kernel**: MNN already ships
the code, its selection heuristic just declines to use it.

**§H.18's prediction was directionally right:** it expected forced Winograd to look worse at full
clock. It does, on the large-spatial 32-core. It did not anticipate the 48-core win.

### §H.29 — Where forcing Winograd wins: a narrow island, not a rule
14-shape sweep, 3×3 s1 p1 +PReLU, 6-deep chain, cooled + interleaved, metric `conv_all_us`
(per conv, every conv kernel). "MNN picks" = what the stock heuristic selects; where it already
picks `wino` the flag is a no-op and the row is a **noise-floor control**.

| shape | default | forced wino | delta | MNN picks |
|---|---|---|---|---|
| 16→16@72×96 | 45.9 | 75.9 | **+65.5%** | ori |
| 24→24@72×96 | 85.7 | 127.0 | **+48.1%** | ori |
| 32→32@72×96 | 126.8 | 140.9 | +11.1% | ori |
| 48→48@72×96 | 262.6 | 294.0 | +11.9% | ori |
| 32→32@144×192 | 381.0 | 577.5 | **+51.6%** | ori |
| 16→16@36×48 | 26.0 | 35.5 | +36.6% | ori |
| **32→32@36×48** | 58.5 | 44.0 | **−24.7%** | ori |
| **48→48@36×48** | 100.9 | 87.4 | **−13.3%** | ori |
| 64→64@72×96 | 283.0 | 283.8 | +0.3% | wino (control) |
| 64→64@36×48 | 82.4 | 82.4 | +0.0% | wino (control) |
| 96→96@36×48 | 124.4 | 125.9 | +1.2% | wino (control) |
| 32→32@18×24 | 29.5 | 30.0 | +1.7% | wino (control) |
| 48→48@18×24 | 41.0 | 38.5 | −6.1% | wino (control) |
| 96→96@18×24 | 51.4 | 51.9 | +0.9% | wino (control) |

**Noise floor** from the six controls: within ±6%, mostly ±2%. The two wins (−24.7%, −13.3%) and
the large losses (+37% … +66%) are far outside it; the +11% pair sits comfortably outside too.

**The rule:** the F(2,3) transform cost scales with **spatial size**, its arithmetic saving with
**channel count**. So Winograd wins only where spatial is small *and* channels are ≥32:
- **36×48 spatial, C ∈ {32,48}: wins, −13% to −25%.** Both are shapes MNN currently refuses.
- 72×96 or larger: loses at every channel count measured (+11% … +66%).
- C ≤ 24: loses everywhere, at any spatial size.
- 18×24 spatial: MNN already picks Winograd and is right to.

**MNN's actual bug is the `input->width() < output->channel()` clause.** It admits Winograd only
when the image is narrower than the channel count, which is a proxy for "small spatial, many
channels" — the right *idea*, but calibrated so tightly that it excludes `32→32@36×48` (48 ≮ 32)
where Winograd wins by 25%. Loosening that one comparison — e.g. `in_w <= 1.5 × out_c` with the
existing `C ≥ 32` floor — captures both wins found here without admitting any measured loss
(`48→48@72×96` would need `96 ≤ 72`: still excluded; `16→16@36×48` is blocked by the C≥32 floor).
**Not implemented as a default change** — this is a 14-shape fit on one device, and MNN's heuristic
is global. Shipped as the measurement flag plus this table.

**For the application:** only `48→48@36×48` is a real model shape, and it gains 13% (−10.3% on the
whole Block2 graph). `32→32@72×96`, the dominant core, is firmly on the losing side, which is why
Block1 regresses 13%. **Winograd is a per-conv decision here, not a per-model one** — worth having
in the autotuner's candidate set, not worth forcing globally.

### §H.30 — c4h1w2 + LDS: §H.9's conclusion holds, but its MECHANISM was wrong
§H.9 closed the LDS-at-constant-blocking question by reasoning ("the constant-blocking isolation
settles it without needing to build it"), concluding LDS "captures NOTHING". Built it:
`conv_2d_3x3s1_lds_w2` (conv_2d_buf.cl, env `MNN_CONV_LDS=w2`) is byte-for-byte the c4h1w2
algorithm — 4 out-channels × 2 adjacent columns, 2 accumulators, same reduction order — with the
input read from `__local` instead of global. Its comparison partner is `conv_2d_c4h1w2` itself, so
**LDS is the only difference**. Bit-exact everywhere (cosine 1.000000).

Cooled, interleaved, 3 reps, `conv_all_us`:
| core | MNN default | c4h1w2 (twin) | **LDS w2** | LDS vs its twin | LDS 1-out/thread (§H.3) |
|---|---|---|---|---|---|
| 32→32@72×96 | 118.7 | 210.4 | **149.6** | **−28.9%** | 213.0 |
| 48→48@36×48 | 101.0 | 101.0 | 118.7 | **+17.5%** | 154.6 |
| 96→96@18×24 | 52.0 | 52.0 | 50.0 | −3.8% (noise) | 50.0 |

**Two findings, and they point opposite ways.**
1. **The mechanism claim was wrong.** At genuinely constant blocking, LDS captures a large amount
   on the big-spatial core: it makes the *same* algorithm **28.9% faster** (210.4 → 149.6). §H.3
   read "LDS bought nothing, the reuse was L2-served"; that is not true at 72×96, where the working
   set stops fitting and staging the halo pays. It flips sign at 36×48 (+17.5%), where the tile is
   small enough that L2 already serves it and the barriers are pure cost.
2. **The conclusion still holds.** LDS-w2 never beats MNN's autotuned default — 149.6 vs 118.7
   (+26%) on the 32-core, +17.5% on the 48-core, tie on the 96-core. What §H.3 actually
   demonstrated was that *c4h1w2 is a bad blocking for these shapes*, and LDS recovers part of that
   deficit without closing it. **LDS remains a documented negative for shipping**, now for the right
   reason and with the confound removed.

Registered in `conv_bench/make_bundle.py` (`LDS_MODES`) so the suite re-decides on new hardware —
a device with less L2 per CU could plausibly flip finding 2.

### §H.31 — Split-K over input channels: mechanism CONFIRMED, magnitude too small to ship
**Precondition test first.** Split-K only pays if the GPU is idle for want of threads, so before
building anything, batch scaling was used as a pure thread-count probe (batch is a multiplier that
changes nothing else; batch>1 is not proposed as an optimization — the app is always batch 1).
Perfect scaling = 2.00x means the machine was already saturated at batch 1.
| conv | N=1 | N=2 | N=4 | N2/N1 |
|---|---|---|---|---|
| **64→96@36×48 s2** | 99 | 132 | 209 | **1.33x** |
| 32→48@72×96 s2 | 78 | 108 | 182 | 1.38x |
| 64→64@72×96 s2 | 156 | 238 | 429 | 1.53x |
| 32→32@72×96 s1 | 125 | 203 | 379 | 1.62x |
| 48→48@36×48 s1 | 98 | 161 | 256 | 1.64x |
| 16→32@144×192 s2 | 69 | 116 | 216 | 1.68x |
Every shape is under 2.00x ⇒ **batch 1 does not saturate this GPU**, and the stride-2 heads are the
most starved, exactly as §H.20 argued. (Caveat: sub-linear scaling also amortises fixed per-launch
cost, so this over-states the pure occupancy headroom.)

**Built it.** `conv_2d_c4h1w1_splitk` + `conv_2d_splitk_reduce` (conv_2d_buf.cl), env
`MNN_CONV_SPLITK=<2|4|8>`, two passes through a DYNAMIC scratch buffer holding SPLITK un-reduced
partials — no atomics (this stack has no fp16 atomics). Bias and activation move to the reduce
pass. **Bit-exact at every split factor on every shape (cosine 1.000000).**

**The honest comparison isolates split-K from blocking.** The split-K kernel is c4h1w1 (1
accumulator, the max-parallel point an occupancy fix wants); MNN's default is an autotuned
8-accumulator kernel. Comparing them directly conflates two changes, so the control is c4h1w1
with and without the split:
| conv | MNN default | c4h1w1 (control) | +splitK=2 | +splitK=4 | split-K effect |
|---|---|---|---|---|---|
| **64→96@36×48 s2** (1.33x) | 100.0 | 111.0 | **98.0** | 100.0 | **−11.7%** |
| 32→48@72×96 s2 (1.38x) | 78.0 | 108.0 | 116.0 | 118.9 | +7.4% |
| 32→32@72×96 s1 (1.62x) | 123.8 | 213.0 | 221.8 | 231.7 | +4.1% |

**Reading — the theory is right and the payoff is still not there.** Split-K helps precisely and
only on the shape the diagnostic flagged as most starved (`64→96@36×48 s2`, −11.7% at constant
blocking), and hurts on the shapes with less idle hardware, with the damage tracking the batch-
scaling ratio in the predicted order. The mechanism is real. But against MNN's actual default the
best result is **98.0 vs 100.0 = −2%**, at the noise floor: the extra write-then-re-read of the
partials costs about as much as the recovered occupancy buys. Split factors above 2 always lose —
more parallelism, more scratch traffic, and the reduce pass grows.

**Verdict: negative for shipping, positive for the model of the device.** It confirms occupancy
starvation is real and measurable on the stride-2 heads, and it puts a ceiling on what any
occupancy-only fix can return here (~2% net). Kept env-gated, default off.

### §H.32 — ION / dma_buf zero-copy input: prize SIZED (~32% of wall), path BLOCKED for this harness
This one never touches kernel time — it removes the host→device input copy — so it is measured on
wall clock, not `conv time`.

**The extensions are real.** `MNN_DUMP_CL_EXT=1` on the device reports `cl_arm_import_memory`,
`cl_arm_import_memory_dma_buf`, `cl_khr_external_memory`, `cl_khr_external_memory_dma_buf`. The
import side is genuinely available.

**Sizing the prize.** Wall vs whole-graph kernel time (profiling build, 60 loops):
| model | input | wall min | kernel | gap | gap % of wall |
|---|---|---|---|---|---|
| Block1 | 7.96 MB fp32 | 3423 us | 1401 | 2022 | 59% |
| Block2 | 3.76 MB | 2535 | 1057 | 1478 | 58% |
| core_32 | 0.88 MB | 1830 | 793 | 1037 | 57% |
The gap is ~58% of wall everywhere, but it does **not** scale with input size the way a pure copy
would — input grows 9x while the gap grows 1.95x. Fitting `gap = a + b·MB` over the three points
gives **a ≈ 915 us fixed** + **b ≈ 139 us/MB** (predicts Block2 at 1438 vs 1478 measured, 3% error).
So for Block1 the size-dependent part is ~1107 us of a 3423 us wall — **an upper bound of ~32% of
wall** for anything that removes per-byte host traffic, against a ~915 us fixed submission cost that
zero-copy cannot touch. Two caveats that both shrink the real prize: the 139 us/MB term also covers
the **output** copy (which scales with these models too) and per-byte driver work that import would
not remove; and this is a **profiling build**, which inflates host overhead (§H.10).

**The path is blocked for this harness, and not for a reason a kernel change can fix.** Every
`/dev/dma_heap` node on the device is root-only (`crw-------`) or owned by `system`/`camera`/
`drmrpc` with no world access; `/dev/ion` is gone (as expected on a modern kernel). Our test binary
runs as uid 2000 `shell`, which is in none of those groups, so it **cannot allocate a dma_buf at
all** — there is nothing to import. This is not specific to the harness: a normal Android app
cannot open the heaps either. The realistic production route is `AHardwareBuffer_allocate` (gralloc,
reachable from an app) → extract the dma_buf fd → `clImportMemoryARM`, which needs an app-context
test binary and changes to MNN's input-tensor allocation, not just a kernel.

**Status: NOT IMPLEMENTED, prize sized, blocker identified.** Highest-value remaining wall-clock
lever found in this investigation (~32% ceiling, and it is the only lever that attacks the 58% of
wall that is not GPU kernel time at all), but it is an app-integration task rather than a conv task.
The other half of that 58% — the ~915 us fixed submission cost — is a separate lever, best attacked
with MNN's record/replay queue (already noted in §H.25).

### §H.33 — Register minimisation: NOT BUILT; the accumulator axis is closed, the weight axis is not
Item 4 of the session queue is the one item **not delivered**. What is and is not settled:

**Settled by measurement — the accumulator-count axis is fully bracketed** (all cooled, all on this
device): 1 acc `c4h1w1` 213.0 us · 4 acc `c4h2w2` +8.8% · **8 acc `c4h4w2` optimum** · 16 acc
`c4h4w4` +233.7% / `c8h8w1` +48.7%. Reducing accumulators below 8 is therefore already measured and
already loses — the "register minimisation" idea in its most obvious form is closed, and §H.10's
point that fp16 accumulators are already in force (Low precision injects `-DCOMPUTE_FLOAT=half`)
removes the other easy halving.

**Not settled — non-accumulator live registers.** Inspecting `conv_2d_c4h4w2` shows it *already*
streams input rows: each input row is loaded inside its own scope (`v0..v3`), so rows are not all
simultaneously live and §H.12's "in-register prefetch moves registers the wrong way" does not
describe the current kernel. The live set per tap is ~8 accumulators + 4 inputs + **4 weight
vectors** (`k0..k3`). The one untested direction is narrowing those weights: load `k0`, apply it to
all 8 accumulators, discard, then `k1`, and so on — ~12 fewer live VGPRs, which matters only if it
crosses a wave-occupancy quantum.

**Why it was not built:** it requires regenerating the whole ~600-line fully-unrolled kernel (these
variants must never index accumulators dynamically — that trap cost 2.8x once already), and it was
the lowest-EV item in the queue by the session's own ordering. **Its expected value is now lower
still:** §H.31 measured the ceiling on any occupancy-only fix at ~2% net on this device, and a
12-VGPR saving is an occupancy-only fix. Left open and explicitly flagged, not silently dropped.

---

## Session B — summary

| # | strategy | outcome | number |
|---|---|---|---|
| 1 | **Force Winograd ON** | **MIXED — real win on one core** | **−13.3% on 48→48@36×48, −24.7% on 32→32@36×48; +15.4% on the 32-core, +11…+66% elsewhere** |
| — | *`conv time` metric* | **BUG FOUND — invalidates prior Winograd comparisons** | *omits the transforms, launched 2x/conv* |
| 2 | Split-K over Cin | mechanism confirmed, too small to ship | −11.7% at constant blocking on the starved head; **−2% vs default** |
| 3 | c4h1w2 + LDS | §H.9's conclusion right, mechanism wrong | LDS = **−28.9% vs its own twin**, still +26% vs default |
| 4 | Register minimisation | **not built** | accumulator axis closed by prior data; weight axis open |
| 5 | ION / dma_buf zero-copy | prize sized, path blocked | **~32% of wall** ceiling; heaps inaccessible to uid `shell` |

**The one shippable result:** `48→48@36×48` — the shape that had resisted every kernel in the
investigation — gains **13%** by using Winograd, which MNN already implements and merely declines to
select. It needs no new kernel, only a loosened `input->width() < output->channel()` clause, and it
is worth **−10.3%** on the whole Block2 graph.

**The most valuable result is the negative one:** `conv time` silently under-counts the Winograd
path, and it had been the primary metric throughout §H. Any §H conclusion that compared a Winograd
arm to a direct arm — including the §H.20 ceiling calibration, where several reference shapes take
the Winograd path — needs re-deriving on `conv_all_us`. The corrected 96→96@18×24 figure is
**51.0 us, not 30.2**.

### §H.34 — The c4h4w2+HARD regression: a silent macro-name mismatch (FIXED)
`c4h4w2 + MNN_CONV_HARD` measured **−4%** instead of §H.23's −18.7%. Cause found and fixed; the
kernel was fine, the shape constants were never reaching it.

`conv_2d_buf.cl` carried a fallback block guarded by `#ifdef HC_IN_H`, defining `HCINH` as the
runtime expression `in_hw.x` when the guard failed. §H.26 renamed the host's build options from
`HC_IN_H`/`HC_OUT_H`/… to `HCINH`/`HCOUTH`/… (no underscores). The guard therefore **never fired
again**, the `#else` branch always ran, and — because `.cl` text is preprocessed *after* the
command-line `-D` macros — `#define HCINH in_hw.x` **silently redefined** the host's `-DHCINH=72`.
Last definition wins, so every shape constant reverted to a runtime argument.

**Net effect: `MNN_CONV_HARD` was a no-op for all four 2-D tile kernels** (c4h4w2/c4h2w2/c4h2w4/
c4h4w4) from commit 9fea7c34 onward. It kept working for the seven `_hc` kernels, which live in
`conv_2d_hc_buf.cl` and have no fallback block — which is why §H.26 still measured a HARD effect and
the breakage went unnoticed. No warning is emitted; the kernel just quietly gets slower.

**Fix:** each fallback is now individually `#ifndef`-guarded, so a host `-D` always wins, while
`conv_2d_3x3s1_fused2` (which builds its own options outside `hcPut` and genuinely needs the
defaults) still compiles.

**Restored, cooled + interleaved, `conv_all_us`, cosine 1.000000:**
| core | MNN default | c4h4w2 | c4h4w2 + HARD |
|---|---|---|---|
| 32→32@72×96 | 119.0 | 110.0 (−7.6%) | **96.8 (−18.7%)** |
| 48→48@36×48 | 101.0 | 115.0 (+13.9%) | 102.0 (+1.0%) |
96.8 us reproduces §H.23 to the digit. **The −18.7% bar is real and is restored.**

**Lesson for the harness:** an `#ifdef`-guarded fallback in a `.cl` that shadows a host `-D` fails
*silently and in the fast direction* — it looks like a performance regression, never a build error.
Any renamed build option must be grepped for in the `.cl` as well as the host.

### §H.35 — LLC / system-cache for inter-layer buffers: NO PRIZE (the traffic already costs ~0)
Idea: keep the intermediate tensor between consecutive convs in a last-level / system cache so the
round-trip beats DRAM.

**Two independent reasons it is dead, one measured and one structural.**

**1. Measured: the intermediate round-trip already costs nothing.** A chain of N identical convs
performs N−1 intermediate write+read round-trips. If those cost anything, per-conv time must rise
with N. It does not:
| shape | intermediate | d1 | d2 | d4 | d6 | spread |
|---|---|---|---|---|---|---|
| 32→32@72×96 | 432 KB | 118.1 | 120.0 | 119.0 | 118.7 | +1.6% |
| 48→48@36×48 | 162 KB | 100.4 | 101.0 | 100.6 | 100.7 | +0.6% |

The obvious objection is that these intermediates are small enough to be cache-resident anyway, so
the size was swept until they cannot possibly be:
| shape | intermediate | depth 1 | depth 6 | d6 vs d1 |
|---|---|---|---|---|
| 32→32@36×48 | 108 KB | 58.0 | 57.0 | −1.7% |
| 32→32@72×96 | 432 KB | 118.1 | 118.7 | +0.5% |
| 32→32@144×192 | 1728 KB | 381.0 | 383.0 | +0.5% |
| **32→32@288×384** | **6912 KB** | 1421.0 | 1415.0 | **−0.4%** |
At **6.9 MB** the intermediate exceeds any plausible cache on this part, and the round-trip is
*still* free to within measurement noise. So the convs are not stalled on intermediate traffic at
any size — the writes and re-reads hide completely behind compute (consistent with §H.20's
occupancy/latency-bound diagnosis and with §H.1's <2% fusion ceiling, now confirmed on the
corrected metric and over a 64x size range). **You cannot make ~0 faster.**

**2. Structural: there is no knob, and on this hardware there would not be one.** The full
extension list is `cl_arm_import_memory{,_dma_buf}`, `cl_khr_external_memory{,_dma_buf}`,
`cl_khr_{fp16,subgroups,priority_hints,integer_dot_product,image2d_from_buffer,…}`,
`cl_qcom_perf_hint` — **no cache-residency or memory-placement extension of any kind** (no
`cl_arm_scheduling_controls`, no MALL/system-cache hint). And the last-level cache on an RDNA-class
part is a hardware-managed victim cache over DRAM traffic, not an address space you allocate into;
there is no "put this buffer in LLC" operation to expose. Going through ANGLE→Vulkan removes even
the theoretical hooks.

**Verdict: FALSIFIED, and cheaply — the precondition test cost one sweep.** Worth noting the shape
of the result: the *reason* is the same one that killed cross-layer fusion (§H.1) and LDS staging
(§H.3/§H.30). Inter-layer and intra-conv memory traffic on this device are simply not where the
time goes; three separate strategies aimed at memory traffic have now all returned ~0.

### §H.36 — Plain NCHW convolution vs NC4HW4: real, shape-dependent, and NOT a win on this model
Built a genuine NCHW path (env `MNN_CONV_NCHW=1`): `cvt_nc4hw4_to_nchw` → `conv_2d_nchw_c4w8` →
`cvt_nchw_to_nc4hw4`, so the conv is **correct end-to-end** (cosine 0.999982–1.000000, including at
unaligned channel counts) rather than a timing hack, and the two conversion kernels are timed
separately — the caller's question was explicitly about the conv kernel, not the layout plumbing.

**The first NCHW kernel was a false negative — again.** v1 used an indexed `acc[4][8]` array and
scalar weight loads strided by `ic*9`; it measured **+213% / +329%**. Rewritten with 8 explicit
float4 accumulators (no array indexing → nothing spills to scratch), weights repacked to
`[ocb][ic][kh][kw][4oc]` so a tap is one aligned float4 load, and input read via 4 aligned float4
loads instead of 10 scalars, the same algorithm measures **−6.7% / +26.7%**. **A 3.4x swing from
implementation quality alone** — the third time in this investigation that a layout/blocking idea
was nearly falsified by its own first implementation (cf. §H.21's 2.8x). *A theory-only rejection is
worth nothing, and so is a one-implementation rejection.*

**Conv kernel only, cooled + interleaved, 3 reps:**
| shape | NC4HW4 default | NC4HW4 c4h4w2+HARD (best) | **NCHW** | NCHW vs best |
|---|---|---|---|---|
| 32→32@72×96 | 120.0 | **97.0** | 111.0 | +14% |
| 34→34@72×96 | 167.0 | **114.0** | 119.7 | +5% |
| **48→48@72×96** | 264.3 | 210.0 | **199.5** | **−5%** |
| 48→48@36×48 | 102.0 | **102.0** | 135.0 | +32% |
| 96→96@18×24 | 50.0 | **50.0** | 50.9 | +2% |

**NC4HW4's channel-padding cliff is real and NCHW does not have it.** Sweeping C at 72×96:
| C | NC4HW4 pads to | NC4HW4 | NCHW |
|---|---|---|---|
| 16 | 16 | 45.5 | 52.9 |
| **18** | **20** | **64.5** | **61.5** |
| 20 | 20 | 64.0 | 70.0 |
| 32 | 32 | 118.8 | 113.5 |
| **34** | **36** | **166.8** | **121.9** |
| 36 | 36 | 169.5 | 127.0 |
| 48 | 48 | 262.6 | 197.5 |
| **50** | **52** | **327.6** | **188.2** |
NC4HW4 costs *exactly* the same at C=18 and C=20, and at C=34 and C=36 — it genuinely computes the
padded channel count. NCHW is monotonic in C, as it should be. At C=50 NC4HW4 pays **+25% over
C=48 for 4% more channels**; NCHW pays nothing.

**Two real bugs were found and fixed while doing this**, both of which would have corrupted the
answer: (1) the v2 kernel wrote a full float4 of output channels unconditionally, overrunning the
scratch buffer whenever C%4≠0 — i.e. precisely at C=18/34/50, the interesting cases (fixed by
padding the NCHW scratch planes to a multiple of 4 and zero-filling); (2) trap 3 struck again — a
`"` inside a kernel comment broke the generated C++.

**Verdict for the application: NO.** NCHW's advantage grows with channel count and spatial size and
crosses over around C≈48 at 72×96. The model's actual stride-1 cores are `32→32@72×96` (NCHW +14%),
`48→48@36×48` (+32%) and `96→96@18×24` (tie) — **NCHW loses or ties on every one of them.** And the
conv-only figure flatters NCHW: with conversions counted it is +30% to +46%. A fully-NCHW pipeline
would amortise the conversions to ~2 per graph instead of 2 per conv (≈7 us/conv here), which is
enough to keep the C=48@72×96 win but not enough to rescue any shape in this model.

**Verdict as a finding: YES, and it should be kept.** The padding cliff is a genuine structural tax
that MNN pays on every non-multiple-of-4 channel count, and it is large (up to +25%). It reinforces
the existing shape rule (§B: keep channels a multiple of 16) with a mechanism, and it means a model
with C=18/34/50-style layers is paying for the layout, not the arithmetic.

### §H.37 — Re-audit of the falsified strategies under NCHW
Asked whether NC4HW4 was a hidden constraint that invalidated the earlier negative results. Working
from §H.36's measurements (NCHW conv ≈ parity, winning only at C≥48 with large spatial) plus
§H.35's finding that memory traffic costs ~0 at every size:

| strategy | why it failed in NC4HW4 | does NCHW change the mechanism? | status |
|---|---|---|---|
| **Cross-layer fusion** (§H.1, §H.25) | inter-layer traffic <1 us/conv, so there is nothing to save | **No.** §H.35 re-measured this over a 64x size range up to a 6.9 MB intermediate: still ~0. The prize is absent in *any* layout. | **stays falsified — closed** |
| **LDS input tiling** (§H.3, §H.30) | barriers + no traffic to save; L2 already serves the reuse | **Partly.** An NCHW tile stages one channel plane, so ~4x more spatial fits in 64 KB LDS than an NC4HW4 float4 tile. The capacity argument genuinely changes. But the *reason* it lost does not: barrier cost is layout-independent, and §H.35 says the traffic it removes is free. | **stays falsified; the capacity variant is the one untested corner** |
| **im2col + GEMM** (§H.13) | GEMM half alone (139 us) already beat by direct conv (119 us) at C=32 | **Possibly, at high C.** NCHW im2col is the natural contiguous form, and NCHW's efficiency *improves* with channel count (§H.36: sublinear scaling 32→48). At C=32 direct NCHW is 111-114 us, so GEMM still has to beat that and previously could not. | **untested at C≥48 — the strongest remaining lead** |
| **Split-K** (§H.31) | extra write+re-read pass costs what the occupancy gain buys | **No.** Purely about thread count and an extra pass; the layout of either does not enter. | **stays falsified** |
| **Winograd selection** (§H.28/§H.29) | transform cost vs channel count | **Unknown.** MNN's Winograd is NC4HW4-specific; an NCHW Winograd is a different implementation, not a re-run. | **out of scope, untested** |

**Conclusion.** NCHW is not a hidden unlock for the falsified strategies. Three of the five fail for
reasons that are provably layout-independent — above all because **memory traffic on this device
costs approximately nothing**, which is now measured directly (§H.35) rather than inferred. That
single fact is what kills fusion, LDS and LLC placement, and it does so in either layout.

**The one lead worth a future session** is im2col+GEMM under NCHW at C≥48, where NCHW's
channel-scaling advantage is largest and where the GEMM form is natural rather than contorted. It
should be judged against `conv_2d_c4h4w2 + MNN_CONV_HARD`, not against MNN's default, and measured
with `conv_all_us`.

### §H.38 — im2col + GEMM under NCHW at C≥48: FALSIFIED, and the reasoning that revived it was wrong
§H.37 flagged this as the strongest remaining lead, on the grounds that (a) NCHW makes the GEMM
shapes natural, (b) NCHW's efficiency improves with channel count, and (c) §H.35 had just shown
memory traffic costs ~0, which was im2col's original disqualifier. Built it: `im2col_nchw` (reads
the NC4HW4 input directly and writes col[K][M]) + `gemm_nchw_c4m8` (4 oc × 8 pixels = 8 float4
accumulators, the same register class as the best direct kernel; weights pre-packed `[ocb][k][4oc]`).
Env `MNN_CONV_IMGEMM=1`. Correct: cosine 0.999968–0.999992.

**A measurement trap first: at C≥64, MNN takes the Winograd path**, which is selected *before*
`ConvBufExecution` is constructed, so no `MNN_CONV_*` flag reaches the code and every arm silently
measures the identical Winograd kernels. The first C=64/96 run showed all four arms within 2% of
each other — that was the tell. Re-run with `MNN_NO_WINOGRAD=1` on every arm.

**All conv-related kernel time, per conv, cooled + interleaved:**
| shape | Winograd (real default) | NC4HW4 direct | **c4h4w2+HARD (best)** | NCHW direct | **im2col+GEMM** |
|---|---|---|---|---|---|
| 48→48@72×96 | n/a (not selected) | 264.3 | **210.0** | 230.5 | **385.2 (+83%)** |
| 64→64@72×96 | 281.8 | 403.5 | **318.1** | 383.5 | **597.4 (+88%)** |
| 96→96@72×96 | 435.0 | 877.6 | **679.0** | 871.0 | **1491.0 (+120%)** |

**It gets worse with C, not better — the exact opposite of the predicted trend.** And the cost is in
the GEMM, not the im2col:
| shape | im2col | GEMM | repack |
|---|---|---|---|
| C=48 | 116.0 | **279.0** | 27.0 |
| C=64 | 153.0 | **413.9** | 29.0 |
| C=96 | 231.0 | **1209.7** | 47.0 |
At C=96 the GEMM alone runs at ~0.95 TFLOP/s against the 3.03 TF practical ceiling.

**Why the §H.35 argument was wrong.** §H.35 showed a *one-time* intermediate write+read round-trip
is free, because it hides behind compute. im2col is not that: the column matrix is **streamed
through the ALUs in the inner loop**, so the 9x data duplication becomes 9x the bytes-per-MAC in the
hot path, not a one-off round trip. Free bulk traffic and free inner-loop bandwidth are different
claims, and only the first was measured. That conflation is what made this look promising.

**The structural ceiling.** The direct conv already achieves the 8-float4 register tile that is the
measured optimum here (§H.22 — 16 accumulators regress +234%, so the tile cannot grow). A GEMM
cannot get more arithmetic intensity per thread than that on this device, and it must additionally
stream a 9x-inflated operand and pay a separate im2col pass. Its best case is therefore *direct
conv minus the im2col*, i.e. it cannot win.

**Honest scope of the falsification:** this GEMM is naive — no LDS tiling of the col matrix, so each
col element is re-read once per oc-block (12–24 times). A properly LDS-tiled GEMM would cut that,
and is the one untested variant. But it must overcome an 83–120% gap while capped at the same
8-accumulator tile, and §H.30 already showed LDS staging plus barriers does not beat the direct path
here. **Recorded as falsified with the residual variant named.**

**Practical note that came out of it:** at C≥64 with large spatial, MNN's Winograd is the real
default and beats every direct kernel tried (281.8 vs 318.1 at C=64; 435.0 vs 679.0 at C=96, −36%).
Combined with §H.29 this completes the picture: **Winograd is right at high C, wrong at low C, and
MNN's selection boundary is merely mis-calibrated in between.**

---

> **Benchmark convention from §H.39 onward — NCHW layout conversions are EXCLUDED.**
> The NCHW path needs an NC4HW4→NCHW conversion before the conv block and an NCHW→NC4HW4 after it.
> In a real deployment those would be folded into custom ops at the *boundaries of a conv block*,
> paid once per block rather than once per conv, so charging them to every conv would not reflect
> any shippable design. Every NCHW number below is therefore **conv-kernel time only**. The
> conversion cost is still measured and printed alongside, never hidden — and it is not small
> (e.g. 271 us/inference on head18), so any decision to adopt NCHW must budget for it separately
> and confirm the fold-into-custom-ops assumption holds.

### §H.39 — NCHW on the STRIDE-2 HEADS: falsified on the model's own shapes
The strongest remaining NCHW hypothesis. The model's heads are stride-2 with **unaligned input
channels** (18 and 34), the regime where NC4HW4's padding tax is worst — and the NCHW path had been
stride-1 only, so it had never been tried there. The stride-1 sweep gave real cause for optimism:
NCHW was **−4.7% at C=18** and **−26.9% at C=34**.

Built `conv_2d_nchw_s2_c4w8` (4 oc × 8 output pixels = 8 float4 accumulators, the same register
class as everything else; 8 outputs span a 20-wide input window covered by 5 aligned float4 loads).
Correct on every head (cosine 0.999997–1.000000), and the kernel dump confirms it actually engaged.

**Conv-kernel time only, per conv, cooled + interleaved:**
| head | conv | NC4HW4 | NCHW | NCHW+HARD | best vs NC4HW4 |
|---|---|---|---|---|---|
| head18 | 18→16@288×384 s2 | **149.0** | 191.0 | 218.0 | +28.2% |
| head18 | 16→32@144×192 s2 | **66.0** | 80.0 | 79.0 | +19.7% |
| head34 | 34→32@144×192 s2 | **139.0** | 148.0 | 148.0 | +6.5% |
| head34 | 32→48@72×96 s2 | **80.0** | 114.0 | 113.0 | +41.2% |
| head64 | 64→64@72×96 s2 | **160.0** | 226.0 | 225.0 | +40.6% |
| head64 | 64→96@36×48 s2 | **103.0** | 202.0 | 206.0 | +96.1% |

**NCHW loses on all six, by +6.5% to +96.1%.** The channel-padding advantage does not survive the
move to stride 2, and this is the cleanest possible test of it: `34→32@144×192 s2` has the same
unaligned C=34 that won by 26.9% at stride 1, and here it loses by 6.5%.

**Why stride 2 reverses it.** At stride 2 the input plane is 4x the output, so the conv is dominated
by input-side traffic — and that is exactly where NC4HW4 wins structurally: one float4 load serves
**4 channels** of the reduction, in both stride regimes. NCHW reads one channel plane at a time, and
at stride 2 it must pull a 20-wide input window to produce 8 outputs (5 aligned float4 loads) versus
a 12-wide window at stride 1 (4 loads) — 25% more load instructions per output, per channel. The
padding saving is only 6–11% of the arithmetic; it cannot pay for that.

**This closes the last model-relevant NCHW hypothesis.** The earlier verdict ("NCHW loses on this
model") was flagged as under-tested because it rested only on the three aligned stride-1 cores. It
has now been tested on the six unaligned stride-2 convs that were the strongest counter-case, and
it holds — with a mechanism, not just a number.

### §H.40 — Shape hardcoding does NOT transfer to the NCHW kernels
The −18.7% constant-folding win (§H.23) applied to the NCHW path via a new `NC_*` macro set
(`ncPut`, same contract as `hcPut`: the runtime arg name by default, a literal under
`MNN_CONV_HARD=1`, with `#ifndef` guards so it cannot be shadowed — the §H.34 bug).

**Verified to actually engage**, because §H.34 taught that a build option which silently fails to
reach the kernel is indistinguishable from a lever that does not work. `MNN_CONV_NCHW_DEBUG=1`
prints the emitted options:
```
hard=0  -DNC_IC=in_channels -DNC_INH=in_h  -DNC_INW=in_w  ...
hard=1  -DNC_IC=34          -DNC_INH=144   -DNC_INW=192   ...
```

| shape | NC4HW4 default | NCHW | NCHW+HARD | hardcoding effect |
|---|---|---|---|---|
| 32→32@72×96 | 118.7 | 115.0 | 116.0 | **+0.9%** (none) |
| 48→48@36×48 | 101.0 | 133.6 | 127.0 | −4.9% |
| 96→96@18×24 | 50.0 | 50.0 | 50.0 | 0% |
| all six stride-2 heads | — | — | — | ±0–14%, no consistent sign |

**At most −5%, never turning a loss into a win** — against −18.7% on the NC4HW4 side. The mechanism
is straightforward once measured: constant folding paid on the NC4HW4 kernels because they carry a
*runtime-dependent halo bounds check per tap* plus a runtime channel-block loop, and folding
collapses both. The NCHW kernels were written with an interior/edge split and a simple channel loop,
so there is far less left to fold. **Constant folding is worth what the kernel's runtime branching
costs — it is not a general multiplier.** Kept, env-gated, default off.

### §H.41 — Reproducibility: every Session-B strategy is now in the standard suite
The whole point of the bundle is that a strategy can be re-decided on new hardware, so nothing was
left as a one-off scratch script and **no existing test was removed or replaced**. Added to
`conv_bench/bundle_run_report.py`:
- **§14 — LDS-at-constant-blocking + split-K.** `MNN_CONV_LDS=w2` against its own twin
  `conv_2d_c4h1w2`, and `MNN_CONV_SPLITK=2|4`. The w2 tile is chosen per shape from the
  `out_w % (2*TILE_W) == 0` constraint, and reported as `n/a (tile)` when no tile fits.
- **§15 — NCHW layout**, cores and heads, `NCHW` and `NCHW+HARD`, with the layout-conversion cost
  in its own column and the exclusion rationale stated inline.
- **§16 — im2col + GEMM**, with `MNN_NO_WINOGRAD=1` forced on every arm.

**Two metric fixes went in with them:**
1. `conv_all_us()` and `conv_kernel_only_us()` added, with the §H.27 trap documented at the call
   site: MNN's `conv time` counter omits the Winograd transforms.
2. **Existing §9 (Winograd vs direct) was measured with the buggy metric** and is now on
   `conv_all_us`. It also gained a `Winograd FORCED` column, so the mis-calibrated selection gate
   (§H.28/§H.29) is visible on any device the bundle is run against, not just this one.

Manifest additions in `make_bundle.py`: `lds_modes`, `splitk_factors`, `impl_switching_envs`,
`NCHW_MODE`, `IMGEMM_MODE`. Bundle regenerated and the full report re-run end to end to confirm
nothing regressed.

### §H.42 — Session-B closing summary

| # | strategy | outcome | best number |
|---|---|---|---|
| 1 | **Force Winograd ON** | **WIN, shape-dependent** | **−13% on 48→48@36×48, −25% on 32→32@36×48**; loses at 72×96+ |
| — | *`conv time` metric bug* | **found; invalidated prior comparisons** | turned +15% into an apparent −45% |
| — | *`c4h4w2+HARD` regression* | **found + fixed** (macro-name shadowing) | **−18.7% restored** |
| 2 | Split-K over Cin | mechanism confirmed, too small | −11.7% at constant blocking; **−2% vs default** |
| 3 | c4h1w2 + LDS | conclusion right, mechanism wrong | LDS −28.9% vs its twin; still +26% vs default |
| 4 | Register minimisation | not built | accumulator axis closed; weight axis open |
| 5 | ION / dma_buf | prize sized, path blocked | **~32% of wall** ceiling; heaps unreachable as uid `shell` |
| 6 | LLC for intermediates | **no prize** | round-trip free even at **6.9 MB** |
| 7 | NCHW (stride 1) | loses on this model | wins only C≥48 @72×96 |
| 8 | NCHW (stride 2, heads) | **falsified on the model's own shapes** | +6.5% … +96.1% |
| 9 | NCHW shape hardcoding | does not transfer | ≤5%, vs −18.7% on NC4HW4 |
| 10 | im2col + GEMM (NCHW) | falsified, worsens with C | +83% … +120% |

**The one shippable result remains §H.28:** let `48→48@36×48` use Winograd. No new kernel — MNN
already has the code and merely declines to select it. Worth −10.3% on the whole Block2 graph.

**The most valuable results are the two bugs**, both of which were silently corrupting conclusions:
the `conv time` metric, and the shadowed `HC_*` macro. Both are now impossible to hit again — the
metric by documentation at the call site plus a safer default in the suite, the macro by `#ifndef`
guards plus an observable `MNN_CONV_NCHW_DEBUG` dump of what actually reached the compiler.

**Method note worth carrying forward.** Three separate strategies in this session were nearly
falsified by their own first implementation — the 2-D tile (2.8x), the NCHW conv (3.4x), and, in the
other direction, forced Winograd (a metric artefact of 3x). A single implementation measuring badly
is not evidence a strategy is bad. Every negative here that matters was re-checked with the obvious
traps removed before being written down.

### §H.43 — Suite validation + a throttling trap in the validation itself
Full `run_report.py` executed end to end after the additions: **18 sections, 0 exceptions**, and the
new sections write their data into the `.json` (`session_b`, `nchw`, `imgemm`, plus `winograd` now
carrying `forced_wino_us`).

**Trap worth recording, because it nearly produced a false "you broke it".** The final default-path
regression check, run immediately after hours of continuous benchmarking, read
`core_32 = 198.5 us` against the 119.0 baseline — a 1.67x apparent regression with every new flag
off. It was pure thermal throttle (the documented ~2.75x, gotcha §9). After a 240 s cooldown the
same binary reads **120.0 / 101.0 / total 1402**, matching pre-session 119.0 / 101.0 / 1400. A
regression check is only meaningful on a cooled device, exactly like a benchmark.

**Related, and the reason the section notes were rewritten:** the validation run itself measured
`MNN default = 189 us` in §14 and, at that inflated baseline, `c4h1w2+LDS` came out FASTER than the
default — the opposite of the cooled §H.30 result. Interleaving protects the *comparison* between
arms but does not stop different kernels from throttling by different amounts. The prose in §14/§15/
§16 originally asserted the reference device's conclusions and would have sat directly above a table
contradicting them. It now describes the **mechanism** and what to look for, and points at FINDINGS
for the reference numbers — which is what the suite is for: re-deciding on new hardware, not
restating this device's answer.

### §H.44 — Fused 2-conv kernel in NCHW: BUILT and FALSIFIED (worse than the NC4HW4 fusion)
Fusion had been closed by argument for NCHW (§H.37) rather than by measurement. Built:
`conv_2d_nchw_fused2` (env `MNN_CONV_NCHW_FUSE2=1`, tile via `MNN_FUSE_TILE`), the direct NCHW
counterpart of `conv_2d_3x3s1_fused2` — one workgroup owns a T×T output tile, stages conv1's
(T+2)²×C output in `__local`, then computes conv2 from it. Same weights applied twice, exactly like
the NC4HW4 original, so it fits in one conv Execution and validates against the shipped numpy
conv(conv(x)) reference.

**Correct: cosine 0.999994** vs the reference (the NC4HW4 fused2 measured 0.993444 in the same run,
reproducing its §H.25 value to the digit — so the harness is sound and the new kernel is actually
*more* accurate than the original).

**Measured (conv-kernel time, all arms with `MNN_NO_WINOGRAD=1`, T=6 ⇒ conv1 recompute 1.78×):**
| shape | NC4HW4 1 conv | NC4HW4 fused2 | NCHW 1 conv | **NCHW fused2** | NCHW fused vs 2× single |
|---|---|---|---|---|---|
| 32→32@72×96 | 120.0 | 1572.0 | 100.0 | 1727.0 | **+764%** |
| 48→48@36×48 | 98.0 | 1245.0 | 114.0 | 2992.0 | **+1212%** |
| 96→96@18×24 | 141.0 | 4768.0 | 198.7 | 11531.0 | **+2802%** |

**NCHW makes fusion WORSE, not better** — 1.1× to 2.4× worse than the already-falsified NC4HW4
fused kernel, and the gap widens with channel count.

**Mechanism.** Phase 2 reads the staged intermediate one scalar at a time: `C×9` LDS reads per
4-channel output block, i.e. `9·C²/4` scalar LDS transactions per thread. The NC4HW4 fused kernel
stores the intermediate as float4, so a single LDS read serves 4 channels — **4× fewer
transactions**. This is the same structural property that makes NCHW lose the single-conv case
(§H.36/§H.39), and fusion amplifies it because the intermediate is read C times rather than once.
Against that, NCHW relaxes nothing: the intermediate needs `(T+2)²·C` values in either layout, so
the capacity bound of §H.25 is untouched.

⇒ **Fusion is now closed by measurement in BOTH layouts**, not by argument. §H.37's conclusion was
right, and the earlier note there suggesting NCHW's 4×-larger LDS tile capacity might help fusion
was wrong: that applies to a single-conv halo tile (one channel plane), whereas fusion needs every
channel of the intermediate.

**Fixture trap (cost one full measurement round, worth recording).** The first run reported cosine
**0.010** for the new kernel *and* for the known-good NC4HW4 fused2. That was the test, not the
kernels: the conv² reference is computed for one specific input, which must be staged as
`tdir/input.txt`; without it MNN generates its own input and every kernel reads as uncorrelated.
The tell was that a previously-validated kernel failed identically — **when a known-good control
fails the same way as the thing under test, suspect the fixture first.** Same run also showed
"fused == 1 conv" at 96 channels, which was the §H.38 Winograd-shadowing trap again; all arms now
carry `MNN_NO_WINOGRAD=1`.

### §H.45 — Harness trap: the bundle ships its own binaries and OVERWRITES the ones you just built
Cost a full round of confusing measurements, and is the same shape as the stale-`input.json` trap.

`run_report.py` stages the libraries from the bundle's own `bin/` directory. `make_bundle.py` copies
those from `build_android_profile` **at bundle-creation time**. So a bundle built before a kernel
change carries binaries *without* that change, and running the report silently replaces whatever
`session_measure.py --push` put on the device.

**How it presented.** `MNN_CONV_NCHW_FUSE2=1` measured 1727 us in the dedicated run (fused kernel
engaged), then 125.0 us afterwards — exactly the plain-conv time, i.e. not engaged. Same binary on
disk, same model, same env. In between, the report validation had run and rolled the device back to
the pre-fuse2 libs. The report's own §10 then printed "NCHW fused2 FASTER −61%", which was the plain
conv being compared against 2× itself.

**Diagnosis that worked.** Kernel-name inspection is USELESS here: every direct-path kernel is
reported by the profiler as `ConvBuf2D-ori-...` regardless of which kernel object runs, so the name
cannot distinguish fused/NCHW/default. What settled it was instrumenting the gate itself
(`MNN_CONV_NCHW_DEBUG=1` now prints `[FUSE2GATE] ... -> 1`) and confirming the branch was entered.

**Rules this establishes:**
1. **Regenerate the bundle after any kernel or host change** before running `run_report.py`, or it
   will measure stale binaries. `make_bundle.py` then `run_report.py`, in that order, always.
2. Never infer "which kernel ran" from the profiler name on the direct path — gate-level
   instrumentation or a timing signature is the only reliable evidence.
3. A flag that reads *exactly* the same as the no-flag arm is the signature of a flag that did not
   engage, not of a lever that does nothing. Treat exact equality as suspicious.

**What this does NOT invalidate:** §H.44's fused-NCHW numbers, which were taken immediately after a
fresh `--push` and reproduce exactly on the current build (NCHW 1 conv 100.0, NCHW fused2 1728.0 vs
1727.0 originally). Only the report-section validation was affected.

**Re-validated after regenerating the bundle** (so `bin/` carries the fused kernel): §10 now reports
`32->32@72x96` as **SLOWER +512%** (NC4HW4 fused) and **SLOWER +736%** (NCHW fused), against the
dedicated cooled measurements of +555% and +764%. The absolute values differ because the validation
runs with a short cooldown and is thermally inflated; the ratios agree, and the verdicts are now
correct where the stale-libs run had printed "FASTER -61%". Report completes with 0 exceptions.

**Harness leak found and fixed while cleaning up.** After the report exits, a
`ModuleBasic.out core_32.mnn ... 20000 ... clk.bin` process was still running on the device — the
clock sampler that `sample_clock()` spawns and kills by PID. The PID kill misses (observed), leaving
a 20000-loop job pinned to the GPU **after the report has finished**, which then throttles or
contends with whatever is measured next and would look like a mysterious regression (cf. the 198 vs
119 scare in §H.43). `sample_clock()` now ends with a name-based `pkill -9 -f ModuleBasic.out`,
which is safe because it is only ever called between sections with nothing else of ours on device.

### §H.46 — IMPLICIT GEMM (the standing prompt), built and measured in BOTH layouts: falsified
`IMPLICIT_GEMM_SESSION_PROMPT.md` had never been executed — no implicit-GEMM kernel existed. Built
`conv_2d_implicit_gemm`: an LDS-tiled GEMM that gathers the im2col columns **on the fly** out of the
input while staging them into `__local`, so the expanded matrix is never materialised. This also
closes the "LDS-tiled GEMM" variant §H.38 named as its one untested residual.

Tile: 64 pixels × 32 output channels per workgroup, 64 threads, 8 pixels × 4 channels per thread =
**8 float4 accumulators**, the register class §H.22 measured as optimal. K-step is 9 (one input
channel's taps), which keeps the gather indexing cheap. **Layout is a compile-time switch and the
only difference between the two variants** — same tiling, same math, same registers — so the
NC4HW4-vs-NCHW comparison is exact. `MNN_CONV_IGEMM=1` (NC4HW4, no conversion kernels at all) /
`=nchw`. Correct in both: cosine 0.999949–0.999989.

**The first implementation was again a false negative — the fourth time this session.** It decomposed
`m -> (b,y,x)` inside the staging loop: two *runtime* integer divisions per element, 9 per thread per
channel step, for a value that is invariant in both loops. Hoisting it (two divisions for the whole
kernel) was worth **1.8×**:
| core | first cut | after hoisting | MNN default |
|---|---|---|---|
| 32→32@72×96 | 444 | **243** | 119 |
| 48→48@36×48 | 596 | **319** | 101 |
| 96→96@18×24 | 1021 | **584** | 146 |

**Final, cooled + interleaved, all arms `MNN_NO_WINOGRAD=1`:**
| core | MNN default | c4h4w2+HARD | iGEMM NC4HW4 | iGEMM NCHW |
|---|---|---|---|---|
| 32→32@72×96 | 119.0 | **97.0 (−18.5%)** | 243.0 (+104%) | 236.0 (+98%) |
| 48→48@36×48 | **101.0** | 101.0 | 319.0 (+216%) | 312.8 (+210%) |
| 96→96@18×24 | **146.0** | 170.0 | 584.0 (+300%) | 574.0 (+293%) |

**Success criterion (>5% under the default on 48→48@36×48): NOT MET, by 3.1×.**

**Why, structurally.** Per k the thread reads 8 scalar A values and one float4 B value from LDS and
issues 8 `mad` instructions — **9 LDS accesses per 8 ALU instructions**. The kernel is LDS-throughput
bound, not ALU bound, so the tiling that makes a GEMM fast on a big matrix cannot pay here: the fp16
register budget (8 float4 before the §H.22 cliff) caps the tile at ~8×4, and that tile is too small
to amortise the LDS traffic. The direct `conv_2d_c*` kernels avoid this entirely by holding inputs in
*registers* across the 9 taps and never touching LDS.

**On the premise.** The prompt's case rested on §H.13's GEMM-reduce beating the direct conv by 21% on
the 48-core *with the data pre-arranged*. That advantage does not survive folding the gather back in
— which is exactly what §H.14 argued on reasoning alone and is now measured.

**Layout verdict (the part that was asked for explicitly): it barely matters.** NCHW is 2–3% ahead of
NC4HW4 across all three cores (236/243, 313/319, 574/584) — the contiguous gather is slightly cheaper
than NC4HW4's stride-4 scalar read, but both funnel through the same LDS bottleneck, which dominates.
**For implicit GEMM the layout is not the lever.**

**Untested residual, named honestly:** a wider output-channel tile (TM=4, TN=8 instead of 8×4) would
cut LDS accesses per mad from ~1.1 to ~0.75 at the same 8-float4 register budget. That is ~1.5× on
the binding resource against a 3.1× gap, so it is very unlikely to flip the result, but it is the one
variant not built.
