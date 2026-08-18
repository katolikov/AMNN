# Conv strategy summary — what we have, and why each works or doesn't

Every item below is **measured on-device** (Xclipse 960 / Exynos 2600, OpenCL fp16), is env-gated
and **default OFF**, and is registered in `conv_bench/` so it re-decides itself on new hardware.
Full detail in `FINDINGS.md`; section numbers are the pointers.

Sections 1–4 were measured in **buffer** mode (`gpuMode=68`) unless stated. **Image/texture mode
(`gpuMode=132`) is a separate, fully-implemented backend**, audited in §H.51 — see the image row in
section 1 and the per-shape table there before assuming a buffer-mode result transfers.

Reference shapes: `32→32@72×96`, `48→48@36×48`, `96→96@18×24` (stride-1 cores) and six stride-2
head convs. Baselines: MNN's autotuned default, and `conv_2d_c4h4w2 + MNN_CONV_HARD` (−18.7%).

---

## 1. WINS — ship these

| strategy | flag | result | why it works |
|---|---|---|---|
| **PReLU fused into conv** | converter `MNN_FUSE_CONV_PRELU=1` | **−8/9% per block** | removes a whole extra kernel + its full output round-trip. Already shipped. §E |
| **Winograd gate fixed** | on by default (`MNN_WINOGRAD_GATE_OLD=1` reverts) | **−33.7% on 48→48@36×48**, **−16.0% on Block2 whole-graph, no flags** | the stock clause `in_w < out_c` had the right idea but was calibrated far too tightly. Re-derived after the GEMM fix as **`in_w <= 2*out_c`**: 0 missed wins, 0 admitted losses over 14 shapes. §H.29's own 1.5 proposal is now stale — `48→48@72×96` flipped +11.9% → −19.0%. §H.53 |
| **Shape hardcoding** | `MNN_CONV_HARD=1` | **−18.7% on 32→32@72×96** | every shape value is a runtime arg by default, so nothing constant-folds. Baking them collapses the per-tap halo bounds checks and the channel loop. Kernel- and shape-dependent: it *hurts* some pairs (+16%, +33%). §H.23/§H.34 |
| **2-D register tile** | `conv_2d_c4h4w2` | −7.6% alone, −18.7% with HARD | at equal accumulator count, spreading the tile over **two spatial axes** reuses each loaded weight better than 1-D. §H.21/§H.22 |
| **Tall-skinny GEMM tile fix** | on by default (`MNN_GEMM_NO_TALLSKINNY=1` reverts) | **−23.9% on 48→48@36×48**; **−19.5% on 48→48@18×24 with no flags**; Block2 forced-wino −4.0% → **−16.5%** | `getGemmParams` requires `NWG` to divide N exactly and never pairs a small NWG with a large MWG, so any padded out-channel count that is an **odd multiple of 16** (48, 80, 112) collapses to a 16×16 tile with **16 threads/group**. Adding `NWG=48` + large-MWG/`NWG=16` candidates fixes it. Buffer now MATCHES image on the 48-core (67.0 vs 67.8). §H.52 |
| **IMAGE (texture) memory mode** | runtime `gpuMode=132` | **−24.5% on Block2 whole-graph** (−44.6% conv-only); −18…−33% on 8 of 14 shapes | two independent effects: image's Winograd gemm beats buffer's general `XgemmBatched` on **tall-skinny** GEMM (small out-channels), and image's **direct** conv beats buffer's at C≤24. **No new kernel — a separate backend that already exists.** Wins C≤48, loses C≥64. §H.51 |

**Combined recommendation:** PReLU fusion everywhere + **image mode on Block2** ≈ **−24.5% on Block2**
whole-graph, with no new kernel code. Block1 stays on buffer (image regresses it 9.5%). `gpuMode` is
per-Interpreter, so a split pipeline picks per model.

---

## 2. NEAR-MISSES — right mechanism, too small

| strategy | flag | result | why it falls short |
|---|---|---|---|
| Split-K over Cin | `MNN_CONV_SPLITK=n` | −11.7% at constant blocking, **−2% vs default** | genuinely fixes occupancy starvation, but the extra partial-sum write+re-read eats the gain. §H.31 |
| LDS input staging | `MNN_CONV_LDS=w2` | **−28.9% vs its own twin**, still +26% vs default | LDS *does* capture reuse (the old "captures nothing" claim was wrong on mechanism), but it never beats a kernel that keeps inputs in registers. §H.30 |
| Constant-memory weights | `MNN_CONV_CONSTW=1` | 0% (NC4HW4), ≤2% (NCHW) | ANGLE reports a 1 GB constant limit ⇒ `__constant` is the same storage as `__global`. Real win exists on Adreno's *physical* constant memory; this stack has none. §H.49 |

---

## 3. FALSIFIED — and the reason is structural

| strategy | flag | result | the mechanism that kills it |
|---|---|---|---|
| **NCHW layout** (s1 + s2) | `MNN_CONV_NCHW=1` | +14% … +96% | NC4HW4's float4 load serves **4 channels of the reduction at once**; NCHW reads one channel plane at a time. Worse at stride 2, where the input plane is 4× the output. Its only advantage — no channel padding — is 6–11%, far too small to pay. §H.36/§H.39 |
| **im2col + GEMM** | `MNN_CONV_IMGEMM=1` | +83% … +120%, worsens with C | im2col inflates **inner-loop bytes-per-MAC 9×**. Bulk traffic is free here; *streamed* traffic is not. §H.38 |
| **Implicit GEMM** (both layouts) | `MNN_CONV_IGEMM=1\|nchw` | +98% … +300% | 9 LDS accesses per 8 ALU instructions ⇒ LDS-throughput bound. The fp16 register budget caps the tile too small to amortise it. Layout barely matters (2–3%). §H.46 |
| **Layer fusion** (both layouts) | `MNN_CONV_FUSED2`, `MNN_CONV_NCHW_FUSE2` | +512% … +2802% | a T×T output tile needs conv1 over a (T+2)² halo ⇒ **1.78× recompute**, to buy back an intermediate round-trip that costs ~0. §H.25/§H.44 |
| **LLC / cache placement** | — | no prize | the intermediate round-trip is **free even at 6.9 MB**. You cannot speed up zero. Also no such extension exists on this stack. §H.35 |
| 16-accumulator tiles | `conv_2d_c4h4w4`, `c8h8w1` | +234%, +49% | past the register cliff; occupancy collapses. §H.8/§H.22 |
| space2depth on heads | — | +12…17% | Cin is already past the crossover. §H.6 |
| Subgroup halo exchange | — | dead | `sub_group_shuffle` does not compile through clspv. §H.11 |

**The single fact behind half of this table:** memory traffic on this device costs approximately
nothing. That is measured (§H.35), and it is why fusion, LDS, and LLC placement all return ~0 — in
either layout.

---

## 4. NOT A STRATEGY — but the highest-value findings

| finding | why it matters |
|---|---|
| **`conv time` excludes Winograd transforms** | it was the primary metric for most of the investigation. Turned a +15% regression into an apparent −45% win, and under-reports every Winograd shape. Some older recorded numbers — including the ~3.03 TFLOP/s ceiling — need re-deriving. §H.27 |
| **A renamed macro silently disabled `MNN_CONV_HARD`** | the −18.7% lever was a no-op for four kernels and nobody noticed; it fails *silently and in the fast direction*. §H.34 |
| **IMAGE mode was never tested at all** | a wide-tensor crash caused buffer mode to be mandated globally, and ~20 strategies were then evaluated inside that assumption. It is worth **−24.5% on Block2**. The lesson is the over-generalisation, not the number: one real failure closed off an entire backend for the rest of the investigation. §H.47/§H.51 |
| **Four ideas nearly died from bad first implementations** | swings of 1.8×–3.4× from implementation quality alone. A single bad measurement is not evidence an idea is bad. |

---

## 5. Open / not built

| idea | status |
|---|---|
| ~~Image (texture) mode audit~~ | **DONE — §H.51.** PReLU fusion ported to the image backend (cost measured: **zero**); win re-measured against the Winograd-gate confound and swept over 14 shapes. Wins C≤48, loses C≥64. |
| ~~Fix buffer's `XgemmBatched` on tall-skinny GEMM~~ | **DONE — §H.52.** Root cause was a divisibility rule in the tile selector, not the kernel. Fixed, default ON, measured safe. |
| ~~Winograd transform kernels~~ | **ATTEMPTED, FALSIFIED — §H.53.** 2-tiles/thread: **+29% slower** (56 live FLOAT4 = register cliff). Bounds-check elimination: **no effect**. They are **34% fixed launch overhead** (`src_us = 7.3 + 2.711/1000 items`) and latency-bound otherwise — at their practical floor for this structure. Further gains need dispatch batching (§H.32) or an intermediate-layout change. |
| **Winograd F(4,3)** | scoped, not built. MNN ships only the `2_3_1` transforms, so it is a from-scratch 6×6 transform pair. §F's rejection is a *cost model*, computed for a different conv, on assumptions §H.35 has since undermined. §H.50 |
| **INT8 + `cl_khr_integer_dot_product`** | extension is exposed and unused; 1.67–2.45× reported elsewhere. **Only idea on the list that trades accuracy** — needs a real output gate, not a cosine check. §H.48 |
| Zero-copy input (dma_buf) | ~32% of wall clock, but the heaps are unreachable as uid `shell`; needs a real app context. §H.33 |
| Dispatch batching | attacks the ~915 µs fixed submission cost, not kernel time. §H.32 |
