# Image (texture) mode — audit + plan (pre-implementation)

Status: **COMPLETE — see `FINDINGS.md` §H.51 for the final results.** This file is the pre-work
audit; the audit table (§3) and build analysis (§4) stand. **§1's conclusion below was superseded:**
it read the 3-core set as "image mode is one shape", and the later 14-shape sweep showed it is a
*region* (image wins C<=48, loses C>=64; 7 of 14 shapes, up to -33%). Kept as written because the
narrowing and its correction are both part of the record.
Device: Samsung Xclipse 960 / Exynos 2600, serial R3CY905E04M.

---

## 0. Device image limits (trap 5, queried FIRST)

Added to the existing `MNN_DUMP_CL_EXT=1` HWINFO block in `OpenCLRuntime.cpp`:

```
[HWINFO] image2d_max_width=16384 image2d_max_height=16384
[HWINFO] image_support=1 max_read_image_args=128
```

MNN's NC4HW4 image mapping is `width = W * ceil(C/4)`, `height = N * H`.

| tensor | image W×H | vs 16384² |
|---|---|---|
| core_32 `1×32×72×96` | 768 × 72 | safe (4.7%) |
| core_48 `1×48×36×48` | 576 × 36 | safe |
| core_96 `1×96×18×24` | 576 × 18 | safe |
| winograd src/dst (32-core, worst in suite) | 3072 × 576 | safe (19%) |
| `pipeline/` 1920-wide @ C=32 | 15360 × H | **93% — marginal** |
| `pipeline/` 1920-wide @ C=64 | 30720 × H | **EXCEEDS — reboot zone** |

⇒ the whole conv_bench suite is safe. `pipeline/` is safe only up to C=32 at 1920 width.
The rule to apply before adding any shape: `W * ceil(C/4) <= 16384 && N*H <= 16384`.

---

## 1. The measurement that reframes the session

§H.47's −32.7% compared **buffer-direct against image-Winograd** — the two modes were not
running the same algorithm. The gates differ:

```c
// ConvBufWinograd::valid   (buffer)
ic>=32 && oc>=32 && input->width() < output->channel()
// ConvWinograd::valid      (image)
ic>=32 && oc>=32                              // <-- no width clause
```

`48→48@36×48` has `in_w=48, out_c=48` ⇒ `48<48` false ⇒ **buffer refuses Winograd, image takes it.**

Re-measured on `single_*.mnn` (**act=none**, depth 1, so both modes do identical work),
3 reps interleaved + cooled, metric `conv_all_us`, every arm gated against the **CPU backend**
(`forwardType 0`) at **cosine 1.000000**:

| core | buffer default | buffer forced-wino | image | **image vs buffer @ matched algorithm** |
|---|---|---|---|---|
| 32→32@72×96 | 125.0 | 149.0 (+19.2%) | 147.5 (+18.0%) | **−1.0%** |
| **48→48@36×48** | 99.0 | 89.0 (−10.1%) | **67.3 (−32.0%)** | **−24.4%** |
| 96→96@18×24 | 54.0 | 51.9 (−4.0%) | 56.4 (+4.5%) | **+8.7%** |

Kernel names confirm the paths: buffer default = `ConvBuf2D-ori-…` on the 32/48 cores and
`Conv-winograd-batchgemm/rearrange` on the 96; image = `Convolution0/1/2` (ConvWinograd) on all three.

**Reading.**
1. The 48-core's −32% is **~10 points Winograd gate + ~24 points image path.** Both real.
2. On the 32-core, image mode contributes **nothing** (−1.0%): the entire delta vs default is the
   Winograd gate, and it is a *regression*.
3. On the 96-core, where both modes already ran Winograd, image is **8.7% worse**.

⇒ **Image mode is not a general win.** *(Superseded: on these three cores it looks like a single
shape, but the 14-shape sweep in §H.51 found a coherent region — image wins at C<=48 and loses at
C>=64, on 7 of 14 shapes. The mechanism claim below was right: it is not "texture memory is faster"
in general.)*

**Caveat on the earlier run:** I first ran this on `core_*.mnn`, which `make_bundle.py` builds with
`fuse_prelu=True`. Those numbers (image −32.8%, matched −22.8%) are contaminated — image drops the
fused PReLU entirely. The table above uses `single_*.mnn` (`act="none"`) and is clean.

---

## 2. The PReLU blocker is smaller than briefed, and points at a different file

The brief assumed the fusion must be ported to "the image conv kernels". Actual state:

| path | PReLU support | reachable from a `leakyReluSlope` model? |
|---|---|---|
| image `conv_2d.cl` (direct) | **already fully implemented** — `#ifdef PRELU`, per-channel slope as `image2d_t` indexed by `out_channel_block_idx`, in every direct variant | **no** — `ConvExecution` only sets `mPrelu` from the `Extra`-op path (`ConvExecution.cpp:135`); it never reads `common()->leakyReluSlope()` |
| image `ConvWinograd` + `winogradTransformDest2_3_1.cl` | **none** — dest kernel has `RELU`/`RELU6` only | no |

And **all three cores take ConvWinograd in image mode**, so the direct-path PReLU that already
exists is not even on the hot path. The port is therefore:

- **(a)** `ConvExecution.cpp`: read `common()->leakyReluSlope()` → set `mPrelu` + upload the slope
  image. ~15 lines; mirrors `ConvBufExecution.cpp:122`. Kernel side already done.
- **(b)** `winogradTransformDest2_3_1.cl` + `ConvWinograd.{hpp,cpp}`: add `#ifdef PRELU` at the 4
  outputs + slope arg + upload. **Direct mirror of the already-shipped buffer patch**
  (`winogradTransform_buf.cl` / `ConvBufWinograd.cpp:215`). This is the one that matters.

Cost of *not* doing (b) is currently unmeasured; the §H.47 "~6 µs" figure is an estimate for an
unfused separate kernel, whereas fused into the dest transform it should be far less.

---

## 3. Audit — every strategy in §B–§H.50

**EXISTS** = already implemented in the image backend · **PORT** = worth porting · **MEANINGLESS** =
structurally n/a in image mode · **DEAD** = falsified for layout-independent reasons, do not redo.

| strategy | § | verdict | why |
|---|---|---|---|
| PReLU fusion | §E | **PORT — blocker** | direct path exists but unwired; ConvWinograd has none. See §2. |
| Force Winograd | §H.28/29 | **EXISTS — it is the image default** | image gate lacks buffer's `in_w<out_c` clause. Accounts for ~10 of the 48-core's 32 points, and is a regression on the 32-core. Image needs the *inverse* control: a way to turn Winograd **off**. |
| Winograd F(2,5) | — | **EXISTS, image-only** | `winogradTransform{Source,Dest}2_5_1` ship in image, not buffer. Irrelevant to this 3×3 model; recorded. |
| 2-D register tile `c4h4w2` | §H.21/22 | **PORT (conditional)** | image direct family = {`c4h1w4`,`c8h4w1`,`c4h4w1`} — no 2-axis 8-accumulator tile. But all three cores take Winograd in image mode, so this only pays on shapes that fall to the direct path. Port *after* establishing which shapes those are. |
| Shape hardcoding | §H.23/34/40 | **PORT** | technique is layout-agnostic (bake shape ints into build options). −18.7% in buffer. Applies to image direct conv *and* the winograd transform/gemm kernels. |
| Weights in a texture | (new) | **EXISTS — untested mechanism** | `mWeightUseBuffer = (gpuType==MALI)`, so on Xclipse image mode puts **weights in an `image2d_t`**. §H.49 falsified `__constant`; weights-as-texture is a *different* mechanism, live by default, never isolated. Candidate explanation for the 48-core win. |
| NCHW layout | §H.36/39 | **MEANINGLESS** | image2d storage is NC4HW4 *by construction* (`w = W·⌈C/4⌉`). An "NCHW image" forfeits the float4 texel fetch that is the entire point. Already +14…+96% in buffer. |
| LDS staging | §H.9/30 | **SKIP — it is the anti-thesis** | the Adreno claim image mode rests on is precisely *texture cache beats LDS*. Falsified in buffer; in image mode it is the strategy the hypothesis predicts is dominated. |
| im2col + GEMM | §H.13/38 | **DEAD** | +83…120%; mechanism is 9× bytes-per-MAC inflation — arithmetic, layout-independent. Extra image blocker: im2col matrix width vs the 16384 limit. |
| Implicit GEMM | §H.46 | **DEAD** | +98…300%; LDS-throughput bound + fp16 register cap. Layout barely mattered (2–3%). |
| Layer fusion (2-conv) | §H.25/44 | **DEAD** | 1.78× recompute to buy back a round-trip that costs ~0. |
| Cross-layer fusion ceiling | §H.1 | **DEAD** | inter-layer traffic ≈0 — layout-independent. Brief says do not redo. |
| LLC / cache placement | §H.35 | **DEAD** | same zero; no such extension on this stack. |
| Split-K over Cin | §H.31 | **PORT (low priority)** | mechanism confirmed, but −2% vs default in buffer. Only if image changes the occupancy picture. |
| 16-accumulator tiles | §H.8/22 | **DEAD** | register cliff is a hardware property. |
| space2depth on heads | §H.6 | **DEAD** | Cin already past the crossover; arithmetic. |
| Subgroup halo exchange | §H.11 | **BLOCKED** | `sub_group_shuffle` does not compile through clspv. Toolchain-level. |
| `__constant` weights | §H.49 | **DEAD** | ANGLE reports a 1 GB constant limit ⇒ `__constant` ≡ `__global`. Mode-independent. |
| Winograd F(4,3) | §H.50 | **OPEN in both modes** | not built anywhere; image would need its own 6×6 transform pair. Out of scope unless asked. |
| Concurrency / dispatch batching / zero-copy | §H.10/32/33 | **ORTHOGONAL** | unaffected by memory mode. |

Net: **1 blocker to port, 2 real ports (+1 conditional), 1 untested mechanism to isolate,
12 already closed.**

---

## 4. Build integration — two folders are not needed

Measured, not assumed:

- `source/backend/opencl/CMakeLists.txt` does `file(GLOB_RECURSE ...)` over the whole backend, so
  **`execution/buffer/` and `execution/image/` are both compiled into the same `libMNN_CL.so`.**
  There is no build-time buffer/image switch (`MNN_OPENCL_SIZE_CUT` only *removes* buffer).
  **Buffer vs image is a pure runtime choice: `gpuMode` 68 vs 132.** One build serves both.
- With `MNN_SEP_BUILD=ON` (already set in `build_android_profile`), touching any OpenCL source
  relinks only `libMNN_CL.so`. Measured incremental rebuild after touching a generated `.cl` cpp:

  ```
  make -j8 MNN_CL   →   2.05 s total
  ```

  A `.cl` edit adds one `opencl_codegen.py` run (~1 s). Push is ~6 MB stripped, a few seconds.

⇒ **Recommendation: keep ONE build directory (`build_android_profile`) and keep the project's
existing env-gate discipline.** Two mode-specific trees would cost disk and sync effort, buy nothing
for mode switching, and actively *harm* measurement quality: fair comparison here requires arms to be
**interleaved within one session on one cooled device** (trap 6, §H.43), which is impossible if the
two arms live in different builds that each need a push. The harness change actually required is the
one already made:

```python
session_measure.run(..., mode=68|132, ftype=3|0)   # gpuMode + forwardType are now parameters
```

The only case for a second tree is bisecting a *stock-vs-modified* kernel, and the established
pattern already covers that better: every strategy is an env-gated build option, default OFF, so
both arms exist in one binary and can be interleaved.

---

## 5. Proposed order of work — cheapest decisive measurement first

1. ✅ **Query image limits** (done — 16384², whole suite safe).
2. ✅ **Separate the Winograd-gate confound from the image-path effect** (done — §1). This was the
   cheapest decisive measurement and it already halves the claim.
3. **Port PReLU into image `ConvWinograd`** (§2b) + wire `leakyReluSlope` into image `ConvExecution`
   (§2a). Gate against CPU on the *fused* models. Then re-run §1's table on `core_*.mnn` and state
   what PReLU costs in image mode. **This is the go/no-go for the real model.**
4. **Isolate the 48-core mechanism.** Image wins 24% there and loses on both neighbours; that is a
   mechanism question, not a mode question. Cheapest probes, in order: which gemm variant the image
   autotuner picks (`ConvWinograd.cpp:313`) vs the buffer batchgemm; then weights-as-texture
   (flip `mWeightUseBuffer` on to move weights back to a buffer and re-measure — a one-line isolation
   of the untested mechanism).
5. **Sweep buffer-vs-image across the §H.29 14-shape grid** to find the boundary of the 48-core
   island, exactly as §H.29 did for Winograd. Determines whether this is one shape or a region.
6. **Port shape hardcoding** to the image winograd/direct kernels (only if 4–5 leave a live case).
7. **Port `c4h4w2`** to image `conv_2d.cl` — only for shapes that provably take the direct path.
8. **Report integration:** new `bundle_run_report.py` sections (buffer-vs-image per shape;
   matched-algorithm control) + `make_bundle.py` model coverage, so the decision re-runs on new HW.
9. **FINDINGS §H.51+**, per-shape verdict.

Steps 6–7 are contingent: if step 5 shows the win is a single shape already covered by
`MNN_FORCE_WINOGRAD` in buffer, the correct deliverable is a **per-conv mode/algorithm decision**,
not a global mode switch — and the kernel ports are not worth building.
