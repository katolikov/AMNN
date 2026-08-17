# Session prompt — build and measure an IMPLICIT GEMM conv (OpenCL, Xclipse 960)

Self-contained brief for a dedicated session. Everything below is measured on-device unless
marked as analysis. Read the "traps" section before writing any kernel — it will save you hours.

## Your task

Build an **implicit GEMM** 3×3 stride-1 convolution for MNN's OpenCL **buffer/fp16** backend: a
kernel that performs the GEMM reduction **without materialising the im2col matrix**, gathering the
3×3 columns on the fly (registers / LDS / whatever works) instead of reading a pre-expanded buffer.
Measure it honestly against MNN's autotuned default, and record the result either way.

## Why this idea exists — and the honest case against it

The explicit two-pass version is already built and measured (`MNN_CONV_IM2COL=1`, §H.13/§H.14):

| core | im2col pass | GEMM pass | total | MNN default | **GEMM alone vs default** |
|---|---|---|---|---|---|
| 32→32@72×96 | 94 µs | 139 µs | 233 | 120 | **+16% (loses)** |
| 48→48@36×48 | 38 µs | 80 µs | 118 | 102 | **−21% (wins)** |
| 96→96@18×24 | 21 µs | 114 µs | 135 | 30 | **+274% (loses badly)** |

The whole idea rests on the middle row: on that one shape the GEMM reduce alone beats the direct
conv, because im2col inflates channels (48 → 432) and lifts the conv out of the occupancy-starved
regime. **Implicit GEMM is the attempt to capture that −21% without paying the 38 µs im2col pass.**

Be sceptical, and size the prize before building much:
- It wins on **1 of 3 cores**. On the other two the GEMM reduce is *slower* than the direct conv.
- The −21% is measured with the data **pre-arranged**. Fusing the gather back in moves you toward
  the direct conv's own efficiency, not away from it. §H.14 argues this reasoning-only; nobody has
  built it, which is exactly why it is worth one honest attempt.
- The ceiling is hard: this GPU's practical conv limit is **~3.03 TFLOP/s** (§H.20, measured). The
  48-core runs at 0.71 TF = 23% of it. So the theoretical headroom is ~4×, but every kernel strategy
  tried so far has landed within ±20%.

**Success criterion:** beat MNN's autotuned default on 48→48@36×48 by >5%, cooled and interleaved,
bit-exact. Anything else is a documented negative — which is a perfectly good outcome, provided the
implementation is a fair test (see traps).

## The target convs

Stride-1 cores, batch=1, 3×3, pad 1, fp16, NC4HW4 (channels packed by 4):
`32→32@72×96` · `48→48@36×48` · `96→96@18×24` (the last already runs at 79% of ceiling — ignore it).

## Device facts

Samsung Xclipse 960 (Exynos 2600, RDNA4-derived), **8 compute units**, 980 MHz, 64 KB LDS,
`max_work_group_size` 1024, fp16 with packed math (`v_pk_fma_f16`) confirmed in use.
OpenCL runs **through ANGLE → clspv → SPIR-V → Vulkan**, not a native CL driver. Consequences:
- **No `sub_group_shuffle`** (`cl_khr_subgroups` exists; broadcast works, shuffle does not compile).
  Register-level neighbour exchange is therefore unavailable in OpenCL here.
- Vulkan *does* expose subgroup shuffle, but that driver **emulates it 41–66× slower** than a global
  reload. `subgroupQuadSwap` is the exception (0.7× a global load) and remains untried.
- No cooperative matrix / WMMA.

## What is already falsified (do not redo)

LDS halo tiling (+0.7%/+9.2%, and 2–2.5× slower at stride 2); cross-layer fusion (<2% ceiling —
inter-layer traffic is under 1 µs/conv); 16-accumulator tiles (`c8h8w1` +48.7%, `c4h4w4` +234%);
space2depth on stride-2 heads (+12–17%); subgroup halo-share (toolchain-blocked); Winograd (MNN
never selects it for these shapes — measured no-op). Full log: `FINDINGS.md` §H.

**What currently wins:** `conv_2d_c4h4w2` — a 2-D register tile (4 channels × 4 rows × 2 cols,
8 accumulators), **−8.6%**, and **−18.7%** with shapes hardcoded via `MNN_CONV_HARD=1` (§H.21/§H.23).
Beat that, not just the stock default, if you want the result to matter.

## Traps that have already cost real time

0. **A stale `tdir/input.json` is indistinguishable from a compiler crash — check it first.**
   The measurement helpers rewrite `tdir/input.json` on every call, so after any measurement loop
   the staged shape is whatever ran last. Running a *different* model against it makes MNN segfault
   **before printing anything**, and stdout is block-buffered, so you get a bare `Segmentation fault`
   with no output at all. This already cost a full session: five "ANGLE/clspv crashes" were retracted
   once the fixture was fixed (§H.24). Verified deterministic, same binary, A/B/A/B:
   `[1,32,24,48]` runs, `[1,48,36,48]` segfaults, repeat. **Always push an `input.json` matching the
   model you are about to run**, and if you see an unexplained segfault, check the shape before
   suspecting anything else:
   ```bash
   adb shell cat /data/local/tmp/convprobe/tdir/input.json
   ```

1. **Never index accumulators dynamically.** The first `c4h4w2` used a runtime loop with an if/else
   accumulator chain and ternary column selects: **308 µs**. Fully unrolled, identical maths:
   **109 µs**. A 2.8× difference from spilling alone — it looked like a clean falsification and
   wasn't. Generate the kernel from a Python template with everything statically expanded.
2. **Constant folding and loop unrolling are separate levers.** Hardcoding the shape is −18.7%;
   adding `opencl_unroll_hint` to the now-constant channel loop is **+13.4%** (I-cache/register
   blowup). Measure them independently.
3. **The `.cl` codegen does not escape `"`** — a double quote in a comment breaks the generated C++.
4. **The first `__kernel` in `conv_2d_buf.cl` is inside `#ifdef CONV_LOCAL_SIZE`** — anything
   inserted "before the first kernel" silently vanishes in normal builds.
5. **`DEAL_NON_UNIFORM_DIM2` is a backslash-continued multi-line macro** — do not split it.
6. **A previously-documented "clspv crashes when you modify conv_2d_buf.cl" trap has been
   RETRACTED** — it was the stale-`input.json` bug above (see the corrected §H.24). Editing,
   relocating and duplicating kernels in that file are all fine as far as anyone has actually
   measured. Do not design around a limitation that does not exist.
7. **Rebuild step:** after editing any `.cl`, run
   `python3 source/backend/opencl/execution/cl/opencl_codegen.py source/backend/opencl/execution/cl/`
   or your edit is silently ignored (the kernels are compiled into a generated `.cpp`).
8. **The device throttles ~2.75×** (119 → 328 µs) under sustained benchmarking. Interleave the arms
   **and** cool between reps, and check raw per-rep values — a median across the throttle point is
   meaningless.

## Harness

Repo `AMNN`, branch `feature/opencl-conv-specialize` (or `MNN`, `opencl-conv-specialize`).

```bash
python3 conv_bench/run_all.py --quick        # build -> push -> measure everything -> report
```

Force one kernel and hardcode the shape:
```bash
adb shell 'cd /data/local/tmp/convprobe && MNN_CONV_SPEC=1 MNN_CONV_HARD=1 MNN_CONV_FORCE=<kernel> \
  LD_LIBRARY_PATH=. ./ModuleBasic.out core_48.mnn tdir 0 3 120 68 2 x.bin 2>&1'
```

Timing comes from `conv time = N us` (GPU kernel time, needs `MNN_GPU_TIME_PROFILE=ON`; it excludes
the NCHW↔NC4HW4 rasters). Divide by 6 for the 6-deep chain models. **Give every forced kernel its
own cache file** — sharing one makes some runs emit no output at all and a correct kernel then reads
as `cosine = nan` (a correct kernel was falsely reported broken this way).

Correctness gate: run the same model with and without your kernel, pull `output/0_0.txt`, require
cosine > 0.999 (the good kernels here are bit-exact, 1.000000).

## Deliverable

Kernel + host wiring behind an env flag (default off), a correctness gate, cooled interleaved
measurements on both stride-1 cores against **both** MNN's default and `c4h4w2+HARD`, and a
`FINDINGS.md` §H.25 entry stating the result plainly — including "slower, and here is the
mechanism" if that is what the numbers say. Register it in `conv_bench/make_bundle.py`'s `VARIANTS`
so every future device measures it, even if it loses here: the winner is device- and clock-dependent.
