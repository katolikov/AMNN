# Session prompt — test the strategies that were rejected WITHOUT being built

Self-contained brief for a dedicated session. Everything below is measured on-device unless
marked as analysis.

## Why this session exists

Across this investigation several strategies were dismissed by **reasoning alone**, never built.
That habit has already been caught being wrong once and right once:

- **Wrong:** the 2-D register tile was rejected after one implementation measured 308 µs
  ("2.6× slower, falsified"). Rewritten fully unrolled — identical maths — it measured **109 µs and
  now beats MNN's autotuned default by 8.6%** (−18.7% with shapes hardcoded). A 2.8× swing from
  implementation quality alone.
- **Right:** all 16-accumulator tiles were dismissed as "expected to regress identically". Built
  (`c4h4w4`): **+234%**. The reasoning held.

So a theory-only rejection is worth roughly nothing until someone builds it. Your job is to close
the remaining ones with measurements, in the order below, and to record each result — including the
negatives, which are the majority outcome here and are genuinely useful.

## The queue, highest value first

### 1. Force Winograd ON (never actually tested)
`MNN_NO_WINOGRAD=1` was measured and is a **no-op** — because MNN's heuristic never *selects*
Winograd for these shapes in the first place (§8 of the report shows it picks the `ori` path on
every core and head). So the standing claim "Winograd is contraindicated for K≤128" has **never
been measured on this device** — only the selector's opinion of it has.
Bypass `ConvBufWinograd`'s selection heuristic (an env flag, mirroring `MNN_CONV_FORCE`) and force
it on for `32→32@72×96` and `48→48@36×48`. Winograd F(2,3) cuts multiplies 2.25× at the cost of
transform passes. §H.18 predicts it should look *worse* at full clock and better at low clock.
Cheap: the kernels already exist, this is selector plumbing.

### 2. Split-K over input channels
The stride-2 heads and the small cores are **output-starved**: `64→96@36×48 s2` produces only
18×24×24 = 10 368 output float4s, far too few to fill 8 CUs. Splitting the Cin reduction across
several workgroups and reducing afterwards multiplies thread count directly. Needs either atomics
(fp32) or a two-pass reduce. This is the only idea in the queue that attacks occupancy starvation
head-on, which §H.20 identifies as the actual limiter (these convs run at 23–35% of the measured
3.03 TFLOP/s ceiling).

### 3. `c4h1w2` + LDS
§H.9 closed this with "the constant-blocking isolation settles it without needing to build it".
That is exactly the reasoning shape that failed for the 2-D tile. LDS lost everywhere it was
measured, so expectations should be low — but it is one kernel and it removes an open claim.

### 4. In-register prefetch / register minimisation
§H.12 dismissed these as "low-EV" because they move register pressure the wrong way. Given that the
current winner is an 8-accumulator tile and 16 accumulators is catastrophic, the register budget is
clearly the binding constraint — so a variant that *reduces* live registers is the one direction not
yet explored. Weakest item in the queue; do it last.

### 5. ION / dma_buf zero-copy input
`cl_arm_import_memory` and `cl_khr_external_memory_dma_buf` are exposed and unused. This does not
touch kernel time at all — it removes the host→device upload. Only worth doing if wall-clock, not
kernel time, is the target. Measure with wall clock (`min= X ms`), not `conv time`.

**Explicitly out of scope for this session:** batch > 1 (batch is always 1 in this application) and
anything Vulkan (`subgroupQuadSwap` etc. — different substrate, different session).

## Device facts

Samsung Xclipse 960 (Exynos 2600, RDNA4-derived), **8 compute units**, 980 MHz, 64 KB LDS, fp16 with
packed math confirmed. OpenCL runs **through ANGLE → clspv → SPIR-V → Vulkan**, not a native driver:
**no `sub_group_shuffle`**, no cooperative matrix. Measured practical conv ceiling: **3.03 TFLOP/s**
(§H.20) — the target convs sit at 23–35% of it, so the honest headroom is ~3×, not the ~6× an
assumed 6.5 TF peak once suggested.

Targets (stride-1 cores, batch 1, 3×3, pad 1, fp16, NC4HW4):
`32→32@72×96` · `48→48@36×48` · `96→96@18×24` (the last is at 79% of ceiling — every one of 13
implementations ties there at 30 µs; skip it).
Stride-2 head pairs: `18→16@288×384` → `16→32@144×192`; `34→32@144×192` → `32→48@72×96`;
`64→64@72×96` → `64→96@36×48`.

## The bar to beat

MNN's autotuned default, **and** `conv_2d_c4h4w2` with `MNN_CONV_HARD=1` (−18.7% on the main core).
Beating only the stock default is not interesting any more.

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

1. **Never index accumulators dynamically** — dynamic accumulator loops/selects spill to scratch and
   cost up to 2.8×. Generate kernels from a Python template, fully unrolled.
2. **Constant folding and loop unrolling are separate levers**: folding is −18.7%, adding
   `opencl_unroll_hint` on the folded loop is **+13.4%**. Measure independently.
3. **The `.cl` codegen does not escape `"`** — a quote in a comment breaks the generated C++.
4. **The first `__kernel` in `conv_2d_buf.cl` sits inside `#ifdef CONV_LOCAL_SIZE`** — text inserted
   "before the first kernel" silently vanishes in normal builds.
5. **`DEAL_NON_UNIFORM_DIM2` is a backslash-continued macro** — do not split it.
6. **A previously-documented "clspv crashes when you modify conv_2d_buf.cl" trap has been
   RETRACTED** — it was the stale-`input.json` bug above (see the corrected §H.24). Editing,
   relocating and duplicating kernels in that file are all fine as far as anyone has actually
   measured. Do not design around a limitation that does not exist.
7. **Always regenerate after editing a `.cl`:**
   `python3 source/backend/opencl/execution/cl/opencl_codegen.py source/backend/opencl/execution/cl/`
8. **Give every forced kernel its own cache file.** Sharing one makes some runs emit no output, so a
   correct kernel reads as `cosine = nan` — this already produced a false "FAIL" once.
9. **The device throttles ~2.75×** (119 → 328 µs) under sustained load. Interleave arms, cool
   between reps, and inspect raw per-rep values; a median spanning the throttle point is meaningless.

## Harness

Repo `AMNN`, branch `feature/opencl-conv-specialize` (or `MNN`, `opencl-conv-specialize`).

```bash
python3 conv_bench/run_all.py --quick        # build -> push -> measure everything -> report
```

```bash
adb shell 'cd /data/local/tmp/convprobe && MNN_CONV_SPEC=1 MNN_CONV_HARD=1 MNN_CONV_FORCE=<kernel> \
  LD_LIBRARY_PATH=. ./ModuleBasic.out core_48.mnn tdir 0 3 120 68 2 x.bin 2>&1'
```

`conv time = N us` is GPU kernel time (excludes the NCHW↔NC4HW4 rasters); divide by 6 for the 6-deep
chain models. Correctness: same model with/without the kernel, pull `output/0_0.txt`, require
cosine > 0.999 (the good kernels here are bit-exact).

## Deliverable

For each item: kernel + host wiring behind an env flag (default off), a correctness gate, cooled
interleaved measurements against both baselines, and a `FINDINGS.md` entry stating the outcome
plainly. Register every new kernel in `conv_bench/make_bundle.py`'s `VARIANTS` even if it loses
here — the winner is device- and clock-dependent, and the suite is meant to re-decide on new
hardware.
