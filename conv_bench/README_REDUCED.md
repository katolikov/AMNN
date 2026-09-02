# Reduced-shape conv benchmark — runbook

Measures this model's 13 distinct convs at a chosen fraction of their original spatial size.

## Choosing the shape family

A family is a **divisor applied to `model_convs_updated.csv`**, the one conv list:

    python3 conv_bench/make_bundle.py --shape-family full   # original sizes
    python3 conv_bench/make_bundle.py --shape-family 1.5    # H/1.5, W/1.5
    python3 conv_bench/make_bundle.py --shape-family 3      # H/3, W/3   (default)

`CONV_BENCH_SHAPES` does the same thing for callers that already set it; the flag wins.

**Set it consistently.** `make_bundle.py` takes `--shape-family`, every other script reads
`CONV_BENCH_SHAPES`, so it is easy to build one family and then run another. preflight refuses that
outright -- it recomputes its case list per run, so a mismatch would launch models of one size while
believing they are another, and the results look like measurements. It presents as impossible
numbers (`Block5 ... direct_convs=4/2`: four direct convs in a two-conv model) rather than an error,
which is why it is now checked. Exporting the variable for the whole session is the simplest way:

    export CONV_BENCH_SHAPES=full

It matters at **bundle-build time** -- the models are pre-converted, so once built the bundle IS
that family, and the manifest records which.

You do not need to clear the tuning cache by hand. Every script fingerprints the binaries and the
manifest before pushing, and drops the cache when either changed -- so a rebuilt libMNN_CL.so or a
rebuilt bundle invalidates it automatically. Clearing it manually used to be a step you had to
remember, and forgetting it meant silently measuring programs compiled by the previous library.

A divisor is refused unless every spatial dimension divides exactly AND stays even. Exactness keeps
the shapes the model's real shapes rather than rounded approximations; evenness keeps stride-2
halving exact, so the pyramid each block encodes survives the reduction. `CONV_BENCH_SHAPES=5` is
rejected naming the conv it fails on.

Families are derived, never hand-maintained. They used to be separate CSVs with cores and heads
hardcoded per family, and they drifted: `full` once built a bundle whose blocks were full-size while
its cores and heads were still 1/3, with nothing in the bundle saying so.

Work per conv: `full` 1x, `1.5` 4/9, `3` 1/9 -- so the /1.5 convs are 4x the /3 ones, and the sweep
takes correspondingly longer than the ~7 min warm figure quoted below.

## The 13 convs (shown at /3)

| # | conv | original | reduced | stride |
|---|---|---|---|---|
| 1 | 1→8   | 576×768 | 192×256 | 2 |
| 2 | 8→16  | 288×384 | 96×128  | 2 |
| 3 | 18→16 | 288×384 | 96×128  | 2 |
| 4 | 16→32 | 144×192 | 48×64   | 2 |
| 5 | 34→32 | 144×192 | 48×64   | 2 |
| 6 | 32→48 | 72×96   | 24×32   | 2 |
| 7 | 64→64 | 72×96   | 24×32   | 2 |
| 8 | 64→96 | 36×48   | 12×16   | 2 |
| 9 | 8→8   | 288×384 | 96×128  | 1 |
| 10 | 16→16 | 144×192 | 48×64  | 1 |
| 11 | 32→32 | 72×96   | 24×32  | 1 |
| 12 | 48→48 | 36×48   | 12×16  | 1 |
| 13 | 96→96 | 18×24   | 6×8    | 1 |

The five stride-1 cores each appear 6× in their block, so the model runs ~30 conv instances.
They are reached through five probe models (Block3/4/1/2/96), which between them hold all 13 —
so one launch times several convs, attributed by the shape tag in each kernel's name.

## 0. Build (once per machine)

    cmake -B build_host -DMNN_BUILD_CONVERTER=ON
    make -C build_host MNNConvert -j10

    cmake -B build_android_profile \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 -DCMAKE_BUILD_TYPE=Release \
      -DMNN_OPENCL=ON -DMNN_ARM82=ON -DMNN_GPU_TIME_PROFILE=ON \
      -DMNN_SEP_BUILD=ON -DMNN_BUILD_SHARED_LIBS=ON -DMNN_USE_LOGCAT=OFF
    make -C build_android_profile MNN MNN_CL MNN_Express ModuleBasic.out -j10

    python3 conv_bench/make_bundle.py

**`MNN_USE_LOGCAT=OFF` is required.** With it on, every `MNN_PRINT` goes to logcat instead of
stdout, the harness parses nothing, and preflight fails with ~70 errors that all look like a
broken device.

Rebuild the bundle after any change to libMNN_CL.so, and clear `/data/local/tmp/convprobe/tune/`
— cached programs from the old library are stale.

## 1. Integrity gate (once per device/build)  — ~40 min

    python3 conv_bench/preflight.py <serial>

Checks each MNN_CONV_* flag actually engages, that forced kernels compute the right answer
(cosine vs CPU), and that the parser sees what it expects. A CLEAN verdict is the precondition
for trusting anything below; it catches wrong-build and wrong-flag problems that otherwise show
up as plausible numbers.

## 2. Noise floors (once per device)  — ~35 min cold, ~10 warm

    python3 conv_bench/variance_probe.py

Runs each configuration's batch five times and measures how far the batch MEDIAN moves. Writes
`noise_floors.json`. Every later verdict is gated on this: without it everything falls back to an
assumed 6%, which is wrong for at least two configurations here (one buffer core measured 17%).

Across-batch spread is the right quantity — reps inside one batch are interleaved and agree with
each other even when the batch as a whole is off, which is how a 54% error once passed every check.

## 3. Full sweep  — ~3.5 h first run, ~5 min after

    python3 conv_bench/full_sweep.py --reps 3

Measured on the reference device: warm-up 191 min, then five batches totalling 271 s. Warm-up cost
is NOT uniform across arms -- the plain kernels compile in seconds while NCHW, im2col and implicit
GEMM take far longer, so the last third of the arm list dominates. Budget ~4 h to bring up a new
device end to end, and ~5 min per sweep afterwards.

43 arms per conv: 15 kernels × 2 memory modes, plus NCHW, im2col+GEMM, implicit GEMM, LDS ×2,
split-K ×2, constant-weights, HARD, force/no-Winograd, and both mode defaults.

Phase 1 warms the tuning cache for all 215 (arm, model) pairs (**~3 h**, once per device) —
compilation is not measurement, and doing it inside a batch heats the device enough to invalidate
the batch. Phase 2 measures: one batch per probe model, 34–80 s each, each certified by a clock
watchdog.

Prints THREE recommendation tables, because gpuMode is set per Interpreter and a single
"best per conv" list mixes buffer and image winners that cannot coexist in one model:

    1. BUFFER  -- best kernel per conv if the model runs in buffer mode
    2. IMAGE   -- best per conv if it runs in image mode
    3. OVERALL -- best arm either way, against the deployed baseline (buffer, no flags)

Table 3 sizes the prize; tables 1 and 2 are what you can actually deploy. Re-render any past run
with `python3 conv_bench/show_results.py --recommend`.

## Reading back every measurement

The sweep prints one winner per conv; every arm it measured is in `conv_bench/results.db`.

    python3 conv_bench/show_results.py                  # latest run, every conv, every arm
    python3 conv_bench/show_results.py --conv 96->96    # one conv
    python3 conv_bench/show_results.py --csv > runs.csv # for a spreadsheet
    python3 conv_bench/show_results.py --runs           # list runs, then --run <id>

Percentages are against that batch's declared baseline, and each is checked against the
configuration's own measured noise floor -- anything smaller is marked `(noise)` rather than
reported as a result. The CSV carries the full provenance of each row: the env that produced it,
which arm was the baseline, whether that baseline was the deployed configuration, and the batch it
belongs to (only rows sharing a batch are comparable).

## If you see no output

`variance_probe` prints one line per model as it finishes. A cold configuration compiles shaders
before its first line appears, so the first model can take minutes -- that silence is normal.

Silence from EVERY script, including `preflight`, means output is going to logcat instead of
stdout. Check the bundle's copy of the library, which is the one that runs:

    strings conv_bench/conv_probe_bundle/bin/libMNN.so | grep -c MNNJNI     # 0 = logcat OFF

or ask the device directly:

    adb shell 'cd /data/local/tmp/convprobe && LD_LIBRARY_PATH=. ./ModuleBasic.out Block1.mnn tdir 0 3 4 68 2 t.bin 2>&1' | grep -c "kernel time"

Zero, with the lines visible in `adb logcat -d -s MNNJNI`, is conclusive. Two things to know about
the fix: `MNN_USE_LOGCAT` lives in libMNN.so, not libMNN_CL.so, so `make MNN_CL` alone will not
change it; and the bundle carries COPIES, so a rebuild without `make_bundle.py` leaves the old
library in place.

    cd build_android_profile && cmake .. -DMNN_USE_LOGCAT=OFF
    make -C build_android_profile MNN MNN_CL MNN_Express ModuleBasic.out -j10
    python3 conv_bench/make_bundle.py

## What it does NOT do

* **Wall-clock.** Everything above measures GPU kernel time. On this model kernel time is only
  13–15% of wall — the rest is CPU submission overhead — and the two metrics can disagree in
  sign. Confirm any candidate at whole-model wall-clock per block before shipping it.
* **Whole-model decisions.** `gpuMode` is per Interpreter, so a block gets one memory mode even
  when its convs disagree. Measure the block, not the sum of its convs.
* **The record queue.** `MNN_GPU_RECORD_BATCH` (gpuMode bit 1<<9) batches ~10 kernels per
  submission, but `ENABLE_OPENCL_TIME_PROFILER` bypasses it — so every number here is measured
  with MNN's own dispatch batching disabled. Measuring it needs a non-profiling build.

## Files

Committed: the scripts, `model_convs_updated.csv` (the single conv list every family derives
from), and the offline tests -- `test_bench_store.py` (21) and `test_shape_families.py` (8), which
pins the /3 derivation to the 13 convs actually measured.
Gitignored: `results.db`, `noise_floors.json`, `*_state.json` — device- and build-specific.
Committing them would let a claim be gated on another device's noise, or a sweep resume onto
results from a different libMNN_CL.so.
