# How to test the convolutions on a device (from this branch)

This branch contains **all the source and scripts** — it does **not** contain binaries. Anything
compiled (`.so`, `MNNConvert`, `ModuleBasic.out`) has to be built on the machine you run from.
Pick one of the two paths below.

---

## Path A — build here, then run (use this if you want AMNN's own code measured)

### 0. Prerequisites on the host machine
* Android NDK (any recent r25+; export `ANDROID_NDK=/path/to/ndk/<version>`)
* cmake ≥ 3.10, make, a C++ toolchain
* `adb` on PATH, device connected & authorized (`adb devices` lists it)
* Python 3 with **numpy** and **onnx**:  `pip3 install numpy onnx`

### 1. Build the host converter (makes `.mnn` models)
```bash
mkdir -p build_host && cd build_host
cmake .. -DMNN_BUILD_CONVERTER=ON -DCMAKE_BUILD_TYPE=Release && make -j8
cd ..
```

### 2. Build the android libs + test tool — **`MNN_GPU_TIME_PROFILE=ON` is mandatory**
Without it there are no per-kernel GPU timings and every number in the report is `0`.
```bash
mkdir -p build_android_profile && cd build_android_profile
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 -DCMAKE_BUILD_TYPE=Release \
  -DMNN_OPENCL=ON -DMNN_ARM82=ON -DMNN_GPU_TIME_PROFILE=ON \
  -DMNN_BUILD_TOOLS=ON -DMNN_SEP_BUILD=ON -DMNN_LOW_MEMORY=ON
make -j8
cd ..
```

### 3. (optional) pin the GPU clock
Comparisons are only meaningful at a known clock, and this GPU class throttles from ~980 MHz to
~400–600 MHz under sustained load. Needs root:
```bash
adb shell 'su -c "echo 980000 > /sys/kernel/gpu/gpu_min_clock"'
adb shell 'su -c "echo 980000 > /sys/kernel/gpu/gpu_max_clock"'
```
Without root, skip it — the scripts sample the clock and report what it actually was.

### 4. Run
```bash
adb devices                       # get the serial
python3 conv_bench/probe_device.py <SERIAL> -o report_<device>.md
```
`probe_device.py` preflights first and prints exactly what is missing if a build/dep is absent.
Build dirs are auto-discovered; override with `MNN_BENCH_CONVERT` / `MNN_BENCH_ANDROID_BUILD`.

---

## Path B — no build on the run machine (portable bundle)

Build the bundle **once on a machine that has the two builds above**, then copy one tarball:
```bash
python3 conv_bench/make_bundle.py          # -> conv_bench/conv_probe_bundle.tar.gz (~37 MB)
```
On the run machine (needs only **python3 + adb**, no numpy/onnx/MNN/NDK):
```bash
tar xzf conv_probe_bundle.tar.gz && cd conv_probe_bundle
python3 run_report.py --list
python3 run_report.py --serial <SERIAL> -o report.md
```
The bundle carries pre-converted models, the arm64 libs, and precomputed correctness references.
It also has the stronger thermal safeguards (interleaved A/B comparisons, cooldowns between
sections, and an end-of-run clock re-check that marks the report VALID or THROTTLED).

---

## What gets measured
Hardware caps (compute units, clocks, LDS, `subgroup_shuffle`), clock stability, subgroup
compile-test, every conv strategy (`c4h1w1…c8h1w4`, `c8h8w1`, `c8h4w1_pa`, LDS + tile sweep)
against MNN's own default, im2col+GEMM with the implicit-GEMM headroom, the fused 2-layer
megakernel, the real model blocks ± PReLU fusion, concurrency, and a correctness gate per kernel.

Send the produced `report.md` back for analysis (mention the device and the clock you pinned).
Background and prior results: `conv_bench/OPTIMIZATION_HANDOFF.md`, `FINDINGS.md` §H.
