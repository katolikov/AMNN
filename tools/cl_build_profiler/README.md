# clBuildProgram profiler

Measures what an OpenCL driver charges MNN for compiling its kernels, and checks that
the binaries it hands back are usable.

Every OpenCL program the MNN backend ships is compiled with the build options the
engine really passes, and every call around the compile is timed separately, so a
device where `clBuildProgram` is ten times slower than another can be compared line by
line instead of by feel.

The tool links no part of libMNN. It takes only the generated kernel sources from
`source/backend/opencl/execution/cl` and loads the vendor ICD with `dlopen`, so it
builds and runs on a device without an MNN build present.

## Build and run on a device

```bash
./run_device.sh -s <serial>
```

That builds for `arm64-v8a`, pushes to `/data/local/tmp/cl_build_profiler`, runs, and
pulls the results into `./cl_profile_<timestamp>/`:

| file | contents |
|------|----------|
| `profile.log` | the full console report |
| `samples.csv` | one row per measured build, for spreadsheets and diffs |
| `summary.json` | device identification plus per program statistics |
| `device_info.txt` | SoC, build fingerprint, CPU governors, hot thermal zones |

Anything after `--` goes to the profiler:

```bash
./run_device.sh -s <serial> -- --programs 'conv_2d*' --repeat 5 --split-compile-link
```

Use `--no-build` to reuse the binary already built, `--no-push` to reuse the one on the
device. `run_device.sh --help` lists the rest.

To build for the host instead:

```bash
cmake -S tools/cl_build_profiler -B build-clprof && cmake --build build-clprof -j
./build-clprof/cl_build_profiler --list
```

## What is measured

For every program, in this order:

| phase | what runs | what it tells you |
|-------|-----------|-------------------|
| `cold` | `clBuildProgram` on source the driver has never seen | the real compiler cost, the number that matters for a first run |
| `cache-fill` | one build of the unsalted source | produces the reference binary |
| `warm` | the same source and options again | how much the driver's own compiler cache saves |
| `binary` | `clCreateProgramWithBinary` then `clBuildProgram` | the path MNN takes when its cache file is present |

Cold builds are salted: a uniquely named `__constant` is appended to the source so the
driver cannot serve a cached binary and every measurement is a genuine compile. Pass
`--no-salt` to measure what a repeated identical build costs instead.

Each pass times `clCreateProgramWithSource`, `clBuildProgram`, the build status query,
`clGetProgramInfo(CL_PROGRAM_BINARIES)`, `clCreateKernelsInProgram`, the per kernel
`clGetKernelInfo` and `clGetKernelWorkGroupInfo` queries, and the releases. The CPU time
the process burns inside `clBuildProgram` is recorded next to the wall time: when the
two match the compiler runs in process and is CPU bound, when they diverge the time is
spent waiting on something else.

`--split-compile-link` additionally measures `clCompileProgram` and `clLinkProgram`
separately, which splits a slow build into front end (parsing and preprocessing) and
back end (code generation) cost.

`--jobs <n>` builds the whole set again from `n` threads and compares the wall time
against the same work done serially, which exposes a global lock inside the driver.

The run also compiles and executes a small built in kernel and checks its output, so a
report always states whether the device could actually run what it compiled.

## What is verified

For every program that builds:

* **round trip** — the produced binary reloads through `clCreateProgramWithBinary` and builds
* **determinism** — building the same source with the same options twice yields the same bytes
* **stable** — the binary read back after the round trip is byte identical to the one that went in
* **kernels** — the reloaded program exposes the same kernel names
* **attributes** — and the same work group size, preferred multiple, local and private memory per kernel

A driver that fails *determinism* or *stable* cannot be trusted with MNN's binary cache
file, because the cached blob would not describe the program that gets used.

## Surviving a driver crash

A GPU compiler that segfaults takes the whole process with it. The builds therefore run
in a forked child that flushes each finished program to disk before starting the next.
If the child dies, the parent names the program that was in flight, marks it
`[DRIVER CRASH]`, and restarts on the rest, so one bad kernel costs one result instead
of the entire run. Pass `--no-crash-guard` to run everything in one process.

## Build options

By default the tool reproduces `OpenCLRuntime::buildKernelWithCache`: the precision
defines for `--precision 0|1|2`, the input and output type defines the engine adds for
float tensors, `-DSET_ATTRIBUTE=false` and the `-cl-mad-enable -w` suffix.

Some programs need options only their `Execution` knows. `binary` does not preprocess
without `-DOPERATOR`, and every kernel of `groupnorm_buf` sits behind
`#if LOCAL_SIZE > 1`. `programDefines` in `src/ProgramCatalog.cpp` carries those, taken
from the call sites, so the whole kernel set compiles. Turn them off with
`--no-program-defines` to see what the bare source does, add your own with `--options`,
or replace the option string entirely with `--options-replace`.

## Exit codes

| code | meaning |
|------|---------|
| 0 | every program built and every check passed |
| 1 | bad usage, or no usable OpenCL device |
| 2 | a program failed to build |
| 3 | a verification or execution check failed |

## Comparing two devices

Run on both, then diff the summaries or load both `samples.csv` files into one sheet and
group by `program`. The `device` and `driver` columns identify each row. The lines to
look at first are the per program `cold` median and the `cold / binary` ratio in the
summary: the first says how expensive the compiler is, the second says how much a
working binary cache is worth on that device.
