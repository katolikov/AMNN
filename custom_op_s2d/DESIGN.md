# Fused `Affine + SpaceToDepth` OpenCL **buffer** op — design (no implementation)

Target device: **S26 — Exynos 2600 / Xclipse 960 (RDNA4), OpenCL = ANGLE-over-Vulkan, buffer mode.**
Proposed name: `FusedMathS2D`, `OpType_FusedMathS2D = 609` (606/607/608 = BinCount / DualRangeHist /
ApplyLUT; `MockChain=607` lives on its own branch — take 609 and resolve that collision if those
branches ever merge).

**Spec locked:** two inputs `[1,1,H,W]` → **`[1,13,H/2,W/2]`**; per-pixel affine math, fp16, dense
NCHW in *and* out, H and W even, inputs produced by upstream OpenCL-buffer ops (already on the GPU),
consumer is not a conv.

---

## 1. Spec

```
in0  A : [1,1,H,W]  fp16, dense NCHW   (output of an upstream OpenCL buffer op)
in1  B : [1,1,H,W]  fp16, dense NCHW   (idem)
out    : [1,13,H/2,W/2] fp16, dense NCHW         (H2 = H/2, W2 = W/2)

P0(h,w) = alpha*A(h,w) + beta*B(h,w) + gamma      // mixed  — reads both
P1(h,w) = delta*B(h,w) + epsilon                  // single — reads B only
P2(h,w) = P0(h,w) - P1(h,w)                       // diff   — reads nothing new

out[p*4 + dy*2 + dx][ho][wo] = Pp( 2*ho + dy, 2*wo + dx )     p∈{0,1,2}, dy,dx∈{0,1}
out[12][ho][wo]              = K                              // uniform / scalar-math channel
```

Channel order within a plane is `(dy,dx) = (0,0),(0,1),(1,0),(1,1)`, planes major. We own this op,
so this is a *definition* — write it into the schema comment and mirror it in the CPU reference.
Wrong channel order is the classic silent bug here; pin it with a hand-computed 4×4 case.

**Coefficients are scalar kernel arguments**, not build options. Constant-folding them buys nothing
(§2.2) and arguments keep one compiled binary for all coefficient values — fewer kernels, fewer
autotune-cache entries, no rebuild to retune. Put them in the op param (`FusedMathS2DParam`) and
pass them straight through. If the shape of the math itself ever changes (not just the constants),
*that* becomes a `-DMATH_MODE=n` build option.

**Channel 12 is a uniform plane** — a constant, or the result of scalar-only arithmetic (no
dependence on `A`/`B` at the pixel level). It costs one extra `vstore` per work-item and no loads.
See §2.3 for the case against materializing it at all.

`P2` collapses algebraically: `P2 = alpha*A + (beta-delta)*B + (gamma-epsilon)` — still affine, so
it is either one extra subtract of two values already in registers, or its own 2-mad chain. Either
way **it needs no new loads and no new inputs**; its entire cost is the 4 extra stores (§2.1).

All three planes being affine means the op is numerically a 1×1 conv followed by space-to-depth.
Nice to know, not actionable — a real conv would be far slower than a handful of `mad`s.

Creator preconditions (else return `nullptr` → CPU fallback, never garbage):
`batch==1`, `C==1` on both inputs, identical shapes, `H%2==0 && W%2==0`, float type.

---

## 2. Roofline — this is the entire design

### 2.1 The floor

Per output pixel the op **must** read 4+4 input elements and write 13. Over the frame:

```
reads  =  2*H*W elements                            (unchanged by the extra channels)
writes = 13*(H/2)*(W/2) = 3.25*H*W elements
total  =  5.25*H*W elements = 10.5*H*W bytes (fp16)  ← hard floor, reachable in ONE kernel
```

**The diff plane is write-only cost.** Reads and ALU are untouched (§1), so the 4 extra channels
add `1*H*W` of stores — **+25% over the 8-channel version, nothing more**. Computing them as a
separate downstream op instead would read the 8 channels back (`2*H*W`) and write 4 (`1*H*W`) =
`+3*H*W`, so folding them in here is **3× cheaper than a second op**.

`P1` reading only `B` does **not** reduce traffic: `B` is read once and serves both planes, and `A`
is still needed in full by `P0`.

Measured streaming ceiling on this device: **65–73 GB/s** (from the ApplyLUT op — same kernel class;
the number holding constant across precisions is what proved it bandwidth-bound).

Predicted kernel time `= 10.5*H*W / 70e9`:

| H×W | fp16 traffic | predicted (13 ch) | 12 ch | 8 ch |
|-----|--------------|-------------------|-------|------|
| 720×1280  | 9.7 MB  | ~138 µs | 132 µs | 105 µs |
| 1080×1920 | 21.8 MB | ~311 µs | 296 µs | 237 µs |
| 1440×1920 | 29.0 MB | ~415 µs | 395 µs | 316 µs |

### 2.3 The uniform channel: cheap here, but probably shouldn't exist

Channel 12 adds `0.25*H*W` of stores — **+5%** over 12 channels. Trivial in isolation. But look at
the round trip: this op writes `0.25*H*W` and the consumer reads `0.25*H*W` back, so **~0.5*H*W of
DRAM traffic (~10% of this op) is spent conveying a single number**.

If the consumer is one of our own kernels, pass it as a **scalar kernel argument** and drop the
channel — that is a straight 5% win here and more downstream. Materialize it only if the consumer is
fixed/opaque and genuinely requires a 13-channel tensor. Flagging it because it is the kind of thing
that is free to fix now and awkward to fix once a downstream op hardcodes 13.

### 2.2 Consequences to internalise before writing code

1. **No algorithmic win exists.** 5.25·H·W is information-theoretic (given the spec — but see §2.3). All effort goes to reaching peak
   bandwidth: layout, coalescing, stream count, occupancy.
2. **The ALU is free — and this math is 4 `mad`s.** ~35 Gelem/s against a multi-TFLOP fp16 ALU is a
   budget of **50+ flops per element**, against a spend of ~4. The math is not merely cheap here,
   it is unmeasurable. Do not optimize it, do not specialize on the coefficients, do not consider
   fp16-vs-fp32 math for *speed* (only for registers, §6 T5).
3. **One kernel launch.** Any design needing two dispatches pays a second full read of the inputs
   (+50% traffic).
4. **The real remaining win is upstream, not inside this kernel — see §3.2.**

---

## 3. What this op must beat, and what beats this op

### 3.1 The decomposed graph (the baseline for the report)

`Scale(A) → Scale(B) → Add → P0`, `Scale(B) → P1`, `SpaceToDepth ×2`, `Concat`. MNN decomposes
`SpaceToDepth` in geometry (`source/geometry/GeometryDepthToSpace.cpp`) into **Raster** regions, and
`Concat` likewise:

| stage | traffic | dispatches |
|-------|---------|-----------|
| affine chain → P0 | 3–7 · HW (depends how many Binary/Scale nodes survive fusion) | 1–3 |
| affine → P1 | 2 · HW | 1 |
| Raster (s2d P0) | 2 · HW | 1+ |
| Raster (s2d P1) | 2 · HW | 1+ |
| diff → P2 | 2R + 1W = 3 · HW | 1 |
| Raster (s2d P2) | 2 · HW | 1+ |
| Concat | up to 6 · HW | 1+ |
| **total** | **~18–26 · HW** | **7–11** |

vs **5.25·HW in 1 dispatch** — a **4–5× traffic reduction**, 6–10 launches saved (launch overhead is not
negligible on this ANGLE stack: ~34% of the small conv transform kernels), and the raster kernels are
strided/scalar and run well below peak. **Expect 4–6× end-to-end.**

### 3.2 The bigger win: absorb the producers

The inputs come from **upstream OpenCL buffer ops we control**. That makes the two full-resolution
planes `A` and `B` *materialized intermediates that exist only to be consumed by this op* — and
materializing a full-res fp16 plane costs `1·HW` to write plus `1·HW` to read.

```
today:   upstream(A) 2*HW + upstream(B) 2*HW + this op 5.25*HW      =  9.25*HW
fused:   read the upstream sources (2*HW) + write the output (3.25*HW) =  5.25*HW
```

**That is another ~1.8× on the pair — larger than anything left inside this kernel.** If the upstream
ops are elementwise (or any per-pixel map), pulling them into this kernel is nearly free: the ALU
budget is untouched (§2.2), the read pattern is identical, and the intermediate tensors disappear
from the memory pool entirely.

Recommendation: **build the op as specified first, measure it, then evaluate absorbing the
producers as step 2.** Note the shape of the extension now so it stays cheap: keep `f`/`h` as inline
functions taking the *raw* inputs, so widening them to "raw upstream sources → P0/P1" is a local
edit, not a redesign. (It does not apply if an upstream op is a conv, a reduction, or anything with
cross-pixel dependencies.)

The consumer is not a conv, so there is no downstream fusion counterpart to chase.

---

## 4. Layout — decided, plus the one thing to verify on-device

**All three tensors dense NCHW. No NC4HW4 anywhere.**

The buffer backend's default is NC4HW4 (`offset = ((((c/4)*N+n)*H+h)*W+w)*4 + c%4`, see
`buffer_convert_buf.cl`). For the `C=1` inputs that pads the channel to 4 → **4× read traffic** —
the single most expensive mistake available here. Force it off in the creator, as
`ApplyLUTBufExecution` does, for inputs **and** output:

```cpp
for (auto t : {in0, in1, out}) {
    if (TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) return nullptr;
    TensorUtils::setTensorSupportPack(t, false);
}
```

Resulting buffer views:

```
A, B : contiguous H*W
out  : 13 contiguous planes of H2*W2 ,  out[c] at c*H2*W2 ,  planeStride = H2*W2
```

**⚠ The one thing to verify early.** `A` and `B` are *shared* tensors — the upstream op writes them.
Marking them unpacked here only works if the producer also emits dense NCHW. You've said the
upstream ops already produce NCHW, which means they are presumably custom ops doing the same
forcing; confirm it before optimizing anything else, because the failure mode is silent and
expensive: MNN inserts a `buffer_convert_*` kernel per input, each costing `2·HW` — **+100% traffic,
wiping out most of the win.** Check the profiler op list for `buffer_convert_*` neighbours in the
very first working run. If a producer is a stock MNN op that insists on NC4HW4, the fix is §3.2
(absorb it), not a layout compromise.

**Cost of NCHW output.** Identical byte count, but **13 concurrent write streams**; with 4 read
streams that is **17 live streams per wave** → many open DRAM pages and real write-combine pressure.
The in-layout mitigation is long per-stream bursts (§5.4, §6 T2), not a wider `VW`.

**The 13th channel changed the NC4HW4 trade.** At `C=12` the packed layout was 3 exact slices with
zero padding — 3 write streams for the same bytes, a clear thing to measure. At `C=13` it pads to
16, so packed output writes `4*H*W` instead of `3.25*H*W`:

| output layout | total traffic | write streams |
|---------------|---------------|---------------|
| dense NCHW (spec) | 5.25·H·W | 13 |
| NC4HW4 (`C=13`→16) | 6·H·W (**+14%**) | 4 |

So NC4HW4 now has to beat a 14% byte penalty on stream count alone. Still worth one measurement, but
NCHW's position is much stronger than it was at 12 channels — and if §2.3 lands and the channel goes
away, `C=12` restores the zero-padding case and the question reopens.

**Never image/texture mode** — independent of speed, it fails allocation and can *reboot the phone*
on tensors wider than the max 2D image size. Buffer only, `gpuMode 68`.

---

## 5. Kernel design

### 5.1 Work decomposition

**One work-item produces `VW` complete output pixels — all 13 channels of each.**

That is the point of the fusion: `A` and `B` are read once and feed all 13 outputs. Two tempting
structures, both strictly worse:

* *split the 13 channels across work-items / a z-dimension* → each group re-reads the inputs → 2× reads;
* *two dispatches, 4 planes each* → +2·HW (+50%), plus a second launch.

Because the 2×2 blocks **partition** the input, every element is read exactly once by exactly one
work-item. Therefore **no reuse, no LDS, no barriers, no subgroup ops** — which is what this stack
wants anyway: LDS+barrier lost to plain cached-global reads in ApplyLUT (occupancy), and
`subgroupShuffle` lowers to `ds_bpermute`, measured **6.6× slower** than redundant L2 loads in the
Vulkan conv work.

### 5.2 The space-to-depth is free

With `VW` output pixels per item, each row load is a `2*VW`-wide vector whose **even lanes are
`dx=0` and odd lanes are `dx=1`**. OpenCL C has built-in `.even` / `.odd` component selectors, so
the deinterleave is a **compile-time register swizzle — zero instructions, zero scratch**:

```c
FLOAT_IN a0 = VLOAD_IN(0, A + t);       // row 2*ho     , 2*VW elements
FLOAT_IN a1 = VLOAD_IN(0, A + t + W);   // row 2*ho + 1
FLOAT_V a00 = a0.even, a01 = a0.odd;    // (dy,dx) = (0,0) and (0,1)
FLOAT_V a10 = a1.even, a11 = a1.odd;    // (1,0) and (1,1)
```

`dy` is simply which of the two row loads you use. So "space2depth" costs **nothing** — it is only a
question of which register feeds which store. This is the key structural insight of the kernel.

### 5.3 Sketch

```c
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifndef VW
#define VW 4                                  // output pixels per work-item; 2, 4 or 8
#endif
#define PASTE_(a,b) a##b
#define PASTE(a,b)  PASTE_(a,b)
#define FLOAT_IN    PASTE(FLOAT, 2*VW)        // half4 / half8 / half16
#define FLOAT_V     PASTE(FLOAT, VW)          // half2 / half4 / half8
#define CFLOAT_V    PASTE(COMPUTE_FLOAT, VW)
#define TO_CFLOAT_V PASTE(CONVERT_COMPUTE_FLOAT, VW)
#define TO_FLOAT_V  PASTE(CONVERT_FLOAT, VW)
#define VLOAD_IN    PASTE(vload,  2*VW)
#define VSTORE_V    PASTE(vstore, VW)

// A handful of mad's against a ~50 flop/element budget -- unmeasurable. Keep them taking
// the RAW inputs so §3.2 (absorbing the upstream ops) stays a local edit.
inline CFLOAT_V p0_of(CFLOAT_V a, CFLOAT_V b, float al, float be, float ga) {
    return mad((CFLOAT_V)al, a, mad((CFLOAT_V)be, b, (CFLOAT_V)ga));
}
inline CFLOAT_V p1_of(CFLOAT_V b, float de, float ep) {
    return mad((CFLOAT_V)de, b, (CFLOAT_V)ep);
}

#define EMIT(ch, v) VSTORE_V(TO_FLOAT_V(v), 0, out + (ch)*planeStride + o)

// One (dy,dx) lane-group -> its three channels. Emitting per lane-group (rather than
// per plane) keeps only ONE fp32 pair live at a time and lets P2 reuse p0/p1 straight
// out of registers -- the diff costs a single subtract.
#define GROUP(k, av, bv)                                              \
    do {                                                              \
        CFLOAT_V _a = TO_CFLOAT_V(av), _b = TO_CFLOAT_V(bv);          \
        CFLOAT_V _p0 = p0_of(_a, _b, alpha, beta, gamma);             \
        CFLOAT_V _p1 = p1_of(_b, delta, epsilon);                     \
        EMIT(0 + (k), _p0);                                           \
        EMIT(4 + (k), _p1);                                           \
        EMIT(8 + (k), _p0 - _p1);                                     \
    } while (0)

__kernel void math_s2d_buf(
    __global const FLOAT* A, __global const FLOAT* B, __global FLOAT* out,
    __private const int W,            // input width = 2*W2
    __private const int W2, __private const int W2v,   // W2v = W2/VW
    __private const int total,        // = H2 * W2v   (exact 1-D grid)
    __private const int planeStride,  // = H2*W2
    __private const float alpha, __private const float beta, __private const float gamma,
    __private const float delta, __private const float epsilon)
{
    const int gid = get_global_id(0);
    if (gid >= total) return;                    // masks only the LWS round-up
    const int ho = gid / W2v;
    const int wo = (gid - ho*W2v) * VW;

    const int t = (ho*2)*W + wo*2;               // top-row element offset
    const FLOAT_IN a0 = VLOAD_IN(0, A + t), a1 = VLOAD_IN(0, A + t + W);
    const FLOAT_IN b0 = VLOAD_IN(0, B + t), b1 = VLOAD_IN(0, B + t + W);

    const int o = ho*W2 + wo;
    // channel = plane*4 + dy*2 + dx ; one lane-group at a time, so at most one fp32
    // vector pair is live (see the VGPR table in §6).
    GROUP(0, a0.even, b0.even);   // (dy,dx) = (0,0) -> channels 0, 4, 8
    GROUP(1, a0.odd , b0.odd );   //           (0,1) -> channels 1, 5, 9
    GROUP(2, a1.even, b1.even);   //           (1,0) -> channels 2, 6, 10
    GROUP(3, a1.odd , b1.odd );   //           (1,1) -> channels 3, 7, 11

    // Channel 12: uniform plane, depends on no load. Issue it FIRST in the real kernel
    // so it retires while the four global loads are still in flight.
    VSTORE_V((FLOAT_V)kConst, 0, out + 12*planeStride + o);
}
```

Rules encoded above, each earned on this device:

1. **Swizzle / `.even` / `.odd` access only — never `(FLOAT*)&vec`.** A pointer-cast into a vector
   makes it addressable, forces scratch and spills; that mistake doubled ApplyLUT's wide-vector time.
2. **Math in `COMPUTE_FLOAT`, store at `FLOAT`.** In Normal precision that is fp32 math over fp16
   storage — matches the CPU reference closely at zero time cost (but see §6 T5: fp16 math buys
   registers, not speed).
3. **1-D exact grid** (`total = H2*W2/VW`): no rounding waste on either axis, one `div/mod` per item
   (free here), and the only masked lanes are the LWS tail.
4. **`int` offsets** — max index `2*H*W` is far below 2³¹; `long` math would cost for nothing.
5. **Branch-free body.** `H`, `W` even is guaranteed; require `W2 % VW == 0` for the fast kernel and
   build a `VW=2` variant for widths that don't divide — never a tail branch in the hot kernel.
6. `VW ∈ {2,4,8}` only: MNN defines `FLOATn`/`COMPUTE_FLOATn` for n ∈ {2,3,4,8,16}, so `2*VW` must
   stay ≤16 and `VW` must be a legal vector width. `VW=1` would need a scalar special case — skip it.

### 5.4 What the memory system sees

Per wave (64 lanes) per instruction, lane `i` touches `base + i*VW` (stores) or `base + i*2*VW`
(loads) → every stream is **fully contiguous across the wave**. At `VW=4`, fp16:

* loads: 4 streams × 16 B/lane (`dwordx4`) → 4 × 1 KB contiguous per wave
* stores: 13 streams × 8 B/lane (`dwordx2`) → 13 × 512 B contiguous per wave

Both well above the 128 B minimum burst — that is what keeps the 13-stream NCHW output tolerable.
**17 live streams is the main risk** introduced by the extra channels; the cheap lever against it
is a larger LWS (longer contiguous run per stream per workgroup) rather than a larger `VW`, because
LWS costs no registers. See §6 T2.

---

## 6. Tuning sweep (measure — the priors only shorten it)

| # | Axis | Candidates | Prior |
|---|------|-----------|-------|
| T1 | `VW` | 2, **4**, 8 | **VW=4.** VW=2 leaves stores at 4 B/lane — far too thin against 13 write streams. VW=8 doubles store width but the register math below turns hostile. |
| T2 | LWS | 128, **256** | Raised from the 8-channel plan: with 13 write streams, longer contiguous runs per stream matter more, and LWS costs no registers (VW=4, LWS=256 → 2 KB/plane/workgroup). Test 128 as the lower bracket. |
| T3 | Grid | **1-D exact** vs 2-D `(W2/VW, H2)` | 1-D avoids rounding waste; 2-D keeps a workgroup inside one row (marginally better DRAM page locality). Cheap to test both. |
| T4 | Output layout | NCHW (spec) vs `-DOUT_NC4HW4` | Run once. At `C=13` the packed layout pads to 16 → **+14% bytes** to buy 4 write streams instead of 13 (§4). Worth knowing, but NCHW is now the favourite; the question reopens if §2.3 drops the channel back to `C=12`. |
| T5 | Precision | Normal (fp16 store / fp32 math) vs Low (fp16/fp16) | Low **halves the live register count** — a plausible occupancy win. Not an ALU question. High (fp32) is the correctness oracle only. |
| T6 | Convert placement | lazy per-channel (sketch) vs all four inputs converted upfront | Lazy should win on registers; verify once and move on. |
| T7 | Coefficients | kernel args (spec) vs `-D` constants | Expected to be a no-op; one run settles it, then never revisit. |

**The register argument behind the VW prior** — this device is occupancy-sensitive (that is what
decided ApplyLUT), so count VGPRs before sweeping:

| VW | inputs live (fp16) | if all converted to fp32 upfront | verdict |
|----|--------------------|----------------------------------|---------|
| 2  | 4×4 halves = 8 VGPRs | +16 | stores too thin |
| 4  | 4×8 halves = 16 VGPRs | +32 | **sweet spot** |
| 8  | 4×16 halves = 32 VGPRs | **+64 for converted inputs alone** | occupancy risk |

At VW=8 with fp32 math, eager conversion alone approaches the point where waves/SIMD drops — which
is exactly why the sketch converts lazily, and why T5 (fp16 math) may beat the "more accurate" build.

**Report effective GB/s = `5.25*H*W*2 / t`**, not just µs — it makes "how close to the ceiling" readable
at a glance and comparable across shapes.

**Stop rule:** once effective bandwidth is within ~10% of 65–73 GB/s, further kernel tuning is dead
weight. Move to §3.2 (absorbing the producers), which is worth ~2× rather than another 3%.

---

## 7. Host side (`FusedMathS2DBufExecution`)

Mirror `ApplyLUTBufExecution` (`source/backend/opencl/execution/buffer/`):

* Derive from **`CommonExecution`**, push a **single `Unit`** into `mUnits`.
* `onEncode`: validate → build options (`-DVW=`, optional `-DOUT_NC4HW4`) →
  `runtime->buildKernel("math_s2d_buf","math_s2d_buf",opts,precision)` → `setArg` (buffers, shape
  scalars, then the five coefficients) → `globalWorkSize = ROUND_UP(total, LWS)` → `recordKernel`
  for the profiler.
* Env knobs for the sweep (`MNN_S2D_VW`, `MNN_S2D_LWS`, `MNN_S2D_GRID`) — the ApplyLUT pattern turns
  §6 into a shell loop instead of a dozen rebuilds. Keep them; ship the measured defaults.
* Register **BUFFER-only**: `REGISTER_OPENCL_OP_CREATOR(..., OpType_FusedMathS2D, BUFFER)`. Image
  mode then falls back to CPU, which is what we want.
* Creator applies the §4 layout forcing and returns `nullptr` on any unsupported shape.

---

## 8. Files to touch

```
schema/default/MNN.fbs                                    OpType_FusedMathS2D=609 + FusedMathS2DParam
source/shape/ShapeFusedMathS2D.cpp                        [1,1,H,W]x2 -> [1,13,H/2,W/2]
source/backend/cpu/CPUFusedMathS2D.{hpp,cpp}              reference impl (oracle + fallback)
source/backend/opencl/execution/cl/math_s2d_buf.cl        the kernel
source/backend/opencl/execution/cl/math_s2d_buf_mnn_cl.cpp    generated
source/backend/opencl/execution/cl/opencl_source_map.hpp      generated
source/backend/opencl/execution/buffer/FusedMathS2DBufExecution.{hpp,cpp}
express/  _FusedMathS2D(VARP, VARP, coeffs)               lets tests build the graph without a converter
test/op/FusedMathS2DTest.cpp                              CPU + OpenCL, all precisions
test/speed/FusedMathS2DSpeed.cpp                          the sweep harness
tools/converter/.../FusedMathS2DOnnx.cpp (+ make_*.py)    only if a real .mnn must carry it
python tools/script/register.py .                         regenerate codegen glue
```

---

## 9. Validation & benchmark plan

**Correctness first — no timing number is trusted before this passes:**
1. Hand-computed 4×4 case pinning the **channel order** (the classic silent bug).
2. CPU reference vs the spec; then `./run_test.out op/FusedMathS2D 3 1 68` (OpenCL, High/fp32) should
   match to ~1e-6. Then precision `0` (Normal) and `2` (Low) with fp16 tolerances.
   ⚠ argv order: the `68` gpuMode flag in the wrong slot silently runs **image** mode instead.
3. Assert **zero CPU fallback** in the op list — a fallback silently "passes" the test.
4. **Check the op list for `buffer_convert_*` neighbours** (§4). This is the highest-value check in
   the whole plan: two converts cost more than every kernel optimization combined.
5. Shape matrix: `W2` divisible and non-divisible by `VW`; small (8×8) and full frame.

**Performance:**
* Profile build, `gpuMode 68`, **min-of-N kernel time**.
* Because the inputs are already on the GPU there are no host copies in the way, so **graph time is
  meaningful here** (unlike ApplyLUT, where wall clock was dominated by transfers). Report both:
  kernel time for the roofline, graph-segment time for the real-world claim.
* Report effective GB/s against the 65–73 GB/s ceiling.
* Baseline arm: the decomposed express graph from §3.1, same device, same session.
* **Carry a `ctrl` arm re-running the default configuration last.** Long unpinned runs throttle this
  GPU (980→747 MHz observed) and inflate late baselines into phantom wins. If `ctrl` ≠ the first
  run, the sweep is invalid — shorten it and re-run.

**Acceptance:**
* correct on CPU + OpenCL buffer at Normal/High/Low, zero fallback, **no neighbouring convert kernel**;
* ≥ **60 GB/s** effective (≈85% of the device ceiling);
* ≥ 4× the decomposed graph on kernel time, ≥ 5× on graph time.

---

## 10. Risks & known gotchas (all previously hit on this device)

| Risk | Mitigation |
|------|-----------|
| A producer emits NC4HW4 → `buffer_convert_*` per input, +100% traffic | §4 check, in the first working run. Fix by forcing the producer's layout, or by absorbing it (§3.2). |
| NC4HW4 sneaks onto a `C=1` tensor → 4× read traffic | §4 forcing; assert `dimensionFormat` in the test. |
| **fp16 overflow** | fp16 stores clip at 65504. `alpha*A + beta*B` overflows the *store* even though the math ran in fp32, if the coefficients or inputs are large. Know your input range; clamp in-kernel if it is not bounded. `pipeline/scripts/detect_fp16_overflow.py` exists for this. |
| `CommonExecution::onExecute` does not flush the queue | Matters if a host readback follows this op directly — the result can be read before the kernel runs (garbage / cos≈0, looks exactly like a precision bug). Confirm the `commandQueue().flush()` fix is on the branch. |
| Autotune cache collision during the sweep | One cache file shared across kernel variants makes some runs write **no output** and report `cosine nan`. Distinct cache path (or kernel name) per variant. |
| Image mode | Do not enable — allocation failure and device reboot on wide tensors. |
| Throttled baselines | The `ctrl` arm above. |

---

## 11. Still open

* Exact coefficient values / whether `P1` needs more than `delta*B + epsilon` — fills in two inline
  functions and the CPU oracle; nothing else in this design depends on it.
* **What the upstream ops actually are** — decides whether §3.2 (absorbing them, worth ~2×) is
  available. Worth answering before the tuning sweep, not after.

---

## 12. Measured results (branch `feature/fused-math-s2d`, off tag 3.5.0)

Device R3CY905E04M (SM-S942B, Exynos 2600 / Xclipse 960, OpenCL = ANGLE-over-Vulkan),
buffer mode `gpuMode 68`, precision Normal (fp16 storage / fp32 math), frame 1080×1920,
min-of-8 kernel time from the `MNN_GPU_TIME_PROFILE=ON` build.

**Shipped result: 358–363 µs, one dispatch, ~60.5 GB/s aggregate — 4.7× the decomposed graph.**

| | kernel time | note |
|---|---|---|
| fused op (VW=4, autotuned LWS) | **358 µs** | single `FusedMathS2D0` kernel |
| decomposed graph (Scale/Add + SpaceToDepth×3 + Concat) | **1686 µs** | 12 kernels, 648 µs of it Raster |
| **speedup** | **4.71×** | wall-clock 8.97 → 5.85 ms (wall is dominated by the 8 MB upload + 13 MB readback) |

Correctness: CPU, and OpenCL buffer at High / Normal / Low, all pass; shapes 4×4, 8×6
(VW→1), 16×20 (VW→2), 64×64, 270×480. Verified in the profiler that the op is **one kernel
with no `buffer_convert_*` neighbour and no CPU fallback** — the §4 layout forcing took.

### 12.1 Where the time goes — the op is write-bound, not read-bound

Measured by dropping written planes while keeping every load alive (13 → 12 → 9 → 5 → 1):

| planes written | kernel time | Δ |
|---|---|---|
| 13 (shipped) | 360 µs | — |
| 12 (no uniform channel) | 353 µs | const plane costs **8 µs, 2%** |
| 9 (no P2) | 299 µs | the diff planes cost **61 µs, 17%** |
| 5 (P0 + const) | 188 µs | |
| 1 | 131 µs | |

Fitting the slope: **writes ≈ 36.6 µs/Melement (~55 GB/s), reads ≈ 27 µs/Melement (~74 GB/s)**.
Writes are ~1.35× the per-element cost of reads and are **69% of the runtime**. The measured
total matches that two-term model to within noise, which is the evidence that the kernel is at
the memory system's limit rather than at a kernel-shape limit.

This revises §2's single-rate assumption: the 65–73 GB/s figure inherited from ApplyLUT is a
*1:1 read/write* rate. This op writes 1.6 elements for every one it reads, so its ceiling is
lower — ~60 GB/s aggregate, which is what it achieves.

### 12.2 Tuning sweep — what won and what was falsified

| axis | result |
|---|---|
| `VW` = 2 / **4** / 8 | 378 / **358** / 365 µs → VW=4, as predicted |
| LWS: autotuned vs forced 32/64/128/256/512/1024 | auto = 1024 = **360 µs**; every smaller forced size worse (372–396). The §6 T2 prior ("larger LWS, longer per-stream runs") held, and the tuner already finds it |
| 2-D grid (T3) | 371 µs — **worse**, 1-D exact grid stays |
| `restrict` on all three pointers | neutral in time; kept, because it is what legalises issuing the uniform-channel store ahead of the loads |
| uniform store first vs last | 358 vs 365 µs → **first** |
| typed aligned vector access instead of `vloadN`/`vstoreN` | neutral — the compiler was already emitting aligned accesses |
| precision Low (fp16 math) — T5 | 385 µs, **worse** than Normal's 358. The register-pressure argument did not pay; ship Normal |
| **multi-tile channel-major store batching** (TILES=2/4) | 401 / 360 µs — **falsified**. This was the §4/§5.4 "13 concurrent write streams are the cap" hypothesis; batching stores per channel changes nothing, and the plane-scaling above is near-linear in bytes. The limit is raw write bandwidth, not stream count |
| output as final tensor vs as a device-side intermediate | 361 vs 373 µs — the final-output allocation is **not** a slow host-visible path |

`ctrl` re-run of the default at the end of the sweep: 360 µs vs 358 µs at the start — no
thermal drift, the sweep is valid.

### 12.3 What is left

Nothing inside the kernel: it sits on the measured write-bandwidth line. The only remaining
levers are the ones that write fewer bytes or read fewer times, and both are spec decisions:

1. **Drop the uniform channel 12** — 8 µs (2%). Still worth doing for the round-trip reason in
   §2.3 (the consumer also spends ~0.25·H·W reading it back), but smaller than predicted.
2. **Drop the P2 diff planes** — 61 µs (17%), if the consumer can subtract for itself.
3. **Absorb the upstream producers** (§3.2) — unchanged, still the largest available win at
   roughly 1.8× on the pair, and now more attractive: it removes writes, which are the
   expensive side.
