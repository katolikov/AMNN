#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// FusedMathS2D: per-pixel affine math on two single-plane inputs fused with
// space-to-depth(block=2). [1,1,H,W] x2 -> [1,13,H/2,W/2], all dense NCHW.
//
//   P0 = alpha*A + beta*B + gamma   -> channels 0..3
//   P1 = delta*B + epsilon          -> channels 4..7
//   P2 = P0 - P1                    -> channels 8..11
//   channel 12                      = kConst (uniform plane)
//
// The op reads 2*H*W elements and writes 13*(H/2)*(W/2) = 3.25*H*W, which is the
// information-theoretic floor, in one dispatch -- there is no algorithmic win
// available, only bandwidth. Measured on Xclipse 960 (ANGLE-over-Vulkan) the
// kernel is write-bound: reads run around 74 GB/s, writes around 55 GB/s, and
// the stores are ~69% of the runtime.
//
// Shape of the kernel and why:
//   * dense NCHW everywhere -- NC4HW4 would pad the C=1 inputs to four channels,
//     i.e. 4x the read traffic;
//   * one work-item owns VW whole output pixels, i.e. all 13 channels, so A and B
//     are read exactly once and feed every output;
//   * the 2x2 blocks partition the input, so nothing is shared between
//     work-items: no local memory, no barriers, no subgroup ops (on this stack
//     subgroup shuffles lower to ds_bpermute and lose to plain cached loads);
//   * space-to-depth is a free register swizzle -- in a 2*VW-wide row vector the
//     even lanes are dx=0 and the odd lanes are dx=1, and dy picks the row load;
//   * stores are emitted per (dy,dx) lane group so a single COMPUTE_FLOAT pair is
//     live at a time and P2 reuses P0/P1 out of registers (one subtract).
// Vector lanes are only ever touched by swizzle: a (FLOAT*)&vec cast would force
// addressable scratch and spill.
//
// Measured and rejected on device (1080x1920, fp16): 2-D grid (371 vs 358 us),
// forced local sizes 32..512 (372-396), multi-tile channel-major store batching
// to shorten the live write-stream mix (360-401), typed aligned vector accesses
// instead of vloadN/vstoreN (neutral), fp16 math instead of fp32 (385).

#ifndef VW
#define VW 4        // output pixels per work-item
#endif
#ifndef VW2
#define VW2 8       // = 2*VW, input elements per row per work-item
#endif

#define PASTE_(a, b) a##b
#define PASTE(a, b)  PASTE_(a, b)

#define FLOAT_IN  PASTE(FLOAT, VW2)
#define VLOAD_IN  PASTE(vload, VW2)

#if VW == 1
#define FLOAT_V     FLOAT
#define CFLOAT_V    COMPUTE_FLOAT
#define TO_CFLOAT_V CONVERT_COMPUTE_FLOAT
#define TO_FLOAT_V  CONVERT_FLOAT
#define VSTORE_V(v, off, ptr) (ptr)[off] = (v)
#else
#define FLOAT_V     PASTE(FLOAT, VW)
#define CFLOAT_V    PASTE(COMPUTE_FLOAT, VW)
#define TO_CFLOAT_V PASTE(CONVERT_COMPUTE_FLOAT, VW)
#define TO_FLOAT_V  PASTE(CONVERT_FLOAT, VW)
#define VSTORE_V    PASTE(vstore, VW)
#endif

#define STORE_OUT(ch, v) VSTORE_V(TO_FLOAT_V(v), 0, output + (ch) * planeStride + o)

// One (dy,dx) lane group -> its three channels.
#define GROUP(k, av, bv)                                            \
    {                                                               \
        CFLOAT_V _a  = TO_CFLOAT_V(av);                             \
        CFLOAT_V _b  = TO_CFLOAT_V(bv);                             \
        CFLOAT_V _p0 = mad(vAlpha, _a, mad(vBeta, _b, vGamma));     \
        CFLOAT_V _p1 = mad(vDelta, _b, vEpsilon);                   \
        STORE_OUT(0 + (k), _p0);                                    \
        STORE_OUT(4 + (k), _p1);                                    \
        STORE_OUT(8 + (k), _p0 - _p1);                              \
    }

__kernel void fused_math_s2d_buf(
    __global const FLOAT* restrict inputA,
    __global const FLOAT* restrict inputB,
    __global FLOAT* restrict output,
    __private const int inWidth,      // = 2 * outWidth
    __private const int outWidth,
    __private const int outWidthV,    // = outWidth / VW
    __private const int total,        // = outHeight * outWidthV
    __private const int planeStride,  // = outHeight * outWidth
    __private const float alpha,
    __private const float beta,
    __private const float gamma,
    __private const float delta,
    __private const float epsilon,
    __private const float kConst) {

    const int gid = get_global_id(0);
    if (gid >= total) {
        return;             // masks only the local-size round-up
    }

    const int ho = gid / outWidthV;
    const int wo = (gid - ho * outWidthV) * VW;
    const int o  = ho * outWidth + wo;

    // Uniform channel: depends on no load, so issue it while the four global
    // loads below are still in flight. `restrict` is what lets the compiler keep
    // it there -- without it this store could alias the inputs.
    VSTORE_V((FLOAT_V)((FLOAT)kConst), 0, output + 12 * planeStride + o);

    const int t = (ho * 2) * inWidth + wo * 2;
    const FLOAT_IN a0 = VLOAD_IN(0, inputA + t);
    const FLOAT_IN a1 = VLOAD_IN(0, inputA + t + inWidth);
    const FLOAT_IN b0 = VLOAD_IN(0, inputB + t);
    const FLOAT_IN b1 = VLOAD_IN(0, inputB + t + inWidth);

    const CFLOAT_V vAlpha   = (CFLOAT_V)alpha;
    const CFLOAT_V vBeta    = (CFLOAT_V)beta;
    const CFLOAT_V vGamma   = (CFLOAT_V)gamma;
    const CFLOAT_V vDelta   = (CFLOAT_V)delta;
    const CFLOAT_V vEpsilon = (CFLOAT_V)epsilon;

    GROUP(0, a0.even, b0.even)   // (dy,dx) = (0,0) -> channels 0, 4,  8
    GROUP(1, a0.odd,  b0.odd)    //           (0,1) -> channels 1, 5,  9
    GROUP(2, a1.even, b1.even)   //           (1,0) -> channels 2, 6, 10
    GROUP(3, a1.odd,  b1.odd)    //           (1,1) -> channels 3, 7, 11
}
