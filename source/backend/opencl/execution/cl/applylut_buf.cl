#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// ApplyLUT: per-element 1-D LUT remap with linear interpolation. A memory-bound
// element-wise op over a dense single-plane float buffer (input 0), remapped
// through a small runtime LUT (input 1, length N in [8,256]) into an identically
// shaped output. Optimized for bandwidth:
//   * dense flat NCHW buffer (no NC4HW4 channel padding) set up host-side,
//   * each work-item streams VEC contiguous elements with one vectorized
//     load + one vectorized store (coalesced), and
//   * the LUT is staged once per workgroup into local memory, so every (data-
//     dependent) gather is on-chip and broadcasts for neighbouring pixels.
// Interp math runs in COMPUTE_FLOAT (fp32 even in fp16-buffer mode) to match the
// CPU reference; only the streamed pixels are stored back at FLOAT precision.
//
// Clamp-index-only semantics:
//   pos  = x * (N-1)
//   i0   = clamp((int)floor(pos), 0, N-1);  i1 = min(i0+1, N-1)
//   frac = pos - (COMPUTE_FLOAT)i0
//   out  = lut[i0] + frac * (lut[i1] - lut[i0])
// In-range x in [0,1] is standard lerp; out-of-range x keeps the clamped
// endpoint's slope (frac may fall outside [0,1)).

#ifndef VEC
#define VEC 4
#endif
#ifndef MAX_LUT
#define MAX_LUT 256
#endif

#define PASTE_(a, b) a##b
#define PASTE(a, b)  PASTE_(a, b)
#define FLOATN   PASTE(FLOAT, VEC)
#define VLOADN   PASTE(vload, VEC)
#define VSTOREN  PASTE(vstore, VEC)

// LUT storage: by default staged into local memory once per workgroup; with
// -DLUT_GLOBAL the tiny table is read straight from (cached) global memory,
// dropping the local buffer and the barrier. Selected host-side per device.
#ifdef LUT_GLOBAL
#define LUT_PTR       __global const FLOAT*
#define LUT_READ(p, i) ((COMPUTE_FLOAT)(p)[i])
#else
#define LUT_PTR       __local const COMPUTE_FLOAT*
#define LUT_READ(p, i) ((p)[i])
#endif

// One vector lane, kept in registers (swizzle access -- no addressable scratch,
// which a pointer-cast into the vector would otherwise force and spill).
#define LANE(v, s) (FLOAT)applylut_one((COMPUTE_FLOAT)(v).s, llut, maxIdx, scaleF)

// One scalar pixel: normalized value -> interpolated LUT output.
inline COMPUTE_FLOAT applylut_one(COMPUTE_FLOAT x, LUT_PTR llut,
                                  const int maxIdx, const COMPUTE_FLOAT scaleF) {
    const COMPUTE_FLOAT pos = x * scaleF;
    int i0 = (int)floor(pos);
    i0 = clamp(i0, 0, maxIdx);
    const int i1 = min(i0 + 1, maxIdx);
    const COMPUTE_FLOAT frac = pos - (COMPUTE_FLOAT)i0;
    const COMPUTE_FLOAT lo = LUT_READ(llut, i0);
    const COMPUTE_FLOAT hi = LUT_READ(llut, i1);
    return mad(frac, hi - lo, lo);
}

__kernel void applylut_buf(
    __global const FLOAT* input,
    __global const FLOAT* lut,
    __global FLOAT* output,
    __private const int n,
    __private const int lutSize) {

#ifdef LUT_GLOBAL
    // Read the small table straight from cached global memory (no barrier).
    LUT_PTR llut = lut;
#else
    // Cooperative LUT stage into local memory (every work-item participates
    // before the barrier -- no early return above it).
    __local COMPUTE_FLOAT llut[MAX_LUT];
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    for (int i = lid; i < lutSize; i += lsize) {
        llut[i] = (COMPUTE_FLOAT)lut[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    const int gid  = get_global_id(0);
    const int base = gid * VEC;
    if (base >= n) {
        return;
    }
    const int maxIdx = lutSize - 1;
    const COMPUTE_FLOAT scaleF = (COMPUTE_FLOAT)maxIdx;

    if (base + VEC <= n) {
        // Fast path: full vector, coalesced load/store. vin/vout stay in
        // registers; each lane is remapped via swizzle access.
        FLOATN vin = VLOADN(gid, input);
        FLOATN vout;
#if VEC == 2
        vout = (FLOATN)(LANE(vin, s0), LANE(vin, s1));
#elif VEC == 4
        vout = (FLOATN)(LANE(vin, s0), LANE(vin, s1), LANE(vin, s2), LANE(vin, s3));
#elif VEC == 8
        vout = (FLOATN)(LANE(vin, s0), LANE(vin, s1), LANE(vin, s2), LANE(vin, s3),
                        LANE(vin, s4), LANE(vin, s5), LANE(vin, s6), LANE(vin, s7));
#elif VEC == 16
        vout = (FLOATN)(LANE(vin, s0), LANE(vin, s1), LANE(vin, s2), LANE(vin, s3),
                        LANE(vin, s4), LANE(vin, s5), LANE(vin, s6), LANE(vin, s7),
                        LANE(vin, s8), LANE(vin, s9), LANE(vin, sa), LANE(vin, sb),
                        LANE(vin, sc), LANE(vin, sd), LANE(vin, se), LANE(vin, sf));
#endif
        VSTOREN(vout, gid, output);
    } else {
        // Scalar tail: dynamic sizes not divisible by VEC.
        for (int c = 0; base + c < n; ++c) {
            output[base + c] = (FLOAT)applylut_one((COMPUTE_FLOAT)input[base + c], llut, maxIdx, scaleF);
        }
    }
}
