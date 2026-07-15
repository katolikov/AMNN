#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// Exact order-statistic selection for up to MAX_TARGETS ranks, computed by
// binary-searching the monotonic (sign-flipped) uint32 encoding of each
// float32 element: for float x, key = as_uint(x) XOR (sign ? 0xFFFFFFFF :
// 0x80000000) is order-preserving over the full float range, so counting how
// many keys are less-or-equal to a midpoint lets us binary-search directly
// for the exact bit pattern of the element at a given rank in 32 passes,
// without ever sorting/gathering the (up to millions-of-elements) array.
// This avoids TopKV2's k<=1024 buffer-kernel cap entirely.
#define LOCAL_SIZE 256
#ifndef MAX_TARGETS
#define MAX_TARGETS 16
#endif

inline uint quantileFloatToKey(float x) {
    uint bits = as_uint(x);
    uint mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

inline float quantileKeyToFloat(uint key) {
    uint bits = (key & 0x80000000u) ? (key ^ 0x80000000u) : (key ^ 0xFFFFFFFFu);
    return as_float(bits);
}

// Resets bisection state at the start of every onExecute (loKey/hiKey brackets
// and countOut are re-initialized per inference call; targetRank/frac are
// shape-derived constants uploaded once in onResize, not touched here).
__kernel void quantile_init_buf(
    __global uint *loKey,
    __global uint *hiKey,
    __global uint *countOut,
    __private const int numTargets) {
    const int t = get_global_id(0);
    if (t >= numTargets) {
        return;
    }
    loKey[t] = 0u;
    hiKey[t] = 0xFFFFFFFFu;
    countOut[t] = 0u;
}

// One pass of the bisection: for each of `numTargets` independent target
// ranks, count how many input elements have key <= mid(target), where
// mid(target) is derived from the target's current [loKey, hiKey] bracket.
// countOut must be zero before this dispatch (quantile_update_buf resets it
// at the end of the previous iteration; the host clears it before pass 0).
__kernel void quantile_count_buf(
    __global const FLOAT *input,
    __global const uint *loKey,
    __global const uint *hiKey,
    __global uint *countOut,
    __private const int n,
    __private const int numTargets) {
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
    const int lid = get_local_id(0);

    __local uint reduceBuf[LOCAL_SIZE];

    uint mid[MAX_TARGETS];
    for (int t = 0; t < numTargets; ++t) {
        uint lo = loKey[t];
        uint hi = hiKey[t];
        mid[t] = lo + ((hi - lo) >> 1);
    }

    uint priv[MAX_TARGETS];
    for (int t = 0; t < numTargets; ++t) {
        priv[t] = 0;
    }

    for (int i = gid; i < n; i += gsize) {
        uint key = quantileFloatToKey((float)input[i]);
        for (int t = 0; t < numTargets; ++t) {
            priv[t] += (key <= mid[t]) ? 1u : 0u;
        }
    }

    for (int t = 0; t < numTargets; ++t) {
        reduceBuf[lid] = priv[t];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = LOCAL_SIZE >> 1; off > 0; off >>= 1) {
            if (lid < off) {
                reduceBuf[lid] += reduceBuf[lid + off];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            atomic_add(&countOut[t], reduceBuf[0]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Narrows each target's [loKey, hiKey] bracket by one bit using the count
// produced by quantile_count_buf, and resets countOut for the next pass.
__kernel void quantile_update_buf(
    __global uint *loKey,
    __global uint *hiKey,
    __global uint *countOut,
    __global const int *targetRank,
    __private const int numTargets) {
    const int t = get_global_id(0);
    if (t >= numTargets) {
        return;
    }
    uint lo = loKey[t];
    uint hi = hiKey[t];
    if (lo < hi) {
        uint mid = lo + ((hi - lo) >> 1);
        uint cnt = countOut[t];
        if ((int)cnt >= targetRank[t] + 1) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
        loKey[t] = lo;
        hiKey[t] = hi;
    }
    countOut[t] = 0;
}

// After convergence loKey[2*i] and loKey[2*i+1] hold the exact keys of the
// floor/ceil order statistics for quantile level i; blend them with the
// (compile-time-known) interpolation weight.
__kernel void quantile_finalize_buf(
    __global const uint *loKey,
    __global const float *frac,
    __global FLOAT *output,
    __private const int numQ) {
    const int i = get_global_id(0);
    if (i >= numQ) {
        return;
    }
    float lower = quantileKeyToFloat(loKey[2 * i]);
    float upper = quantileKeyToFloat(loKey[2 * i + 1]);
    output[i] = (FLOAT)(lower + frac[i] * (upper - lower));
}
