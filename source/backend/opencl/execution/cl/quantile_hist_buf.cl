#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// Experimental: single shared histogram (built once, not once per target,
// unlike the earlier failed radix-select attempt) over the top HIST_BITS
// bits of the same monotonic key used by quantile_buf.cl's exact bisection.
// Bucketing by the monotonic key (not the raw linear value) preserves the
// same log-uniform relative-precision behavior the bisection has -- a plain
// linear-value histogram would badly lose relative precision for values
// near 0. One pass resolves HIST_BITS bits for every target at once; the
// remaining bits are then finished by a few passes of the existing exact
// bisection kernels (quantile_count_buf/quantile_update_buf), reusing
// already-verified code for the refinement tail instead of duplicating it.
#ifndef HIST_BITS
#define HIST_BITS 12
#endif
#define HIST_BUCKETS (1 << HIST_BITS)

inline uint quantileHistFloatToKey(float x) {
    uint bits = as_uint(x);
    uint mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

inline uint quantileHistBucket(uint key) {
    return key >> (32 - HIST_BITS);
}

__kernel void quantile_hist_init_buf(__global uint *hist) {
    const int i = get_global_id(0);
    if (i < HIST_BUCKETS) {
        hist[i] = 0u;
    }
}

__kernel void quantile_hist_build_buf(
    __global const FLOAT *input,
    __global uint *hist,
    __private const int n) {
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    __local uint localHist[HIST_BUCKETS];
    for (int i = lid; i < HIST_BUCKETS; i += lsize) {
        localHist[i] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = gid; i < n; i += gsize) {
        uint key = quantileHistFloatToKey((float)input[i]);
        atomic_inc(&localHist[quantileHistBucket(key)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < HIST_BUCKETS; i += lsize) {
        if (localHist[i] != 0u) {
            atomic_add(&hist[i], localHist[i]);
        }
    }
}

// One work-item per target scans the (small, shared) histogram to find the
// bucket containing its target rank, and seeds loKey/hiKey to that bucket's
// key range for the existing bisection kernels to refine further.
__kernel void quantile_hist_scan_buf(
    __global const uint *hist,
    __global uint *loKey,
    __global uint *hiKey,
    __global uint *countOut,
    __global const int *targetRank,
    __private const int numTargets) {
    const int t = get_global_id(0);
    if (t >= numTargets) {
        return;
    }
    int rank = targetRank[t];
    uint cum = 0u;
    int chosen = HIST_BUCKETS - 1;
    for (int b = 0; b < HIST_BUCKETS; ++b) {
        uint c = hist[b];
        if ((int)(cum + c) >= rank + 1) {
            chosen = b;
            break;
        }
        cum += c;
    }
    uint lo = ((uint)chosen) << (32 - HIST_BITS);
    uint hi = (chosen == HIST_BUCKETS - 1) ? 0xFFFFFFFFu : ((((uint)chosen + 1) << (32 - HIST_BITS)) - 1u);
    loKey[t] = lo;
    hiKey[t] = hi;
    countOut[t] = 0u;
}
