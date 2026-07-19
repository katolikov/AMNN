#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// Opt-in fast path (QuantileParam.assumeUint8Source): the caller asserts
// every input element is already exactly round(level)/255 for some uint8
// level in [0,255] (e.g. a uint8 image plane normalized to [0,1]). Under
// that assumption there are only 256 possible distinct values total, so a
// single EXACT shared histogram (not per-target, same sharing idea as
// quantile_hist_buf.cl) resolves every quantile level directly -- no
// bisection/refinement pass needed at all, unlike the general-input path.
// Must not be used for inputs outside [0,1] or not uint8-sourced: values
// get clamped/rounded to the nearest of the 256 levels, which is silently
// wrong for anything else.
#define U8_BUCKETS 256

inline int quantileU8Level(float x) {
    int level = (int)(x * 255.0f + 0.5f);
    level = max(0, min(255, level));
    return level;
}

__kernel void quantile_u8_hist_init_buf(__global uint *hist) {
    const int i = get_global_id(0);
    if (i < U8_BUCKETS) {
        hist[i] = 0u;
    }
}

__kernel void quantile_u8_hist_build_buf(
    __global const FLOAT *input,
    __global uint *hist,
    __private const int n) {
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    __local uint localHist[U8_BUCKETS];
    for (int i = lid; i < U8_BUCKETS; i += lsize) {
        localHist[i] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = gid; i < n; i += gsize) {
        atomic_inc(&localHist[quantileU8Level((float)input[i])]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < U8_BUCKETS; i += lsize) {
        if (localHist[i] != 0u) {
            atomic_add(&hist[i], localHist[i]);
        }
    }
}

// One work-item per quantile level: scans the shared 256-bucket histogram
// once, finding both the floor- and ceil-rank buckets in the same pass, and
// writes the interpolated result directly -- no separate refinement stage.
__kernel void quantile_u8_finalize_buf(
    __global const uint *hist,
    __global const int *loRank,
    __global const int *hiRank,
    __global const float *frac,
    __global FLOAT *output,
    __private const int numQ) {
    const int i = get_global_id(0);
    if (i >= numQ) {
        return;
    }
    const int loR = loRank[i];
    const int hiR = hiRank[i];

    uint cum = 0u;
    int loLevel = U8_BUCKETS - 1;
    int hiLevel = U8_BUCKETS - 1;
    bool loFound = false;
    for (int b = 0; b < U8_BUCKETS; ++b) {
        uint next = cum + hist[b];
        if (!loFound && (int)next >= loR + 1) {
            loLevel = b;
            loFound = true;
        }
        if ((int)next >= hiR + 1) {
            hiLevel = b;
            break;
        }
        cum = next;
    }

    float lower = loLevel / 255.0f;
    float upper = hiLevel / 255.0f;
    output[i] = (FLOAT)(lower + frac[i] * (upper - lower));
}
