#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// DualRangeHist: single-pass dual masked range-histogram over two float frames
// A, B (values ~[0,1]) plus an optional per-element base validity mask. Per
// pixel it computes a SHARED keep bit
//   keep = base & (low < A < high) & (low < B < high)   (raw values, exclusive)
// and, when kept, increments histA[rint(A*(binNum-1))] and histB[rint(B*(binNum-1))]
// (round-half-to-even, matching torch.round, done inside the op). Optionally it
// also accumulates validCount = sum(keep) == sum(histA) == sum(histB).
//
// Structure mirrors bincount_buf.cl's register-histogram fast path, extended to
// two histograms: each work-item keeps two private register histograms
// (privA/privB[BIN_NUM]) plus a private counter, avoids a data-dependent array
// index by comparing the rounded bin against every bin unrolled
// (priv[b] += (bin==b)*keep), and merges across work-items with a fused
// single-tree log reduction in local memory (run once per histogram, reusing one
// BIN_NUM*LOCAL_SIZE buffer) plus one global atomic per bin per workgroup. The
// mask & discretized frames are never materialized.
#define LOCAL_SIZE 256
#ifndef BIN_NUM
#define BIN_NUM 16
#endif

// Frames are logically float; read through the precision-dependent FLOAT macro
// (half in fp16 buffer mode, float in high precision). The range test and the
// bin math are done at FLOAT precision (NOT force-upcast to float) so that in
// fp16 buffer mode they run in half -- bit-exact to a PyTorch reference that
// keeps fp16 tensors: torch computes `round(A_f16 * (binNum-1))` and the
// low/high compares in fp16 (the python scalars demote to the tensor dtype).
// In high precision FLOAT == float, so the fp32 behaviour is unchanged.
#define IN_T   FLOAT
#define IN_T4  FLOAT4

// Optional base mask (third input): int32 or float (fp16 buffer). A component is
// gated by (m != 0).
#ifdef HAS_BASE
#ifdef BASE_FLOAT
#define BASE_T  FLOAT
#define BASE_T4 FLOAT4
#else
#define BASE_T  int
#define BASE_T4 int4
#endif
#endif

// Round a non-negative value to the nearest integer, ties-to-even. Matches
// torch.round / std::rint (default rounding) exactly, and -- unlike the GPU's
// built-in rint on some devices under -cl-mad-enable -- is deterministic at the
// k+0.5 ties, so the bin decision is reproducible across devices.
static inline int roundHalfEven(float x) {
    int   fl   = (int)floor(x);
    float frac = x - (float)fl;
    if (frac < 0.5f) return fl;
    if (frac > 0.5f) return fl + 1;
    return (fl & 1) ? (fl + 1) : fl;   // exact .5 -> nearest even
}

// keep bit for one component; bin index via round-half-to-even. `a`,`b` are the
// raw frame values; `mkeep` folds in the optional base mask (1 when no base).
// Compares are at FLOAT precision (half in fp16 mode) against fp16-rounded
// low/high, matching torch's fp16 range test. For the bin, torch evaluates
// `A_f16 * (binNum-1)` in fp16 THEN rounds; so in fp16 mode we round the product
// to half (RTE) before roundHalfEven -- otherwise the un-rounded product (e.g.
// 4.5007 instead of 4.5) would round up and disagree with torch at the ties.
#define COMPUTE_KEEP(a, b, mkeep) \
    (((FLOAT)(a) > lowF) & ((FLOAT)(a) < highF) & ((FLOAT)(b) > lowF) & ((FLOAT)(b) < highF) & (mkeep))
#ifdef MNN_SUPPORT_FP16
#define TO_BIN(v) roundHalfEven((float)convert_half_rte((float)(v) * (float)scaleF))
#else
#define TO_BIN(v) roundHalfEven((float)(v) * (float)scaleF)
#endif

__kernel void dualrangehist_init_buf(
    __global int *histA,
    __global int *histB,
#ifdef EMIT_VALIDCOUNT
    __global int *validCount,
#endif
    __private const int binNum) {
    const int idx = get_global_id(0);
    if (idx < binNum) {
        histA[idx] = 0;
    } else if (idx < 2 * binNum) {
        histB[idx - binNum] = 0;
    }
#ifdef EMIT_VALIDCOUNT
    else if (idx == 2 * binNum) {
        validCount[0] = 0;
    }
#endif
}

// Cross-work-item merge shared by both count kernels. Assumes BIN_NUM*LOCAL_SIZE
// <= 4096 (guaranteed: the register path is only used for small BIN_NUM). Reduces
// privA into histA, then reuses the same local buffer for privB into histB, and
// (optionally) the counter into validCount.
#define MERGE_HISTOGRAMS()                                                     \
    __local int reduceBuf[BIN_NUM * LOCAL_SIZE];                               \
    for (int b = 0; b < BIN_NUM; ++b) {                                        \
        reduceBuf[b * LOCAL_SIZE + lid] = privA[b];                           \
    }                                                                          \
    barrier(CLK_LOCAL_MEM_FENCE);                                              \
    for (int off = LOCAL_SIZE >> 1; off > 0; off >>= 1) {                      \
        if (lid < off) {                                                       \
            for (int b = 0; b < BIN_NUM; ++b) {                               \
                reduceBuf[b * LOCAL_SIZE + lid] += reduceBuf[b * LOCAL_SIZE + lid + off]; \
            }                                                                  \
        }                                                                      \
        barrier(CLK_LOCAL_MEM_FENCE);                                          \
    }                                                                          \
    if (lid == 0) {                                                            \
        for (int b = 0; b < BIN_NUM; ++b) {                                   \
            atomic_add(&histA[b], reduceBuf[b * LOCAL_SIZE]);                 \
        }                                                                      \
    }                                                                          \
    barrier(CLK_LOCAL_MEM_FENCE);                                              \
    for (int b = 0; b < BIN_NUM; ++b) {                                        \
        reduceBuf[b * LOCAL_SIZE + lid] = privB[b];                           \
    }                                                                          \
    barrier(CLK_LOCAL_MEM_FENCE);                                              \
    for (int off = LOCAL_SIZE >> 1; off > 0; off >>= 1) {                      \
        if (lid < off) {                                                       \
            for (int b = 0; b < BIN_NUM; ++b) {                               \
                reduceBuf[b * LOCAL_SIZE + lid] += reduceBuf[b * LOCAL_SIZE + lid + off]; \
            }                                                                  \
        }                                                                      \
        barrier(CLK_LOCAL_MEM_FENCE);                                          \
    }                                                                          \
    if (lid == 0) {                                                            \
        for (int b = 0; b < BIN_NUM; ++b) {                                   \
            atomic_add(&histB[b], reduceBuf[b * LOCAL_SIZE]);                 \
        }                                                                      \
    }

#ifdef EMIT_VALIDCOUNT
#define MERGE_COUNTER()                                                        \
    barrier(CLK_LOCAL_MEM_FENCE);                                              \
    reduceBuf[lid] = counter;                                                  \
    barrier(CLK_LOCAL_MEM_FENCE);                                              \
    for (int off = LOCAL_SIZE >> 1; off > 0; off >>= 1) {                      \
        if (lid < off) {                                                       \
            reduceBuf[lid] += reduceBuf[lid + off];                           \
        }                                                                      \
        barrier(CLK_LOCAL_MEM_FENCE);                                          \
    }                                                                          \
    if (lid == 0) {                                                            \
        atomic_add(&validCount[0], reduceBuf[0]);                             \
    }
#else
#define MERGE_COUNTER()
#endif

// Contiguous fast path (stride == 1): vectorized 4-wide loads over A, B, base.
__kernel void dualrangehist_count_buf(
    __global const IN_T *A,
    __global const IN_T *B,
#ifdef HAS_BASE
    __global const BASE_T *base,
#endif
    __global int *histA,
    __global int *histB,
#ifdef EMIT_VALIDCOUNT
    __global int *validCount,
#endif
    __private const int n,
    __private const float low,
    __private const float high,
    __private const int binNum) {
    const int gid   = get_global_id(0);
    const int gsize = get_global_size(0);
    const int lid   = get_local_id(0);
    const FLOAT scaleF = (FLOAT)(binNum - 1);
    const FLOAT lowF   = (FLOAT)low;
    const FLOAT highF  = (FLOAT)high;

    int privA[BIN_NUM];
    int privB[BIN_NUM];
    for (int b = 0; b < BIN_NUM; ++b) {
        privA[b] = 0;
        privB[b] = 0;
    }
    int counter = 0;

    const int n4 = n >> 2;
    for (int i = gid; i < n4; i += gsize) {
        IN_T4 va = vload4(i, A);
        IN_T4 vb = vload4(i, B);
#ifdef HAS_BASE
        BASE_T4 vm = vload4(i, base);
        int kx = COMPUTE_KEEP(va.x, vb.x, (vm.x != 0));
        int ky = COMPUTE_KEEP(va.y, vb.y, (vm.y != 0));
        int kz = COMPUTE_KEEP(va.z, vb.z, (vm.z != 0));
        int kw = COMPUTE_KEEP(va.w, vb.w, (vm.w != 0));
#else
        int kx = COMPUTE_KEEP(va.x, vb.x, 1);
        int ky = COMPUTE_KEEP(va.y, vb.y, 1);
        int kz = COMPUTE_KEEP(va.z, vb.z, 1);
        int kw = COMPUTE_KEEP(va.w, vb.w, 1);
#endif
        int ax = TO_BIN(va.x), ay = TO_BIN(va.y), az = TO_BIN(va.z), aw = TO_BIN(va.w);
        int bx = TO_BIN(vb.x), by = TO_BIN(vb.y), bz = TO_BIN(vb.z), bw = TO_BIN(vb.w);
        for (int b = 0; b < BIN_NUM; ++b) {
            privA[b] += (ax == b) * kx + (ay == b) * ky + (az == b) * kz + (aw == b) * kw;
            privB[b] += (bx == b) * kx + (by == b) * ky + (bz == b) * kz + (bw == b) * kw;
        }
        counter += kx + ky + kz + kw;
    }
    // Tail: up to 3 trailing elements when n is not a multiple of 4.
    for (int i = (n4 << 2) + gid; i < n; i += gsize) {
        float a = (float)A[i];
        float b = (float)B[i];
#ifdef HAS_BASE
        int keep = COMPUTE_KEEP(a, b, (base[i] != 0));
#else
        int keep = COMPUTE_KEEP(a, b, 1);
#endif
        int ba = TO_BIN(a), bb = TO_BIN(b);
        for (int c = 0; c < BIN_NUM; ++c) {
            privA[c] += (ba == c) * keep;
            privB[c] += (bb == c) * keep;
        }
        counter += keep;
    }

    MERGE_HISTOGRAMS();
    MERGE_COUNTER();
}

// Downsampled path (stride > 1): strided gather over the last two dims,
// A[..,::s,::s] (and B, base). batch == 1, so nSampled == Hs*Ws.
__kernel void dualrangehist_sample_buf(
    __global const IN_T *A,
    __global const IN_T *B,
#ifdef HAS_BASE
    __global const BASE_T *base,
#endif
    __global int *histA,
    __global int *histB,
#ifdef EMIT_VALIDCOUNT
    __global int *validCount,
#endif
    __private const int nSampled,
    __private const int stride,
    __private const int W,
    __private const int Ws,
    __private const float low,
    __private const float high,
    __private const int binNum) {
    const int gid   = get_global_id(0);
    const int gsize = get_global_size(0);
    const int lid   = get_local_id(0);
    const FLOAT scaleF = (FLOAT)(binNum - 1);
    const FLOAT lowF   = (FLOAT)low;
    const FLOAT highF  = (FLOAT)high;

    int privA[BIN_NUM];
    int privB[BIN_NUM];
    for (int b = 0; b < BIN_NUM; ++b) {
        privA[b] = 0;
        privB[b] = 0;
    }
    int counter = 0;

    for (int k = gid; k < nSampled; k += gsize) {
        const int ph = k / Ws;
        const int pw = k - ph * Ws;
        const int flat = (ph * stride) * W + pw * stride;
        float a = (float)A[flat];
        float b = (float)B[flat];
#ifdef HAS_BASE
        int keep = COMPUTE_KEEP(a, b, (base[flat] != 0));
#else
        int keep = COMPUTE_KEEP(a, b, 1);
#endif
        int ba = TO_BIN(a), bb = TO_BIN(b);
        for (int c = 0; c < BIN_NUM; ++c) {
            privA[c] += (ba == c) * keep;
            privB[c] += (bb == c) * keep;
        }
        counter += keep;
    }

    MERGE_HISTOGRAMS();
    MERGE_COUNTER();
}
