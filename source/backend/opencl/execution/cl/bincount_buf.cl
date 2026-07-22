#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// BinCount: count occurrences of each integer value of a flattened int32 input
// into a fixed number of bins (BIN_NUM). Values outside [0, BIN_NUM) are
// dropped. Output is int32 of length BIN_NUM.
//
// The fast path keeps a per-work-item private histogram in registers. A
// data-dependent array index (priv[input[i]]++) would spill that array out of
// registers into (slow) private memory, so instead each element is compared
// against every bin and the matching counter conditionally incremented
// (priv[b] += (v == b)). With BIN_NUM a small compile-time constant and the
// loop unrolled, all counters stay in registers with zero atomic contention --
// the same register-resident, contention-free structure that makes the
// Quantile count kernel fast. Cross-work-item merge is a log-tree reduction in
// local memory plus a single global atomic per bin per workgroup, so the total
// number of global atomics is only workGroups * BIN_NUM (vs. one per input
// element in the naive baseline below).
#define LOCAL_SIZE 256
#ifndef BIN_NUM
#define BIN_NUM 16
#endif

// Input element type. Integer inputs are read as int32 directly. Float inputs
// are read through the precision-dependent FLOAT macro (half in fp16 buffer
// mode, float in high precision) -- so a logically-float input whose device
// buffer is stored as half is supported -- and truncated to an int bin index
// (matching the CPU reference's static_cast<int>). Out-of-[0,BIN_NUM) indices
// match no bin in the compare below and are naturally dropped.
#ifdef BINCOUNT_IN_FLOAT
#define IN_T   FLOAT
#define IN_T4  FLOAT4
#define TO_BIN(x) ((int)(x))
#else
#define IN_T   int
#define IN_T4  int4
#define TO_BIN(x) (x)
#endif

// Optional binary mask (second input): elements with mask != 0 are counted,
// others dropped, output stays int32 counts. Mask may be int32 or float
// (fp16 buffer); a component's contribution is gated by (m != 0).
#ifdef BINCOUNT_MASK
#ifdef BINCOUNT_MASK_FLOAT
#define MASK_T  FLOAT
#define MASK_T4 FLOAT4
#else
#define MASK_T  int
#define MASK_T4 int4
#endif
#endif

__kernel void bincount_init_buf(
    __global int *output,
    __private const int binNum) {
    const int b = get_global_id(0);
    if (b >= binNum) {
        return;
    }
    output[b] = 0;
}

__kernel void bincount_count_buf(
    __global const IN_T *input,
#ifdef BINCOUNT_MASK
    __global const MASK_T *mask,
#endif
    __global int *output,
    __private const int n,
    __private const int binNum) {
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
    const int lid = get_local_id(0);

    int priv[BIN_NUM];
    for (int b = 0; b < BIN_NUM; ++b) {
        priv[b] = 0;
    }

    // Vectorized loads: one 128-bit (int4) / 64-bit (half4) transaction per 4
    // elements improves memory throughput. Scalar-component comparisons each
    // return 0/1 in OpenCL C, so summing them gives this group's contribution
    // to bin b. With a mask, each component is additionally gated by (m != 0).
    const int n4 = n >> 2;
    for (int i = gid; i < n4; i += gsize) {
        IN_T4 v = vload4(i, input);
        int vx = TO_BIN(v.x), vy = TO_BIN(v.y), vz = TO_BIN(v.z), vw = TO_BIN(v.w);
#ifdef BINCOUNT_MASK
        MASK_T4 m = vload4(i, mask);
        int mx = (m.x != 0), my = (m.y != 0), mz = (m.z != 0), mw = (m.w != 0);
        for (int b = 0; b < BIN_NUM; ++b) {
            priv[b] += (vx == b) * mx + (vy == b) * my + (vz == b) * mz + (vw == b) * mw;
        }
#else
        for (int b = 0; b < BIN_NUM; ++b) {
            priv[b] += (vx == b) + (vy == b) + (vz == b) + (vw == b);
        }
#endif
    }
    // Tail: the up-to-3 trailing elements when n is not a multiple of 4.
    for (int i = (n4 << 2) + gid; i < n; i += gsize) {
        int v = TO_BIN(input[i]);
#ifdef BINCOUNT_MASK
        int keep = (mask[i] != 0);
        for (int b = 0; b < BIN_NUM; ++b) {
            priv[b] += ((v == b) ? 1 : 0) * keep;
        }
#else
        for (int b = 0; b < BIN_NUM; ++b) {
            priv[b] += (v == b) ? 1 : 0;
        }
#endif
    }

    // ---- cross-work-item merge ----
#if (BIN_NUM * LOCAL_SIZE) <= 4096
    // Fused reduction: collapse all BIN_NUM bins in a single log-tree (8 barrier
    // rounds for LOCAL_SIZE=256) instead of one tree per bin (~BIN_NUM*8
    // barriers). This attacks the dominant cost: with each work-item doing only
    // a few elements of accumulation, the per-workgroup reduction barriers were
    // the bottleneck, not loads or ALU. Bin-major layout [b*LOCAL_SIZE + lid]
    // keeps every reduction step's accesses contiguous across threads
    // (coalesced, bank-conflict-free).
    __local int reduceBuf[BIN_NUM * LOCAL_SIZE];
    for (int b = 0; b < BIN_NUM; ++b) {
        reduceBuf[b * LOCAL_SIZE + lid] = priv[b];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = LOCAL_SIZE >> 1; off > 0; off >>= 1) {
        if (lid < off) {
            for (int b = 0; b < BIN_NUM; ++b) {
                reduceBuf[b * LOCAL_SIZE + lid] += reduceBuf[b * LOCAL_SIZE + lid + off];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        for (int b = 0; b < BIN_NUM; ++b) {
            atomic_add(&output[b], reduceBuf[b * LOCAL_SIZE]);
        }
    }
#else
    // Fallback for large BIN_NUM (fused buffer would exceed local memory):
    // sequential per-bin reduction over a single LOCAL_SIZE scratch buffer.
    __local int reduceBuf[LOCAL_SIZE];
    for (int b = 0; b < BIN_NUM; ++b) {
        reduceBuf[lid] = priv[b];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = LOCAL_SIZE >> 1; off > 0; off >>= 1) {
            if (lid < off) {
                reduceBuf[lid] += reduceBuf[lid + off];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            atomic_add(&output[b], reduceBuf[0]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif
}

// Downsampled variant: counts only input[..., ::stride, ::stride] over the last
// two dims (nSampled ~= N/stride^2 elements), an approximate histogram that is
// far cheaper at large stride. Each work-item maps a linear sampled index k to
// a full-tensor flat offset via the sampled-grid geometry (Ws sampled columns
// per row, HsWs sampled elements per HW plane) and gathers that element. Reads
// are strided (not vectorizable/coalesced), but at stride^2-fewer elements the
// gather cost is small. Merge is identical to bincount_count_buf.
__kernel void bincount_sample_buf(
    __global const IN_T *input,
#ifdef BINCOUNT_MASK
    __global const MASK_T *mask,
#endif
    __global int *output,
    __private const int nSampled,
    __private const int stride,
    __private const int W,
    __private const int Ws,
    __private const int HsWs,
    __private const int HW,
    __private const int binNum) {
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
    const int lid = get_local_id(0);

    int priv[BIN_NUM];
    for (int b = 0; b < BIN_NUM; ++b) {
        priv[b] = 0;
    }

    for (int k = gid; k < nSampled; k += gsize) {
        const int plane = k / HsWs;
        const int r  = k - plane * HsWs;
        const int ph = r / Ws;
        const int pw = r - ph * Ws;
        const int flat = plane * HW + (ph * stride) * W + pw * stride;
        int v = TO_BIN(input[flat]);
#ifdef BINCOUNT_MASK
        int keep = (mask[flat] != 0);
        for (int b = 0; b < BIN_NUM; ++b) {
            priv[b] += ((v == b) ? 1 : 0) * keep;
        }
#else
        for (int b = 0; b < BIN_NUM; ++b) {
            priv[b] += (v == b) ? 1 : 0;
        }
#endif
    }

    // ---- cross-work-item merge (identical to bincount_count_buf) ----
#if (BIN_NUM * LOCAL_SIZE) <= 4096
    __local int reduceBuf[BIN_NUM * LOCAL_SIZE];
    for (int b = 0; b < BIN_NUM; ++b) {
        reduceBuf[b * LOCAL_SIZE + lid] = priv[b];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = LOCAL_SIZE >> 1; off > 0; off >>= 1) {
        if (lid < off) {
            for (int b = 0; b < BIN_NUM; ++b) {
                reduceBuf[b * LOCAL_SIZE + lid] += reduceBuf[b * LOCAL_SIZE + lid + off];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        for (int b = 0; b < BIN_NUM; ++b) {
            atomic_add(&output[b], reduceBuf[b * LOCAL_SIZE]);
        }
    }
#else
    __local int reduceBuf[LOCAL_SIZE];
    for (int b = 0; b < BIN_NUM; ++b) {
        reduceBuf[lid] = priv[b];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = LOCAL_SIZE >> 1; off > 0; off >>= 1) {
            if (lid < off) {
                reduceBuf[lid] += reduceBuf[lid + off];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            atomic_add(&output[b], reduceBuf[0]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif
}

// Naive baseline kept for on-device benchmarking only: one global atomic per
// input element straight into the BIN_NUM-sized output. Suffers heavy
// contention when BIN_NUM is small (few addresses hammered by many threads).
__kernel void bincount_naive_buf(
    __global const IN_T *input,
    __global int *output,
    __private const int n,
    __private const int binNum) {
    const int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    int v = TO_BIN(input[gid]);
    if (v >= 0 && v < binNum) {
        atomic_add(&output[v], 1);
    }
}
