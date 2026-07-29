//
//  DualRangeHistTest.cpp
//  MNNTests
//

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

// float -> fp16 (round-to-nearest-even) -> float. The result lands exactly on
// the fp16 grid, so feeding it to a Low/fp16-buffer backend is a no-op upload.
// This lets the fp16 stress case assert bit-exactness across CPU (fp32), OpenCL
// High (fp32) and OpenCL Low (fp16): all three see identical grid values, and
// the op does its bin/range math in fp32 on every path.
static float quantizeF16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    uint16_t h;
    if (exp >= 0x1f) {                       // inf/NaN or overflow -> inf/NaN
        h = (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
    } else if (exp <= 0) {                    // subnormal / underflow
        if (exp < -10) {
            h = (uint16_t)sign;
        } else {
            mant |= 0x800000u;
            const int shift = 14 - exp;
            uint32_t half = mant >> shift;
            const uint32_t rem = mant & ((1u << shift) - 1);
            const uint32_t mid = 1u << (shift - 1);
            if (rem > mid || (rem == mid && (half & 1))) half++;
            h = (uint16_t)(sign | half);
        }
    } else {                                  // normal
        uint32_t half = (uint32_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
        const uint32_t rem = mant & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) half++;  // carry into exp is fine
        h = (uint16_t)half;
    }
    // fp16 -> float32 (exact)
    const uint32_t hs = (uint32_t)(h & 0x8000) << 16;
    uint32_t he = (h >> 10) & 0x1f;
    uint32_t hm = h & 0x3ff;
    uint32_t out;
    if (he == 0) {
        if (hm == 0) {
            out = hs;
        } else {
            int e = -1;
            do { e++; hm <<= 1; } while ((hm & 0x400) == 0);
            hm &= 0x3ff;
            out = hs | ((uint32_t)(127 - 15 - e) << 23) | (hm << 13);
        }
    } else if (he == 0x1f) {
        out = hs | 0x7f800000u | (hm << 13);
    } else {
        out = hs | ((uint32_t)(he - 15 + 127) << 23) | (hm << 13);
    }
    float r; std::memcpy(&r, &out, 4); return r;
}

// Independent reference oracle (separate code path from the op) implementing the
// exact PyTorch semantics: keep = base & (low<A<high) & (low<B<high) on the
// raw values, histA[rint(A*(binNum-1))] / histB[rint(B*(binNum-1))] gated by
// keep, validCount = sum(keep). round-half-to-even via std::rint.
static void referenceDualRangeHist(const std::vector<float>& A, const std::vector<float>& B,
                                   const std::vector<float>* base, int H, int W, int binNum,
                                   float low, float high, int stride,
                                   std::vector<int>& histA, std::vector<int>& histB, int& validCount) {
    histA.assign(binNum, 0);
    histB.assign(binNum, 0);
    validCount = 0;
    const int Ws = (W + stride - 1) / stride;
    const int Hs = (H + stride - 1) / stride;
    const float scale = (float)(binNum - 1);
    for (int hs = 0; hs < Hs; ++hs) {
        for (int ws = 0; ws < Ws; ++ws) {
            const int flat = (hs * stride) * W + ws * stride;
            const float a = A[flat];
            const float b = B[flat];
            const bool inRange = (a > low && a < high && b > low && b < high);
            const bool keepBase = (base == nullptr) ? true : ((*base)[flat] != 0.0f);
            if (!(inRange && keepBase)) {
                continue;
            }
            const int binA = (int)std::rint(a * scale);
            const int binB = (int)std::rint(b * scale);
            if (binA >= 0 && binA < binNum) histA[binA] += 1;
            if (binB >= 0 && binB < binNum) histB[binB] += 1;
            ++validCount;
        }
    }
}

// fp16 reference: mirrors the OpenCL Low/fp16-buffer path (and a PyTorch reference
// that keeps fp16 tensors). The range compare is done against fp16-rounded low/high
// and the bin product A*(binNum-1) is evaluated in fp16, then round-half-to-even.
// Inputs are assumed already on the fp16 grid (callers quantize them).
static void referenceDualRangeHistF16(const std::vector<float>& A, const std::vector<float>& B,
                                      const std::vector<float>* base, int H, int W, int binNum,
                                      float low, float high, int stride,
                                      std::vector<int>& histA, std::vector<int>& histB, int& validCount) {
    histA.assign(binNum, 0);
    histB.assign(binNum, 0);
    validCount = 0;
    const int Ws = (W + stride - 1) / stride;
    const int Hs = (H + stride - 1) / stride;
    const float lowH  = quantizeF16(low);
    const float highH = quantizeF16(high);
    const float scaleH = quantizeF16((float)(binNum - 1));
    for (int hs = 0; hs < Hs; ++hs) {
        for (int ws = 0; ws < Ws; ++ws) {
            const int flat = (hs * stride) * W + ws * stride;
            const float a = A[flat];
            const float b = B[flat];
            const bool inRange = (a > lowH && a < highH && b > lowH && b < highH);
            const bool keepBase = (base == nullptr) ? true : ((*base)[flat] != 0.0f);
            if (!(inRange && keepBase)) {
                continue;
            }
            const int binA = (int)std::rint(quantizeF16(a * scaleH));
            const int binB = (int)std::rint(quantizeF16(b * scaleH));
            if (binA >= 0 && binA < binNum) histA[binA] += 1;
            if (binB >= 0 && binB < binNum) histB[binB] += 1;
            ++validCount;
        }
    }
}

class DualRangeHistTest : public MNNTestCase {
public:
    virtual ~DualRangeHistTest() = default;

    // Compare op output {histA, histB[, validCount]} against the reference.
    static bool checkAgainstReference(const char* tag, std::vector<VARP>& out, int binNum,
                                      const std::vector<int>& refA, const std::vector<int>& refB,
                                      int refCount, bool checkCount) {
        auto gotA = out[0]->readMap<int>();
        auto gotB = out[1]->readMap<int>();
        for (int b = 0; b < binNum; ++b) {
            if (gotA[b] != refA[b]) {
                MNN_ERROR("DualRangeHistTest %s failed: histA[%d] got %d want %d\n", tag, b, gotA[b], refA[b]);
                return false;
            }
            if (gotB[b] != refB[b]) {
                MNN_ERROR("DualRangeHistTest %s failed: histB[%d] got %d want %d\n", tag, b, gotB[b], refB[b]);
                return false;
            }
        }
        if (checkCount) {
            int gotCount = out[2]->readMap<int>()[0];
            if (gotCount != refCount) {
                MNN_ERROR("DualRangeHistTest %s failed: validCount got %d want %d\n", tag, gotCount, refCount);
                return false;
            }
        }
        return true;
    }

    virtual bool run(int precision) {
        // ===== case 1: tiny, no base mask, hand-verifiable =====
        // binNum=4 (scale=3), low=0.1 high=0.9.
        // A = {0, 1/3, 2/3, 1};  B = {1/3, 1/3, 2/3, 1}
        // range membership (both in (0.1,0.9)): idx0 A=0 out; idx1 both 1/3 in;
        // idx2 both 2/3 in; idx3 A=1,B=1 out. So kept idx = {1,2}.
        // bins: A[1]=rint(1)=1, A[2]=rint(2)=2; B[1]=1, B[2]=2.
        // histA={0,1,1,0} histB={0,1,1,0} validCount=2.
        {
            const int binNum = 4;
            std::vector<float> A = {0.0f, 1.0f/3, 2.0f/3, 1.0f};
            std::vector<float> B = {1.0f/3, 1.0f/3, 2.0f/3, 1.0f};
            VARP va = _Input({1, 1, 1, 4}, NCHW, halide_type_of<float>());
            VARP vb = _Input({1, 1, 1, 4}, NCHW, halide_type_of<float>());
            ::memcpy(va->writeMap<float>(), A.data(), A.size() * sizeof(float));
            ::memcpy(vb->writeMap<float>(), B.data(), B.size() * sizeof(float));
            auto out = _DualRangeHist(va, vb, nullptr, binNum, 0.1f, 0.9f, 1, true);
            std::vector<int> refA, refB; int refCount;
            referenceDualRangeHist(A, B, nullptr, 1, 4, binNum, 0.1f, 0.9f, 1, refA, refB, refCount);
            if (!checkAgainstReference("case1", out, binNum, refA, refB, refCount, true)) return false;
        }

        // ===== case 2: tiny with base mask =====
        // base = {1,0,1,1}; combined with case-1 range keeps idx {2} only
        // (idx1 masked out). histA={0,0,1,0} histB={0,0,1,0} validCount=1.
        {
            const int binNum = 4;
            std::vector<float> A = {0.0f, 1.0f/3, 2.0f/3, 1.0f};
            std::vector<float> B = {1.0f/3, 1.0f/3, 2.0f/3, 1.0f};
            std::vector<float> base = {1.0f, 0.0f, 1.0f, 1.0f};
            VARP va = _Input({1, 1, 1, 4}, NCHW, halide_type_of<float>());
            VARP vb = _Input({1, 1, 1, 4}, NCHW, halide_type_of<float>());
            VARP vm = _Input({1, 1, 1, 4}, NCHW, halide_type_of<float>());
            ::memcpy(va->writeMap<float>(), A.data(), A.size() * sizeof(float));
            ::memcpy(vb->writeMap<float>(), B.data(), B.size() * sizeof(float));
            ::memcpy(vm->writeMap<float>(), base.data(), base.size() * sizeof(float));
            auto out = _DualRangeHist(va, vb, vm, binNum, 0.1f, 0.9f, 1, true);
            std::vector<int> refA, refB; int refCount;
            referenceDualRangeHist(A, B, &base, 1, 4, binNum, 0.1f, 0.9f, 1, refA, refB, refCount);
            if (!checkAgainstReference("case2", out, binNum, refA, refB, refCount, true)) return false;
        }

        // ===== full-frame builders (fp16-stable): values k/15 are >> fp16 error
        // away from the .5 bin half-boundaries and from low/high, so High and Low
        // precision agree with the fp32 reference. =====
        const int H = 1440, W = 1920;
        const int n = H * W;
        const float low = 0.05f, high = 0.95f;
        const int binNum = 16;
        std::vector<float> A(n), B(n), base(n);
        for (int i = 0; i < n; ++i) {
            A[i]    = (float)(i % 16) / 15.0f;
            B[i]    = (float)((i * 7) % 16) / 15.0f;
            base[i] = (i % 3 == 0) ? 0.0f : 1.0f;   // ~2/3 kept by base
        }

        // ===== case 3: full [1,1,1440,1920], base mask, binNum=16, stride=1 =====
        {
            VARP va = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            VARP vb = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            VARP vm = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            ::memcpy(va->writeMap<float>(), A.data(), n * sizeof(float));
            ::memcpy(vb->writeMap<float>(), B.data(), n * sizeof(float));
            ::memcpy(vm->writeMap<float>(), base.data(), n * sizeof(float));
            auto out = _DualRangeHist(va, vb, vm, binNum, low, high, 1, true);
            std::vector<int> refA, refB; int refCount;
            referenceDualRangeHist(A, B, &base, H, W, binNum, low, high, 1, refA, refB, refCount);
            if (!checkAgainstReference("case3(full/stride1)", out, binNum, refA, refB, refCount, true)) return false;
        }

        // ===== case 4: full frame, base mask, stride=8, no explicit validCount =====
        {
            VARP va = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            VARP vb = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            VARP vm = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            ::memcpy(va->writeMap<float>(), A.data(), n * sizeof(float));
            ::memcpy(vb->writeMap<float>(), B.data(), n * sizeof(float));
            ::memcpy(vm->writeMap<float>(), base.data(), n * sizeof(float));
            auto out = _DualRangeHist(va, vb, vm, binNum, low, high, 8, false);
            std::vector<int> refA, refB; int refCount;
            referenceDualRangeHist(A, B, &base, H, W, binNum, low, high, 8, refA, refB, refCount);
            if (!checkAgainstReference("case4(full/stride8)", out, binNum, refA, refB, refCount, false)) return false;
        }

        // ===== case 5: full frame, NO base mask (absent => all-ones), stride=1 =====
        {
            VARP va = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            VARP vb = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            ::memcpy(va->writeMap<float>(), A.data(), n * sizeof(float));
            ::memcpy(vb->writeMap<float>(), B.data(), n * sizeof(float));
            auto out = _DualRangeHist(va, vb, nullptr, binNum, low, high, 1, true);
            std::vector<int> refA, refB; int refCount;
            referenceDualRangeHist(A, B, nullptr, H, W, binNum, low, high, 1, refA, refB, refCount);
            if (!checkAgainstReference("case5(full/nobase)", out, binNum, refA, refB, refCount, true)) return false;
        }

        // ===== case 6: fp16 boundary stress =====
        // Unlike cases 3-5 (values k/15, deliberately fp16-stable), this sweeps
        // [0,1] with 4096 fine levels (step ~2.4e-4, finer than the fp16 ULP near
        // 0.5) so many pixels sit right on bin half-boundaries (k+0.5)/15 and on
        // the low/high edges -- exactly where fp32-vs-fp16 histograms diverge.
        // Both the op input and the reference are fed the SAME fp16-quantized
        // grid values, so the op must be bit-exact on CPU (fp32), OpenCL High
        // (fp32) and OpenCL Low (fp16). (A reference kept in fp32 while the op
        // stores fp16 would legitimately differ here by a few counts per bin.)
        {
            std::vector<float> Aq(n), Bq(n), baseq(n);
            for (int i = 0; i < n; ++i) {
                Aq[i]    = quantizeF16((float)(i % 4096) / 4095.0f);
                Bq[i]    = quantizeF16((float)((i * 7 + 1) % 4096) / 4095.0f);
                baseq[i] = (i % 3 == 0) ? 0.0f : 1.0f;
            }
            VARP va = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            VARP vb = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            VARP vm = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            ::memcpy(va->writeMap<float>(), Aq.data(), n * sizeof(float));
            ::memcpy(vb->writeMap<float>(), Bq.data(), n * sizeof(float));
            ::memcpy(vm->writeMap<float>(), baseq.data(), n * sizeof(float));
            auto out = _DualRangeHist(va, vb, vm, binNum, low, high, 1, true);
            // The op runs fp16 only on an fp16-capable OpenCL device at Low
            // precision; on CPU, at High precision, OR on a device without
            // cl_khr_fp16 (MNN forces fp32 there) it runs fp32. Both references
            // are exact ground truth for their precision, so accept whichever
            // one the op actually matches; a kernel matching NEITHER is a bug.
            std::vector<int> f16A, f16B; int f16Count;
            std::vector<int> f32A, f32B; int f32Count;
            referenceDualRangeHistF16(Aq, Bq, &baseq, H, W, binNum, low, high, 1, f16A, f16B, f16Count);
            referenceDualRangeHist(Aq, Bq, &baseq, H, W, binNum, low, high, 1, f32A, f32B, f32Count);
            auto eq = [&](const std::vector<int>& rA, const std::vector<int>& rB, int rc) {
                auto gA = out[0]->readMap<int>();
                auto gB = out[1]->readMap<int>();
                if (out[2]->readMap<int>()[0] != rc) return false;
                for (int b = 0; b < binNum; ++b) {
                    if (gA[b] != rA[b] || gB[b] != rB[b]) return false;
                }
                return true;
            };
            const bool okF16 = eq(f16A, f16B, f16Count);
            const bool okF32 = eq(f32A, f32B, f32Count);
            if (!okF16 && !okF32) {
                auto gA = out[0]->readMap<int>();
                MNN_ERROR("DualRangeHistTest case6(fp16-boundary) matched neither reference "
                          "(type=%d prec=%d). bin4: op=%d f16=%d f32=%d\n",
                          (int)getCurrentType(), precision, gA[4], f16A[4], f32A[4]);
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(DualRangeHistTest, "op/DualRangeHist");
