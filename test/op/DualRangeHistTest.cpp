//
//  DualRangeHistTest.cpp
//  MNNTests
//

#include <cmath>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

// Independent reference oracle (separate code path from the op) implementing the
// exact PyTorch semantics: keep = base & (low<=A<=high) & (low<=B<=high) on the
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
            const bool inRange = (a >= low && a <= high && b >= low && b <= high);
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
        // range membership (both in [0.1,0.9]): idx0 A=0 out; idx1 both 1/3 in;
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

        return true;
    }
};

MNNTestSuiteRegister(DualRangeHistTest, "op/DualRangeHist");
