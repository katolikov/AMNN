//
//  CPUDualRangeHist.cpp
//  MNN
//

#include <cmath>
#include <cstring>
#include "backend/cpu/CPUDualRangeHist.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUDualRangeHist::CPUDualRangeHist(Backend* backend, const Op* op) : Execution(backend) {
    auto param = op->main_as_DualRangeHistParam();
    mBinNum         = param->binNum();
    mLow            = param->low();
    mHigh           = param->high();
    mSampleStride   = param->sampleStride() > 1 ? param->sampleStride() : 1;
    mEmitValidCount = param->emitValidCount();
}

// Read the i-th element of a (int32 or float) tensor as a float; used for both
// the frames and the optional base mask.
static inline float readF(const Tensor* t, int i) {
    if (t->getType() == halide_type_of<float>()) {
        return t->host<float>()[i];
    }
    return static_cast<float>(t->host<int>()[i]);
}

ErrorCode CPUDualRangeHist::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto A = inputs[0];
    auto B = inputs[1];
    const Tensor* base = (inputs.size() == 3) ? inputs[2] : nullptr;

    int* histA = outputs[0]->host<int>();
    int* histB = outputs[1]->host<int>();
    ::memset(histA, 0, mBinNum * sizeof(int));
    ::memset(histB, 0, mBinNum * sizeof(int));
    int* validCount = mEmitValidCount ? outputs[2]->host<int>() : nullptr;

    // Spatial downsampling over the last two dims: A[..., ::s, ::s] (and B, base).
    const int stride = mSampleStride;
    const int rank = A->dimensions();
    const int W  = rank >= 1 ? A->length(rank - 1) : 1;
    const int H  = rank >= 2 ? A->length(rank - 2) : 1;
    const int Ws = (W + stride - 1) / stride;
    const int Hs = (H + stride - 1) / stride;

    // value -> bin index, done inside the op: rint() matches torch.round
    // (round-half-to-even under the default rounding mode). Valid pixels are
    // guaranteed in [0, binNum-1] (low, high in [0,1]), so no clamp is needed.
    const float scale = (float)(mBinNum - 1);
    int kept = 0;
    for (int hs = 0; hs < Hs; ++hs) {
        for (int ws = 0; ws < Ws; ++ws) {
            const int flat = (hs * stride) * W + ws * stride;   // batch == 1
            const float a = readF(A, flat);
            const float b = readF(B, flat);
            const bool inRange = (a >= mLow && a <= mHigh && b >= mLow && b <= mHigh);
            const bool keepBase = (base == nullptr) ? true : (readF(base, flat) != 0.0f);
            if (!(inRange && keepBase)) {
                continue;
            }
            const int binA = (int)std::rint(a * scale);
            const int binB = (int)std::rint(b * scale);
            if (binA >= 0 && binA < mBinNum) {
                histA[binA] += 1;
            }
            if (binB >= 0 && binB < mBinNum) {
                histB[binB] += 1;
            }
            ++kept;
        }
    }
    if (validCount != nullptr) {
        validCount[0] = kept;
    }
    return NO_ERROR;
}

class CPUDualRangeHistCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUDualRangeHist(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDualRangeHistCreator, OpType_DualRangeHist);

} // namespace MNN
