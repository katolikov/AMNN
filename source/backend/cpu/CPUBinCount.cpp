//
//  CPUBinCount.cpp
//  MNN
//

#include <cstring>
#include "backend/cpu/CPUBinCount.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUBinCount::CPUBinCount(Backend* backend, const Op* op) : Execution(backend) {
    auto param = op->main_as_BinCountParam();
    mBinNum = param->binNum();
    mBinaryMask = param->binaryMask();
    mSampleStride = param->sampleStride() > 1 ? param->sampleStride() : 1;
}

// Read the i-th element of a (int32 or float) tensor as a "kept" flag.
static inline bool masked(const Tensor* mask, int i) {
    if (mask->getType() == halide_type_of<float>()) {
        return mask->host<float>()[i] != 0.0f;
    }
    return mask->host<int>()[i] != 0;
}

// Read the i-th input element as an integer bin index regardless of whether the
// input tensor is stored as int32 or float32.
static inline int readBin(const Tensor* input, int i) {
    if (input->getType() == halide_type_of<float>()) {
        return static_cast<int>(input->host<float>()[i]);
    }
    return input->host<int>()[i];
}

ErrorCode CPUBinCount::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    const int n = input->elementSize();
    const bool hasSecond = (inputs.size() == 2);
    const bool maskMode  = hasSecond && mBinaryMask;
    const bool weighted  = hasSecond && !mBinaryMask;

    // Spatial downsampling over the last two dims: input[..., ::s, ::s].
    const int stride = mSampleStride;
    const int rank = input->dimensions();
    const int W  = rank >= 1 ? input->length(rank - 1) : 1;
    const int H  = rank >= 2 ? input->length(rank - 2) : 1;
    const int HW = H * W;
    const int B  = HW > 0 ? n / HW : 1;
    const int Ws = (W + stride - 1) / stride;
    const int Hs = (H + stride - 1) / stride;

    int*   optrI = nullptr;
    float* optrF = nullptr;
    if (weighted) {
        optrF = output->host<float>();
        ::memset(optrF, 0, mBinNum * sizeof(float));
    } else {
        optrI = output->host<int>();
        ::memset(optrI, 0, mBinNum * sizeof(int));
    }
    auto mask = hasSecond ? inputs[1] : nullptr;
    const float* wptr = weighted ? inputs[1]->host<float>() : nullptr;

    for (int b = 0; b < B; ++b) {
        for (int hs = 0; hs < Hs; ++hs) {
            for (int ws = 0; ws < Ws; ++ws) {
                const int flat = b * HW + (hs * stride) * W + ws * stride;
                const int bin = readBin(input, flat);
                if (bin < 0 || bin >= mBinNum) {
                    continue;
                }
                if (maskMode) {
                    if (masked(mask, flat)) {
                        optrI[bin] += 1;
                    }
                } else if (weighted) {
                    optrF[bin] += wptr[flat];
                } else {
                    optrI[bin] += 1;
                }
            }
        }
    }
    return NO_ERROR;
}

class CPUBinCountCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUBinCount(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUBinCountCreator, OpType_BinCount);

} // namespace MNN
