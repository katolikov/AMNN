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

    if (maskMode) {
        // Binary mask -> int32 counts of kept (mask != 0) elements per bin.
        auto optr = output->host<int>();
        ::memset(optr, 0, mBinNum * sizeof(int));
        auto mask = inputs[1];
        for (int i = 0; i < n; ++i) {
            const int bin = readBin(input, i);
            if (bin >= 0 && bin < mBinNum && masked(mask, i)) {
                optr[bin] += 1;
            }
        }
    } else if (weighted) {
        auto optr = output->host<float>();
        ::memset(optr, 0, mBinNum * sizeof(float));
        auto wptr = inputs[1]->host<float>();
        for (int i = 0; i < n; ++i) {
            const int bin = readBin(input, i);
            if (bin >= 0 && bin < mBinNum) {
                optr[bin] += wptr[i];
            }
        }
    } else {
        auto optr = output->host<int>();
        ::memset(optr, 0, mBinNum * sizeof(int));
        for (int i = 0; i < n; ++i) {
            const int bin = readBin(input, i);
            if (bin >= 0 && bin < mBinNum) {
                optr[bin] += 1;
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
