//
//  CPUQuantile.cpp
//  MNN
//

#include <algorithm>
#include <cmath>
#include "backend/cpu/CPUQuantile.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

namespace MNN {

CPUQuantile::CPUQuantile(Backend* backend, const Op* op) : Execution(backend) {
    auto param = op->main_as_QuantileParam();
    mQLevels.resize(param->qLevels()->size());
    for (int i = 0; i < mQLevels.size(); ++i) {
        mQLevels[i] = param->qLevels()->Get(i);
    }
}

ErrorCode CPUQuantile::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    const int n = input->elementSize();
    MNN_ASSERT(n > 0);

    std::vector<float> sorted(n);
    ::memcpy(sorted.data(), input->host<float>(), n * sizeof(float));
    std::sort(sorted.begin(), sorted.end());

    auto outPtr = output->host<float>();
    for (int i = 0; i < mQLevels.size(); ++i) {
        const float idx = mQLevels[i] * (n - 1);
        const int lo = static_cast<int>(std::floor(idx));
        const int hi = static_cast<int>(std::ceil(idx));
        const float frac = idx - lo;
        outPtr[i] = sorted[lo] + frac * (sorted[hi] - sorted[lo]);
    }
    return NO_ERROR;
}

class CPUQuantileCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUQuantile(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUQuantileCreator, OpType_Quantile);

} // namespace MNN
