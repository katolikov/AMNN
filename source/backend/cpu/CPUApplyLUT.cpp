//
//  CPUApplyLUT.cpp
//  MNN
//

#include <cmath>
#include <algorithm>
#include "backend/cpu/CPUApplyLUT.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUApplyLUT::CPUApplyLUT(Backend* backend, const Op* op) : Execution(backend) {
    auto param = op->main_as_ApplyLUTParam();
    if (param != nullptr) {
        mInterp = param->interp();
    }
}

ErrorCode CPUApplyLUT::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto lut    = inputs[1];
    auto output = outputs[0];

    const float* in  = input->host<float>();
    const float* tab = lut->host<float>();
    float* out       = output->host<float>();

    const int n = input->elementSize();
    const int N = lut->elementSize();
    const int maxIdx = N - 1;
    const float scale = (float)maxIdx;

    // Clamp-index-only linear interpolation (see header). In-range x in [0,1] is
    // standard lerp; out-of-range x keeps the clamped endpoint's slope (frac may
    // fall outside [0,1)). Matches the OpenCL kernel bit-for-bit in fp32.
    for (int i = 0; i < n; ++i) {
        const float pos = in[i] * scale;
        int i0 = (int)std::floor(pos);
        i0 = std::min(std::max(i0, 0), maxIdx);
        const int i1 = std::min(i0 + 1, maxIdx);
        const float frac = pos - (float)i0;
        const float lo = tab[i0];
        const float hi = tab[i1];
        out[i] = lo + frac * (hi - lo);
    }
    return NO_ERROR;
}

class CPUApplyLUTCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUApplyLUT(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUApplyLUTCreator, OpType_ApplyLUT);

} // namespace MNN
