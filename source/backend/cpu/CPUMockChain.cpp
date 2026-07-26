//
//  CPUMockChain.cpp
//  MNN
//

#include "backend/cpu/CPUMockChain.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

namespace MNN {

CPUMockChain::CPUMockChain(Backend* backend, const Op* op) : Execution(backend) {
    auto param = op->main_as_MockChainParam();
    if (param != nullptr) {
        mOffset0 = param->offset0();
        mOffset1 = param->offset1();
    }
}

// The elementwise mock chain, mirrored exactly by the OpenCL kernels:
//   func1(x) = x + 1
//   func2(x) = x * 2
//   func3(a,b) = a + b
//   func4(x,k) = x + k
//   func5(x) = x * 3
// branchA: t2a = func2(func1(A));  branchB: t2b = func2(func1(B))
// t3 = func3(t2a, t2b)
// out0 = func5(func4(t3, offset0));  out1 = func5(func4(t3, offset1))
ErrorCode CPUMockChain::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto A = inputs[0];
    auto B = inputs[1];
    const int n = A->elementSize();
    const float* a = A->host<float>();
    const float* b = B->host<float>();
    float* out0 = outputs[0]->host<float>();
    float* out1 = outputs[1]->host<float>();

    for (int i = 0; i < n; ++i) {
        const float t2a = (a[i] + 1.0f) * 2.0f;
        const float t2b = (b[i] + 1.0f) * 2.0f;
        const float t3  = t2a + t2b;
        out0[i] = (t3 + mOffset0) * 3.0f;
        out1[i] = (t3 + mOffset1) * 3.0f;
    }
    return NO_ERROR;
}

class CPUMockChainCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUMockChain(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUMockChainCreator, OpType_MockChain);

} // namespace MNN
