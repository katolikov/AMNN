//
//  CPUMockChain.hpp
//  MNN
//
//  CPU reference for the MockChain template op. Computes the same 5-function
//  elementwise chain the OpenCL path dispatches across multiple kernels, so the
//  unit test can compare GPU output against a trivially-correct baseline.
//

#ifndef CPUMockChain_hpp
#define CPUMockChain_hpp

#include "core/Execution.hpp"

namespace MNN {

class CPUMockChain : public Execution {
public:
    CPUMockChain(Backend* backend, const Op* op);
    virtual ~CPUMockChain() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    float mOffset0 = 10.0f;
    float mOffset1 = 20.0f;
};

} // namespace MNN

#endif /* CPUMockChain_hpp */
