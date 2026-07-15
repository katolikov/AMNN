//
//  CPUQuantile.hpp
//  MNN
//

#ifndef CPUQuantile_hpp
#define CPUQuantile_hpp

#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
// Reference implementation (exact, via full sort). Not performance-critical:
// this backend exists as the correctness oracle for the OpenCL buffer kernel,
// not as a production fast path for the [1,1,1440,1920]-scale use case.
class CPUQuantile : public Execution {
public:
    CPUQuantile(Backend* backend, const Op* op);
    virtual ~CPUQuantile() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    std::vector<float> mQLevels;
};
} // namespace MNN

#endif // CPUQuantile_hpp
