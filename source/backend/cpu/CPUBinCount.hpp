//
//  CPUBinCount.hpp
//  MNN
//

#ifndef CPUBinCount_hpp
#define CPUBinCount_hpp

#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
// Reference implementation. Serves as the correctness oracle for the OpenCL
// buffer kernel; not the production fast path for the [1,1,1440,1920]-scale
// use case. Counts occurrences of each integer value in the flattened input
// into a fixed number of bins (binNum). Values outside [0, binNum) are
// dropped. With a second (weights) input the output holds float weight-sums;
// otherwise int32 counts.
class CPUBinCount : public Execution {
public:
    CPUBinCount(Backend* backend, const Op* op);
    virtual ~CPUBinCount() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    int mBinNum;
};
} // namespace MNN

#endif // CPUBinCount_hpp
