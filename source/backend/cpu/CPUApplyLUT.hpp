//
//  CPUApplyLUT.hpp
//  MNN
//
//  Reference implementation of ApplyLUT: per-element 1-D lookup-table remap with
//  linear interpolation. Serves as the correctness oracle for the OpenCL buffer
//  kernel; it is not the production fast path for the [1,1,1440,1920]-scale use
//  case. Input 0 is the float image, input 1 the [N] float LUT (runtime tensor).
//  Semantics (clamp-index-only): pos = x*(N-1); i0 = clamp(floor(pos),0,N-1);
//  i1 = min(i0+1,N-1); frac = pos - i0; out = lut[i0] + frac*(lut[i1]-lut[i0]).
//

#ifndef CPUApplyLUT_hpp
#define CPUApplyLUT_hpp

#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class CPUApplyLUT : public Execution {
public:
    CPUApplyLUT(Backend* backend, const Op* op);
    virtual ~CPUApplyLUT() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    int mInterp = 1;
};
} // namespace MNN

#endif // CPUApplyLUT_hpp
