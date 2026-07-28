//
//  CPUDualRangeHist.hpp
//  MNN
//

#ifndef CPUDualRangeHist_hpp
#define CPUDualRangeHist_hpp

#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
// Reference implementation of the dual masked range-histogram. Serves as the
// correctness oracle for the OpenCL buffer kernel; not the production fast path
// for the [1,1,1440,1920]-scale use case. Single pass over two float frames A,B
// (+ optional base mask): keep = base & (low<=A<=high) & (low<=B<=high) on the
// raw values, and when kept increments histA[rint(A*(binNum-1))] and
// histB[rint(B*(binNum-1))] (round-half-to-even). Outputs int32 histA, histB and
// optionally validCount = sum(keep).
class CPUDualRangeHist : public Execution {
public:
    CPUDualRangeHist(Backend* backend, const Op* op);
    virtual ~CPUDualRangeHist() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    int   mBinNum;
    float mLow;
    float mHigh;
    int   mSampleStride;
    bool  mEmitValidCount;
};
} // namespace MNN

#endif // CPUDualRangeHist_hpp
