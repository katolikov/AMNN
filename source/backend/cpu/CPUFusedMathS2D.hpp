//
//  CPUFusedMathS2D.hpp
//  MNN
//

#ifndef CPUFusedMathS2D_hpp
#define CPUFusedMathS2D_hpp

#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class CPUFusedMathS2D : public Execution {
public:
    CPUFusedMathS2D(Backend* b, const FusedMathS2DParam* param);
    virtual ~CPUFusedMathS2D() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    float mAlpha;
    float mBeta;
    float mGamma;
    float mDelta;
    float mEpsilon;
    float mConst;
};
} // namespace MNN

#endif /* CPUFusedMathS2D_hpp */
