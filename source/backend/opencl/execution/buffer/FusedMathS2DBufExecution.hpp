//
//  FusedMathS2DBufExecution.hpp
//  MNN
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifndef FusedMathS2DBufExecution_hpp
#define FusedMathS2DBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class FusedMathS2DBufExecution : public CommonExecution {
public:
    FusedMathS2DBufExecution(const MNN::Op *op, Backend *backend);
    virtual ~FusedMathS2DBufExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mAlpha   = 1.0f;
    float mBeta    = 1.0f;
    float mGamma   = 0.0f;
    float mDelta   = 1.0f;
    float mEpsilon = 0.0f;
    float mConst   = 0.0f;
    int mVecWidth  = 4;   // output pixels per work-item (env: MNN_S2D_VW)
    int mForcedLws = 0;   // 0 = autotune            (env: MNN_S2D_LWS)
};

} // namespace OpenCL
} // namespace MNN
#endif /* FusedMathS2DBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
