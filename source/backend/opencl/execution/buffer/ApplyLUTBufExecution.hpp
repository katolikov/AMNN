//
//  ApplyLUTBufExecution.hpp
//  MNN
//
//  OpenCL buffer-path implementation of ApplyLUT: per-element 1-D LUT remap with
//  linear interpolation. A bandwidth-bound element-wise op -- input 0 is streamed
//  as a dense flat buffer (NC4HW4 rejected / packing disabled host-side so a
//  [1,1,H,W] tensor is contiguous H*W with no channel padding), VEC elements per
//  work-item via one vectorized load/store, and the runtime LUT (input 1) is
//  staged once per workgroup into local memory. Single kernel, single pass.
//

#ifndef ApplyLUTBufExecution_hpp
#define ApplyLUTBufExecution_hpp

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class ApplyLUTBufExecution : public CommonExecution {
public:
    ApplyLUTBufExecution(const MNN::Op *op, Backend *backend);
    virtual ~ApplyLUTBufExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    OpenCLBackend *mOpenCLBackend = nullptr;
    int  mInterp    = 1;
    int  mVec       = 2;      // elements streamed per work-item (2/4/8/16)
    bool mLutGlobal = true;   // read LUT from cached global instead of local
};

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* ApplyLUTBufExecution_hpp */
