//
//  BinCountBufExecution.hpp
//  MNN
//
//  OpenCL buffer-path implementation of BinCount. Unweighted (int32 counts)
//  only; the weighted path falls back to CPU. Uses a per-work-item
//  register-resident histogram (contention-free) merged via log-tree reduction
//  plus one global atomic per bin per workgroup.
//

#ifndef BinCountBufExecution_hpp
#define BinCountBufExecution_hpp

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class BinCountBufExecution : public CommonExecution {
public:
    BinCountBufExecution(const MNN::Op *op, Backend *backend);
    virtual ~BinCountBufExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    OpenCLBackend *mOpenCLBackend = nullptr;
    int mBinNum = 0;
};

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* BinCountBufExecution_hpp */
