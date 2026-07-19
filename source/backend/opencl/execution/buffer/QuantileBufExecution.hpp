//
//  QuantileBufExecution.hpp
//  MNN
//
//  OpenCL buffer-path implementation of Quantile. Selects exact order
//  statistics via monotonic-key binary search instead of a full sort/TopK,
//  so it has no k<=1024-style size cap (see TopKV2BufExecution).
//

#ifndef QuantileBufExecution_hpp
#define QuantileBufExecution_hpp

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class QuantileBufExecution : public CommonExecution {
public:
    QuantileBufExecution(const MNN::Op *op, Backend *backend);
    virtual ~QuantileBufExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    OpenCLBackend *mOpenCLBackend = nullptr;
    std::vector<float> mQLevels;
    int mNumTargets = 0;
    std::vector<int> mTargetRank;   // size mNumTargets
    std::vector<float> mFrac;       // size mQLevels.size()

    std::shared_ptr<cl::Buffer> mLoKeyBuffer;
    std::shared_ptr<cl::Buffer> mHiKeyBuffer;
    std::shared_ptr<cl::Buffer> mCountBuffer;
    std::shared_ptr<cl::Buffer> mTargetRankBuffer;
    std::shared_ptr<cl::Buffer> mFracBuffer;

    // 16 (not 32) bits of the monotonic key resolved: measured to already
    // match the fp16-tolerance reference exactly while halving runtime (32
    // iters: ~15.4ms GPU / 12 iters started showing visible error; see
    // algorithm-selection investigation). Each iteration costs ~1 count +
    // 1 update kernel dispatch.
    static const int kIters = 16;
};

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* QuantileBufExecution_hpp */
