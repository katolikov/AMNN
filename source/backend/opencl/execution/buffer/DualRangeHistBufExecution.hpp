//
//  DualRangeHistBufExecution.hpp
//  MNN
//
//  OpenCL buffer-path implementation of DualRangeHist: single-pass dual masked
//  range-histogram over two float frames A, B (+ optional base mask). Reuses the
//  BinCount register-histogram machinery extended to two histograms (two private
//  register arrays merged via a fused log-tree reduction plus one global atomic
//  per bin per workgroup), the CU-derived grid heuristic, fp16 FLOAT-macro reads,
//  and the sampleStride sampled-grid geometry. Register path only (small
//  binNum); larger binNum falls back to the CPU reference.
//

#ifndef DualRangeHistBufExecution_hpp
#define DualRangeHistBufExecution_hpp

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class DualRangeHistBufExecution : public CommonExecution {
public:
    DualRangeHistBufExecution(const MNN::Op *op, Backend *backend);
    virtual ~DualRangeHistBufExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    OpenCLBackend *mOpenCLBackend = nullptr;
    int   mBinNum = 0;
    float mLow = 0.0f;
    float mHigh = 1.0f;
    int   mSampleStride = 1;
    bool  mEmitValidCount = false;
};

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* DualRangeHistBufExecution_hpp */
