//
//  MockChainBufExecution.hpp
//  MNN
//
//  TEMPLATE op: shows how ONE custom OpType dispatches several OpenCL kernels
//  sequentially (fan-out / fan-in DAG) via CommonExecution's mUnits, with
//  DYNAMIC scratch buffers wiring the stages together. The kernels themselves
//  are trivial elementwise arithmetic (see mockchain_buf.cl) -- to reuse this
//  for a real op, keep the dispatch/sync skeleton and swap in your own kernels,
//  scratch-tensor shapes, and per-unit global/local work sizes.
//

#ifndef MockChainBufExecution_hpp
#define MockChainBufExecution_hpp

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class MockChainBufExecution : public CommonExecution {
public:
    MockChainBufExecution(const MNN::Op *op, Backend *backend);
    virtual ~MockChainBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    OpenCLBackend *mOpenCLBackend = nullptr;
    float mOffset0 = 10.0f;
    float mOffset1 = 20.0f;

    // Scratch device tensors passed between kernels. Held as members so the
    // Tensor wrappers (and the cl_mem the DYNAMIC pool assigns them at resize)
    // stay valid for the op's lifetime. Lifetimes are managed by acquire/release
    // ordering in onEncode so the pool can overlap non-overlapping buffers.
    std::shared_ptr<Tensor> mT1a, mT2a; // branch A: func1 -> func2
    std::shared_ptr<Tensor> mT1b, mT2b; // branch B: func1 -> func2
    std::shared_ptr<Tensor> mT3;        // func3 fan-in result
    std::shared_ptr<Tensor> mT4a, mT4b; // func4 result for each output run
};

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MockChainBufExecution_hpp */
