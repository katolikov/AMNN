//
//  MockChainBufExecution.cpp
//  MNN
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "MockChainBufExecution.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace OpenCL {

// Local work size for the 1-D elementwise kernels. Small and fixed keeps the
// template simple; a real op would tune this (see localWS2DDefault usage in the
// other buffer executions).
static const uint32_t kLocalSize = 64;

MockChainBufExecution::MockChainBufExecution(const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto param = op->main_as_MockChainParam();
    if (param != nullptr) {
        mOffset0 = param->offset0();
        mOffset1 = param->offset1();
    }
}

ErrorCode MockChainBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 2 && outputs.size() == 2);
    auto A = inputs[0];
    auto B = inputs[1];
    auto out0 = outputs[0];
    auto out1 = outputs[1];
    const int size = A->elementSize();

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const int prec = mOpenCLBackend->getPrecision();
    std::set<std::string> buildOptions; // no specialization needed for the mock

    // --- scratch tensors, one element-buffer per intermediate ---
    // createDevice + onAcquireBuffer(DYNAMIC) reserves pool memory; the matching
    // onReleaseBuffer(DYNAMIC) after a buffer's last use lets the pool reuse it.
    auto acquire = [&](std::shared_ptr<Tensor> &t) -> bool {
        t.reset(Tensor::createDevice<float>(std::vector<int>{size}));
        return mOpenCLBackend->onAcquireBuffer(t.get(), Backend::DYNAMIC);
    };
    auto release = [&](std::shared_ptr<Tensor> &t) {
        mOpenCLBackend->onReleaseBuffer(t.get(), Backend::DYNAMIC);
    };

    // --- build one KernelWrap PER DISPATCH ---
    // A cl::Kernel holds a single set of bound args, so the two runs of func1/
    // func2/func4/func5 each need their OWN instance: setArg-ing one kernel twice
    // for two enqueues would make the second binding clobber the first before
    // either runs. func3 is dispatched once. buildKernel is cheap (programs are
    // cached), so we just build a fresh handle wherever a dispatch needs one.
    auto build = [&](const char *name) {
        return runtime->buildKernel("mockchain_buf", name, buildOptions, prec, A, out0);
    };
    auto k1a = build("mockchain_func1_buf");
    auto k2a = build("mockchain_func2_buf");
    auto k1b = build("mockchain_func1_buf");
    auto k2b = build("mockchain_func2_buf");
    auto k3  = build("mockchain_func3_buf");
    auto k4a = build("mockchain_func4_buf");
    auto k5a = build("mockchain_func5_buf");
    auto k4b = build("mockchain_func4_buf");
    auto k5b = build("mockchain_func5_buf");
    if (k1a == nullptr || k2a == nullptr || k1b == nullptr || k2b == nullptr || k3 == nullptr ||
        k4a == nullptr || k5a == nullptr || k4b == nullptr || k5b == nullptr) {
        return NOT_SUPPORT;
    }

    // Helper: bind a unary elementwise kernel (in, out, size) and push its Unit.
    // Every dispatch is pushed in topological order; because they run on one
    // in-order queue, that ordering IS the synchronization -- no barriers, no
    // readbacks between kernels.
    mUnits.clear();
    mUnits.reserve(9);
    cl_int ret = CL_SUCCESS;
    const uint32_t global = ROUND_UP((uint32_t)size, kLocalSize);

    auto pushUnary = [&](std::shared_ptr<KernelWrap> k, const Tensor *in, const Tensor *out) {
        uint32_t idx = 0;
        ret |= k->get().setArg(idx++, openCLBuffer(in));
        ret |= k->get().setArg(idx++, openCLBuffer(out));
        ret |= k->get().setArg(idx++, size);
        Unit unit;
        unit.kernel = k;
        unit.globalWorkSize = {global, 1, 1};
        unit.localWorkSize  = {kLocalSize, 1, 1};
        mUnits.emplace_back(std::move(unit));
    };
    auto pushFunc4 = [&](std::shared_ptr<KernelWrap> k, const Tensor *in, const Tensor *out, float offset) {
        uint32_t idx = 0;
        ret |= k->get().setArg(idx++, openCLBuffer(in));
        ret |= k->get().setArg(idx++, openCLBuffer(out));
        ret |= k->get().setArg(idx++, offset);
        ret |= k->get().setArg(idx++, size);
        Unit unit;
        unit.kernel = k;
        unit.globalWorkSize = {global, 1, 1};
        unit.localWorkSize  = {kLocalSize, 1, 1};
        mUnits.emplace_back(std::move(unit));
    };
    auto pushFunc3 = [&](std::shared_ptr<KernelWrap> k, const Tensor *a, const Tensor *b, const Tensor *out) {
        uint32_t idx = 0;
        ret |= k->get().setArg(idx++, openCLBuffer(a));
        ret |= k->get().setArg(idx++, openCLBuffer(b));
        ret |= k->get().setArg(idx++, openCLBuffer(out));
        ret |= k->get().setArg(idx++, size);
        Unit unit;
        unit.kernel = k;
        unit.globalWorkSize = {global, 1, 1};
        unit.localWorkSize  = {kLocalSize, 1, 1};
        mUnits.emplace_back(std::move(unit));
    };

    // ===== branch A: func1 -> func2 =====
    if (!acquire(mT1a) || !acquire(mT2a)) { return OUT_OF_MEMORY; }
    pushUnary(k1a, A, mT1a.get());        // t1a = A + 1
    pushUnary(k2a, mT1a.get(), mT2a.get());// t2a = t1a * 2
    release(mT1a);                         // t1a dead after func2a

    // ===== branch B: func1 -> func2 (reuses t1a's freed slot for t1b) =====
    if (!acquire(mT1b) || !acquire(mT2b)) { return OUT_OF_MEMORY; }
    pushUnary(k1b, B, mT1b.get());         // t1b = B + 1
    pushUnary(k2b, mT1b.get(), mT2b.get());// t2b = t1b * 2
    release(mT1b);

    // ===== func3: fan-in of both branch results (needs t2a AND t2b live) =====
    if (!acquire(mT3)) { return OUT_OF_MEMORY; }
    pushFunc3(k3, mT2a.get(), mT2b.get(), mT3.get()); // t3 = t2a + t2b
    release(mT2a);
    release(mT2b);

    // ===== output run 0: func4(offset0) -> func5 -> out0 =====
    if (!acquire(mT4a)) { return OUT_OF_MEMORY; }
    pushFunc4(k4a, mT3.get(), mT4a.get(), mOffset0); // t4a = t3 + offset0
    pushUnary(k5a, mT4a.get(), out0);                // out0 = t4a * 3
    release(mT4a);

    // ===== output run 1: func4(offset1) -> func5 -> out1 =====
    if (!acquire(mT4b)) { return OUT_OF_MEMORY; }
    pushFunc4(k4b, mT3.get(), mT4b.get(), mOffset1); // t4b = t3 + offset1
    pushUnary(k5b, mT4b.get(), out1);                // out1 = t4b * 3
    release(mT4b);
    release(mT3);

    MNN_CHECK_CL_SUCCESS(ret, "setArg MockChainBufExecution");
    return NO_ERROR;
}

class MockChainBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (inputs.size() != 2 || outputs.size() != 2) {
            return nullptr;
        }
        // Template runs on plain (non-NC4HW4) float buffers; anything else falls
        // back to the CPU reference.
        for (int i = 0; i < inputs.size(); ++i) {
            if (inputs[i]->getType().code != halide_type_float) {
                return nullptr;
            }
            if (TensorUtils::getDescribe(inputs[i])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                return nullptr;
            }
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        OPENCL_CREATOR_CHECK(new MockChainBufExecution(op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(MockChainBufCreator, OpType_MockChain, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
