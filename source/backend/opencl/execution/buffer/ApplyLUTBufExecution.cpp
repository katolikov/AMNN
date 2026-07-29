//
//  ApplyLUTBufExecution.cpp
//  MNN
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "ApplyLUTBufExecution.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include <cstdlib>

namespace MNN {
namespace OpenCL {

static const int kLocalSize = 256;
static const int kMaxLut    = 256;   // matches MAX_LUT in applylut_buf.cl

ApplyLUTBufExecution::ApplyLUTBufExecution(const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto param = op->main_as_ApplyLUTParam();
    if (param != nullptr) {
        mInterp = param->interp();
    }
    // Tuned defaults for Xclipse 960 / ANGLE-over-Vulkan (measured, min kernel
    // time over the [1,1,1440,1920] frame): the op is occupancy/latency-bound,
    // not transaction-width-bound, so narrow vectors (more work-items) plus
    // cached-global LUT reads (no local buffer or barrier -> higher occupancy)
    // win. VEC=2 + global ~315us vs VEC=8 + local ~410us. Both are env-tunable.
    mVec = 2;
    if (const char *e = getenv("MNN_APPLYLUT_VEC")) {
        int v = atoi(e);
        if (v == 2 || v == 4 || v == 8 || v == 16) {
            mVec = v;
        }
    }
    // LUT storage: cached-global reads (default) vs local-memory staging.
    mLutGlobal = true;
    if (const char *e = getenv("MNN_APPLYLUT_LUTGLOBAL")) {
        mLutGlobal = atoi(e) != 0;
    }
}

ApplyLUTBufExecution::~ApplyLUTBufExecution() = default;

ErrorCode ApplyLUTBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 2);
    MNN_ASSERT(outputs.size() == 1);
    auto input  = inputs[0];
    auto lut    = inputs[1];
    auto output = outputs[0];

    const int n = input->elementSize();
    const int lutSize = lut->elementSize();
    MNN_ASSERT(n > 0);
    MNN_ASSERT(lutSize >= 2 && lutSize <= kMaxLut);

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    std::set<std::string> buildOptions;
    buildOptions.insert("-DVEC=" + std::to_string(mVec));
    buildOptions.insert("-DMAX_LUT=" + std::to_string(kMaxLut));
    if (mLutGlobal) {
        buildOptions.insert("-DLUT_GLOBAL");
    }

    auto kernel = runtime->buildKernel("applylut_buf", "applylut_buf", buildOptions, mOpenCLBackend->getPrecision());
    if (kernel == nullptr) {
        return NOT_SUPPORT;
    }

    // One work-item per VEC contiguous elements; the kernel guards the tail and
    // any padding work-items rounded up to the local size.
    const int nVec = UP_DIV(n, mVec);
    const uint32_t global = (uint32_t)ROUND_UP(nVec, kLocalSize);
    const uint32_t local  = (uint32_t)kLocalSize;

    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= kernel->get().setArg(idx++, openCLBuffer(lut));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, n);
    ret |= kernel->get().setArg(idx++, lutSize);
    MNN_CHECK_CL_SUCCESS(ret, "setArg ApplyLUTBufExecution");

    mUnits.clear();
    Unit unit;
    unit.kernel         = kernel;
    unit.globalWorkSize = {global, 1, 1};
    unit.localWorkSize  = {local, 1, 1};
    mUnits.emplace_back(std::move(unit));

    return NO_ERROR;
}

class ApplyLUTBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (inputs.size() != 2 || outputs.size() != 1) {
            return nullptr;
        }
        // Image (input 0) and LUT (input 1) must both be float; LUT length in
        // [2, kMaxLut] to fit the local-memory stage.
        if (inputs[0]->getType().code != halide_type_float ||
            inputs[1]->getType().code != halide_type_float) {
            return nullptr;
        }
        const int lutSize = inputs[1]->elementSize();
        if (lutSize < 2 || lutSize > kMaxLut) {
            return nullptr;
        }
        // Force the dense flat layout the kernel assumes.
        for (int i = 0; i < (int)inputs.size(); ++i) {
            if (TensorUtils::getDescribe(inputs[i])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                return nullptr;
            }
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < (int)outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        OPENCL_CREATOR_CHECK(new ApplyLUTBufExecution(op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(ApplyLUTBufCreator, OpType_ApplyLUT, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
