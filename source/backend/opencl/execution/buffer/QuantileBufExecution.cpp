//
//  QuantileBufExecution.cpp
//  MNN
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "QuantileBufExecution.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "MNN_generated.h"
#include <cmath>

namespace MNN {
namespace OpenCL {

static const int kLocalSize = 256;
static const int kMaxTargets = 16;

QuantileBufExecution::QuantileBufExecution(const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto param = op->main_as_QuantileParam();
    mQLevels.resize(param->qLevels()->size());
    for (int i = 0; i < mQLevels.size(); ++i) {
        mQLevels[i] = param->qLevels()->Get(i);
    }
}

QuantileBufExecution::~QuantileBufExecution() = default;

ErrorCode QuantileBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNumTargets = 2 * (int)mQLevels.size();
    if (mNumTargets > kMaxTargets) {
        MNN_ERROR("Quantile: at most %d quantile levels supported on OpenCL buffer backend (got %d)\n",
                   kMaxTargets / 2, (int)mQLevels.size());
        return NOT_SUPPORT;
    }
    return CommonExecution::onResize(inputs, outputs);
}

ErrorCode QuantileBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto input  = inputs[0];
    auto output = outputs[0];
    const int n = input->elementSize();
    MNN_ASSERT(n > 0);

    // rank/interpolation-weight computation depends only on n and the
    // (compile-time-fixed) q levels, not on tensor contents, so it happens
    // once here at resize time -- same math as the CPU reference / the
    // ONNX-decomposition baseline's per-quantile constants.
    mTargetRank.resize(mNumTargets);
    mFrac.resize(mQLevels.size());
    for (int i = 0; i < (int)mQLevels.size(); ++i) {
        const float idx = mQLevels[i] * (n - 1);
        const int lo = (int)std::floor(idx);
        const int hi = (int)std::ceil(idx);
        mTargetRank[2 * i]     = lo;
        mTargetRank[2 * i + 1] = hi;
        mFrac[i] = idx - lo;
    }

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto context = &runtime->context();

    mLoKeyBuffer.reset(new cl::Buffer(*context, CL_MEM_READ_WRITE, mNumTargets * sizeof(uint32_t)));
    mHiKeyBuffer.reset(new cl::Buffer(*context, CL_MEM_READ_WRITE, mNumTargets * sizeof(uint32_t)));
    mCountBuffer.reset(new cl::Buffer(*context, CL_MEM_READ_WRITE, mNumTargets * sizeof(uint32_t)));
    mTargetRankBuffer.reset(new cl::Buffer(*context, CL_MEM_READ_WRITE, mNumTargets * sizeof(int32_t)));
    mFracBuffer.reset(new cl::Buffer(*context, CL_MEM_READ_WRITE, mQLevels.size() * sizeof(float)));

    runtime->commandQueue().enqueueWriteBuffer(*mTargetRankBuffer, CL_TRUE, 0, mNumTargets * sizeof(int32_t),
                                                mTargetRank.data());
    runtime->commandQueue().enqueueWriteBuffer(*mFracBuffer, CL_TRUE, 0, mFrac.size() * sizeof(float), mFrac.data());

    std::set<std::string> buildOptions;
    // Size private arrays to the actual target count rather than the fixed
    // kMaxTargets(16) cap: measured ~10% faster at low target counts with no
    // regression at 16, since it avoids allocating/spilling unused lanes.
    buildOptions.insert("-DMAX_TARGETS=" + std::to_string(mNumTargets));

    auto initKernel     = runtime->buildKernel("quantile_buf", "quantile_init_buf", buildOptions, mOpenCLBackend->getPrecision());
    auto countKernel     = runtime->buildKernel("quantile_buf", "quantile_count_buf", buildOptions, mOpenCLBackend->getPrecision());
    auto updateKernel    = runtime->buildKernel("quantile_buf", "quantile_update_buf", buildOptions, mOpenCLBackend->getPrecision());
    auto finalizeKernel  = runtime->buildKernel("quantile_buf", "quantile_finalize_buf", buildOptions, mOpenCLBackend->getPrecision());
    if (initKernel == nullptr || countKernel == nullptr || updateKernel == nullptr || finalizeKernel == nullptr) {
        return NOT_SUPPORT;
    }

    // enough workgroups to keep the GPU busy; each work-item grid-strides
    // over the remainder, so this doesn't need to divide n evenly.
    const int workGroups = std::max(1, std::min(512, (n + kLocalSize - 1) / kLocalSize));
    const uint32_t countGlobal = (uint32_t)workGroups * kLocalSize;

    cl_int ret = CL_SUCCESS;
    {
        uint32_t idx = 0;
        ret |= initKernel->get().setArg(idx++, *mLoKeyBuffer);
        ret |= initKernel->get().setArg(idx++, *mHiKeyBuffer);
        ret |= initKernel->get().setArg(idx++, *mCountBuffer);
        ret |= initKernel->get().setArg(idx++, mNumTargets);
    }
    {
        uint32_t idx = 0;
        ret |= countKernel->get().setArg(idx++, openCLBuffer(input));
        ret |= countKernel->get().setArg(idx++, *mLoKeyBuffer);
        ret |= countKernel->get().setArg(idx++, *mHiKeyBuffer);
        ret |= countKernel->get().setArg(idx++, *mCountBuffer);
        ret |= countKernel->get().setArg(idx++, n);
        ret |= countKernel->get().setArg(idx++, mNumTargets);
    }
    {
        uint32_t idx = 0;
        ret |= updateKernel->get().setArg(idx++, *mLoKeyBuffer);
        ret |= updateKernel->get().setArg(idx++, *mHiKeyBuffer);
        ret |= updateKernel->get().setArg(idx++, *mCountBuffer);
        ret |= updateKernel->get().setArg(idx++, *mTargetRankBuffer);
        ret |= updateKernel->get().setArg(idx++, mNumTargets);
    }
    {
        uint32_t idx = 0;
        ret |= finalizeKernel->get().setArg(idx++, *mLoKeyBuffer);
        ret |= finalizeKernel->get().setArg(idx++, *mFracBuffer);
        ret |= finalizeKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= finalizeKernel->get().setArg(idx++, (int)mQLevels.size());
    }
    MNN_CHECK_CL_SUCCESS(ret, "setArg QuantileBufExecution");

    mUnits.clear();
    mUnits.reserve(2 + 2 * kIters);

    auto pushUnit = [&](std::shared_ptr<KernelWrap> kernel, uint32_t global, uint32_t local) {
        Unit unit;
        unit.kernel = kernel;
        unit.globalWorkSize = {global, 1, 1};
        unit.localWorkSize  = local > 0 ? cl::NDRange(local, 1, 1) : cl::NDRange(1, 1, 1);
        mUnits.emplace_back(std::move(unit));
    };

    pushUnit(initKernel, (uint32_t)mNumTargets, 0);
    for (int it = 0; it < kIters; ++it) {
        pushUnit(countKernel, countGlobal, kLocalSize);
        pushUnit(updateKernel, (uint32_t)mNumTargets, 0);
    }
    pushUnit(finalizeKernel, (uint32_t)mQLevels.size(), 0);

    return NO_ERROR;
}

class QuantileBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        OPENCL_CREATOR_CHECK(new QuantileBufExecution(op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(QuantileBufCreator, OpType_Quantile, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
