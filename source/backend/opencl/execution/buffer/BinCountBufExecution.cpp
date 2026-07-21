//
//  BinCountBufExecution.cpp
//  MNN
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "BinCountBufExecution.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include <cstdlib>

namespace MNN {
namespace OpenCL {

static const int kLocalSize = 256;
// Register array size cap: above this the per-work-item private histogram is
// too large to stay in registers, so we defer to the CPU reference instead.
static const int kMaxBins = 256;

BinCountBufExecution::BinCountBufExecution(const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto param = op->main_as_BinCountParam();
    mBinNum = param->binNum();
}

BinCountBufExecution::~BinCountBufExecution() = default;

ErrorCode BinCountBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto input  = inputs[0];
    auto output = outputs[0];
    const int n = input->elementSize();
    MNN_ASSERT(n > 0);

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    std::set<std::string> buildOptions;
    buildOptions.insert("-DBIN_NUM=" + std::to_string(mBinNum));
    // Float input (fp32 logically; stored as half in fp16 buffer mode) is read
    // through the precision FLOAT macro and truncated to a bin index; integer
    // input is read as int32 directly.
    const bool floatInput = (input->getType().code == halide_type_float);
    if (floatInput) {
        buildOptions.insert("-DBINCOUNT_IN_FLOAT");
    }

    auto initKernel  = runtime->buildKernel("bincount_buf", "bincount_init_buf", buildOptions, mOpenCLBackend->getPrecision());
    // Env toggle: MNN_BINCOUNT_NAIVE=1 selects the global-atomic-per-element
    // baseline instead of the register-histogram path (on-device benchmarking).
    const bool useNaive = (getenv("MNN_BINCOUNT_NAIVE") != nullptr);
    auto countKernel = runtime->buildKernel("bincount_buf",
                                            useNaive ? "bincount_naive_buf" : "bincount_count_buf",
                                            buildOptions, mOpenCLBackend->getPrecision());
    if (initKernel == nullptr || countKernel == nullptr) {
        return NOT_SUPPORT;
    }

    cl_int ret = CL_SUCCESS;
    {
        uint32_t idx = 0;
        ret |= initKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= initKernel->get().setArg(idx++, mBinNum);
    }

    uint32_t countGlobal;
    uint32_t countLocal;
    if (useNaive) {
        countLocal  = kLocalSize;
        countGlobal = (uint32_t)((n + kLocalSize - 1) / kLocalSize) * kLocalSize;
    } else {
        // Grid sizing: the merge (per-workgroup reduction + BIN_NUM global
        // atomics) is fixed overhead per workgroup, so the sweet spot is a
        // small multiple of the compute-unit count that lets each work-item
        // grid-stride over many elements -- NOT maxed-out occupancy. An
        // on-device sweep (N=2.76M) bottomed out around 4x compute units;
        // above that the redundant reduction + atomic traffic dominates.
        const int cu = (int)runtime->deviceComputeUnits();
        int wgCap = std::max(1, cu) * 4;
        if (const char* e = getenv("MNN_BINCOUNT_WG")) {
            wgCap = std::max(1, atoi(e));
        }
        const int workGroups = std::max(1, std::min(wgCap, (n + kLocalSize - 1) / kLocalSize));
        countLocal  = kLocalSize;
        countGlobal = (uint32_t)workGroups * kLocalSize;
    }
    {
        uint32_t idx = 0;
        ret |= countKernel->get().setArg(idx++, openCLBuffer(input));
        ret |= countKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= countKernel->get().setArg(idx++, n);
        ret |= countKernel->get().setArg(idx++, mBinNum);
    }
    MNN_CHECK_CL_SUCCESS(ret, "setArg BinCountBufExecution");

    mUnits.clear();
    mUnits.reserve(2);

    auto pushUnit = [&](std::shared_ptr<KernelWrap> kernel, uint32_t global, uint32_t local) {
        Unit unit;
        unit.kernel = kernel;
        unit.globalWorkSize = {global, 1, 1};
        unit.localWorkSize  = local > 0 ? cl::NDRange(local, 1, 1) : cl::NDRange(1, 1, 1);
        mUnits.emplace_back(std::move(unit));
    };

    pushUnit(initKernel, (uint32_t)mBinNum, 0);
    pushUnit(countKernel, countGlobal, countLocal);

    return NO_ERROR;
}

class BinCountBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        // Unweighted only on GPU; weighted (2 inputs) falls back to CPU.
        if (inputs.size() != 1) {
            return nullptr;
        }
        // Register-histogram path requires a small, compile-time-fixed bin count
        // and an int32 input; otherwise fall back to CPU.
        auto param = op->main_as_BinCountParam();
        if (param == nullptr || param->binNum() <= 0 || param->binNum() > kMaxBins) {
            return nullptr;
        }
        // Accept int32 or float (fp32/fp16-buffer) input; the count kernel is
        // specialized per input type. Other dtypes fall back to CPU.
        const auto t = inputs[0]->getType();
        const bool okType = (t == halide_type_of<int>()) || (t.code == halide_type_float);
        if (!okType) {
            return nullptr;
        }
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        OPENCL_CREATOR_CHECK(new BinCountBufExecution(op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(BinCountBufCreator, OpType_BinCount, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
