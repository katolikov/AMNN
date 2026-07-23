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
// Above this bin count the register-histogram spills; use the local-memory
// histogram kernel instead. (16 is where the fused reduction stops fitting.)
static const int kRegisterBinCap = 16;

BinCountBufExecution::BinCountBufExecution(const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto param = op->main_as_BinCountParam();
    mBinNum = param->binNum();
    mBinaryMask = param->binaryMask();
    mSampleStride = param->sampleStride() > 1 ? param->sampleStride() : 1;
}

BinCountBufExecution::~BinCountBufExecution() = default;

ErrorCode BinCountBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const bool masked = mBinaryMask && (inputs.size() == 2);
    MNN_ASSERT((inputs.size() == 1 || masked) && outputs.size() == 1);
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
    if (masked) {
        buildOptions.insert("-DBINCOUNT_MASK");
        if (inputs[1]->getType().code == halide_type_float) {
            buildOptions.insert("-DBINCOUNT_MASK_FLOAT");
        }
    }

    // Spatial downsampling geometry over the last two dims (input[..,::s,::s]).
    const int stride = mSampleStride;
    const bool sampled = stride > 1;
    const int rank = input->dimensions();
    const int W  = rank >= 1 ? input->length(rank - 1) : 1;
    const int H  = rank >= 2 ? input->length(rank - 2) : 1;
    const int HW = H * W;
    const int planes = HW > 0 ? n / HW : 1;
    const int Ws = (W + stride - 1) / stride;
    const int Hs = (H + stride - 1) / stride;
    const int HsWs = Hs * Ws;
    const int nSampled = planes * HsWs;   // == n when stride == 1

    // Large binNum: the register-histogram (priv[BIN_NUM] + BIN_NUM-way
    // reduction) spills and collapses above ~16 bins, so switch to a
    // local-memory histogram (atomic-add into a per-workgroup on-chip
    // histogram). Env MNN_BINCOUNT_LOCAL=1/0 forces the path for A/B benchmarks.
    bool useLocal = mBinNum > kRegisterBinCap;
    if (const char* e = getenv("MNN_BINCOUNT_LOCAL")) {
        useLocal = atoi(e) != 0;
    }
    // The local and sampled kernels both take the sampled-grid geometry.
    const bool geomArgs = sampled || useLocal;
    const int workItems = geomArgs ? nSampled : n;

    auto initKernel  = runtime->buildKernel("bincount_buf", "bincount_init_buf", buildOptions, mOpenCLBackend->getPrecision());
    // Env toggle: MNN_BINCOUNT_NAIVE=1 selects the global-atomic-per-element
    // baseline instead of the register-histogram path (on-device benchmarking).
    // The naive kernel has no mask/sample/local variant, so it is disabled then.
    const bool useNaive = (getenv("MNN_BINCOUNT_NAIVE") != nullptr) && !masked && !sampled && !useLocal;
    const char* countName = useLocal  ? "bincount_local_buf"
                          : sampled   ? "bincount_sample_buf"
                          : useNaive  ? "bincount_naive_buf"
                                      : "bincount_count_buf";
    auto countKernel = runtime->buildKernel("bincount_buf", countName, buildOptions, mOpenCLBackend->getPrecision());
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
        countGlobal = (uint32_t)((workItems + kLocalSize - 1) / kLocalSize) * kLocalSize;
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
        const int workGroups = std::max(1, std::min(wgCap, (workItems + kLocalSize - 1) / kLocalSize));
        countLocal  = kLocalSize;
        countGlobal = (uint32_t)workGroups * kLocalSize;
    }
    {
        uint32_t idx = 0;
        ret |= countKernel->get().setArg(idx++, openCLBuffer(input));
        if (masked) {
            ret |= countKernel->get().setArg(idx++, openCLBuffer(inputs[1]));
        }
        ret |= countKernel->get().setArg(idx++, openCLBuffer(output));
        if (geomArgs) {
            // sample/local kernel(input, [mask], output, nSampled, stride, W, Ws, HsWs, HW, binNum)
            ret |= countKernel->get().setArg(idx++, nSampled);
            ret |= countKernel->get().setArg(idx++, stride);
            ret |= countKernel->get().setArg(idx++, W);
            ret |= countKernel->get().setArg(idx++, Ws);
            ret |= countKernel->get().setArg(idx++, HsWs);
            ret |= countKernel->get().setArg(idx++, HW);
        } else {
            ret |= countKernel->get().setArg(idx++, n);
        }
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
        // Register-histogram path requires a small, compile-time-fixed bin count
        // and an int32/float input; otherwise fall back to CPU.
        auto param = op->main_as_BinCountParam();
        if (param == nullptr || param->binNum() <= 0 || param->binNum() > kMaxBins) {
            return nullptr;
        }
        // One input (counts) or two inputs with binaryMask (masked counts) run
        // on GPU; float weight-sums (two inputs, no mask) fall back to CPU.
        const bool masked = (inputs.size() == 2) && param->binaryMask();
        if (inputs.size() != 1 && !masked) {
            return nullptr;
        }
        // Accept int32 or float (fp32/fp16-buffer) value/mask; the count kernel
        // is specialized per type. Other dtypes fall back to CPU.
        auto typeOk = [](const Tensor* t) {
            const auto h = t->getType();
            return (h == halide_type_of<int>()) || (h.code == halide_type_float);
        };
        if (!typeOk(inputs[0]) || (masked && !typeOk(inputs[1]))) {
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
