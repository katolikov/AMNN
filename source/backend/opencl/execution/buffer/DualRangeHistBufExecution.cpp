//
//  DualRangeHistBufExecution.cpp
//  MNN
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "DualRangeHistBufExecution.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include <cstdlib>

namespace MNN {
namespace OpenCL {

static const int kLocalSize = 256;
// Register path holds two priv[BIN_NUM] histograms and a fused reduction buffer
// of BIN_NUM*LOCAL_SIZE ints; both stay valid only while BIN_NUM stays small.
// (BIN_NUM*LOCAL_SIZE <= 4096 at binNum 16.) Larger binNum falls back to CPU.
static const int kRegisterBinCap = 16;

DualRangeHistBufExecution::DualRangeHistBufExecution(const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto param = op->main_as_DualRangeHistParam();
    mBinNum         = param->binNum();
    mLow            = param->low();
    mHigh           = param->high();
    mSampleStride   = param->sampleStride() > 1 ? param->sampleStride() : 1;
    mEmitValidCount = param->emitValidCount();
}

DualRangeHistBufExecution::~DualRangeHistBufExecution() = default;

ErrorCode DualRangeHistBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const bool hasBase = (inputs.size() == 3);
    MNN_ASSERT((inputs.size() == 2 || hasBase));
    MNN_ASSERT((int)outputs.size() == (mEmitValidCount ? 3 : 2));
    auto A = inputs[0];
    auto B = inputs[1];
    auto histA = outputs[0];
    auto histB = outputs[1];
    const int n = A->elementSize();
    MNN_ASSERT(n > 0);

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    std::set<std::string> buildOptions;
    buildOptions.insert("-DBIN_NUM=" + std::to_string(mBinNum));
    if (hasBase) {
        buildOptions.insert("-DHAS_BASE");
        if (inputs[2]->getType().code == halide_type_float) {
            buildOptions.insert("-DBASE_FLOAT");
        }
    }
    if (mEmitValidCount) {
        buildOptions.insert("-DEMIT_VALIDCOUNT");
    }

    // Spatial downsampling geometry over the last two dims (batch == 1).
    const int stride = mSampleStride;
    const bool sampled = stride > 1;
    const int rank = A->dimensions();
    const int W  = rank >= 1 ? A->length(rank - 1) : 1;
    const int H  = rank >= 2 ? A->length(rank - 2) : 1;
    const int Ws = (W + stride - 1) / stride;
    const int Hs = (H + stride - 1) / stride;
    const int nSampled = Hs * Ws;   // == n when stride == 1 (batch == 1)
    const int workItems = sampled ? nSampled : n;

    auto initKernel  = runtime->buildKernel("dualrangehist_buf", "dualrangehist_init_buf", buildOptions, mOpenCLBackend->getPrecision());
    const char* countName = sampled ? "dualrangehist_sample_buf" : "dualrangehist_count_buf";
    auto countKernel = runtime->buildKernel("dualrangehist_buf", countName, buildOptions, mOpenCLBackend->getPrecision());
    if (initKernel == nullptr || countKernel == nullptr) {
        return NOT_SUPPORT;
    }

    const float low = mLow, high = mHigh;
    cl_int ret = CL_SUCCESS;
    {
        uint32_t idx = 0;
        ret |= initKernel->get().setArg(idx++, openCLBuffer(histA));
        ret |= initKernel->get().setArg(idx++, openCLBuffer(histB));
        if (mEmitValidCount) {
            ret |= initKernel->get().setArg(idx++, openCLBuffer(outputs[2]));
        }
        ret |= initKernel->get().setArg(idx++, mBinNum);
    }

    // Grid sizing: the merge (per-workgroup reduction + BIN_NUM global atomics)
    // is fixed overhead per workgroup; the sweet spot is a small multiple of the
    // compute-unit count that lets each work-item grid-stride over many elements.
    // (Same CU*4 basin found for BinCount on this device.)
    const int cu = (int)runtime->deviceComputeUnits();
    int wgCap = std::max(1, cu) * 4;
    if (const char* e = getenv("MNN_DUALHIST_WG")) {
        wgCap = std::max(1, atoi(e));
    }
    const int workGroups = std::max(1, std::min(wgCap, (workItems + kLocalSize - 1) / kLocalSize));
    const uint32_t countGlobal = (uint32_t)workGroups * kLocalSize;
    const uint32_t countLocal  = kLocalSize;
    {
        uint32_t idx = 0;
        ret |= countKernel->get().setArg(idx++, openCLBuffer(A));
        ret |= countKernel->get().setArg(idx++, openCLBuffer(B));
        if (hasBase) {
            ret |= countKernel->get().setArg(idx++, openCLBuffer(inputs[2]));
        }
        ret |= countKernel->get().setArg(idx++, openCLBuffer(histA));
        ret |= countKernel->get().setArg(idx++, openCLBuffer(histB));
        if (mEmitValidCount) {
            ret |= countKernel->get().setArg(idx++, openCLBuffer(outputs[2]));
        }
        if (sampled) {
            // sample(A, B, [base], histA, histB, [vc], nSampled, stride, W, Ws, low, high, binNum)
            ret |= countKernel->get().setArg(idx++, nSampled);
            ret |= countKernel->get().setArg(idx++, stride);
            ret |= countKernel->get().setArg(idx++, W);
            ret |= countKernel->get().setArg(idx++, Ws);
        } else {
            // count(A, B, [base], histA, histB, [vc], n, low, high, binNum)
            ret |= countKernel->get().setArg(idx++, n);
        }
        ret |= countKernel->get().setArg(idx++, low);
        ret |= countKernel->get().setArg(idx++, high);
        ret |= countKernel->get().setArg(idx++, mBinNum);
    }
    MNN_CHECK_CL_SUCCESS(ret, "setArg DualRangeHistBufExecution");

    mUnits.clear();
    mUnits.reserve(2);

    auto pushUnit = [&](std::shared_ptr<KernelWrap> kernel, uint32_t global, uint32_t local) {
        Unit unit;
        unit.kernel = kernel;
        unit.globalWorkSize = {global, 1, 1};
        unit.localWorkSize  = local > 0 ? cl::NDRange(local, 1, 1) : cl::NDRange(1, 1, 1);
        mUnits.emplace_back(std::move(unit));
    };

    const uint32_t initGlobal = (uint32_t)(2 * mBinNum + (mEmitValidCount ? 1 : 0));
    pushUnit(initKernel, initGlobal, 0);
    pushUnit(countKernel, countGlobal, countLocal);

    return NO_ERROR;
}

class DualRangeHistBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto param = op->main_as_DualRangeHistParam();
        // Register-fused path requires a small, compile-time-fixed bin count.
        if (param == nullptr || param->binNum() <= 0 || param->binNum() > kRegisterBinCap) {
            return nullptr;
        }
        // Two frames (+ optional base mask); frames must be float (fp32/fp16-buffer).
        const bool hasBase = (inputs.size() == 3);
        if (inputs.size() != 2 && !hasBase) {
            return nullptr;
        }
        if (inputs[0]->getType().code != halide_type_float || inputs[1]->getType().code != halide_type_float) {
            return nullptr;
        }
        auto typeOk = [](const Tensor* t) {
            const auto h = t->getType();
            return (h == halide_type_of<int>()) || (h.code == halide_type_float);
        };
        if (hasBase && !typeOk(inputs[2])) {
            return nullptr;
        }
        for (int i = 0; i < inputs.size(); ++i) {
            if (TensorUtils::getDescribe(inputs[i])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                return nullptr;
            }
        }
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        OPENCL_CREATOR_CHECK(new DualRangeHistBufExecution(op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(DualRangeHistBufCreator, OpType_DualRangeHist, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
