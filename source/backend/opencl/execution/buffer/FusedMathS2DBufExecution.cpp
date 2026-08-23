//
//  FusedMathS2DBufExecution.cpp
//  MNN
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/FusedMathS2DBufExecution.hpp"
#include "core/TensorUtils.hpp"
#include <cstdlib>

namespace MNN {
namespace OpenCL {

static int envInt(const char *name, int fallback) {
    const char *e = getenv(name);
    if (e == nullptr) {
        return fallback;
    }
    return atoi(e);
}

FusedMathS2DBufExecution::FusedMathS2DBufExecution(const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    auto param = op->main_as_FusedMathS2DParam();
    if (param != nullptr) {
        mAlpha   = param->alpha();
        mBeta    = param->beta();
        mGamma   = param->gamma();
        mDelta   = param->delta();
        mEpsilon = param->epsilon();
        mConst   = param->kConst();
    }
    // Tuning knobs; the shipped defaults are the measured ones on Xclipse 960.
    int vw = envInt("MNN_S2D_VW", 4);
    if (vw == 1 || vw == 2 || vw == 4 || vw == 8) {
        mVecWidth = vw;
    }
    mForcedLws = envInt("MNN_S2D_LWS", 0);

}

ErrorCode FusedMathS2DBufExecution::onEncode(const std::vector<Tensor *> &inputs,
                                             const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto openCLBackend = static_cast<OpenCLBackend *>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();

    auto input  = inputs[0];
    auto output = outputs[0];
    const int inHeight  = input->length(2);
    const int inWidth   = input->length(3);
    const int outHeight = inHeight / 2;
    const int outWidth  = inWidth / 2;
    const int planeStride = outHeight * outWidth;

    // Widest vector the output width divides exactly; no tail branch in the hot kernel.
    int vw = mVecWidth;
    while (vw > 1 && (outWidth % vw) != 0) {
        vw >>= 1;
    }
    const int outWidthV = outWidth / vw;
    const int total     = outHeight * outWidthV;

    std::set<std::string> buildOption;
    buildOption.emplace("-DVW=" + std::to_string(vw));
    buildOption.emplace("-DVW2=" + std::to_string(2 * vw));


    unit.kernel = runtime->buildKernel("fused_math_s2d_buf", "fused_math_s2d_buf", buildOption,
                                       openCLBackend->getPrecision());
    if (unit.kernel == nullptr) {
        return NOT_SUPPORT;
    }
    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    std::vector<uint32_t> globalWorkSize = {static_cast<uint32_t>(total), 1};

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[0]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[1]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, inWidth);
    ret |= unit.kernel->get().setArg(idx++, outWidth);
    ret |= unit.kernel->get().setArg(idx++, outWidthV);
    ret |= unit.kernel->get().setArg(idx++, total);
    ret |= unit.kernel->get().setArg(idx++, planeStride);
    ret |= unit.kernel->get().setArg(idx++, mAlpha);
    ret |= unit.kernel->get().setArg(idx++, mBeta);
    ret |= unit.kernel->get().setArg(idx++, mGamma);
    ret |= unit.kernel->get().setArg(idx++, mDelta);
    ret |= unit.kernel->get().setArg(idx++, mEpsilon);
    ret |= unit.kernel->get().setArg(idx++, mConst);
    MNN_CHECK_CL_SUCCESS(ret, "setArg FusedMathS2DBufExecution");

    std::vector<uint32_t> localWorkSize;
    if (mForcedLws > 0) {
        localWorkSize = {static_cast<uint32_t>(std::min<int>(mForcedLws, (int)maxWorkGroupSize)), 1};
    } else {
        std::string kernelName = "fused_math_s2d_buf";
        localWorkSize = localWS2DDefault(globalWorkSize, maxWorkGroupSize, runtime, kernelName,
                                         unit.kernel, openCLBackend->getCLTuneLevel(),
                                         "fused_math_s2d_buf").first;
    }

    openCLBackend->recordKernel2d(unit.kernel, globalWorkSize, localWorkSize);
    unit.globalWorkSize = {globalWorkSize[0], globalWorkSize[1]};
    unit.localWorkSize  = {localWorkSize[0], localWorkSize[1]};
    return NO_ERROR;
}

class FusedMathS2DBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (inputs.size() != 2 || outputs.size() != 1) {
            return nullptr;
        }
        for (int i = 0; i < (int)inputs.size(); ++i) {
            if (inputs[i]->dimensions() != 4 || inputs[i]->length(0) != 1 || inputs[i]->length(1) != 1) {
                return nullptr;
            }
            if (inputs[i]->getType().code != halide_type_float) {
                return nullptr;
            }
        }
        if ((inputs[0]->length(2) % 2) != 0 || (inputs[0]->length(3) % 2) != 0) {
            return nullptr;
        }
        // The kernel indexes dense NCHW. NC4HW4 would pad the C=1 inputs to four
        // channels, i.e. 4x the read traffic on a purely bandwidth-bound op.
        for (int i = 0; i < (int)inputs.size(); ++i) {
            if (TensorUtils::getDescribe(inputs[i])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                return nullptr;
            }
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < (int)outputs.size(); ++i) {
            if (TensorUtils::getDescribe(outputs[i])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                return nullptr;
            }
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        OPENCL_CREATOR_CHECK(new FusedMathS2DBufExecution(op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(FusedMathS2DBufCreator, OpType_FusedMathS2D, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
