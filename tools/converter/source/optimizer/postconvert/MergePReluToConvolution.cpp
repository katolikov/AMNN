//
//  MergePReluToConvolution.cpp
//  MNNConverter
//
//  Fuse a per-channel PReLU (LeakyReLU) into a preceding dense 3x3 stride-1 convolution
//  by storing the slopes on the conv (Convolution2DCommon.leakyReluSlope). Opt-in only:
//  the pass is registered into the post-convert chain ONLY when env MNN_FUSE_CONV_PRELU
//  is set (see PostConverter.cpp), because not every backend reads leakyReluSlope.
//
//  Fuses into any dense (group==1, non-quantized) convolution: on the OpenCL buffer backend
//  such a conv runs either through ConvBufWinograd (3x3 s1 d1, in/out>=64) or ConvBufExecution
//  (all other shapes incl. 1x1 / general / gemm), and BOTH apply the per-channel slope. Depthwise
//  / grouped / quantized convs keep their separate PReLU op and are unaffected.
//

#include "../PostTreatUtils.hpp"
#include "MergeToConvolution.hpp"

using namespace MNN;

class MergePReluToConvolution : public MergeToConvolution {
public:
    bool merge2Convolution(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const {
        if (convolutionOp->type != MNN::OpType_Convolution) {
            return false; // dense conv only (winograd path)
        }
        if (inplaceOp->type != MNN::OpType_PReLU) {
            return false;
        }
        auto conv = convolutionOp->main.AsConvolution2D();
        auto common = conv->common.get();
        // Dense conv only (group==1). Any kernel/stride/dilation/channel count is fine: a conv
        // carrying leakyReluSlope routes either to ConvBufWinograd (3x3 s1 d1, in/out>=64) or to
        // ConvBufExecution (everything else) on the OpenCL buffer backend, and both apply the
        // per-channel slope. Quantized convs (low-memory path) do NOT, so skip those.
        if (common->group != 1) return false;
        if (conv->quanParameter != nullptr) return false;
        if (common->outputCount <= 0) return false;
        if (common->relu || common->relu6) return false;            // already has an activation
        if (!common->leakyReluSlope.empty()) return false;          // already fused

        auto prelu = inplaceOp->main.AsPRelu();
        const int co = common->outputCount;
        if (prelu->slopeCount == co) {
            common->leakyReluSlope = prelu->slope;                  // per-channel
        } else if (prelu->slopeCount == 1) {
            common->leakyReluSlope.assign(co, prelu->slope[0]);     // scalar LeakyReLU broadcast
        } else {
            return false;                                           // not per-channel / not broadcastable
        }
        return true;
    }

    bool merge2Convolution3D(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const {
        return false; // 3D conv has no winograd PReLU path
    }
};
static PostConverterRegister<MergePReluToConvolution> __l("MergePReluToConvolution");
