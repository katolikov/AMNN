//
//  MergePReluToConvolution.cpp
//  MNNConverter
//
//  Fuse a per-channel PReLU, or a scalar LeakyReLU, into the preceding convolution by storing
//  the slopes on the conv
//  (Convolution2DCommon.leakyReluSlope), so the activation costs one FMA at the point where the
//  conv already holds the result instead of a second full pass over the output tensor.
//
//  Opt-in only: the pass is registered into the post-convert chain ONLY when MNNConvert is
//  given --fusePreluToConv (see PostConverter.cpp), because only the OpenCL backend reads
//  leakyReluSlope -- a fused model run on CPU/Metal/Vulkan/CUDA loses the activation silently.
//
//  Fuses into any dense (group==1, non-quantized) convolution, whatever its kernel/stride/
//  dilation: on OpenCL such a conv runs through ConvBufWinograd / ConvWinograd (3x3 s1 d1) or
//  through ConvBufExecution / ConvExecution (everything else, incl. 1x1 / general / gemm), and
//  all four apply the per-channel slope, in both buffer and image memory mode. Depthwise,
//  grouped and quantized convs keep their separate activation op and are unaffected.
//

#include "../PostTreatUtils.hpp"
#include "MergeToConvolution.hpp"

using namespace MNN;

class MergePReluToConvolution : public MergeToConvolution {
public:
    bool merge2Convolution(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const {
        if (convolutionOp->type != MNN::OpType_Convolution) {
            return false; // dense conv only; depthwise/deconv/3D have no fused-slope path
        }
        // Two shapes of the same activation. A LeakyReLU arrives as OpType_ReLU carrying a
        // non-zero slope -- ONNX lowers a 1-element PRelu to it (OnnxPrelu.cpp), TF/TFLite emit
        // it directly. slope == 0 is plain ReLU, which MergeReluToConvolution already folds into
        // common->relu; that is cheaper (no slope buffer at all), so leave it alone.
        const bool isPRelu = (inplaceOp->type == MNN::OpType_PReLU);
        const bool isLeakyRelu = (inplaceOp->type == MNN::OpType_ReLU &&
                                  nullptr != inplaceOp->main.AsRelu() &&
                                  inplaceOp->main.AsRelu()->slope != 0.0f);
        if (!isPRelu && !isLeakyRelu) {
            return false;
        }
        auto conv = convolutionOp->main.AsConvolution2D();
        auto common = conv->common.get();
        // Dense conv only (group==1). Any kernel/stride/dilation/channel count is fine -- see
        // the header comment for the execution paths that consume the slope. Quantized convs
        // (the low-memory path) do NOT read it, so skip those.
        if (common->group != 1) return false;
        if (conv->quanParameter != nullptr) return false;
        if (common->outputCount <= 0) return false;
        if (common->relu || common->relu6) return false;            // already has an activation
        if (!common->leakyReluSlope.empty()) return false;          // already fused

        const int co = common->outputCount;
        if (isLeakyRelu) {
            common->leakyReluSlope.assign(co, inplaceOp->main.AsRelu()->slope);
            return true;
        }
        auto prelu = inplaceOp->main.AsPRelu();
        if (prelu->slopeCount == co) {
            common->leakyReluSlope = prelu->slope;                  // per-channel
        } else if (prelu->slopeCount == 1) {
            common->leakyReluSlope.assign(co, prelu->slope[0]);     // 1-slope PReLU broadcast
        } else {
            return false;                                           // not per-channel / not broadcastable
        }
        return true;
    }

    bool merge2Convolution3D(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const {
        return false; // no 3D conv execution reads leakyReluSlope
    }
};
static PostConverterRegister<MergePReluToConvolution> __l("MergePReluToConvolution");
