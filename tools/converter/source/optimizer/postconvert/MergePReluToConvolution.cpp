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
#include "../Global.hpp"
#include "MergeToConvolution.hpp"
#include "config.hpp"

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
        // MergeToConvolution::onExecute mutates nothing itself: it calls us first and only
        // then checks _isSingleInputOutput, dropping the merge if that fails. Since we write
        // the slope into the conv, a dropped merge would leave BOTH the slope and the original
        // activation op in the graph -- applied twice. Check it here, before touching anything.
        if (!PostTreatUtils::_isSingleInputOutput(inplaceOp)) {
            return false;
        }
        auto conv = convolutionOp->main.AsConvolution2D();
        auto common = conv->common.get();
        // Dense conv only (group==1). Any kernel/stride/dilation/channel count is fine -- see
        // the header comment for the execution paths that consume the slope. Quantized convs
        // (the low-memory path) do NOT read it, so skip those.
        if (common->group != 1) return false;
        if (conv->quanParameter != nullptr) return false;
        // Dynamic weights arrive as a second input, which the OpenCL image creator declines;
        // the op then falls back to cpu and the runtime gate refuses the session. Fusing here
        // would turn a working model into one that cannot load.
        if (convolutionOp->inputIndexes.size() > 1) return false;
        // Weight quantisation is applied by writeFb AFTER this pass, so the check above cannot
        // see it. A conv that ends up with BOTH a slope and a quanParameter routes to the OpenCL
        // low-memory execution, which does not read the slope. Weight quantisation is the larger
        // win of the two, so keep it and skip the fusion -- loudly, not silently.
        auto converterConfig = Global<modelConfig>::Get();
        if (nullptr != converterConfig && converterConfig->weightQuantBits > 0) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                MNN_PRINT("Warning: --fusePreluToConv is ignored because the weights are being "
                          "quantized (--weightQuantBits). Quantized convs run through the OpenCL "
                          "low-memory path, which does not apply a fused activation.\n");
            }
            return false;
        }
        if (common->outputCount <= 0) return false;
        if (common->relu || common->relu6) return false;            // already has an activation
        if (!common->leakyReluSlope.empty()) return false;          // already fused

        const int co = common->outputCount;
        if (isLeakyRelu) {
            common->leakyReluSlope.assign(co, inplaceOp->main.AsRelu()->slope);
            return true;
        }
        auto prelu = inplaceOp->main.AsPRelu();
        // slopeCount is metadata; the vector holds the values. Copying a short one would emit a
        // model the runtime gate refuses at load time, so decline the fusion instead.
        if (prelu->slope.size() < (size_t)prelu->slopeCount) return false;
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
