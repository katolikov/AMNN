//
//  DualRangeHistOnnx.cpp
//  MNN
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(DualRangeHistOnnx);

MNN::OpType DualRangeHistOnnx::opType() {
    return MNN::OpType_DualRangeHist;
}
MNN::OpParameter DualRangeHistOnnx::type() {
    return MNN::OpParameter_DualRangeHistParam;
}

// ONNX has no standard dual masked range-histogram operator, so this maps a
// custom node with op_type "DualRangeHist". Inputs 0 and 1 are the float frames
// A and B; an optional input 2 is a per-element base validity mask. Attributes:
//   binNum (alias minlength) : int   number of bins per histogram (output length)
//   low, high                : float shared inclusive validity range on raw values
//   sampleStride (sample_stride): int >1 downsamples the last two dims
//   emitValidCount (valid_count): int 0/1 - emit a 3rd int32 output = sum(keep)
// Outputs: histA[binNum], histB[binNum] (int32), and optionally validCount[1].
void DualRangeHistOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                            OnnxScope* scope) {
    auto param = new MNN::DualRangeHistParamT;
    param->binNum         = 0;
    param->low            = 0.0f;
    param->high           = 1.0f;
    param->sampleStride   = 1;
    // Default emitValidCount from the node's output count (3 outputs => on); an
    // explicit "emitValidCount"/"valid_count" attribute below overrides it.
    param->emitValidCount = (onnxNode->output_size() >= 3);
    for (const auto& attr : onnxNode->attribute()) {
        if (attr.name() == "binNum" || attr.name() == "minlength") {
            param->binNum = attr.i();
        } else if (attr.name() == "low") {
            param->low = attr.f();
        } else if (attr.name() == "high") {
            param->high = attr.f();
        } else if (attr.name() == "sampleStride" || attr.name() == "sample_stride") {
            param->sampleStride = attr.i();
        } else if (attr.name() == "emitValidCount" || attr.name() == "valid_count") {
            param->emitValidCount = (attr.i() != 0);
        }
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(DualRangeHistOnnx, DualRangeHist);
