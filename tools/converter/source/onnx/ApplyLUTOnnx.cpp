//
//  ApplyLUTOnnx.cpp
//  MNN
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ApplyLUTOnnx);

MNN::OpType ApplyLUTOnnx::opType() {
    return MNN::OpType_ApplyLUT;
}
MNN::OpParameter ApplyLUTOnnx::type() {
    return MNN::OpParameter_ApplyLUTParam;
}

// ONNX has no standard 1-D interpolated LUT operator, so this maps a custom node
// with op_type "ApplyLUT". Input 0 is the float image ([1,1,H,W], values ~[0,1]);
// input 1 is the LUT, a 1-D float tensor of length N. Optional attribute:
//   interp : int   1 = linear (default, only mode implemented), 0 = nearest.
// N is derived from the LUT tensor at runtime; there is no bin-count attribute.
void ApplyLUTOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                       OnnxScope* scope) {
    auto param    = new MNN::ApplyLUTParamT;
    param->interp = 1;
    for (const auto& attr : onnxNode->attribute()) {
        if (attr.name() == "interp" || attr.name() == "mode") {
            param->interp = attr.i();
        }
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ApplyLUTOnnx, ApplyLUT);
