//
//  BinCountOnnx.cpp
//  MNN
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(BinCountOnnx);

MNN::OpType BinCountOnnx::opType() {
    return MNN::OpType_BinCount;
}
MNN::OpParameter BinCountOnnx::type() {
    return MNN::OpParameter_BinCountParam;
}

// ONNX has no standard BinCount operator, so this maps a custom node with
// op_type "BinCount" and an integer "binNum" (alias "minlength") attribute
// giving the fixed number of output bins. Input 0 is the integer values; an
// optional input 1 provides per-element weights (float weight-sums) or, when
// the "binaryMask" attribute is set, a binary mask (int32 counts of kept
// elements).
void BinCountOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                       OnnxScope* scope) {
    auto param = new MNN::BinCountParamT;
    param->binNum = 0;
    param->binaryMask = false;
    param->sampleStride = 1;
    for (const auto& attr : onnxNode->attribute()) {
        if (attr.name() == "binNum" || attr.name() == "minlength") {
            param->binNum = attr.i();
        } else if (attr.name() == "binaryMask") {
            param->binaryMask = (attr.i() != 0);
        } else if (attr.name() == "sampleStride" || attr.name() == "sample_stride") {
            param->sampleStride = attr.i();
        }
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(BinCountOnnx, BinCount);
