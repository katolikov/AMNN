//
//  ShapeQuantile.cpp
//  MNN
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

// Quantile reduces the fully-flattened input tensor to one scalar per
// requested quantile level, output is always a 1-D float tensor of length
// qLevels.size().
class QuantileSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto param = op->main_as_QuantileParam();
        MNN_ASSERT(nullptr != param && nullptr != param->qLevels());
        auto output = outputs[0];
        output->buffer().dimensions = 1;
        output->setLength(0, param->qLevels()->size());
        output->buffer().type = halide_type_of<float>();
        TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        return true;
    }
};

REGISTER_SHAPE(QuantileSizeComputer, OpType_Quantile);

} // namespace MNN
