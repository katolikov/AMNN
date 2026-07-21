//
//  ShapeBinCount.cpp
//  MNN
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

// BinCount counts occurrences of each integer value in the fully-flattened
// input into a fixed number of bins. Output is always a 1-D tensor of length
// binNum. Unweighted (1 input) -> int32 counts; weighted (2 inputs, the second
// being per-element float weights) -> float32 weight-sums.
class BinCountSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto param = op->main_as_BinCountParam();
        MNN_ASSERT(nullptr != param);
        const int binNum = param->binNum();
        if (binNum <= 0) {
            return false;
        }
        auto output = outputs[0];
        output->buffer().dimensions = 1;
        output->setLength(0, binNum);
        const bool weighted = (inputs.size() == 2);
        output->buffer().type = weighted ? halide_type_of<float>() : halide_type_of<int>();
        TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        return true;
    }
};

REGISTER_SHAPE(BinCountSizeComputer, OpType_BinCount);

} // namespace MNN
