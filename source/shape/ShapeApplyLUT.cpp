//
//  ShapeApplyLUT.cpp
//  MNN
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

// ApplyLUT: per-element 1-D lookup-table remap. Input 0 is the image, input 1 is
// the LUT ([N] float). The output is element-wise identical in shape, dtype and
// format to input 0 (the LUT only supplies values, never geometry).
class ApplyLUTSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        if (inputs.size() != 2 || outputs.size() != 1) {
            return false;
        }
        auto input  = inputs[0];
        auto lut    = inputs[1];
        auto output = outputs[0];
        // LUT must be a non-empty 1-D-ish float table.
        if (lut->elementSize() < 2 || lut->getType().code != halide_type_float) {
            return false;
        }
        TensorUtils::copyShape(input, output, true);
        output->buffer().type = input->buffer().type;
        return true;
    }
};

REGISTER_SHAPE(ApplyLUTSizeComputer, OpType_ApplyLUT);

} // namespace MNN
