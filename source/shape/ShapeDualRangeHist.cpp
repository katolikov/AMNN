//
//  ShapeDualRangeHist.cpp
//  MNN
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

// DualRangeHist: single-pass dual masked range-histogram over two float frames
// A (input 0) and B (input 1), plus an optional per-element base mask (input 2).
// Outputs are two 1-D int32 histograms of length binNum (histA, histB) and,
// when emitValidCount is set, a third 1-D int32 scalar-length output holding
// sum(keep). Only batch == 1 frames are supported.
class DualRangeHistSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto param = op->main_as_DualRangeHistParam();
        MNN_ASSERT(nullptr != param);
        if (param == nullptr) {
            return false;
        }
        const int binNum = param->binNum();
        if (binNum <= 0) {
            return false;
        }
        // Two frames (+optional base mask) in; two histograms (+optional count) out.
        if (inputs.size() != 2 && inputs.size() != 3) {
            return false;
        }
        const bool emitValidCount = param->emitValidCount();
        const int expectedOutputs = emitValidCount ? 3 : 2;
        if ((int)outputs.size() != expectedOutputs) {
            return false;
        }

        // Only batch == 1 is supported: everything outside the last two dims must
        // collapse to a single plane.
        auto a = inputs[0];
        const int rank = a->dimensions();
        const int W  = rank >= 1 ? a->length(rank - 1) : 1;
        const int H  = rank >= 2 ? a->length(rank - 2) : 1;
        const int HW = H * W;
        const int planes = HW > 0 ? a->elementSize() / HW : 1;
        if (planes != 1) {
            return false;
        }

        auto setHist = [&](Tensor* t) {
            t->buffer().dimensions = 1;
            t->setLength(0, binNum);
            t->buffer().type = halide_type_of<int>();
            TensorUtils::getDescribe(t)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        };
        setHist(outputs[0]);
        setHist(outputs[1]);
        if (emitValidCount) {
            auto c = outputs[2];
            c->buffer().dimensions = 1;
            c->setLength(0, 1);
            c->buffer().type = halide_type_of<int>();
            TensorUtils::getDescribe(c)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        }
        return true;
    }
};

REGISTER_SHAPE(DualRangeHistSizeComputer, OpType_DualRangeHist);

} // namespace MNN
