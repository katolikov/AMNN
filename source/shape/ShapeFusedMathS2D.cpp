//
//  ShapeFusedMathS2D.cpp
//  MNN
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

// Two single-plane inputs [1,1,H,W] -> [1,13,H/2,W/2]
class FusedMathS2DSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        if (inputs.size() != 2 || outputs.size() != 1) {
            return false;
        }
        auto& ia = inputs[0]->buffer();
        auto& ib = inputs[1]->buffer();
        if (ia.dimensions != 4 || ib.dimensions != 4) {
            MNN_ERROR("FusedMathS2D requires 4D inputs\n");
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            if (ia.dim[i].extent != ib.dim[i].extent) {
                MNN_ERROR("FusedMathS2D: inputs must have identical shapes\n");
                return false;
            }
        }
        const int batch = ia.dim[0].extent;
        const int channel = ia.dim[1].extent;
        const int height = ia.dim[2].extent;
        const int width = ia.dim[3].extent;
        if (batch != 1 || channel != 1) {
            MNN_ERROR("FusedMathS2D: only [1,1,H,W] inputs are supported\n");
            return false;
        }
        if ((height % 2) != 0 || (width % 2) != 0) {
            MNN_ERROR("FusedMathS2D: H(%d) and W(%d) must be even\n", height, width);
            return false;
        }
        auto& ob = outputs[0]->buffer();
        ob.dimensions = 4;
        ob.type = ia.type;
        ob.dim[0].extent = 1;
        ob.dim[1].extent = 13;
        ob.dim[2].extent = height / 2;
        ob.dim[3].extent = width / 2;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        return true;
    }
};

REGISTER_SHAPE(FusedMathS2DSizeComputer, OpType_FusedMathS2D);

} // namespace MNN
