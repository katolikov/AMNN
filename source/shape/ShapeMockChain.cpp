//
//  ShapeMockChain.cpp
//  MNN
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

// MockChain: two inputs (A, B), two outputs. All tensors share the same shape
// and dtype -- every kernel in the chain is elementwise -- so the output shapes
// are just copies of the input shapes. (See MockChainParam in MNN.fbs for the
// intent: this is a dispatch-skeleton template, not a real operator.)
class MockChainSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(2 == outputs.size());
        auto input = inputs[0];
        for (int o = 0; o < 2; ++o) {
            auto output = outputs[o];
            TensorUtils::copyShape(input, output, true);
            output->buffer().type = input->buffer().type;
        }
        return true;
    }
};

REGISTER_SHAPE(MockChainSizeComputer, OpType_MockChain);

} // namespace MNN
