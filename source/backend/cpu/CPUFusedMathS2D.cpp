//
//  CPUFusedMathS2D.cpp
//  MNN
//

#include "backend/cpu/CPUFusedMathS2D.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUFusedMathS2D::CPUFusedMathS2D(Backend* b, const FusedMathS2DParam* param) : Execution(b) {
    mAlpha   = param != nullptr ? param->alpha() : 1.0f;
    mBeta    = param != nullptr ? param->beta() : 1.0f;
    mGamma   = param != nullptr ? param->gamma() : 0.0f;
    mDelta   = param != nullptr ? param->delta() : 1.0f;
    mEpsilon = param != nullptr ? param->epsilon() : 0.0f;
    mConst   = param != nullptr ? param->kConst() : 0.0f;
}

// Reference semantics, mirrored bit-for-bit by fused_math_s2d_buf.cl:
//   P0 = alpha*A + beta*B + gamma   -> channels 0..3
//   P1 = delta*B + epsilon          -> channels 4..7
//   P2 = P0 - P1                    -> channels 8..11
//   channel 12 = kConst
// Inside a plane the 2x2 block maps to (dy,dx) = 00,01,10,11.
ErrorCode CPUFusedMathS2D::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto inputA = inputs[0]->host<float>();
    auto inputB = inputs[1]->host<float>();
    auto output = outputs[0]->host<float>();

    const int height = inputs[0]->length(2);
    const int width  = inputs[0]->length(3);
    const int outH   = height / 2;
    const int outW   = width / 2;
    const int planeStride = outH * outW;

    for (int ho = 0; ho < outH; ++ho) {
        for (int wo = 0; wo < outW; ++wo) {
            const int o = ho * outW + wo;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int lane = dy * 2 + dx;
                    const float a = inputA[(2 * ho + dy) * width + (2 * wo + dx)];
                    const float b = inputB[(2 * ho + dy) * width + (2 * wo + dx)];
                    const float p0 = mAlpha * a + mBeta * b + mGamma;
                    const float p1 = mDelta * b + mEpsilon;
                    output[(0 + lane) * planeStride + o] = p0;
                    output[(4 + lane) * planeStride + o] = p1;
                    output[(8 + lane) * planeStride + o] = p0 - p1;
                }
            }
            output[12 * planeStride + o] = mConst;
        }
    }
    return NO_ERROR;
}

class CPUFusedMathS2DCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (inputs.size() != 2 || outputs.size() != 1) {
            return nullptr;
        }
        return new CPUFusedMathS2D(backend, op->main_as_FusedMathS2DParam());
    }
};

REGISTER_CPU_OP_CREATOR(CPUFusedMathS2DCreator, OpType_FusedMathS2D);

} // namespace MNN
