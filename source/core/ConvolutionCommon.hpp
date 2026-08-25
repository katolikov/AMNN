//
//  ConvolutionCommon.hpp
//  MNN
//
//  Created by MNN on 2020/03/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionCommon_hpp
#define ConvolutionCommon_hpp
#include "AutoStorage.h"
#include "Execution.hpp"
#include "MNN_generated.h"
namespace MNN {
class MNN_PUBLIC ConvolutionCommon : public Execution {
public:
    // Classification of Convolution2DCommon.leakyReluSlope -- the per-channel PReLU/LeakyReLU
    // folded into a conv by MNNConvert --fusePreluToConv. EVERY consumer (the Pipeline gate, the
    // OpenCL op creators, and the executions that upload the slope) must agree on this. When the
    // predicates disagree, a conv gets approved by one gate and ignored by another, and the
    // activation is dropped with no diagnostic -- output that is wrong but plausible.
    enum FusedActivation {
        FusedActivation_None = 0,       // no slope; nothing to apply, any backend is fine
        FusedActivation_PRelu,          // usable: one slope per output channel, no relu/relu6
        // The two ways a slope can be present but unusable. Both must be refused rather than
        // ignored, and they are kept apart so the diagnostic can name the actual problem
        // instead of listing every rule the model might have broken.
        FusedActivation_InvalidWithRelu,    // relu/relu6 on the same conv wins in every kernel
        FusedActivation_InvalidSlopeCount,  // fewer slopes than the conv has output channels
    };
    static bool fusedActivationInvalid(FusedActivation act) {
        return FusedActivation_InvalidWithRelu == act || FusedActivation_InvalidSlopeCount == act;
    }
    static FusedActivation fusedActivation(const Convolution2DCommon* common);

    struct Int8Common {
        AutoStorage<int8_t> weight;
        AutoStorage<float> alpha;
        AutoStorage<float> weightFloat;
        const IDSTQuan* quan;
        bool asymmetric;
        std::vector<int8_t> weightMap;
        bool canUseInt4 = false;
        Backend* backend = nullptr;
        int originBits = 8;
        int alphaSize;
    };
    static std::shared_ptr<Int8Common> load(const Op* op, Backend* backend = nullptr, bool forceFloat = false, bool forceInt8 = false, void* weightPtr = nullptr);
    // if can not get quant bits, return 0
    static int getQuantBitFromExternalFile(const Op* op);
    static void getConvParameters(std::shared_ptr<ConvolutionCommon::Int8Common> *quanCommon, Backend* backend, const MNN::Op *op, const float** originWeight, int* originWeightSize);
    static bool getConvInt8Parameters(const MNN::Op* op, std::shared_ptr<Int8Common>& quanCommon, Backend* backend,
                                      const int8_t*& weight, int& weightSize, float* scale, int32_t* bias, int ocUpHp);

    // Return padX, padY
    static std::pair<int, int> convolutionPad(const Tensor* input, const Tensor* output,
                                              const Convolution2DCommon* common);
    // Return padLeft, padTop, padRight, padBottom
    static std::tuple<int, int, int, int> convolutionPadFull(const Tensor* input, const Tensor* output,
                                              const Convolution2DCommon* common);
    static std::pair<int, int> convolutionTransposePad(const Tensor* input, const Tensor* output,
                                                       const Convolution2DCommon* common);
    struct Im2ColParameter {
        int32_t padX;
        int32_t padY;
        int32_t dilateX;
        int32_t dilateY;
        int32_t strideX;
        int32_t strideY;
        int32_t kernelX;
        int32_t kernelY;
        int32_t icDiv4;
        int32_t kernelCountUnit;
        int32_t iw;
        int32_t ih;
        int32_t ow;
        int32_t oh;
        int32_t srcZStep;
        int32_t srcYStep;
        int32_t packCUnit;
        int32_t destICStride;
        int32_t ic; // ic packed by LP, used by im2col dst data.
        int32_t icup4; // ic packed by LP, used by im2col src data.
    };
};
} // namespace MNN
#endif
