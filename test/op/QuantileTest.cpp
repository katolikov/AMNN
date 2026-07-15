//
//  QuantileTest.cpp
//  MNNTests
//

#include <cmath>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class QuantileTest : public MNNTestCase {
public:
    virtual ~QuantileTest() = default;

    virtual bool run(int precision) {
        // ===== case 1: small, hand-computed (linear interpolation, matches numpy.quantile) =====
        {
            VARP input = _Input({10}, NCHW, halide_type_of<float>());
            std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            ::memcpy(input->writeMap<float>(), data.data(), data.size() * sizeof(float));

            auto output = _Quantile(input, {0.0f, 0.25f, 0.5f, 0.75f, 1.0f});
            auto got = output->readMap<float>();
            std::vector<float> expected = {1.0f, 3.25f, 5.5f, 7.75f, 10.0f};
            if (!checkVectorByRelativeError<float>(got, expected.data(), expected.size(), 0.001f)) {
                MNN_ERROR("QuantileTest case1 (small, hand-computed) failed!\n");
                return false;
            }
        }

        // ===== case 2: constant tensor -> every quantile equals the constant =====
        {
            const int n = 100;
            VARP input = _Input({n}, NCHW, halide_type_of<float>());
            auto ptr = input->writeMap<float>();
            for (int i = 0; i < n; ++i) {
                ptr[i] = 3.0f;
            }
            auto output = _Quantile(input, {0.1f, 0.5f, 0.9f});
            auto got = output->readMap<float>();
            std::vector<float> expected = {3.0f, 3.0f, 3.0f};
            if (!checkVectorByRelativeError<float>(got, expected.data(), expected.size(), 0.001f)) {
                MNN_ERROR("QuantileTest case2 (constant tensor) failed!\n");
                return false;
            }
        }

        // ===== case 3: full [1,1,1440,1920] shape (the target on-device use case), =====
        // ===== deterministic input, expected values cross-checked against numpy.quantile =====
        {
            const int H = 1440, W = 1920;
            const int n = H * W;
            VARP input = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            auto ptr = input->writeMap<float>();
            for (int i = 0; i < n; ++i) {
                ptr[i] = sinf(i * 0.001f);
            }
            std::vector<float> qLevels = {0.05f, 0.25f, 0.5f, 0.75f, 0.95f};
            auto output = _Quantile(input, std::move(qLevels));
            auto got = output->readMap<float>();
            // computed via: numpy.quantile(sin(arange(n)*0.001).astype(f32), qs, method='linear')
            std::vector<float> expected = {-0.98768635f, -0.70706722f, 0.00011102f, 0.70706646f, 0.98768652f};
            if (!checkVectorByRelativeError<float>(got, expected.data(), expected.size(), 0.005f)) {
                MNN_ERROR("QuantileTest case3 (full [1,1,1440,1920] shape) failed!\n");
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(QuantileTest, "op/Quantile");
