//
//  ApplyLUTTest.cpp
//  MNNTests
//

#include <cmath>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

// Independent reference oracle (separate code path from the op): clamp-index-only
// linear-interp LUT remap. pos = x*(N-1); i0 = clamp(floor(pos),0,N-1);
// i1 = min(i0+1,N-1); frac = pos - i0; out = lut[i0] + frac*(lut[i1]-lut[i0]).
static void referenceApplyLut(const std::vector<float>& img, const std::vector<float>& lut,
                              std::vector<float>& out) {
    const int N = (int)lut.size();
    const int maxIdx = N - 1;
    const float scale = (float)maxIdx;
    out.resize(img.size());
    for (size_t p = 0; p < img.size(); ++p) {
        const float pos = img[p] * scale;
        int i0 = (int)std::floor(pos);
        i0 = std::min(std::max(i0, 0), maxIdx);
        const int i1 = std::min(i0 + 1, maxIdx);
        const float frac = pos - (float)i0;
        out[p] = lut[i0] + frac * (lut[i1] - lut[i0]);
    }
}

class ApplyLUTTest : public MNNTestCase {
public:
    virtual ~ApplyLUTTest() = default;

    static bool checkCase(const char* tag, const std::vector<int>& shape,
                          const std::vector<float>& img, const std::vector<float>& lut,
                          float threshold) {
        VARP vin  = _Input(shape, NCHW, halide_type_of<float>());
        VARP vlut = _Input({(int)lut.size()}, NCHW, halide_type_of<float>());
        ::memcpy(vin->writeMap<float>(),  img.data(), img.size() * sizeof(float));
        ::memcpy(vlut->writeMap<float>(), lut.data(), lut.size() * sizeof(float));
        auto out = _ApplyLUT(vin, vlut);

        std::vector<float> ref;
        referenceApplyLut(img, lut, ref);
        auto got = out->readMap<float>();
        if (!checkVector<float>(got, ref.data(), (int)ref.size(), threshold)) {
            MNN_ERROR("ApplyLUTTest %s failed (threshold %f)\n", tag, threshold);
            return false;
        }
        return true;
    }

    virtual bool run(int precision) {
        // OpenCL buffer mode always stores activations in fp16, so allow an fp16-
        // sized tolerance; on CPU (fp32) the actual error is far below this. A
        // formula/indexing bug produces errors well above threshold.
        const float thr = 0.02f;

        // ===== case 1: tiny, hand-verifiable (N=4, scale=3) =====
        // lut = {0, 0.25, 0.5, 1.0}
        // x=0    -> pos0    -> lut[0]=0
        // x=1/6  -> pos0.5  -> lerp(lut0,lut1,0.5)=0.125
        // x=0.5  -> pos1.5  -> lerp(lut1,lut2,0.5)=0.375
        // x=1    -> pos3    -> lut[3]=1.0
        {
            std::vector<float> lut = {0.0f, 0.25f, 0.5f, 1.0f};
            std::vector<float> img = {0.0f, 1.0f/6, 0.5f, 1.0f};
            if (!checkCase("case1(tiny)", {1, 1, 1, 4}, img, lut, 0.01f)) return false;
        }

        // ===== case 2: out-of-range inputs exercise clamp-index-only (N=8) =====
        // Includes x<0 and x>1: high end pins to lut[N-1] (i0==i1); low end keeps
        // the [lut0,lut1] slope (mild extrapolation). Reference mirrors exactly.
        {
            const int N = 8;
            std::vector<float> lut(N);
            for (int i = 0; i < N; ++i) lut[i] = std::pow((float)i / (N - 1), 1.5f);
            std::vector<float> img = {-0.3f, -0.01f, 0.0f, 0.2f, 0.5f, 0.83f, 1.0f, 1.4f};
            if (!checkCase("case2(oob)", {1, 1, 1, (int)img.size()}, img, lut, 0.01f)) return false;
        }

        // ===== case 3: full [1,1,1440,1920], N=16, smooth gamma LUT =====
        {
            const int H = 1440, W = 1920, n = H * W, N = 16;
            std::vector<float> lut(N);
            for (int i = 0; i < N; ++i) lut[i] = std::pow((float)i / (N - 1), 2.2f);  // tone curve
            std::vector<float> img(n);
            for (int i = 0; i < n; ++i) {
                // deterministic sweep across [0,1], smooth so fp16 input noise maps
                // to a proportional (bounded) output difference.
                img[i] = (float)((i * 2654435761u) & 0xffff) / 65535.0f;
            }
            if (!checkCase("case3(full)", {1, 1, H, W}, img, lut, thr)) return false;
        }

        return true;
    }
};

MNNTestSuiteRegister(ApplyLUTTest, "op/ApplyLUT");
