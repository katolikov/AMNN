//
//  ApplyLUTSpeed.cpp
//  MNNTests
//
//  On-device benchmark: fused ApplyLUT (single vectorized pass, LUT in local
//  memory) vs a naive decomposition built from existing ops (floor + clamp + two
//  gathers + lerp). Reports end-to-end wall-clock over the [1,1,1440,1920] frame;
//  run with the OpenCL profiler (gpuMode 68) to also see the per-kernel time.
//
//  Sweep the vector width with the env var MNN_APPLYLUT_VEC in {2,4,8,16}
//  (default 8) across separate runs to find the bandwidth sweet spot.
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <cmath>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "MNNTestSuite.h"

using namespace MNN::Express;
#define WARMUP 8
#define TIME 50

class ApplyLUTSpeed : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int H = 1440, W = 1920, n = H * W, N = 16;

        auto makeImage = [&]() {
            VARP v = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            auto p = v->writeMap<float>();
            for (int i = 0; i < n; ++i) p[i] = (float)((i * 2654435761u) & 0xffff) / 65535.0f;
            return v;
        };
        auto makeLut = [&]() {
            VARP v = _Input({N}, NCHW, halide_type_of<float>());
            auto p = v->writeMap<float>();
            for (int i = 0; i < N; ++i) p[i] = std::pow((float)i / (N - 1), 2.2f);
            return v;
        };

        // ---- fused op ----
        auto benchFused = [&]() {
            VARP img = makeImage(), lut = makeLut();
            auto out = _ApplyLUT(img, lut);
            auto touch = [&]() {
                img->writeMap<float>();
                out->readMap<float>();
            };
            for (int i = 0; i < WARMUP; ++i) touch();
            MNN::Timer t;
            for (int i = 0; i < TIME; ++i) touch();
            MNN_PRINT("[FUSED ApplyLUT   ] avg: %f ms\n", (float)t.durationInUs() / 1000.0f / TIME);
        };

        // ---- naive decomposition from existing ops ----
        auto benchNaive = [&]() {
            VARP img = makeImage(), lut = makeLut();
            auto scaleV = _Scalar<float>((float)(N - 1));
            auto zeroV  = _Scalar<float>(0.0f);
            auto maxV   = _Scalar<float>((float)(N - 1));
            auto oneI   = _Scalar<int>(1);
            auto maxI   = _Scalar<int>(N - 1);
            auto pos  = img * scaleV;
            auto i0f  = _Maximum(_Minimum(_Floor(pos), maxV), zeroV);
            auto i0   = _Cast<int>(i0f);
            auto i1   = _Minimum(i0 + oneI, maxI);
            auto frac = pos - i0f;
            auto lo   = _Gather(lut, i0);
            auto hi   = _Gather(lut, i1);
            auto out  = lo + frac * (hi - lo);
            auto touch = [&]() {
                img->writeMap<float>();
                out->readMap<float>();
            };
            for (int i = 0; i < WARMUP; ++i) touch();
            MNN::Timer t;
            for (int i = 0; i < TIME; ++i) touch();
            MNN_PRINT("[NAIVE decompose  ] avg: %f ms\n", (float)t.durationInUs() / 1000.0f / TIME);
        };

        benchFused();
        benchNaive();
        return true;
    }
};

MNNTestSuiteRegister(ApplyLUTSpeed, "speed/ApplyLUT");
