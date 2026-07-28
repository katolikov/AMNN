//
//  DualRangeHistSpeed.cpp
//  MNNTests
//
//  On-device benchmark: fused DualRangeHist vs the naive multi-pass baseline
//  (compose keep-mask + discretize + two masked BinCounts built from existing
//  ops), plus the validCount emit/derive A/B. Reports end-to-end wall-clock over
//  the [1,1,1440,1920] frame; run with the OpenCL profiler (mode 68) to also see
//  per-kernel times.
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "MNNTestSuite.h"

using namespace MNN::Express;
#define WARMUP 8
#define TIME 50

class DualRangeHistSpeed : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int H = 1440, W = 1920, n = H * W, binNum = 16;
        const float low = 0.05f, high = 0.95f, scale = (float)(binNum - 1);

        auto makeFrame = [&](int seed) {
            VARP v = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            auto p = v->writeMap<float>();
            for (int i = 0; i < n; ++i) p[i] = (float)((i * seed) % 16) / 15.0f;
            return v;
        };
        auto makeBase = [&]() {
            VARP v = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            auto p = v->writeMap<float>();
            for (int i = 0; i < n; ++i) p[i] = (i % 3 == 0) ? 0.0f : 1.0f;
            return v;
        };

        auto benchFused = [&](const char* tag, int stride, bool emitVc) {
            VARP A = makeFrame(1), B = makeFrame(7), base = makeBase();
            auto out = _DualRangeHist(A, B, base, binNum, low, high, stride, emitVc);
            auto touch = [&]() {
                A->writeMap<float>();
                out[0]->readMap<int>();
                out[1]->readMap<int>();
                if (emitVc) out[2]->readMap<int>();
            };
            for (int i = 0; i < WARMUP; ++i) touch();
            MNN::Timer t;
            for (int i = 0; i < TIME; ++i) touch();
            MNN_PRINT("[FUSED %-16s] stride=%d vc=%d  avg: %f ms\n", tag, stride, (int)emitVc,
                      (float)t.durationInUs() / 1000.0f / TIME);
        };

        // Naive multi-pass baseline built from existing ops: compose the shared
        // keep-mask, discretize each frame, then form each masked histogram. This
        // repo has no BinCount op, so each histogram is built as binNum reductions
        // of (bin==b)*keep -- a heavier but faithful "multi-pass from primitives"
        // baseline (materialized mask + discretized frames + 2*binNum reductions).
        auto benchNaive = [&]() {
            VARP A = makeFrame(1), B = makeFrame(7), base = makeBase();
            auto lowV = _Scalar<float>(low), hiV = _Scalar<float>(high), scaleV = _Scalar<float>(scale);
            auto inA  = _Cast<float>(_GreaterEqual(A, lowV)) * _Cast<float>(_LessEqual(A, hiV));
            auto inB  = _Cast<float>(_GreaterEqual(B, lowV)) * _Cast<float>(_LessEqual(B, hiV));
            auto keep = base * inA * inB;
            auto binA = _Round(A * scaleV);
            auto binB = _Round(B * scaleV);
            auto histOf = [&](VARP bin) {
                std::vector<VARP> cols;
                cols.reserve(binNum);
                for (int b = 0; b < binNum; ++b) {
                    auto eq = _Cast<float>(_Equal(bin, _Scalar<float>((float)b)));
                    cols.emplace_back(_ReduceSum(eq * keep, {}, false));
                }
                return _Stack(cols, 0);
            };
            auto histA = histOf(binA);
            auto histB = histOf(binB);
            auto touch = [&]() {
                A->writeMap<float>();
                histA->readMap<float>();
                histB->readMap<float>();
            };
            for (int i = 0; i < WARMUP; ++i) touch();
            MNN::Timer t;
            for (int i = 0; i < TIME; ++i) touch();
            MNN_PRINT("[NAIVE multi-pass ] stride=1 vc=0  avg: %f ms\n",
                      (float)t.durationInUs() / 1000.0f / TIME);
        };

        benchFused("full", 1, false);
        benchFused("full+validCount", 1, true);
        benchFused("downsample", 8, false);
        benchNaive();
        return true;
    }
};

MNNTestSuiteRegister(DualRangeHistSpeed, "speed/DualRangeHist");
