//
//  FusedMathS2DTest.cpp
//  MNNTests
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/MNNDefine.h>
#include <chrono>
#include <random>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "MNN_generated.h"

using namespace MNN;
using namespace MNN::Express;

namespace {

struct Coeff {
    float alpha = 1.25f;
    float beta = -0.75f;
    float gamma = 0.5f;
    float delta = 2.0f;
    float epsilon = -0.25f;
    float kConst = 0.125f;
};

VARP buildOp(VARP a, VARP b, const Coeff& c) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_FusedMathS2D;
    op->main.type  = OpParameter_FusedMathS2DParam;
    auto param     = new FusedMathS2DParamT;
    param->alpha   = c.alpha;
    param->beta    = c.beta;
    param->gamma   = c.gamma;
    param->delta   = c.delta;
    param->epsilon = c.epsilon;
    param->kConst  = c.kConst;
    op->main.value = param;
    return Variable::create(Expr::create(op.get(), {a, b}, 1));
}

// Host reference, identical to CPUFusedMathS2D.
void reference(const std::vector<float>& a, const std::vector<float>& b, int h, int w, const Coeff& c,
               std::vector<float>& out) {
    const int outH = h / 2, outW = w / 2, plane = outH * outW;
    out.assign(13 * plane, 0.0f);
    for (int ho = 0; ho < outH; ++ho) {
        for (int wo = 0; wo < outW; ++wo) {
            const int o = ho * outW + wo;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int lane = dy * 2 + dx;
                    const float av = a[(2 * ho + dy) * w + (2 * wo + dx)];
                    const float bv = b[(2 * ho + dy) * w + (2 * wo + dx)];
                    const float p0 = c.alpha * av + c.beta * bv + c.gamma;
                    const float p1 = c.delta * bv + c.epsilon;
                    out[(0 + lane) * plane + o] = p0;
                    out[(4 + lane) * plane + o] = p1;
                    out[(8 + lane) * plane + o] = p0 - p1;
                }
            }
            out[12 * plane + o] = c.kConst;
        }
    }
}

bool runCase(int h, int w, int precision, const char* tag) {
    Coeff c;
    const int n = h * w;
    std::vector<float> a(n), b(n);
    std::mt19937 gen(1234 + h * 7919 + w);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    for (int i = 0; i < n; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    auto va = _Input({1, 1, h, w}, NCHW, halide_type_of<float>());
    auto vb = _Input({1, 1, h, w}, NCHW, halide_type_of<float>());
    ::memcpy(va->writeMap<float>(), a.data(), n * sizeof(float));
    ::memcpy(vb->writeMap<float>(), b.data(), n * sizeof(float));

    auto out = buildOp(va, vb, c);
    auto info = out->getInfo();
    if (info == nullptr || info->dim.size() != 4 || info->dim[0] != 1 || info->dim[1] != 13 ||
        info->dim[2] != h / 2 || info->dim[3] != w / 2) {
        MNN_ERROR("FusedMathS2D[%s]: wrong output shape\n", tag);
        return false;
    }
    auto got = out->readMap<float>();
    if (got == nullptr) {
        MNN_ERROR("FusedMathS2D[%s]: null output\n", tag);
        return false;
    }
    std::vector<float> expect;
    reference(a, b, h, w, c, expect);

    // fp16 storage (precision Normal/Low) keeps ~3 decimal digits; High is fp32.
    const float tol = (precision == 1) ? 1e-4f : 2e-2f;
    for (size_t i = 0; i < expect.size(); ++i) {
        const float diff = ::fabsf(got[i] - expect[i]);
        const float rel  = diff / std::max(1.0f, ::fabsf(expect[i]));
        if (rel > tol) {
            MNN_ERROR("FusedMathS2D[%s] mismatch at %zu (ch %zu): expect %f got %f\n", tag, i,
                      i / ((size_t)(h / 2) * (w / 2)), expect[i], got[i]);
            return false;
        }
    }
    return true;
}

} // namespace

class FusedMathS2DTest : public MNNTestCase {
public:
    virtual ~FusedMathS2DTest() = default;
    virtual bool run(int precision) {
        // Small hand-checkable case pins the channel order, then odd/even widths
        // exercise the VW fallback, then a realistic frame.
        if (!runCase(4, 4, precision, "4x4")) return false;
        if (!runCase(8, 6, precision, "8x6")) return false;      // outW=3 -> VW falls back to 1
        if (!runCase(16, 20, precision, "16x20")) return false;  // outW=10 -> VW falls back to 2
        if (!runCase(64, 64, precision, "64x64")) return false;
        if (!runCase(270, 480, precision, "270x480")) return false;
        return true;
    }
};
MNNTestSuiteRegister(FusedMathS2DTest, "op/FusedMathS2D");

class FusedMathS2DSpeedTest : public MNNTestCase {
public:
    virtual ~FusedMathS2DSpeedTest() = default;
    virtual bool run(int precision) {
        int h = 1080, w = 1920;
        if (getenv("MNN_S2D_H")) h = atoi(getenv("MNN_S2D_H"));
        if (getenv("MNN_S2D_W")) w = atoi(getenv("MNN_S2D_W"));
        int loop = 20;
        if (getenv("MNN_S2D_LOOP")) loop = atoi(getenv("MNN_S2D_LOOP"));

        Coeff c;
        const int n = h * w;
        auto va = _Input({1, 1, h, w}, NCHW, halide_type_of<float>());
        auto vb = _Input({1, 1, h, w}, NCHW, halide_type_of<float>());
        auto pa = va->writeMap<float>();
        auto pb = vb->writeMap<float>();
        for (int i = 0; i < n; ++i) {
            pa[i] = (float)((i % 251) - 125) * 0.01f;
            pb[i] = (float)((i % 197) - 98) * 0.01f;
        }
        auto out = buildOp(va, vb, c);
        // With a consumer appended, our output stops being the graph's final
        // (host-visible) tensor and becomes an ordinary device-side intermediate.
        if (getenv("MNN_S2D_CONSUMER") && atoi(getenv("MNN_S2D_CONSUMER")) != 0) {
            out = _Relu(out);
        }
        if (out->readMap<float>() == nullptr) {
            MNN_ERROR("FusedMathS2D speed: null output\n");
            return false;
        }
        // Traffic floor: read 2*H*W + write 13*(H/2)*(W/2) elements.
        const double elems = 2.0 * n + 13.0 * (h / 2) * (w / 2);
        double best = 1e30, total = 0.0;
        for (int i = 0; i < loop; ++i) {
            va->writeMap<float>()[0] = (float)i * 0.001f;
            auto t0 = std::chrono::high_resolution_clock::now();
            auto ptr = out->readMap<float>();
            auto t1 = std::chrono::high_resolution_clock::now();
            if (ptr == nullptr) return false;
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            best = std::min(best, ms);
            total += ms;
        }
        MNN_PRINT("[FusedMathS2D speed] %dx%d loop=%d  best %.3f ms  avg %.3f ms  (wall, includes host copies)\n",
                  h, w, loop, best, total / loop);
        MNN_PRINT("[FusedMathS2D speed] traffic floor %.2f MB fp16 / %.2f MB fp32\n",
                  elems * 2.0 / 1048576.0, elems * 4.0 / 1048576.0);
        return true;
    }
};
MNNTestSuiteRegister(FusedMathS2DSpeedTest, "speed/FusedMathS2D");

// Decomposed reference graph: the same result built out of stock MNN ops, so the
// fused op has a baseline to beat on the same device in the same session.
class FusedMathS2DBaselineTest : public MNNTestCase {
public:
    virtual ~FusedMathS2DBaselineTest() = default;
    virtual bool run(int precision) {
        int h = 1080, w = 1920;
        if (getenv("MNN_S2D_H")) h = atoi(getenv("MNN_S2D_H"));
        if (getenv("MNN_S2D_W")) w = atoi(getenv("MNN_S2D_W"));
        int loop = 20;
        if (getenv("MNN_S2D_LOOP")) loop = atoi(getenv("MNN_S2D_LOOP"));

        Coeff c;
        const int n = h * w;
        auto va = _Input({1, 1, h, w}, NCHW, halide_type_of<float>());
        auto vb = _Input({1, 1, h, w}, NCHW, halide_type_of<float>());
        auto pa = va->writeMap<float>();
        auto pb = vb->writeMap<float>();
        for (int i = 0; i < n; ++i) {
            pa[i] = (float)((i % 251) - 125) * 0.01f;
            pb[i] = (float)((i % 197) - 98) * 0.01f;
        }
        auto p0 = va * _Scalar<float>(c.alpha) + vb * _Scalar<float>(c.beta) + _Scalar<float>(c.gamma);
        auto p1 = vb * _Scalar<float>(c.delta) + _Scalar<float>(c.epsilon);
        auto p2 = p0 - p1;
        std::vector<int> constDims = {1, 1, h / 2, w / 2};
        auto kc = _Fill(_Const(constDims.data(), {4}, NCHW, halide_type_of<int>()),
                        _Scalar<float>(c.kConst));
        auto out = _Concat({_SpaceToDepth(p0, 2), _SpaceToDepth(p1, 2), _SpaceToDepth(p2, 2), kc}, 1);
        if (out->readMap<float>() == nullptr) {
            MNN_ERROR("FusedMathS2D baseline: null output\n");
            return false;
        }
        double best = 1e30;
        for (int i = 0; i < loop; ++i) {
            va->writeMap<float>()[0] = (float)i * 0.001f;
            auto t0 = std::chrono::high_resolution_clock::now();
            auto ptr = out->readMap<float>();
            auto t1 = std::chrono::high_resolution_clock::now();
            if (ptr == nullptr) return false;
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        MNN_PRINT("[FusedMathS2D baseline] %dx%d best %.3f ms (wall)\n", h, w, best);
        return true;
    }
};
MNNTestSuiteRegister(FusedMathS2DBaselineTest, "speed/FusedMathS2DBaseline");
