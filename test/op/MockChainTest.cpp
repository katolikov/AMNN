//
//  MockChainTest.cpp
//  MNNTests
//

#include <vector>
#include <cmath>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

// Reference for the mock chain (must match CPUMockChain / mockchain_buf.cl):
//   out0 = (((A+1)*2 + (B+1)*2) + offset0) * 3
//   out1 = (((A+1)*2 + (B+1)*2) + offset1) * 3
static void mockChainRef(const std::vector<float>& a, const std::vector<float>& b,
                         float o0, float o1, std::vector<float>& out0, std::vector<float>& out1) {
    out0.resize(a.size());
    out1.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        const float t3 = (a[i] + 1.0f) * 2.0f + (b[i] + 1.0f) * 2.0f;
        out0[i] = (t3 + o0) * 3.0f;
        out1[i] = (t3 + o1) * 3.0f;
    }
}

class MockChainTest : public MNNTestCase {
public:
    virtual ~MockChainTest() = default;

    bool runCase(const std::vector<int>& shape, float o0, float o1, int precision, const char* tag) {
        int n = 1;
        for (auto d : shape) n *= d;
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; ++i) {
            a[i] = (float)((i % 7) - 3);        // small mixed-sign values (fp16-exact)
            b[i] = (float)((i % 5));            // 0..4
        }

        VARP A = _Input(shape, NCHW, halide_type_of<float>());
        VARP B = _Input(shape, NCHW, halide_type_of<float>());
        ::memcpy(A->writeMap<float>(), a.data(), n * sizeof(float));
        ::memcpy(B->writeMap<float>(), b.data(), n * sizeof(float));

        auto outs = _MockChain(A, B, o0, o1);
        auto g0 = outs[0]->readMap<float>();
        auto g1 = outs[1]->readMap<float>();

        std::vector<float> e0, e1;
        mockChainRef(a, b, o0, o1, e0, e1);

        // fp16 buffer mode (OpenCL Low) needs relative tolerance; values here
        // stay well under 65504 so no overflow.
        const float tol = 0.01f;
        if (!checkVectorByRelativeError<float>(g0, e0.data(), n, tol)) {
            MNN_ERROR("MockChainTest [%s] out0 mismatch\n", tag);
            return false;
        }
        if (!checkVectorByRelativeError<float>(g1, e1.data(), n, tol)) {
            MNN_ERROR("MockChainTest [%s] out1 mismatch\n", tag);
            return false;
        }
        return true;
    }

    virtual bool run(int precision) {
        // 1: tiny hand-sized vector
        if (!runCase({8}, 10.0f, 20.0f, precision, "tiny")) return false;
        // 2: size not a multiple of the local work size (64) -> exercises the
        //    kernels' bounds guard
        if (!runCase({100}, 10.0f, 20.0f, precision, "unaligned")) return false;
        // 3: custom offsets
        if (!runCase({256}, -5.0f, 7.5f, precision, "custom-offsets")) return false;
        // 4: a realistic 2-D frame, exercises many workgroups
        if (!runCase({1, 1, 128, 160}, 10.0f, 20.0f, precision, "frame")) return false;
        return true;
    }
};

MNNTestSuiteRegister(MockChainTest, "op/MockChain");
