// Standalone builder for the custom-op benchmark model: saves a .mnn graph
// containing a single OpType_Quantile node over a 1-D tensor of N elements,
// since the op has no ONNX/TF converter (it's MNN-native, added specifically
// for this benchmark). Usage: build_quantile_model <outPath> [N] [numQLevels]
// (N defaults to 1440*1920, the real production shape; numQLevels defaults
// to 5, evenly spaced in (0,1), to vary target count independently of N).
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

using namespace MNN::Express;

int main(int argc, char** argv) {
    const char* outPath = argc > 1 ? argv[1] : "quantile_custom.mnn";
    const int n = argc > 2 ? atoi(argv[2]) : 1440 * 1920;
    const int numQ = argc > 3 ? atoi(argv[3]) : 5;
    auto x = _Input({n}, NCHW, halide_type_of<float>());
    x->setName("x");
    std::vector<float> qLevels(numQ);
    for (int i = 0; i < numQ; ++i) {
        qLevels[i] = (i + 1.0f) / (numQ + 1.0f);
    }
    auto y = _Quantile(x, std::move(qLevels));
    y->setName("quantiles");
    Variable::save({y}, outPath);
    return 0;
}
