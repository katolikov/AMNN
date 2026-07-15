// Standalone builder for the custom-op benchmark model: saves a .mnn graph
// containing a single OpType_Quantile node over [1,1,1440,1920], since the
// op has no ONNX/TF converter (it's MNN-native, added specifically for this
// benchmark).
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

using namespace MNN::Express;

int main(int argc, char** argv) {
    const char* outPath = argc > 1 ? argv[1] : "quantile_custom.mnn";
    auto x = _Input({1, 1, 1440, 1920}, NCHW, halide_type_of<float>());
    x->setName("x");
    auto y = _Quantile(x, {0.05f, 0.25f, 0.5f, 0.75f, 0.95f});
    y->setName("quantiles");
    Variable::save({y}, outPath);
    return 0;
}
