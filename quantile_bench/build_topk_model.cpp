// Benchmarks MNN's existing local-bitonic-sort TopKV2 buffer kernel as the
// "local bitonic sort, one workgroup per slice" candidate for small N (it's
// already proven to hard-fail above k=1024, so this only makes sense in the
// small-N regime).
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

using namespace MNN::Express;

int main(int argc, char** argv) {
    const char* outPath = argc > 1 ? argv[1] : "topk_bench.mnn";
    const int n = argc > 2 ? atoi(argv[2]) : 1024;
    auto x = _Input({n}, NCHW, halide_type_of<float>());
    x->setName("x");
    auto k = _Const(&n, {1}, NCHW, halide_type_of<int32_t>());
    auto res = _TopKV2(x, k);
    res[0]->setName("sorted");
    Variable::save({res[0]}, outPath);
    return 0;
}
