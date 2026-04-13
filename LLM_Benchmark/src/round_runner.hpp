#pragma once
//  round_runner.hpp — Benchmark round execution and VLM preloading

#include "bench_args.hpp"
#include "utils.hpp"
#include "llm/llm.hpp"

using namespace MNN::Transformer;

struct RoundMetrics {
    double load_ms = 0, prefill_ms = 0, decode_ms = 0;
    double total_ms = 0, vision_ms = 0;
    int prompt_len = 0, gen_len = 0;
    int exit_status = 0;
};

RoundMetrics run_one_round_tokens(Llm* llm, const BenchArgs& a);
RoundMetrics run_one_round_text(Llm* llm, const BenchArgs& a);

#ifdef LLM_SUPPORT_VISION
struct PreloadedVlm {
    MultimodalPrompt mp;
    bool valid = false;
};

PreloadedVlm preload_vlm(const BenchArgs& a);
RoundMetrics run_one_round_vlm(Llm* llm, const BenchArgs& a, const PreloadedVlm& vlm);

// Global VLM state — heap-allocated for explicit cleanup before CLRuntime teardown
extern PreloadedVlm* g_vlm_ptr;
#define g_vlm (*g_vlm_ptr)
#endif

RoundMetrics run_one_round(Llm* llm, const BenchArgs& a);
