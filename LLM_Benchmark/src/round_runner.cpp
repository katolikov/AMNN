#include "round_runner.hpp"
#include <chrono>

#ifdef LLM_SUPPORT_VISION
#include <cv/cv.hpp>
PreloadedVlm* g_vlm_ptr = nullptr;
#endif

static RoundMetrics extract_metrics(Llm* llm, Clock::time_point t0, Clock::time_point t1) {
    RoundMetrics m;
    auto ctx = llm->getContext();
    m.prefill_ms = ctx->prefill_us / 1000.0;
    m.decode_ms  = ctx->decode_us  / 1000.0;
    m.vision_ms  = ctx->vision_us  / 1000.0;
    m.prompt_len = ctx->prompt_len;
    m.gen_len    = ctx->gen_seq_len;
    m.total_ms   = std::chrono::duration_cast<Us>(t1 - t0).count() / 1000.0;
    m.exit_status = static_cast<int>(ctx->status);
    return m;
}

RoundMetrics run_one_round_tokens(Llm* llm, const BenchArgs& a) {
    llm->reset();
    auto tokens = make_synthetic_tokens(a.prompt_tokens);
    auto t0 = Clock::now();
    llm->response(tokens, nullptr, nullptr, a.max_gen_tokens);
    return extract_metrics(llm, t0, Clock::now());
}

RoundMetrics run_one_round_text(Llm* llm, const BenchArgs& a) {
    llm->reset();
    auto t0 = Clock::now();
    llm->response(a.prompt_text, nullptr, nullptr, a.max_gen_tokens);
    return extract_metrics(llm, t0, Clock::now());
}

#ifdef LLM_SUPPORT_VISION
PreloadedVlm preload_vlm(const BenchArgs& a) {
    PreloadedVlm vlm;
    vlm.mp.prompt_template = a.vlm_template;
    PromptImagePart img;
    img.image_data = MNN::CV::imread(a.vlm_image);
    if (!img.image_data.get()) {
        fprintf(stderr, "[BENCH] ERROR: cannot load image: %s\n", a.vlm_image.c_str());
        return vlm;
    }
    auto info = img.image_data->getInfo();
    printf("[BENCH] Image pre-loaded: %s (%d x %d)\n",
           a.vlm_image.c_str(), info->dim[1], info->dim[0]);
    fflush(stdout);
    img.height = 0;
    img.width  = 0;
    vlm.mp.images["img0"] = img;
    vlm.valid = true;
    return vlm;
}

RoundMetrics run_one_round_vlm(Llm* llm, const BenchArgs& a, const PreloadedVlm& vlm) {
    llm->reset();
    auto t0 = Clock::now();
    llm->response(vlm.mp, nullptr, nullptr, a.max_gen_tokens);
    return extract_metrics(llm, t0, Clock::now());
}
#endif

RoundMetrics run_one_round(Llm* llm, const BenchArgs& a) {
#ifdef LLM_SUPPORT_VISION
    if (g_vlm_ptr && g_vlm.valid)
        return run_one_round_vlm(llm, a, g_vlm);
#endif
    if (!a.prompt_text.empty()) return run_one_round_text(llm, a);
    return run_one_round_tokens(llm, a);
}
