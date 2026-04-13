//
//  llm_benchmark.cpp — Main entry point for MNN LLM/VLM Benchmark
//
//  All logic lives in src/ modules. This file contains only main().
//

#define MNN_OPEN_TIME_TRACE
#include "llm/llm.hpp"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Executor.hpp>
#include "Profiler.hpp"

#include "src/utils.hpp"
#include "src/enum_maps.hpp"
#include "src/bench_args.hpp"
#include "src/round_runner.hpp"

#include <iostream>
#include <limits>
#include <memory>

using namespace MNN::Transformer;

int main(int argc, const char* argv[]) {
    auto a = parse_args(argc, argv);
    int eff = compute_effective_numthread(a);

    // ── Print config header ──
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║            MNN LLM/VLM Benchmark                       ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  %-18s │ %-35s ║\n", "Config", a.config_path.c_str());
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║             RUNTIME CONFIGURATION                       ║\n");
    printf("╟──────────────────────────────────────────────────────────╢\n");
    printf("║  %-18s │ %-35s ║\n", "Backend", a.backend.c_str());
    if (is_gpu_backend(a.backend)) {
        std::string gm_str;
        for (auto& n : a.gpu_mode_names) { if (!gm_str.empty()) gm_str += " | "; gm_str += n; }
        printf("║  %-18s │ %-35s ║\n", "GPU mode", gm_str.c_str());
        char nt[64]; snprintf(nt, sizeof(nt), "%d (0x%X)", eff, eff);
        printf("║  %-18s │ %-35s ║\n", "thread_num", nt);
    } else {
        char th[16]; snprintf(th, sizeof(th), "%d", a.threads);
        printf("║  %-18s │ %-35s ║\n", "Threads", th);
    }
    printf("║  %-18s │ %-35s ║\n", "Precision", a.precision.c_str());
    printf("║  %-18s │ %-35s ║\n", "Memory", a.memory.c_str());
    printf("║  %-18s │ %-35s ║\n", "Power", a.power.c_str());
    { char qb[16]; snprintf(qb, sizeof(qb), "%d", a.attention_mode);
      printf("║  %-18s │ %-35s ║\n", "Attention mode", qb); }
    if (a.use_mmap) printf("║  %-18s │ %-35s ║\n", "Use mmap", "ON");
    if (a.max_all_tokens > 0) { char b[16]; snprintf(b,sizeof(b),"%d",a.max_all_tokens); printf("║  %-18s │ %-35s ║\n", "Max all tokens", b); }
    if (a.dynamic_quant != 0) { char b[16]; snprintf(b,sizeof(b),"%d",a.dynamic_quant); printf("║  %-18s │ %-35s ║\n", "Dynamic quant", b); }
    if (!a.hints.empty()) {
        for (size_t i = 0; i < a.hints.size(); ++i) {
            char hval[48]; snprintf(hval, sizeof(hval), "%s = %d", a.hint_names[i].first.c_str(), a.hints[i].second);
            printf("║  %-18s │ %-35s ║\n", i == 0 ? "Hints" : "", hval);
        }
    }
    if (!a.mllm_backend.empty()) {
        char mb[64]; snprintf(mb, sizeof(mb), "%s (t=%d, %s)", a.mllm_backend.c_str(), a.mllm_threads, a.mllm_precision.c_str());
        printf("║  %-18s │ %-35s ║\n", "MLLM backend", mb);
    }
    printf("╟──────────────────────────────────────────────────────────╢\n");
    printf("║             BENCHMARK PARAMETERS                        ║\n");
    printf("╟──────────────────────────────────────────────────────────╢\n");
    { char wb[32]; snprintf(wb, sizeof(wb), "%d warmup + %d measure", a.warmup, a.rounds);
      printf("║  %-18s │ %-35s ║\n", "Rounds", wb); }
    { char pt[16]; snprintf(pt, sizeof(pt), "%d", a.prompt_tokens);
      printf("║  %-18s │ %-35s ║\n", "Prompt tokens", pt); }
    { char gt[16]; snprintf(gt, sizeof(gt), "%d", a.max_gen_tokens);
      printf("║  %-18s │ %-35s ║\n", "Max gen tokens", gt); }
    printf("║  %-18s │ %-35s ║\n", "OP profiling", a.op_profile ? "ON" : "OFF");
    if (!a.vlm_image.empty()) {
        const char* img = strrchr(a.vlm_image.c_str(), '/');
        printf("║  %-18s │ %-35s ║\n", "VLM image", img ? img+1 : a.vlm_image.c_str());
    }
    printf("╚══════════════════════════════════════════════════════════╝\n");
    flush_log();

    // ── 1. Executor ──
    MNN::BackendConfig bc;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, bc, 1);
    MNN::Express::ExecutorScope scope(executor);

    // ── 2. Create & configure model ──
    std::unique_ptr<Llm> llm(Llm::createLLM(a.config_path));
    if (!llm) { fprintf(stderr, "[BENCH] FATAL: createLLM null\n"); return 1; }
    llm->set_config(build_runtime_json(a, a.op_profile));

    // ── 3. Load ──
    printf("[BENCH] Loading model...\n"); flush_log();
    auto tL0 = Clock::now();
    if (!llm->load()) { fprintf(stderr, "[BENCH] FATAL: load failed\n"); return 1; }
    double load_ms = std::chrono::duration_cast<Us>(Clock::now() - tL0).count() / 1000.0;
    printf("[BENCH] Model loaded in %.2f ms\n", load_ms); flush_log();

    // ── 3b. Hints ──
    for (auto& kv : a.hints) {
        char hbuf[256];
        switch (kv.first) {
            case 5: snprintf(hbuf,sizeof(hbuf),"{\"dynamic_option\":%d}",kv.second); llm->set_config(hbuf); break;
            case 7: snprintf(hbuf,sizeof(hbuf),"{\"attention_mode\":%d}",kv.second); llm->set_config(hbuf); break;
            case 8: snprintf(hbuf,sizeof(hbuf),"{\"kvcache_limit\":%d}",kv.second); llm->set_config(hbuf); break;
            default: break;
        }
    }

    // ── 3c. VLM image ──
#ifdef LLM_SUPPORT_VISION
    if (!a.vlm_image.empty() && !a.vlm_template.empty()) {
        g_vlm_ptr = new PreloadedVlm();
        g_vlm = preload_vlm(a);
        if (!g_vlm.valid) { fprintf(stderr, "[BENCH] FATAL: VLM pre-load failed\n"); return 1; }
    }
#endif

    // ── 4. Tuning ──
    if (!a.no_tuning) {
        printf("[BENCH] Tuning kernels...\n"); flush_log();
        MNN::Timer tt;
        llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 50, 100});
        printf("[BENCH] Tuning done in %.2f ms\n", tt.durationInUs() / 1000.0); flush_log();
    } else {
        printf("[BENCH] Tuning skipped (--no_tuning)\n"); flush_log();
    }

    // ── 5. Warmup ──
    printf("\n┌─── Warmup (%d rounds) ──────────────────────────────────┐\n", a.warmup);
    for (int w = 0; w < a.warmup; ++w) {
        auto m = run_one_round(llm.get(), a);
        printf("│  #%d  prefill=%7.1f ms  decode=%7.1f ms  total=%7.1f ms │\n", w+1, m.prefill_ms, m.decode_ms, m.total_ms);
        flush_log();
    }
    printf("└──────────────────────────────────────────────────────────┘\n"); flush_log();

    // ── 6. Measure ──
    printf("\n┌─── Measurement (%d rounds) ────────────────────────────────────────────────────────────────┐\n", a.rounds);
    printf("│  #   │ Prefill ms │   tok/s │ Decode ms │   tok/s │ Vision ms │ Total ms  │ P/G tok │ Status           │\n");
    printf("├──────┼────────────┼─────────┼───────────┼─────────┼───────────┼───────────┼─────────┼──────────────────┤\n");
    std::vector<double> prefills, decodes, totals, visions, ptps, dtps, ttfts;
    for (int r = 0; r < a.rounds; ++r) {
        auto m = run_one_round(llm.get(), a);
        prefills.push_back(m.prefill_ms); decodes.push_back(m.decode_ms);
        totals.push_back(m.total_ms); visions.push_back(m.vision_ms);
        double ttft = m.total_ms - m.decode_ms;
        ttfts.push_back(ttft);
        double pt = m.prefill_ms > 0 ? m.prompt_len / (m.prefill_ms / 1e3) : 0;
        double dt = m.decode_ms  > 0 ? m.gen_len    / (m.decode_ms  / 1e3) : 0;
        ptps.push_back(pt); dtps.push_back(dt);
        printf("│  %2d  │ %10.1f │ %7.1f │ %9.1f │ %7.1f │ %9.1f │ %9.1f │ %3d/%-3d │ %-16s │\n",
               r+1, m.prefill_ms, pt, m.decode_ms, dt, m.vision_ms, m.total_ms,
               m.prompt_len, m.gen_len, llm_status_name(m.exit_status));
        flush_log();
    }
    printf("└──────┴────────────┴─────────┴───────────┴─────────┴───────────┴───────────┴─────────┴──────────────────┘\n");
    flush_log();

    // ── 7. Stats ──
    auto sp = compute_stats(prefills), sd = compute_stats(decodes);
    auto st = compute_stats(totals), sv = compute_stats(visions);
    auto spt = compute_stats(ptps), sdt = compute_stats(dtps), sttft = compute_stats(ttfts);
    bool has_vision = sv.avg_ms > 0;

    printf("\n╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  RESULTS  (n = %d measurement rounds)                               ║\n", a.rounds);
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Model load time : %8.1f ms                                      ║\n", load_ms);
    printf("╟──────────────────────────────────────────────────────────────────────╢\n");
    printf("║  Metric                   │     Min │     Avg │     Max │   Stdev   ║\n");
    printf("╟───────────────────────────┼─────────┼─────────┼─────────┼───────────╢\n");
    if (has_vision) {
        printf("║  TTFT (total-decode) ms   │ %7.1f │ %7.1f │ %7.1f │ %8.2f  ║\n", sttft.min_ms, sttft.avg_ms, sttft.max_ms, sttft.std_ms);
        printf("║    ├ Vision encode  ms    │ %7.1f │ %7.1f │ %7.1f │ %8.2f  ║\n", sv.min_ms, sv.avg_ms, sv.max_ms, sv.std_ms);
        printf("║    ├ LLM forward    ms    │ %7.1f │ %7.1f │ %7.1f │ %8.2f  ║\n", sp.min_ms, sp.avg_ms, sp.max_ms, sp.std_ms);
        printf("║    └ Embed+overhead ms    │         │ %7.1f │         │          ║\n", sttft.avg_ms - sv.avg_ms - sp.avg_ms);
    } else {
        printf("║  Prefill  (TTFT) ms       │ %7.1f │ %7.1f │ %7.1f │ %8.2f  ║\n", sp.min_ms, sp.avg_ms, sp.max_ms, sp.std_ms);
    }
    printf("║  Decode   (total) ms      │ %7.1f │ %7.1f │ %7.1f │ %8.2f  ║\n", sd.min_ms, sd.avg_ms, sd.max_ms, sd.std_ms);
    printf("║  End-to-end       ms      │ %7.1f │ %7.1f │ %7.1f │ %8.2f  ║\n", st.min_ms, st.avg_ms, st.max_ms, st.std_ms);
    printf("╟───────────────────────────┼─────────┼─────────┼─────────┼───────────╢\n");
    printf("║  Prefill throughput tok/s  │ %7.1f │ %7.1f │ %7.1f │ %8.2f  ║\n", spt.min_ms, spt.avg_ms, spt.max_ms, spt.std_ms);
    printf("║  Decode  throughput tok/s  │ %7.1f │ %7.1f │ %7.1f │ %8.2f  ║\n", sdt.min_ms, sdt.avg_ms, sdt.max_ms, sdt.std_ms);
    if (sdt.avg_ms > 0) {
        printf("╟──────────────────────────────────────────────────────────────────────╢\n");
        printf("║  Avg TPOT (time per output token) : %8.3f ms                     ║\n", 1000.0 / sdt.avg_ms);
    }
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
    flush_log();

    // ── 8. Model output ──
    {
        printf("\n┌─── Model Output (separate inference run) ────────────────────────────┐\n");
        flush_log();
        llm->reset();
        MNN::Express::ExecutorScope::Current()->gc();
#ifdef LLM_SUPPORT_VISION
        if (g_vlm_ptr && g_vlm.valid) {
            const int kOutputMaxTokens = 512;
            std::string str_prompt = g_vlm.mp.prompt_template;
            const std::string placeholder = "<img>img0</img>";
            const std::string img_tag = "<img>" + a.vlm_image + "</img>";
            auto pos = str_prompt.find(placeholder);
            if (pos != std::string::npos) str_prompt.replace(pos, placeholder.size(), img_tag);
            llm->response(str_prompt, &std::cout, nullptr, kOutputMaxTokens);
        } else
#endif
        if (!a.prompt_text.empty()) {
            llm->response(a.prompt_text, &std::cout, nullptr, a.max_gen_tokens);
        } else {
            auto tokens = make_synthetic_tokens(a.prompt_tokens > 0 ? a.prompt_tokens : 32);
            llm->response(tokens, &std::cout, nullptr, a.max_gen_tokens);
        }
        printf("\n└───────────────────────────────────────────────────────────────────────┘\n");
        auto ctx = llm->getContext();
        printf("[BENCH] Output run stats: gen_seq_len=%d  status=%s  generate_str_len=%zu\n",
               ctx->gen_seq_len, llm_status_name(static_cast<int>(ctx->status)), ctx->generate_str.size());
        // Dump tokens and logits for diagnostics
        printf("[BENCH] Output tokens (first 20):"); for (int i = 0; i < std::min((int)ctx->output_tokens.size(), 20); i++) printf(" %d", ctx->output_tokens[i]); printf("\n");
        auto lastOutputs = llm->getOutputs();
        if (!lastOutputs.empty() && lastOutputs[0].get()) {
            auto info = lastOutputs[0]->getInfo(); auto ptr = lastOutputs[0]->readMap<float>();
            if (info && ptr) {
                int sz = info->size;
                printf("[BENCH] Last logits shape: dims=["); for (int i = 0; i < (int)info->dim.size(); i++) printf("%d%s", info->dim[i], i+1<(int)info->dim.size()?",":""); printf("] total=%d\n", sz);
                float mn=1e30f, mx=-1e30f, sm=0; int zc=0, nc=0, ic=0;
                for (int i = 0; i < sz; i++) { float v=ptr[i]; if(v!=v){nc++;continue;} if(v==std::numeric_limits<float>::infinity()||v==-std::numeric_limits<float>::infinity()){ic++;continue;} if(v==0)zc++; if(v<mn)mn=v; if(v>mx)mx=v; sm+=v; }
                printf("[BENCH] Logits stats: min=%.6f max=%.6f mean=%.6f zeros=%d nan=%d inf=%d\n", mn, mx, sz>0?sm/sz:0.f, zc, nc, ic);
                printf("[BENCH] First 10 logits:"); for (int i=0;i<std::min(sz,10);i++) printf(" %.4f",ptr[i]); printf("\n");
                if(sz>10){ printf("[BENCH] Last 10 logits:"); for(int i=std::max(0,sz-10);i<sz;i++) printf(" %.4f",ptr[i]); printf("\n"); }
            }
        }
        printf("[BENCH] Output run generate_str:\n---BEGIN---\n%s\n---END---\n", ctx->generate_str.c_str());
        flush_log();
    }

    // ── 9. OP profiling ──
    if (a.op_profile) {
        printf("\n[BENCH] OP-level profiling run...\n"); flush_log();
        auto profiler = MNN::Profiler::getInstance();
        auto exec = MNN::Express::ExecutorScope::Current();
        exec->setCallBack(
            [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) { profiler->start(info); return true; },
            [&](const std::vector<MNN::Tensor*>& out, const MNN::OperatorInfo* info) { for(auto o:out) o->wait(MNN::Tensor::MAP_TENSOR_READ,true); profiler->end(info); return true; });
        auto m = run_one_round(llm.get(), a);
        exec->setCallBack(nullptr, nullptr);
        printf("[BENCH] OP-profile: prefill=%.1f  decode=%.1f  total=%.1f ms\n", m.prefill_ms, m.decode_ms, m.total_ms);
        printf("\n[BENCH] === OP Time by Type ===\n"); profiler->printTimeByType(1);
        printf("\n[BENCH] === OP Time by Name (top ops) ===\n"); profiler->printTimeByName(1);
        flush_log();
    }

    printf("[BENCH] Done.\n"); flush_log();

    // ── 10. Cleanup ──
#ifdef LLM_SUPPORT_VISION
    delete g_vlm_ptr; g_vlm_ptr = nullptr;
#endif
    llm.reset();
    return 0;
}
