#include "bench_args.hpp"
#include "enum_maps.hpp"
#include "utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

BenchArgs parse_args(int argc, const char* argv[]) {
    BenchArgs a;
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <model_config.json> [OPTIONS]\n\n"
            "  --backend <cpu|metal|cuda|opencl|vulkan>\n"
            "  --threads <N>   --gpu_mode <FLAG>[,<FLAG>]\n"
            "  --hints <NAME:V>[,<NAME:V>]\n"
            "  --precision/--memory/--power <low|normal|high>\n"
            "  --attention_mode <0-12>  --quant_qkv <N> (deprecated)\n"
            "  --warmup <N>  --rounds <N>  --prompt_tokens <N>  --max_gen_tokens <N>\n"
            "  --use_mmap  --max_all_tokens <N>  --dynamic_quant <N>  --kvcache_limit <N>\n"
            "  --op_profile  --no_tuning\n"
            "  --prompt_text <str>  --vlm_image <path>  --vlm_template <str>\n"
            "  --mllm_backend <backend>  --mllm_precision <prec>  --mllm_threads <N>\n"
            "  --temperature <f>  --repetition_penalty <f>  --penalty_window <N>\n"
            "  --tmp_path <dir>\n",
            argv[0]);
        exit(1);
    }
    a.config_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        auto is = [&](const char* n) { return strcmp(argv[i], n) == 0; };
        auto nxt = [&]() -> const char* { return (i+1 < argc) ? argv[++i] : ""; };

        if      (is("--backend"))        a.backend        = nxt();
        else if (is("--threads"))        a.threads        = atoi(nxt());
        else if (is("--gpu_mode"))       a.gpu_mode_names = split_csv(nxt());
        else if (is("--hints"))          a.hint_names     = split_kv_csv(nxt());
        else if (is("--precision"))      a.precision      = nxt();
        else if (is("--memory"))         a.memory         = nxt();
        else if (is("--power"))          a.power          = nxt();
        else if (is("--attention_mode")) a.attention_mode  = atoi(nxt());
        else if (is("--quant_qkv"))      a.attention_mode  = atoi(nxt());
        else if (is("--warmup"))         a.warmup         = atoi(nxt());
        else if (is("--rounds"))         a.rounds         = atoi(nxt());
        else if (is("--prompt_tokens"))  a.prompt_tokens  = atoi(nxt());
        else if (is("--max_gen_tokens")) a.max_gen_tokens = atoi(nxt());
        else if (is("--op_profile"))     a.op_profile     = true;
        else if (is("--prompt_text"))    a.prompt_text    = nxt();
        else if (is("--vlm_image"))      a.vlm_image      = nxt();
        else if (is("--vlm_template"))   a.vlm_template   = nxt();
        else if (is("--tmp_path"))       a.tmp_path       = nxt();
        else if (is("--mllm_backend"))   a.mllm_backend   = nxt();
        else if (is("--mllm_precision")) a.mllm_precision = nxt();
        else if (is("--mllm_threads"))   a.mllm_threads   = atoi(nxt());
        else if (is("--repetition_penalty"))  a.repetition_penalty  = atof(nxt());
        else if (is("--presence_penalty"))    a.presence_penalty    = atof(nxt());
        else if (is("--frequency_penalty"))   a.frequency_penalty   = atof(nxt());
        else if (is("--temperature"))         a.temperature         = atof(nxt());
        else if (is("--no_tuning"))           a.no_tuning           = true;
        else if (is("--penalty_window"))      a.penalty_window      = atoi(nxt());
        else if (is("--use_mmap"))            a.use_mmap            = true;
        else if (is("--max_all_tokens"))      a.max_all_tokens      = atoi(nxt());
        else if (is("--dynamic_quant"))       a.dynamic_quant       = atoi(nxt());
        else if (is("--kvcache_limit"))       a.kvcache_limit       = atoi(nxt());
    }

    a.gpu_mode = 0;
    for (auto& name : a.gpu_mode_names)
        a.gpu_mode |= resolve(gpu_mode_map(), name, "gpu_mode");
    for (auto& kv : a.hint_names) {
        int k = resolve(hint_mode_map(), kv.first, "hint");
        int v = atoi(kv.second.c_str());
        a.hints.push_back({k, v});
    }
    return a;
}

bool is_gpu_backend(const std::string& b) {
    return b == "opencl" || b == "vulkan" || b == "metal" || b == "cuda";
}

int compute_effective_numthread(const BenchArgs& a) {
    return is_gpu_backend(a.backend) ? a.gpu_mode : a.threads;
}

std::string build_runtime_json(const BenchArgs& a, bool debug_mode) {
    int eff = compute_effective_numthread(a);
    std::string mllm_block;
    if (!a.mllm_backend.empty()) {
        char mbuf[256];
        snprintf(mbuf, sizeof(mbuf),
            ",\"mllm\":{\"backend_type\":\"%s\",\"thread_num\":%d,\"precision\":\"%s\"}",
            a.mllm_backend.c_str(), a.mllm_threads, a.mllm_precision.c_str());
        mllm_block = mbuf;
    }
    std::string extras;
    if (a.use_mmap) extras += ",\"use_mmap\":true";
    if (a.max_all_tokens > 0) { char t[64]; snprintf(t, sizeof(t), ",\"max_all_tokens\":%d", a.max_all_tokens); extras += t; }
    if (a.dynamic_quant != 0) { char t[64]; snprintf(t, sizeof(t), ",\"dynamic_option\":%d", a.dynamic_quant); extras += t; }
    if (a.kvcache_limit != -1) { char t[64]; snprintf(t, sizeof(t), ",\"kvcache_limit\":%d", a.kvcache_limit); extras += t; }

    char buf[2048];
    snprintf(buf, sizeof(buf),
        "{\"backend_type\":\"%s\",\"thread_num\":%d,\"precision\":\"%s\","
        "\"memory\":\"%s\",\"power\":\"%s\",\"attention_mode\":%d,"
        "\"tmp_path\":\"%s\",\"max_new_tokens\":%d,"
        "\"repetition_penalty\":%.4f,\"presence_penalty\":%.4f,"
        "\"frequency_penalty\":%.4f,\"temperature\":%.4f,\"penalty_window\":%d"
        "%s%s%s}",
        a.backend.c_str(), eff, a.precision.c_str(),
        a.memory.c_str(), a.power.c_str(), a.attention_mode,
        a.tmp_path.c_str(), a.max_gen_tokens,
        a.repetition_penalty, a.presence_penalty,
        a.frequency_penalty, a.temperature, a.penalty_window,
        mllm_block.c_str(), extras.c_str(),
        debug_mode ? ",\"enable_debug\":true" : "");
    return std::string(buf);
}
