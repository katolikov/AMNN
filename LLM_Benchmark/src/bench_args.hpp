#pragma once
//  bench_args.hpp — CLI argument parsing and runtime JSON builder

#include <string>
#include <vector>
#include <utility>

struct BenchArgs {
    std::string config_path;
    std::string backend    = "cpu";
    int         threads    = 4;
    int         gpu_mode   = 0;
    std::vector<std::string> gpu_mode_names;
    std::vector<std::pair<int,int>> hints;
    std::vector<std::pair<std::string,std::string>> hint_names;
    std::string precision  = "low";
    std::string memory     = "low";
    std::string power      = "normal";
    int         attention_mode = 0;
    int         warmup     = 2;
    int         rounds     = 5;
    int         prompt_tokens = 128;
    int         max_gen_tokens = 64;
    bool        op_profile = false;
    std::string prompt_text;
    std::string vlm_image;
    std::string vlm_template;
    std::string tmp_path   = "/data/local/tmp/mnn_bench/tmp";
    std::string mllm_backend;
    std::string mllm_precision = "low";
    int         mllm_threads   = 4;
    float       repetition_penalty  = 1.0f;
    float       presence_penalty    = 0.0f;
    float       frequency_penalty   = 0.0f;
    float       temperature         = 1.0f;
    int         penalty_window      = 0;
    bool        no_tuning           = false;
    bool        use_mmap           = false;
    int         max_all_tokens     = 0;
    int         dynamic_quant      = 0;
    int         kvcache_limit      = -1;
};

BenchArgs parse_args(int argc, const char* argv[]);
bool is_gpu_backend(const std::string& b);
int compute_effective_numthread(const BenchArgs& a);
std::string build_runtime_json(const BenchArgs& a, bool debug_mode);
