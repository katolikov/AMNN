#pragma once
//  utils.hpp — Small helpers: stats, string splitting, token generation, status names

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Us    = std::chrono::microseconds;

inline void flush_log() { fflush(stdout); fflush(stderr); }

struct Stats {
    double min_ms = 0, max_ms = 0, avg_ms = 0, std_ms = 0;
};

inline Stats compute_stats(const std::vector<double>& v) {
    Stats s;
    if (v.empty()) return s;
    s.min_ms = *std::min_element(v.begin(), v.end());
    s.max_ms = *std::max_element(v.begin(), v.end());
    s.avg_ms = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0;
    for (auto x : v) var += (x - s.avg_ms) * (x - s.avg_ms);
    s.std_ms = std::sqrt(var / v.size());
    return s;
}

inline std::vector<std::string> split_csv(const char* s) {
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

inline std::vector<std::pair<std::string,std::string>> split_kv_csv(const char* s) {
    std::vector<std::pair<std::string,std::string>> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        auto c = tok.find(':');
        if (c != std::string::npos)
            out.push_back({tok.substr(0, c), tok.substr(c + 1)});
    }
    return out;
}

inline std::vector<int> make_synthetic_tokens(int n) {
    return std::vector<int>(n, 16);
}

inline const char* llm_status_name(int status) {
    switch (status) {
        case -1: return "NOT_LOADED";
        case  0: return "RUNNING";
        case  1: return "NORMAL_FINISHED";
        case  2: return "MAX_TOKENS_FINISHED";
        case  3: return "USER_CANCEL";
        case  4: return "INTERNAL_ERROR";
        case  5: return "TIMEOUT";
        default: return "UNKNOWN";
    }
}
