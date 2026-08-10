//
//  Metrics.cpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#ifdef _WIN32
#include <ctime>
#else
#include <sys/resource.h>
#include <sys/time.h>
#endif

namespace clprof {

double wallMs() {
    using Clock = std::chrono::steady_clock;
    const auto now = Clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

#ifdef _WIN32

double cpuMs() {
    return static_cast<double>(std::clock()) * 1000.0 / static_cast<double>(CLOCKS_PER_SEC);
}

long peakRssKb() {
    return 0;
}

#else

double cpuMs() {
    struct rusage usage;
    if (0 != getrusage(RUSAGE_SELF, &usage)) {
        return 0.0;
    }
    const double user = usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0;
    const double sys  = usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
    return user + sys;
}

long peakRssKb() {
    struct rusage usage;
    if (0 != getrusage(RUSAGE_SELF, &usage)) {
        return 0;
    }
#if defined(__APPLE__)
    // Darwin reports ru_maxrss in bytes, every other POSIX platform in kilobytes.
    return static_cast<long>(usage.ru_maxrss / 1024);
#else
    return static_cast<long>(usage.ru_maxrss);
#endif
}

#endif  // _WIN32

double Stats::spreadPercent() const {
    if (mean <= 0.0) {
        return 0.0;
    }
    return stddev * 100.0 / mean;
}

Stats summarize(std::vector<double> samples) {
    Stats stats;
    if (samples.empty()) {
        return stats;
    }
    std::sort(samples.begin(), samples.end());
    stats.count = samples.size();
    stats.min   = samples.front();
    stats.max   = samples.back();
    for (double sample : samples) {
        stats.sum += sample;
    }
    stats.mean = stats.sum / static_cast<double>(stats.count);

    const size_t middle = stats.count / 2;
    stats.median = (stats.count % 2 == 0) ? (samples[middle - 1] + samples[middle]) * 0.5 : samples[middle];

    // Nearest-rank p90: with the small sample counts used here interpolation would
    // invent values that were never measured.
    size_t rank = static_cast<size_t>(std::ceil(0.9 * static_cast<double>(stats.count)));
    if (rank == 0) {
        rank = 1;
    }
    stats.p90 = samples[rank - 1];

    double variance = 0.0;
    for (double sample : samples) {
        const double delta = sample - stats.mean;
        variance += delta * delta;
    }
    stats.stddev = std::sqrt(variance / static_cast<double>(stats.count));
    return stats;
}

std::string formatOptional(double value, int decimals) {
    if (value < 0.0) {
        return "-";
    }
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
    return std::string(buffer);
}

}  // namespace clprof
