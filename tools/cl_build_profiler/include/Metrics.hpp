//
//  Metrics.hpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CLPROF_METRICS_HPP
#define CLPROF_METRICS_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace clprof {

/** Monotonic wall clock in milliseconds, counted from an unspecified origin. */
double wallMs();

/** CPU time charged to this process (user + system, all threads) in milliseconds. */
double cpuMs();

/** Peak resident set size of this process in kilobytes, 0 when the platform does not report it. */
long peakRssKb();

/** Descriptive statistics of a sample set, all values in the unit of the samples. */
struct Stats {
    size_t count  = 0;
    double min    = 0.0;
    double max    = 0.0;
    double mean   = 0.0;
    double median = 0.0;
    double p90    = 0.0;
    double stddev = 0.0;
    double sum    = 0.0;

    /** Standard deviation relative to the mean, in percent. 0 when there is no signal. */
    double spreadPercent() const;
};

/** Summarises `samples`. The vector is taken by value because it is sorted in place. */
Stats summarize(std::vector<double> samples);

/** Formats `value` with `decimals` digits, or "-" when the value is negative (not measured). */
std::string formatOptional(double value, int decimals);

}  // namespace clprof

#endif  // CLPROF_METRICS_HPP
