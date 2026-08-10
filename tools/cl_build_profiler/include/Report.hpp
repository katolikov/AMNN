//
//  Report.hpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CLPROF_REPORT_HPP
#define CLPROF_REPORT_HPP

#include <cstdio>
#include <string>
#include <vector>

#include "BuildProfiler.hpp"
#include "CLApi.hpp"
#include "CLEnvironment.hpp"

namespace clprof {

/** Run wide facts that do not belong to any single program. */
struct RunSummary {
    std::string commandLine;
    std::string buildOptions;
    ProfilerConfig config;
    double totalWallMs      = 0.0;
    double unloadCompilerMs = -1.0;
    long peakRssKb          = 0;
    int programsRequested   = 0;

    /** True when the concurrent build test was asked for; it may still not have run. */
    bool contentionRequested = false;
    ExecutionCheck execution;
    ContentionReport contention;
};

namespace report {

void printEnvironment(FILE* out, const CLApi& api, const DeviceDescription& device,
                      const EnvironmentTimings& timings);
void printConfiguration(FILE* out, const RunSummary& summary);
void printProgramTable(FILE* out, const std::vector<ProgramReport>& reports);
void printCallBreakdown(FILE* out, const std::vector<ProgramReport>& reports);
void printSplitBuild(FILE* out, const std::vector<ProgramReport>& reports);
void printVerification(FILE* out, const std::vector<ProgramReport>& reports);
void printFailures(FILE* out, const std::vector<ProgramReport>& reports);
void printKernels(FILE* out, const std::vector<ProgramReport>& reports);
void printSummary(FILE* out, const std::vector<ProgramReport>& reports, const RunSummary& summary);

/** One row per measured pass, for cross device comparison in a spreadsheet. */
bool writeCsv(const std::string& path, const DeviceDescription& device,
              const std::vector<ProgramReport>& reports, std::string& error);

/** The full result set, including device identification and every check. */
bool writeJson(const std::string& path, const DeviceDescription& device, const EnvironmentTimings& timings,
               const std::vector<ProgramReport>& reports, const RunSummary& summary, std::string& error);

}  // namespace report

}  // namespace clprof

#endif  // CLPROF_REPORT_HPP
