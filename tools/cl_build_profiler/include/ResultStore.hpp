//
//  ResultStore.hpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CLPROF_RESULT_STORE_HPP
#define CLPROF_RESULT_STORE_HPP

#include <cstdio>
#include <string>
#include <vector>

#include "BuildProfiler.hpp"
#include "CLEnvironment.hpp"

namespace clprof {

/** Everything one profiling pass produces, whether it ran in this process or a child. */
struct RunOutcome {
    DeviceDescription device;
    EnvironmentTimings timings;
    std::string libraryPath;
    std::vector<ProgramReport> reports;
    ExecutionCheck execution;
    ContentionReport contention;
    CacheRestoreReport cacheRestore;
    double unloadCompilerMs = -1.0;
    long peakRssKb          = 0;

    /** Program that was being built when the file ended, empty when none was. */
    std::string inFlight;
};

/**
 * Append only record file that lets the profiler survive a driver crash.
 *
 * A GPU compiler that segfaults takes the whole process with it, so the builds run
 * in a forked child that flushes every finished program to disk before starting the
 * next one. The parent reads what arrived, notes which program was in flight, and
 * restarts the child on the rest. Records are length prefixed, so the half written
 * record a crash leaves behind is discarded instead of being read as data.
 */
class ResultStore {
public:
    ~ResultStore();

    bool openForAppend(const std::string& path, std::string& error);
    void close();

    bool writeEnvironment(const DeviceDescription& device, const EnvironmentTimings& timings,
                          const std::string& libraryPath);

    /** Names the program about to be built, so a crash can be attributed to it. */
    bool writeMarker(const std::string& programName);
    bool writeProgram(const ProgramReport& report);
    bool writeCacheRestore(const CacheRestoreReport& restore);
    bool writeRun(const ExecutionCheck& execution, const ContentionReport& contention, double unloadCompilerMs,
                  long peakRssKb);

    /** Reads every complete record of `path` into `outcome`. */
    static bool read(const std::string& path, RunOutcome& outcome, std::string& error);

private:
    bool writeRecord(char tag, const std::string& payload);

    FILE* mFile = nullptr;
};

}  // namespace clprof

#endif  // CLPROF_RESULT_STORE_HPP
