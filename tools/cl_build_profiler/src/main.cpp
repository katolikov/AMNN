//
//  main.cpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
//  Profiles clBuildProgram end to end: how long a device takes to turn every OpenCL
//  program MNN ships into a binary, how much of that a driver cache removes, and
//  whether the produced binary is a faithful, reusable copy of the built program.
//

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "BuildProfiler.hpp"
#include "CLApi.hpp"
#include "CLEnvironment.hpp"
#include "Metrics.hpp"
#include "ProgramCatalog.hpp"
#include "Report.hpp"
#include "ResultStore.hpp"

namespace {

using namespace clprof;

/** Exit codes, so a script can tell a slow device from a broken one. */
enum ExitCode {
    kExitOk          = 0,
    kExitFatal       = 1,  ///< bad usage, or no usable OpenCL device
    kExitBuildFailed = 2,  ///< at least one program did not compile
    kExitCheckFailed = 3   ///< at least one verification or execution check failed
};

struct Options {
    std::string libraryPath;
    std::string directory;
    std::vector<std::string> files;
    std::vector<std::string> patterns;
    std::vector<std::string> skipPatterns;
    std::string extraOptions;
    std::string replacementOptions;
    std::string csvPath;
    std::string jsonPath;
    std::string workDir       = ".";
    int precision             = 2;
    int platformIndex         = 0;
    int deviceIndex           = 0;
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    int jobs                  = 1;
    bool listOnly             = false;
    bool showKernels          = false;
    bool useReplacement       = false;
    bool programDefines       = true;
    bool crashGuard           = true;
    ProfilerConfig profiler;
};

void printUsage(FILE* out, const char* executable) {
    fprintf(out,
            "clBuildProgram profiler for the MNN OpenCL backend\n"
            "\n"
            "usage: %s [options]\n"
            "\n"
            "program selection\n"
            "  -p, --programs <glob>[,<glob>]  only profile programs matching these patterns\n"
            "      --skip <glob>[,<glob>]      exclude programs matching these patterns\n"
            "      --dir <path>                profile every *.cl file in a directory instead\n"
            "      --file <path>               profile one source file (may be repeated)\n"
            "      --list                      list the selected programs and exit\n"
            "\n"
            "build options\n"
            "      --precision <0|1|2>         MNN precision level, 2 = fp16 (default 2)\n"
            "      --options <string>          append to the MNN build options\n"
            "      --options-replace <string>  use this build option string verbatim\n"
            "      --no-program-defines        do not add the per program defines MNN passes\n"
            "\n"
            "measurement\n"
            "  -r, --repeat <n>                cold builds per program (default 2)\n"
            "      --warm-repeat <n>           repeated identical builds per program (default 2)\n"
            "      --binary-repeat <n>         builds from the produced binary (default 2)\n"
            "      --warmup <n>                discarded builds before measuring (default 0)\n"
            "      --no-salt                   do not defeat the driver compiler cache\n"
            "      --no-binary                 skip the clCreateProgramWithBinary phase\n"
            "      --no-verify                 skip the binary and kernel cross checks\n"
            "      --split-compile-link        also measure clCompileProgram and clLinkProgram\n"
            "      --jobs <n>                  build with n threads to expose driver locking\n"
            "      --no-crash-guard            do not isolate the builds in a child process\n"
            "      --work-dir <path>           where the crash guard keeps its result file\n"
            "\n"
            "device selection\n"
            "      --library <path>            load this OpenCL library instead of probing\n"
            "      --platform <index>          platform index (default 0)\n"
            "      --device <index>            device index (default 0)\n"
            "      --device-type <gpu|cpu|accelerator|all>   default gpu\n"
            "\n"
            "output\n"
            "      --csv <path>                write one row per measured build\n"
            "      --json <path>               write the full result set\n"
            "      --kernels                   list every kernel of every program\n"
            "  -v, --verbose                   print each build as it happens\n"
            "  -h, --help                      show this message\n"
            "\n"
            "exit codes: 0 ok, 1 fatal, 2 a program failed to build, 3 a check failed\n",
            executable);
}

bool parseInt(const char* text, int& value) {
    char* end         = nullptr;
    const long parsed = strtol(text, &end, 10);
    if (nullptr == end || '\0' != *end) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find(separator, start);
        if (std::string::npos == end) {
            parts.push_back(text.substr(start));
            break;
        }
        parts.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return parts;
}

void appendPatterns(const char* value, std::vector<std::string>& patterns) {
    for (const std::string& pattern : split(value, ',')) {
        if (!pattern.empty() && pattern != "all") {
            patterns.push_back(pattern);
        }
    }
}

/** Returns false when the arguments are malformed; `message` then says why. */
bool parseArguments(int argc, char* argv[], Options& options, std::string& message) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const bool hasValue        = (i + 1 < argc);
        const char* value          = hasValue ? argv[i + 1] : nullptr;

        auto requireValue = [&](const char* name) {
            if (!hasValue) {
                message = std::string(name) + " needs a value";
                return false;
            }
            ++i;
            return true;
        };

        if (argument == "-h" || argument == "--help") {
            printUsage(stdout, argv[0]);
            exit(kExitOk);
        } else if (argument == "--list") {
            options.listOnly = true;
        } else if (argument == "-p" || argument == "--programs") {
            if (!requireValue("--programs")) return false;
            appendPatterns(value, options.patterns);
        } else if (argument == "--skip") {
            if (!requireValue("--skip")) return false;
            appendPatterns(value, options.skipPatterns);
        } else if (argument == "--dir") {
            if (!requireValue("--dir")) return false;
            options.directory = value;
        } else if (argument == "--file") {
            if (!requireValue("--file")) return false;
            options.files.push_back(value);
        } else if (argument == "--precision") {
            if (!requireValue("--precision")) return false;
            if (!parseInt(value, options.precision) || options.precision < 0 || options.precision > 2) {
                message = "--precision must be 0, 1 or 2";
                return false;
            }
        } else if (argument == "--options") {
            if (!requireValue("--options")) return false;
            options.extraOptions = value;
        } else if (argument == "--options-replace") {
            if (!requireValue("--options-replace")) return false;
            options.replacementOptions = value;
            options.useReplacement     = true;
        } else if (argument == "--no-program-defines") {
            options.programDefines = false;
        } else if (argument == "-r" || argument == "--repeat") {
            if (!requireValue("--repeat")) return false;
            if (!parseInt(value, options.profiler.coldRepeat) || options.profiler.coldRepeat < 1) {
                message = "--repeat must be at least 1";
                return false;
            }
        } else if (argument == "--warm-repeat") {
            if (!requireValue("--warm-repeat")) return false;
            if (!parseInt(value, options.profiler.warmRepeat) || options.profiler.warmRepeat < 0) {
                message = "--warm-repeat must not be negative";
                return false;
            }
        } else if (argument == "--binary-repeat") {
            if (!requireValue("--binary-repeat")) return false;
            if (!parseInt(value, options.profiler.binaryRepeat) || options.profiler.binaryRepeat < 0) {
                message = "--binary-repeat must not be negative";
                return false;
            }
        } else if (argument == "--warmup") {
            if (!requireValue("--warmup")) return false;
            if (!parseInt(value, options.profiler.warmup) || options.profiler.warmup < 0) {
                message = "--warmup must not be negative";
                return false;
            }
        } else if (argument == "--no-salt") {
            options.profiler.salt = false;
        } else if (argument == "--no-binary") {
            options.profiler.binaryPhase = false;
        } else if (argument == "--no-verify") {
            options.profiler.verify = false;
        } else if (argument == "--split-compile-link") {
            options.profiler.splitCompileLink = true;
        } else if (argument == "--no-crash-guard") {
            options.crashGuard = false;
        } else if (argument == "--work-dir") {
            if (!requireValue("--work-dir")) return false;
            options.workDir = value;
        } else if (argument == "--jobs") {
            if (!requireValue("--jobs")) return false;
            if (!parseInt(value, options.jobs) || options.jobs < 1) {
                message = "--jobs must be at least 1";
                return false;
            }
        } else if (argument == "--library") {
            if (!requireValue("--library")) return false;
            options.libraryPath = value;
        } else if (argument == "--platform") {
            if (!requireValue("--platform")) return false;
            if (!parseInt(value, options.platformIndex)) {
                message = "--platform needs a number";
                return false;
            }
        } else if (argument == "--device") {
            if (!requireValue("--device")) return false;
            if (!parseInt(value, options.deviceIndex)) {
                message = "--device needs a number";
                return false;
            }
        } else if (argument == "--device-type") {
            if (!requireValue("--device-type")) return false;
            if (!parseDeviceType(value, options.deviceType)) {
                message = "--device-type must be gpu, cpu, accelerator, all or default";
                return false;
            }
        } else if (argument == "--csv") {
            if (!requireValue("--csv")) return false;
            options.csvPath = value;
        } else if (argument == "--json") {
            if (!requireValue("--json")) return false;
            options.jsonPath = value;
        } else if (argument == "--kernels") {
            options.showKernels = true;
        } else if (argument == "-v" || argument == "--verbose") {
            options.profiler.verbose = true;
        } else {
            message = "unknown argument: " + argument;
            return false;
        }
    }
    return true;
}

bool collectPrograms(const Options& options, std::vector<ProgramSource>& programs, std::string& error) {
    if (!options.directory.empty()) {
        if (!catalog::fromDirectory(options.directory, programs, error)) {
            return false;
        }
    }
    for (const std::string& file : options.files) {
        ProgramSource program;
        if (!catalog::fromFile(file, program, error)) {
            return false;
        }
        programs.push_back(program);
    }
    if (programs.empty()) {
        programs = catalog::embedded();
    }
    programs = catalog::filter(programs, options.patterns);

    if (!options.skipPatterns.empty()) {
        std::vector<ProgramSource> kept;
        for (const ProgramSource& program : programs) {
            bool skip = false;
            for (const std::string& pattern : options.skipPatterns) {
                skip = skip || globMatch(pattern, program.name);
            }
            if (!skip) {
                kept.push_back(program);
            }
        }
        programs.swap(kept);
    }

    if (programs.empty()) {
        error = "no program matched the selection";
        return false;
    }
    return true;
}

void printProgramList(const std::vector<ProgramSource>& programs) {
    printf("%-40s %10s %8s  %s\n", "PROGRAM", "BYTES", "LINES", "ORIGIN");
    size_t total = 0;
    for (const ProgramSource& program : programs) {
        printf("%-40s %10zu %8d  %s\n", program.name.c_str(), program.bytes(), program.lines(),
               program.origin.c_str());
        total += program.bytes();
    }
    printf("\n%zu programs, %.1f KB of OpenCL C\n", programs.size(), total / 1024.0);
}

std::string optionsFor(const Options& options, const std::string& baseOptions, const ProgramSource& program) {
    if (!options.programDefines) {
        return baseOptions;
    }
    const std::string extra = programDefines(program.name);
    return extra.empty() ? baseOptions : baseOptions + " " + extra;
}

/**
 * Runs every measurement. Called either directly or, when the crash guard is on,
 * inside a forked child; `store` is then the channel back to the parent.
 */
int runProfiling(const Options& options, const std::vector<ProgramSource>& programs,
                 const std::string& baseOptions, const std::string& commandLine, bool printPreamble,
                 ResultStore* store, RunOutcome& outcome) {
    std::string message;

    CLApi api;
    if (!api.load(options.libraryPath, message)) {
        fprintf(stderr, "error: %s\n", message.c_str());
        for (const std::string& probed : api.probeLog()) {
            fprintf(stderr, "  tried %s\n", probed.c_str());
        }
        return kExitFatal;
    }

    CLEnvironment environment;
    if (!environment.initialize(api, options.platformIndex, options.deviceIndex, options.deviceType, message)) {
        fprintf(stderr, "error: %s\n", message.c_str());
        return kExitFatal;
    }

    outcome.device                = environment.description();
    outcome.timings               = environment.timings();
    outcome.timings.libraryLoadMs = api.loadCostMs();
    outcome.libraryPath           = api.libraryPath();
    if (nullptr != store) {
        store->writeEnvironment(outcome.device, outcome.timings, outcome.libraryPath);
    }

    if (printPreamble) {
        RunSummary preamble;
        preamble.commandLine  = commandLine;
        preamble.buildOptions = baseOptions;
        preamble.config       = options.profiler;
        report::printEnvironment(stdout, api, outcome.device, outcome.timings);
        report::printConfiguration(stdout, preamble);
    }

    BuildProfiler profiler(api, environment, options.profiler);
    outcome.execution = profiler.runExecutionCheck();

    fprintf(stdout, "\nbuilding %zu programs\n", programs.size());
    fflush(stdout);

    for (size_t i = 0; i < programs.size(); ++i) {
        fprintf(stdout, "  [%3zu/%3zu] %-34s ", i + 1, programs.size(), programs[i].name.c_str());
        fflush(stdout);
        if (nullptr != store) {
            store->writeMarker(programs[i].name);
        }

        ProgramReport report = profiler.profile(programs[i], optionsFor(options, baseOptions, programs[i]));
        if (report.compiled) {
            const Stats cold   = summarize(report.phaseTimings(Phase::kCold));
            const Stats warm   = summarize(report.phaseTimings(Phase::kWarm));
            const Stats binary = summarize(report.phaseTimings(Phase::kBinary));
            fprintf(stdout, "cold %8.1f ms   warm %8.2f ms   binary %8.2f ms   %2zu kernels%s\n", cold.median,
                    warm.median, binary.median, report.kernels.size(),
                    report.verification.anyFailed() ? "   CHECK FAILED" : "");
        } else {
            fprintf(stdout, "FAILED: %s\n", report.failure.c_str());
        }
        fflush(stdout);

        if (nullptr != store) {
            store->writeProgram(report);
        }
        outcome.reports.push_back(report);
    }

    // Written before the concurrent test, which is the phase most likely to take a
    // fragile driver down: losing it must not cost the results gathered so far.
    if (nullptr != store) {
        store->writeRun(outcome.execution, outcome.contention, -1.0, peakRssKb());
    }

    if (options.jobs > 1) {
        fprintf(stdout, "\nrunning the concurrent build test with %d threads\n", options.jobs);
        fflush(stdout);
        outcome.contention = profiler.profileContention(programs, baseOptions, options.jobs);
    }

    outcome.unloadCompilerMs = profiler.unloadCompiler();
    outcome.peakRssKb        = peakRssKb();
    if (nullptr != store) {
        store->writeRun(outcome.execution, outcome.contention, outcome.unloadCompilerMs, outcome.peakRssKb);
    }
    return kExitOk;
}

#ifndef _WIN32

/** Synthesises the report of a program whose build killed the process. */
ProgramReport crashedReport(const ProgramSource& program, const std::string& buildOptions, int signalNumber) {
    ProgramReport report;
    report.name         = program.name;
    report.origin       = program.origin;
    report.sourceBytes  = program.bytes();
    report.sourceLines  = program.lines();
    report.buildOptions = buildOptions;
    report.crashed      = true;

    char message[160];
    snprintf(message, sizeof(message), "the build crashed the OpenCL driver, the process died on signal %d",
             signalNumber);
    report.failure = message;
    return report;
}

/**
 * Runs the profiling in a child process and restarts it past any program whose
 * build takes the driver down, so one bad kernel costs one result instead of all
 * of them. Returns the exit code of the last child that finished normally.
 */
int runWithCrashGuard(const Options& options, const std::vector<ProgramSource>& programs,
                      const std::string& baseOptions, const std::string& commandLine, RunOutcome& outcome) {
    const std::string storePath = options.workDir + "/cl_build_profiler_results.bin";
    ::remove(storePath.c_str());

    std::vector<ProgramSource> remaining = programs;
    std::vector<ProgramReport> crashes;
    bool printPreamble = true;

    for (size_t attempt = 0; attempt <= programs.size(); ++attempt) {
        const pid_t child = fork();
        if (child < 0) {
            fprintf(stderr, "warning: fork failed (%s), running without the crash guard\n", strerror(errno));
            return runProfiling(options, remaining, baseOptions, commandLine, printPreamble, nullptr, outcome);
        }

        if (0 == child) {
            ResultStore store;
            std::string error;
            if (!store.openForAppend(storePath, error)) {
                fprintf(stderr, "error: %s (use --work-dir <writable path>)\n", error.c_str());
                _exit(kExitFatal);
            }
            RunOutcome childOutcome;
            const int code =
                runProfiling(options, remaining, baseOptions, commandLine, printPreamble, &store, childOutcome);
            _exit(code);
        }

        int status = 0;
        while (waitpid(child, &status, 0) < 0 && EINTR == errno) {
            continue;
        }
        if (WIFEXITED(status)) {
            const int code = WEXITSTATUS(status);
            if (kExitOk != code) {
                return code;
            }
            break;
        }

        const int signalNumber = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
        RunOutcome partial;
        std::string error;
        ResultStore::read(storePath, partial, error);

        if (partial.inFlight.empty()) {
            fprintf(stdout,
                    "\n  the profiler died on signal %d outside a program build, most likely in the\n"
                    "  concurrent build test; reporting everything that was measured before that\n",
                    signalNumber);
            fflush(stdout);
            break;
        }

        size_t crashedIndex = remaining.size();
        for (size_t i = 0; i < remaining.size(); ++i) {
            if (remaining[i].name == partial.inFlight) {
                crashedIndex = i;
                break;
            }
        }
        if (crashedIndex >= remaining.size()) {
            break;
        }

        fprintf(stdout, "\n  building %s crashed the OpenCL driver (signal %d), continuing with the rest\n",
                remaining[crashedIndex].name.c_str(), signalNumber);
        fflush(stdout);
        crashes.push_back(crashedReport(remaining[crashedIndex],
                                        optionsFor(options, baseOptions, remaining[crashedIndex]), signalNumber));

        remaining.erase(remaining.begin(), remaining.begin() + crashedIndex + 1);
        printPreamble = false;
        if (remaining.empty()) {
            break;
        }
    }

    std::string error;
    if (!ResultStore::read(storePath, outcome, error)) {
        fprintf(stderr, "error: %s\n", error.c_str());
        return kExitFatal;
    }
    ::remove(storePath.c_str());

    outcome.reports.insert(outcome.reports.end(), crashes.begin(), crashes.end());
    if (outcome.reports.empty()) {
        fprintf(stderr, "error: no program could be measured\n");
        return kExitFatal;
    }
    return kExitOk;
}

#endif  // _WIN32

}  // namespace

int main(int argc, char* argv[]) {
    const double startWall = wallMs();

    Options options;
    std::string message;
    if (!parseArguments(argc, argv, options, message)) {
        fprintf(stderr, "error: %s\n\n", message.c_str());
        printUsage(stderr, argv[0]);
        return kExitFatal;
    }

    std::vector<ProgramSource> programs;
    if (!collectPrograms(options, programs, message)) {
        fprintf(stderr, "error: %s\n", message.c_str());
        return kExitFatal;
    }
    if (options.listOnly) {
        printProgramList(programs);
        return kExitOk;
    }

    std::string buildOptions =
        options.useReplacement ? options.replacementOptions : mnnBuildOptions(options.precision);
    if (!options.extraOptions.empty()) {
        buildOptions += " " + options.extraOptions;
    }

    RunSummary summary;
    for (int i = 0; i < argc; ++i) {
        summary.commandLine += (0 == i ? "" : " ");
        summary.commandLine += argv[i];
    }
    summary.buildOptions         = buildOptions;
    summary.config               = options.profiler;
    summary.programsRequested    = static_cast<int>(programs.size());
    summary.contentionRequested  = options.jobs > 1;

    RunOutcome outcome;
    int status = kExitOk;
#ifndef _WIN32
    if (options.crashGuard) {
        status = runWithCrashGuard(options, programs, buildOptions, summary.commandLine, outcome);
    } else {
        status = runProfiling(options, programs, buildOptions, summary.commandLine, true, nullptr, outcome);
    }
#else
    status = runProfiling(options, programs, buildOptions, summary.commandLine, true, nullptr, outcome);
#endif
    if (kExitOk != status) {
        return status;
    }

    summary.execution        = outcome.execution;
    summary.contention       = outcome.contention;
    summary.unloadCompilerMs = outcome.unloadCompilerMs;
    summary.peakRssKb        = std::max(outcome.peakRssKb, peakRssKb());
    summary.totalWallMs      = wallMs() - startWall;

    report::printProgramTable(stdout, outcome.reports);
    report::printCallBreakdown(stdout, outcome.reports);
    report::printSplitBuild(stdout, outcome.reports);
    if (options.profiler.verify) {
        report::printVerification(stdout, outcome.reports);
    }
    if (options.showKernels) {
        report::printKernels(stdout, outcome.reports);
    }
    report::printFailures(stdout, outcome.reports);
    report::printSummary(stdout, outcome.reports, summary);

    if (!options.csvPath.empty()) {
        if (report::writeCsv(options.csvPath, outcome.device, outcome.reports, message)) {
            fprintf(stdout, "\n  csv written to %s\n", options.csvPath.c_str());
        } else {
            fprintf(stderr, "error: %s\n", message.c_str());
        }
    }
    if (!options.jsonPath.empty()) {
        if (report::writeJson(options.jsonPath, outcome.device, outcome.timings, outcome.reports, summary,
                              message)) {
            fprintf(stdout, "  json written to %s\n", options.jsonPath.c_str());
        } else {
            fprintf(stderr, "error: %s\n", message.c_str());
        }
    }

    bool buildFailed = false;
    bool checkFailed = summary.execution.measured && !summary.execution.ok;
    for (const ProgramReport& report : outcome.reports) {
        buildFailed = buildFailed || !report.compiled;
        checkFailed = checkFailed || report.verification.anyFailed();
    }

    if (checkFailed) {
        fprintf(stdout, "\nRESULT: a verification check failed\n");
        return kExitCheckFailed;
    }
    if (buildFailed) {
        fprintf(stdout, "\nRESULT: every check passed, but some programs did not build\n");
        return kExitBuildFailed;
    }
    fprintf(stdout, "\nRESULT: all %zu programs built and verified\n", outcome.reports.size());
    return kExitOk;
}
