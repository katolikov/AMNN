//
//  Report.cpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Report.hpp"

#include <algorithm>
#include <fstream>

#include "Metrics.hpp"

namespace clprof {
namespace report {

namespace {

const char* kRule =
    "--------------------------------------------------------------------------------------------------------";

void printSection(FILE* out, const char* title) {
    fprintf(out, "\n%s\n%s\n%s\n", kRule, title, kRule);
}

std::string truncate(const std::string& text, size_t width) {
    if (text.size() <= width) {
        return text;
    }
    return text.substr(0, width - 3) + "...";
}

double ratio(double numerator, double denominator) {
    if (denominator <= 0.0) {
        return 0.0;
    }
    return numerator / denominator;
}

/** Sorted copy, slowest cold build first: the interesting programs come out on top. */
std::vector<const ProgramReport*> sortedByCost(const std::vector<ProgramReport>& reports) {
    std::vector<const ProgramReport*> sorted;
    sorted.reserve(reports.size());
    for (const ProgramReport& report : reports) {
        sorted.push_back(&report);
    }
    std::sort(sorted.begin(), sorted.end(), [](const ProgramReport* lhs, const ProgramReport* rhs) {
        const double left  = summarize(lhs->phaseTimings(Phase::kCold)).median;
        const double right = summarize(rhs->phaseTimings(Phase::kCold)).median;
        if (left != right) {
            return left > right;
        }
        return lhs->name < rhs->name;
    });
    return sorted;
}

Stats callStats(const ProgramReport& report, Phase phase, double BuildSample::*field) {
    std::vector<double> values;
    for (const BuildSample& sample : report.samples) {
        if (sample.phase == phase && sample.ok) {
            values.push_back(sample.*field);
        }
    }
    return summarize(values);
}

std::string jsonEscape(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() + 8);
    for (char character : text) {
        switch (character) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(character) < 0x20) {
                    char buffer[8];
                    snprintf(buffer, sizeof(buffer), "\\u%04x", character);
                    escaped += buffer;
                } else {
                    escaped += character;
                }
        }
    }
    return escaped;
}

std::string csvEscape(const std::string& text) {
    if (text.find_first_of(",\"\n") == std::string::npos) {
        return text;
    }
    std::string escaped = "\"";
    for (char character : text) {
        if ('"' == character) {
            escaped += "\"\"";
        } else if ('\n' == character) {
            escaped += ' ';
        } else {
            escaped += character;
        }
    }
    escaped += '"';
    return escaped;
}

}  // namespace

void printEnvironment(FILE* out, const CLApi& api, const DeviceDescription& device,
                      const EnvironmentTimings& timings) {
    printSection(out, "OPENCL ENVIRONMENT");
    fprintf(out, "  library            : %s\n", api.libraryPath().c_str());
    for (const std::string& rejected : api.probeLog()) {
        fprintf(out, "  probed (rejected)  : %s\n", rejected.c_str());
    }
    fprintf(out, "  platform           : %s [%d of %u]\n", device.platformName.c_str(), device.platformIndex,
            device.platformCount);
    fprintf(out, "  platform vendor    : %s\n", device.platformVendor.c_str());
    fprintf(out, "  platform version   : %s\n", device.platformVersion.c_str());
    fprintf(out, "  device             : %s [%d of %u, %s]\n", device.deviceName.c_str(), device.deviceIndex,
            device.deviceCount, device.deviceTypeName.c_str());
    fprintf(out, "  device vendor      : %s\n", device.deviceVendor.c_str());
    fprintf(out, "  device version     : %s\n", device.deviceVersion.c_str());
    fprintf(out, "  driver version     : %s\n", device.driverVersion.c_str());
    fprintf(out, "  OpenCL C version   : %s\n", device.openclCVersion.c_str());
    fprintf(out, "  compute units      : %u @ %u MHz\n", device.computeUnits, device.clockMHz);
    fprintf(out, "  global memory      : %llu MB (cache %llu KB)\n",
            static_cast<unsigned long long>(device.globalMem >> 20),
            static_cast<unsigned long long>(device.globalCache >> 10));
    fprintf(out, "  local memory       : %llu KB\n", static_cast<unsigned long long>(device.localMem >> 10));
    fprintf(out, "  max alloc          : %llu MB\n", static_cast<unsigned long long>(device.maxAlloc >> 20));
    fprintf(out, "  max work group     : %zu\n", device.maxWorkGroup);
    fprintf(out, "  compiler / linker  : %s / %s\n", device.compilerAvailable ? "available" : "MISSING",
            device.linkerAvailable ? "available" : "MISSING");
    if (!device.builtInKernels.empty()) {
        fprintf(out, "  built in kernels   : %s\n", device.builtInKernels.c_str());
    }
    fprintf(out, "  extensions         : %s\n", device.deviceExtensions.c_str());

    fprintf(out, "\n  one-off setup cost (ms)\n");
    fprintf(out, "    dlopen + dlsym            : %8.3f\n", timings.libraryLoadMs);
    fprintf(out, "    clGetPlatformIDs          : %8.3f\n", timings.platformIdsMs);
    fprintf(out, "    clGetDeviceIDs            : %8.3f\n", timings.deviceIdsMs);
    fprintf(out, "    clGetDeviceInfo (all)     : %8.3f\n", timings.deviceInfoMs);
    fprintf(out, "    clCreateContext           : %8.3f\n", timings.contextMs);
    fprintf(out, "    clCreateCommandQueue      : %8s\n", formatOptional(timings.queueMs, 3).c_str());
}

void printConfiguration(FILE* out, const RunSummary& summary) {
    printSection(out, "CONFIGURATION");
    fprintf(out, "  command            : %s\n", summary.commandLine.c_str());
    fprintf(out, "  cold repeats       : %d (salt %s)\n", summary.config.coldRepeat,
            summary.config.salt ? "on, every build is a real compile" : "off, driver cache may serve builds");
    fprintf(out, "  warm repeats       : %d\n", summary.config.warmRepeat);
    fprintf(out, "  binary repeats     : %d\n", summary.config.binaryPhase ? summary.config.binaryRepeat : 0);
    fprintf(out, "  warmup builds      : %d\n", summary.config.warmup);
    fprintf(out, "  verification       : %s\n", summary.config.verify ? "on" : "off");
    fprintf(out, "  compile/link split : %s\n", summary.config.splitCompileLink ? "on" : "off");
    fprintf(out, "  build options      : %s\n", summary.buildOptions.c_str());
}

void printProgramTable(FILE* out, const std::vector<ProgramReport>& reports) {
    printSection(out, "BUILD COST PER PROGRAM (milliseconds, sorted by cold median)");
    fprintf(out, "  %-30s %7s %5s %8s %9s %9s %9s %8s %9s %9s %7s %7s\n", "PROGRAM", "SRC KB", "KERN", "BIN KB",
            "COLD med", "COLD min", "COLD max", "SD %", "CPU med", "WARM med", "BIN med", "COLD/W");
    for (const ProgramReport* report : sortedByCost(reports)) {
        if (!report->compiled) {
            continue;
        }
        const Stats cold   = summarize(report->phaseTimings(Phase::kCold));
        const Stats warm   = summarize(report->phaseTimings(Phase::kWarm));
        const Stats binary = summarize(report->phaseTimings(Phase::kBinary));
        const Stats cpu    = callStats(*report, Phase::kCold, &BuildSample::buildCpuMs);

        size_t binaryBytes = 0;
        for (const BuildSample& sample : report->samples) {
            if (Phase::kCacheFill == sample.phase) {
                binaryBytes = sample.binaryBytes;
            }
        }

        fprintf(out, "  %-30s %7.1f %5zu %8.1f %9.1f %9.1f %9.1f %8.1f %9.1f %9.2f %7.2f %7.1f\n",
                truncate(report->name, 30).c_str(), report->sourceBytes / 1024.0, report->kernels.size(),
                binaryBytes / 1024.0, cold.median, cold.min, cold.max, cold.spreadPercent(), cpu.median,
                warm.median, binary.median, ratio(cold.median, warm.median));
    }
}

void printCallBreakdown(FILE* out, const std::vector<ProgramReport>& reports) {
    printSection(out, "COLD BUILD CALL BREAKDOWN (median milliseconds per call)");
    fprintf(out, "  %-30s %9s %9s %8s %8s %9s %9s %8s %9s\n", "PROGRAM", "create", "build", "status", "binary",
            "kernels", "kern info", "release", "TOTAL");
    for (const ProgramReport* report : sortedByCost(reports)) {
        if (!report->compiled) {
            continue;
        }
        const Stats create   = callStats(*report, Phase::kCold, &BuildSample::createMs);
        const Stats build    = callStats(*report, Phase::kCold, &BuildSample::buildMs);
        const Stats status   = callStats(*report, Phase::kCold, &BuildSample::statusMs);
        const Stats binary   = callStats(*report, Phase::kCold, &BuildSample::binaryMs);
        const Stats kernels  = callStats(*report, Phase::kCold, &BuildSample::kernelsMs);
        const Stats kinfo    = callStats(*report, Phase::kCold, &BuildSample::kernelInfoMs);
        const Stats release  = callStats(*report, Phase::kCold, &BuildSample::releaseMs);
        const double total = create.median + build.median + status.median + binary.median + kernels.median +
                             kinfo.median + release.median;
        fprintf(out, "  %-30s %9.3f %9.1f %8.3f %8.3f %9.3f %9.3f %8.3f %9.1f\n", truncate(report->name, 30).c_str(),
                create.median, build.median, status.median, binary.median, kernels.median, kinfo.median,
                release.median, total);
    }
}

void printSplitBuild(FILE* out, const std::vector<ProgramReport>& reports) {
    bool any = false;
    for (const ProgramReport& report : reports) {
        if (report.split.measured) {
            any = true;
            break;
        }
    }
    if (!any) {
        return;
    }

    printSection(out, "FRONT END VERSUS BACK END (clCompileProgram / clLinkProgram, milliseconds)");
    fprintf(out, "  %-30s %10s %10s %10s  %s\n", "PROGRAM", "compile", "link", "sum", "note");
    for (const ProgramReport* report : sortedByCost(reports)) {
        if (!report->split.measured) {
            continue;
        }
        const double sum = report->split.ok ? report->split.compileMs + report->split.linkMs : -1.0;
        fprintf(out, "  %-30s %10s %10s %10s  %s\n", truncate(report->name, 30).c_str(),
                formatOptional(report->split.compileMs, 1).c_str(),
                formatOptional(report->split.linkMs, 1).c_str(), formatOptional(sum, 1).c_str(),
                report->split.failure.c_str());
    }
}

void printCacheRestore(FILE* out, const CacheRestoreReport& restore, const std::vector<ProgramReport>& reports) {
    if (!restore.measured) {
        return;
    }
    printSection(out, "WARM START FROM AN MNN CACHE FILE");
    fprintf(out, "  file               : %s (%.1f KB)\n", restore.path.c_str(), restore.fileBytes / 1024.0);
    fprintf(out, "  programs in file   : %d\n", restore.entriesInFile);
    fprintf(out, "  accepted by MNN    : %d\n", restore.entriesUsable);
    fprintf(out, "  rejected by MNN    : %d  (these are recompiled from source at runtime)\n",
            restore.entriesRejected);

    if (!restore.samples.empty()) {
        fprintf(out, "\n  %-30s %8s %10s %10s %10s %7s\n", "PROGRAM", "BIN KB", "create ms", "build ms",
                "restore ms", "KERNELS");
        for (const CacheRestoreSample& sample : restore.samples) {
            if (!sample.ok) {
                fprintf(out, "  %-30s %8.1f  FAILED: %s\n", truncate(sample.program, 30).c_str(),
                        sample.binaryBytes / 1024.0, sample.failure.c_str());
                continue;
            }
            fprintf(out, "  %-30s %8.1f %10.3f %10.3f %10.3f %7d\n", truncate(sample.program, 30).c_str(),
                    sample.binaryBytes / 1024.0, sample.createMs, sample.buildMs, sample.restoreMs(),
                    sample.kernelCount);
        }
    }

    // The cost the engine avoids is the cold build of exactly those programs the cache
    // could restore, which is the only fair comparison.
    double avoided = 0.0;
    for (const CacheRestoreSample& sample : restore.samples) {
        if (!sample.ok) {
            continue;
        }
        for (const ProgramReport& report : reports) {
            if (report.name == sample.program && report.compiled) {
                avoided += summarize(report.phaseTimings(Phase::kCold)).median;
                break;
            }
        }
    }

    fprintf(out, "\n  restored           : %d of %d programs in %.1f ms\n", restore.restored(),
            restore.entriesInFile, restore.totalRestoreMs());
    if (avoided > 0.0) {
        fprintf(out, "  compiling them     : %.1f ms\n", avoided);
        fprintf(out, "  the cache is worth : %.1fx, %.1f ms of start up time\n",
                ratio(avoided, restore.totalRestoreMs()), avoided - restore.totalRestoreMs());
    }
    for (const std::string& note : restore.notes) {
        fprintf(out, "    %s\n", note.c_str());
    }
}

void printVerification(FILE* out, const std::vector<ProgramReport>& reports) {
    printSection(out, "VERIFICATION OF THE BUILT PROGRAMS");
    fprintf(out, "  round trip  : clCreateProgramWithBinary + clBuildProgram on the produced binary\n");
    fprintf(out, "  determinism : two builds of the same source and options produce the same binary\n");
    fprintf(out, "  stable      : the binary read back after the round trip is byte identical\n");
    fprintf(out, "  kernels     : the reloaded program exposes the same kernel names\n");
    fprintf(out, "  attributes  : and the same work group size, preferred multiple and memory usage\n\n");
    fprintf(out, "  %-30s %11s %12s %9s %9s %11s\n", "PROGRAM", "round trip", "determinism", "stable", "kernels",
            "attributes");

    int passed = 0;
    int failed = 0;
    for (const ProgramReport* report : sortedByCost(reports)) {
        if (!report->compiled) {
            continue;
        }
        const Verification& check = report->verification;
        if (check.anyFailed()) {
            ++failed;
        } else {
            ++passed;
        }
        fprintf(out, "  %-30s %11s %12s %9s %9s %11s\n", truncate(report->name, 30).c_str(),
                checkStateName(check.binaryRoundTrip), checkStateName(check.binaryDeterministic),
                checkStateName(check.binaryStable), checkStateName(check.kernelNamesMatch),
                checkStateName(check.kernelAttributesMatch));
    }
    fprintf(out, "\n  %d program(s) fully verified, %d with at least one failed check\n", passed, failed);

    for (const ProgramReport& report : reports) {
        for (const std::string& note : report.verification.notes) {
            fprintf(out, "    %-28s %s\n", truncate(report.name, 28).c_str(), note.c_str());
        }
    }
}

void printFailures(FILE* out, const std::vector<ProgramReport>& reports) {
    std::vector<const ProgramReport*> failures;
    for (const ProgramReport& report : reports) {
        if (!report.compiled) {
            failures.push_back(&report);
        }
    }
    if (failures.empty()) {
        return;
    }

    printSection(out, "PROGRAMS THAT DID NOT BUILD");
    for (const ProgramReport* report : failures) {
        fprintf(out, "  %s%s: %s\n", report->name.c_str(), report->crashed ? " [DRIVER CRASH]" : "",
                report->failure.c_str());
        if (!report->buildLog.empty()) {
            fprintf(out, "    build log: %s\n", truncate(report->buildLog, 2000).c_str());
        }
    }
}

void printKernels(FILE* out, const std::vector<ProgramReport>& reports) {
    printSection(out, "KERNELS EXPOSED BY EACH PROGRAM");
    for (const ProgramReport* report : sortedByCost(reports)) {
        if (!report->compiled) {
            continue;
        }
        fprintf(out, "  %s (%zu kernels)\n", report->name.c_str(), report->kernels.size());
        for (const KernelInfo& kernel : report->kernels) {
            fprintf(out, "    %-44s args %2u  wg %5zu  mult %4zu  local %6llu B  private %6llu B\n",
                    truncate(kernel.name, 44).c_str(), kernel.numArgs, kernel.workGroupSize,
                    kernel.preferredMultiple, static_cast<unsigned long long>(kernel.localMemBytes),
                    static_cast<unsigned long long>(kernel.privateMemBytes));
        }
    }
}

namespace {

/** The totals of the source build phases; nothing to say when no program was built. */
void printBuildTotals(FILE* out, const std::vector<ProgramReport>& reports, const RunSummary& summary) {
    std::vector<double> coldMedians;
    double coldTotal   = 0.0;
    double warmTotal   = 0.0;
    double binaryTotal = 0.0;
    double cpuTotal    = 0.0;
    size_t sourceTotal = 0;
    size_t binaryBytes = 0;
    int compiled       = 0;
    int kernelTotal    = 0;
    int crashed        = 0;
    int emptyPrograms  = 0;

    for (const ProgramReport& report : reports) {
        crashed += report.crashed ? 1 : 0;
        if (!report.compiled) {
            continue;
        }
        emptyPrograms += report.kernels.empty() ? 1 : 0;
        ++compiled;
        const Stats cold = summarize(report.phaseTimings(Phase::kCold));
        const Stats warm = summarize(report.phaseTimings(Phase::kWarm));
        const Stats bin  = summarize(report.phaseTimings(Phase::kBinary));
        const Stats cpu  = callStats(report, Phase::kCold, &BuildSample::buildCpuMs);
        coldMedians.push_back(cold.median);
        coldTotal += cold.median;
        warmTotal += warm.median;
        binaryTotal += bin.median;
        cpuTotal += cpu.median;
        sourceTotal += report.sourceBytes;
        kernelTotal += static_cast<int>(report.kernels.size());
        for (const BuildSample& sample : report.samples) {
            if (Phase::kCacheFill == sample.phase) {
                binaryBytes += sample.binaryBytes;
            }
        }
    }

    const Stats acrossPrograms = summarize(coldMedians);
    fprintf(out, "  programs requested         : %d\n", summary.programsRequested);
    fprintf(out, "  programs built             : %d\n", compiled);
    fprintf(out, "  programs failed            : %d\n", static_cast<int>(reports.size()) - compiled);
    fprintf(out, "  programs that crashed      : %d\n", crashed);
    fprintf(out, "  programs with no kernel    : %d  (every kernel guarded out by the build options)\n",
            emptyPrograms);
    fprintf(out, "  kernels discovered         : %d\n", kernelTotal);
    fprintf(out, "  source compiled            : %.1f KB\n", sourceTotal / 1024.0);
    fprintf(out, "  binaries produced          : %.1f KB\n", binaryBytes / 1024.0);
    fprintf(out, "\n");
    fprintf(out, "  cold build, all programs   : %9.1f ms  (a cache-less MNN cold start)\n", coldTotal);
    fprintf(out, "  warm build, all programs   : %9.1f ms  (driver compiler cache hit)\n", warmTotal);
    fprintf(out, "  binary build, all programs : %9.1f ms  (MNN cache file hit)\n", binaryTotal);
    fprintf(out, "  cpu time inside cold build : %9.1f ms  (%.0f%% of the wall time)\n", cpuTotal,
            100.0 * ratio(cpuTotal, coldTotal));
    fprintf(out, "  cold / warm                : %9.1fx\n", ratio(coldTotal, warmTotal));
    fprintf(out, "  cold / binary              : %9.1fx\n", ratio(coldTotal, binaryTotal));
    if (sourceTotal > 0) {
        fprintf(out, "  throughput                 : %9.1f KB of source per second\n",
                (sourceTotal / 1024.0) / (coldTotal / 1000.0));
    }
    fprintf(out, "  slowest program            : %9.1f ms\n", acrossPrograms.max);
    fprintf(out, "  median program             : %9.1f ms\n", acrossPrograms.median);
}

}  // namespace

void printSummary(FILE* out, const std::vector<ProgramReport>& reports, const RunSummary& summary) {
    printSection(out, "SUMMARY");

    if (reports.empty()) {
        fprintf(out, "  source builds              : not run\n");
    } else {
        printBuildTotals(out, reports, summary);
    }

    if (summary.cacheRestore.measured) {
        fprintf(out, "\n  MNN cache file             : %d of %d programs restored in %.1f ms\n",
                summary.cacheRestore.restored(), summary.cacheRestore.entriesInFile,
                summary.cacheRestore.totalRestoreMs());
        if (summary.cacheRestore.entriesRejected > 0) {
            fprintf(out, "    rejected                 : %d, recompiled from source at runtime\n",
                    summary.cacheRestore.entriesRejected);
        }
    }

    if (summary.execution.measured) {
        const ExecutionCheck& check = summary.execution;
        fprintf(out, "\n  execution check            : %s\n", check.ok ? "PASS" : "FAIL");
        if (!check.ok) {
            fprintf(out, "    reason                   : %s\n", check.failure.c_str());
        }
        fprintf(out,
                "    build %.1f ms, clCreateKernel %.3f ms, buffers %.3f ms, args %.3f ms,\n"
                "    enqueue %.3f ms, finish %.3f ms, read back %.3f ms\n",
                check.buildMs, check.kernelMs, check.bufferMs, check.argsMs, check.enqueueMs, check.finishMs,
                check.readMs);
    } else if (!summary.execution.failure.empty()) {
        fprintf(out, "\n  execution check            : skipped (%s)\n", summary.execution.failure.c_str());
    }

    if (summary.contention.measured) {
        const ContentionReport& contention = summary.contention;
        fprintf(out, "\n  concurrent build test      : %d threads over %d programs\n", contention.jobs,
                contention.programCount);
        fprintf(out, "    serial wall time         : %9.1f ms\n", contention.serialWallMs);
        fprintf(out, "    concurrent wall time     : %9.1f ms\n", contention.wallMs);
        fprintf(out, "    summed build time        : %9.1f ms\n", contention.summedBuildMs);
        fprintf(out, "    effective parallelism    : %9.2fx of %d (1.0 means the driver serialises)\n",
                contention.effectiveParallelism(), contention.jobs);
        fprintf(out, "    failed builds            : %d  (serial and concurrent pass together)\n",
                contention.failures);
    } else if (summary.contentionRequested) {
        fprintf(out,
                "\n  concurrent build test      : DID NOT FINISH\n"
                "    the process died during it, so this driver cannot compile from several\n"
                "    threads at once with this program set\n");
    }

    fprintf(out, "\n  clUnloadPlatformCompiler   : %s ms\n", formatOptional(summary.unloadCompilerMs, 3).c_str());
    fprintf(out, "  peak resident memory       : %.1f MB\n", summary.peakRssKb / 1024.0);
    fprintf(out, "  total wall time            : %.1f s\n", summary.totalWallMs / 1000.0);
}

bool writeCsv(const std::string& path, const DeviceDescription& device,
              const std::vector<ProgramReport>& reports, std::string& error) {
    std::ofstream file(path.c_str());
    if (!file.is_open()) {
        error = "cannot write " + path;
        return false;
    }
    file << "device,driver,program,phase,iteration,ok,create_ms,build_ms,build_cpu_ms,status_ms,binary_ms,"
            "kernels_ms,kernel_info_ms,release_ms,total_ms,binary_bytes,kernel_count,source_bytes\n";
    for (const ProgramReport& report : reports) {
        for (const BuildSample& sample : report.samples) {
            file << csvEscape(device.deviceName) << ',' << csvEscape(device.driverVersion) << ','
                 << csvEscape(report.name) << ',' << phaseName(sample.phase) << ',' << sample.iteration << ','
                 << (sample.ok ? 1 : 0) << ',' << sample.createMs << ',' << sample.buildMs << ','
                 << sample.buildCpuMs << ',' << sample.statusMs << ',' << sample.binaryMs << ','
                 << sample.kernelsMs << ',' << sample.kernelInfoMs << ',' << sample.releaseMs << ','
                 << sample.totalMs() << ',' << sample.binaryBytes << ',' << sample.kernelCount << ','
                 << report.sourceBytes << '\n';
        }
    }
    return true;
}

bool writeJson(const std::string& path, const DeviceDescription& device, const EnvironmentTimings& timings,
               const std::vector<ProgramReport>& reports, const RunSummary& summary, std::string& error) {
    std::ofstream file(path.c_str());
    if (!file.is_open()) {
        error = "cannot write " + path;
        return false;
    }

    file << "{\n";
    file << "  \"device\": {\n";
    file << "    \"platform\": \"" << jsonEscape(device.platformName) << "\",\n";
    file << "    \"name\": \"" << jsonEscape(device.deviceName) << "\",\n";
    file << "    \"vendor\": \"" << jsonEscape(device.deviceVendor) << "\",\n";
    file << "    \"version\": \"" << jsonEscape(device.deviceVersion) << "\",\n";
    file << "    \"driver\": \"" << jsonEscape(device.driverVersion) << "\",\n";
    file << "    \"openclC\": \"" << jsonEscape(device.openclCVersion) << "\",\n";
    file << "    \"computeUnits\": " << device.computeUnits << ",\n";
    file << "    \"clockMHz\": " << device.clockMHz << ",\n";
    file << "    \"extensions\": \"" << jsonEscape(device.deviceExtensions) << "\"\n";
    file << "  },\n";

    file << "  \"setupMs\": {\n";
    file << "    \"libraryLoad\": " << timings.libraryLoadMs << ",\n";
    file << "    \"platformIds\": " << timings.platformIdsMs << ",\n";
    file << "    \"deviceIds\": " << timings.deviceIdsMs << ",\n";
    file << "    \"deviceInfo\": " << timings.deviceInfoMs << ",\n";
    file << "    \"context\": " << timings.contextMs << ",\n";
    file << "    \"queue\": " << timings.queueMs << "\n";
    file << "  },\n";

    file << "  \"run\": {\n";
    file << "    \"command\": \"" << jsonEscape(summary.commandLine) << "\",\n";
    file << "    \"buildOptions\": \"" << jsonEscape(summary.buildOptions) << "\",\n";
    file << "    \"coldRepeat\": " << summary.config.coldRepeat << ",\n";
    file << "    \"warmRepeat\": " << summary.config.warmRepeat << ",\n";
    file << "    \"binaryRepeat\": " << (summary.config.binaryPhase ? summary.config.binaryRepeat : 0) << ",\n";
    file << "    \"salt\": " << (summary.config.salt ? "true" : "false") << ",\n";
    file << "    \"totalWallMs\": " << summary.totalWallMs << ",\n";
    file << "    \"unloadCompilerMs\": " << summary.unloadCompilerMs << ",\n";
    file << "    \"peakRssKb\": " << summary.peakRssKb << ",\n";
    file << "    \"executionCheck\": \""
         << (summary.execution.measured ? (summary.execution.ok ? "pass" : "fail") : "skipped") << "\"\n";
    file << "  },\n";

    file << "  \"programs\": [\n";
    for (size_t i = 0; i < reports.size(); ++i) {
        const ProgramReport& report = reports[i];
        const Stats cold            = summarize(report.phaseTimings(Phase::kCold));
        const Stats warm            = summarize(report.phaseTimings(Phase::kWarm));
        const Stats binary          = summarize(report.phaseTimings(Phase::kBinary));
        const Stats cpu             = callStats(report, Phase::kCold, &BuildSample::buildCpuMs);

        size_t binaryBytes = 0;
        for (const BuildSample& sample : report.samples) {
            if (Phase::kCacheFill == sample.phase) {
                binaryBytes = sample.binaryBytes;
            }
        }

        file << "    {\n";
        file << "      \"name\": \"" << jsonEscape(report.name) << "\",\n";
        file << "      \"origin\": \"" << jsonEscape(report.origin) << "\",\n";
        file << "      \"sourceBytes\": " << report.sourceBytes << ",\n";
        file << "      \"sourceLines\": " << report.sourceLines << ",\n";
        file << "      \"compiled\": " << (report.compiled ? "true" : "false") << ",\n";
        file << "      \"failure\": \"" << jsonEscape(report.failure) << "\",\n";
        file << "      \"binaryBytes\": " << binaryBytes << ",\n";
        file << "      \"kernelCount\": " << report.kernels.size() << ",\n";
        file << "      \"coldMs\": {\"min\": " << cold.min << ", \"median\": " << cold.median
             << ", \"max\": " << cold.max << ", \"stddev\": " << cold.stddev << "},\n";
        file << "      \"warmMs\": {\"median\": " << warm.median << "},\n";
        file << "      \"binaryMs\": {\"median\": " << binary.median << "},\n";
        file << "      \"coldCpuMs\": {\"median\": " << cpu.median << "},\n";
        file << "      \"verification\": {\"roundTrip\": \"" << checkStateName(report.verification.binaryRoundTrip)
             << "\", \"deterministic\": \"" << checkStateName(report.verification.binaryDeterministic)
             << "\", \"stable\": \"" << checkStateName(report.verification.binaryStable)
             << "\", \"kernelNames\": \"" << checkStateName(report.verification.kernelNamesMatch)
             << "\", \"kernelAttributes\": \"" << checkStateName(report.verification.kernelAttributesMatch)
             << "\"}\n";
        file << "    }" << (i + 1 == reports.size() ? "" : ",") << "\n";
    }
    file << "  ]\n";
    file << "}\n";
    return true;
}

}  // namespace report
}  // namespace clprof
