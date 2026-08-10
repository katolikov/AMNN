//
//  BuildProfiler.hpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CLPROF_BUILD_PROFILER_HPP
#define CLPROF_BUILD_PROFILER_HPP

#include <atomic>
#include <string>
#include <vector>

#include "CLApi.hpp"
#include "CLEnvironment.hpp"
#include "ProgramCatalog.hpp"

namespace clprof {

/** Which compilation path a sample was taken on. */
enum class Phase {
    kCold,       ///< clBuildProgram on source the driver has never seen (salted)
    kCacheFill,  ///< first build of the unsalted source: fills the driver compiler cache
    kWarm,       ///< repeated build of the same unsalted source: driver cache hit
    kBinary      ///< clCreateProgramWithBinary + clBuildProgram, the MNN cache path
};

const char* phaseName(Phase phase);

/** Wall time of every OpenCL call made during one full build of one program. */
struct BuildSample {
    Phase phase       = Phase::kCold;
    int iteration     = 0;
    double createMs   = 0.0;  ///< clCreateProgramWithSource or clCreateProgramWithBinary
    double buildMs    = 0.0;  ///< clBuildProgram
    double buildCpuMs = 0.0;  ///< process CPU time consumed while clBuildProgram ran
    double statusMs   = 0.0;  ///< clGetProgramBuildInfo(CL_PROGRAM_BUILD_STATUS)
    double binaryMs   = 0.0;  ///< clGetProgramInfo(CL_PROGRAM_BINARY_SIZES, CL_PROGRAM_BINARIES)
    double kernelsMs  = 0.0;  ///< clCreateKernelsInProgram
    double kernelInfoMs = 0.0;  ///< clGetKernelInfo and clGetKernelWorkGroupInfo for every kernel
    double releaseMs  = 0.0;  ///< clReleaseKernel for every kernel plus clReleaseProgram
    size_t binaryBytes = 0;
    int kernelCount    = 0;
    bool ok            = false;

    double totalMs() const;
};

/** What a built program exposes for one kernel. */
struct KernelInfo {
    std::string name;
    cl_uint numArgs                = 0;
    size_t workGroupSize           = 0;
    size_t preferredMultiple       = 0;
    cl_ulong localMemBytes         = 0;
    cl_ulong privateMemBytes       = 0;
    size_t compileWorkGroupSize[3] = {0, 0, 0};

    bool sameAttributes(const KernelInfo& other) const;
};

/** Timings of the OpenCL 1.2 two step compile and link path, measured separately. */
struct SplitBuildSample {
    bool measured    = false;
    bool ok          = false;
    double compileMs = -1.0;
    double linkMs    = -1.0;
    std::string failure;
};

/** Outcome of one cross check performed on a built program. */
enum class CheckState { kSkipped, kPass, kFail };

const char* checkStateName(CheckState state);

/** Cross checks that a binary really is a usable, faithful copy of the built program. */
struct Verification {
    CheckState binaryRoundTrip      = CheckState::kSkipped;  ///< binary reloads and builds
    CheckState binaryDeterministic  = CheckState::kSkipped;  ///< same source twice, same bytes
    CheckState binaryStable         = CheckState::kSkipped;  ///< binary survives a reload unchanged
    CheckState kernelNamesMatch     = CheckState::kSkipped;  ///< binary exposes the same kernels
    CheckState kernelAttributesMatch = CheckState::kSkipped; ///< and the same work group limits
    std::vector<std::string> notes;

    bool anyFailed() const;
};

/** Everything measured and checked for a single program. */
struct ProgramReport {
    std::string name;
    std::string origin;
    size_t sourceBytes = 0;
    int sourceLines    = 0;
    std::string buildOptions;
    bool compiled = false;
    bool crashed  = false;  ///< the build took the driver, and the process, down with it
    std::string failure;    ///< error name when the program does not build
    std::string buildLog;   ///< CL_PROGRAM_BUILD_LOG of the failing build

    std::vector<BuildSample> samples;  ///< every pass of every phase, in execution order
    SplitBuildSample split;
    std::vector<KernelInfo> kernels;
    Verification verification;

    std::vector<double> phaseTimings(Phase phase) const;
    std::vector<BuildSample> phaseSamples(Phase phase) const;
};

/** Result of building several programs at once to expose driver side serialisation. */
struct ContentionReport {
    bool measured        = false;
    int jobs             = 0;
    int programCount     = 0;
    double wallMs        = 0.0;  ///< wall time of the whole concurrent run
    double summedBuildMs = 0.0;  ///< sum of the individual clBuildProgram times
    double serialWallMs  = 0.0;  ///< wall time of the same set built one after another
    int failures         = 0;

    /** Builds actually overlapping, 1.0 means the driver serialised everything. */
    double effectiveParallelism() const;
};

/** End to end check that a program built by this tool produces correct results. */
struct ExecutionCheck {
    bool measured   = false;
    bool ok         = false;
    double buildMs  = 0.0;
    double kernelMs = 0.0;
    double bufferMs = 0.0;
    double argsMs   = 0.0;
    double enqueueMs = 0.0;
    double finishMs = 0.0;
    double readMs   = 0.0;
    int mismatches  = 0;
    std::string failure;
};

/** Knobs that control how much work the profiler does per program. */
struct ProfilerConfig {
    int coldRepeat     = 2;
    int warmRepeat     = 2;
    int binaryRepeat   = 2;
    int warmup         = 0;
    bool salt          = true;
    bool binaryPhase   = true;
    bool verify        = true;
    bool splitCompileLink = false;
    bool verbose       = false;
};

/**
 * Runs the measurements. One instance drives a single device; all state that
 * survives between programs (the salt counter) lives here so repeated runs inside
 * one process never collide on a cache key.
 */
class BuildProfiler {
public:
    BuildProfiler(const CLApi& api, const CLEnvironment& environment, const ProfilerConfig& config);

    /** Measures every configured phase for one program. */
    ProgramReport profile(const ProgramSource& program, const std::string& options);

    /** Builds `programs` on `jobs` threads and then serially, to compare the two. */
    ContentionReport profileContention(const std::vector<ProgramSource>& programs, const std::string& options,
                                       int jobs);

    /** Compiles and runs a small built in kernel and validates its output. */
    ExecutionCheck runExecutionCheck();

    /** Wall time of clUnloadPlatformCompiler, negative when the entry point is missing. */
    double unloadCompiler();

private:
    struct BuildArtifacts {
        std::vector<unsigned char> binary;
        std::vector<KernelInfo> kernels;
    };

    bool buildFromSource(const std::string& source, const std::string& options, bool collectArtifacts,
                         BuildSample& sample, BuildArtifacts& artifacts, ProgramReport& report);
    bool buildFromBinary(const std::vector<unsigned char>& binary, const std::string& options,
                         BuildSample& sample, BuildArtifacts& artifacts, ProgramReport& report);
    bool finishProgram(cl_program program, bool collectArtifacts, BuildSample& sample,
                       BuildArtifacts& artifacts, ProgramReport& report);
    void measureSplitBuild(const ProgramSource& program, const std::string& options, ProgramReport& report);
    void verify(const BuildArtifacts& reference, const BuildArtifacts& fromBinary, bool deterministic,
                bool deterministicMeasured, Verification& verification);
    std::string readBuildLog(cl_program program);

    /** Unique per build; atomic because the contention test builds from several threads. */
    unsigned int nextSalt();

    const CLApi& mApi;
    const CLEnvironment& mEnvironment;
    ProfilerConfig mConfig;
    std::atomic<unsigned int> mSalt;
};

}  // namespace clprof

#endif  // CLPROF_BUILD_PROFILER_HPP
