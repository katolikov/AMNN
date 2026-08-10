//
//  BuildProfiler.cpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BuildProfiler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

#include "Metrics.hpp"

namespace clprof {

namespace {

/** Kernel used by the execution check: trivially verifiable, one work dimension. */
const char* kSelfTestSource =
    "__kernel void clprof_selftest(__global const float* lhs, __global const float* rhs,\n"
    "                              __global float* out, const int count) {\n"
    "    const int index = get_global_id(0);\n"
    "    if (index < count) {\n"
    "        out[index] = lhs[index] * 2.0f + rhs[index];\n"
    "    }\n"
    "}\n";

std::string formatError(const char* call, cl_int status) {
    char message[128];
    snprintf(message, sizeof(message), "%s failed: %s (%d)", call, clErrorName(status), status);
    return message;
}

}  // namespace

const char* phaseName(Phase phase) {
    switch (phase) {
        case Phase::kCold:
            return "cold";
        case Phase::kCacheFill:
            return "cache-fill";
        case Phase::kWarm:
            return "warm";
        case Phase::kBinary:
            return "binary";
    }
    return "unknown";
}

const char* checkStateName(CheckState state) {
    switch (state) {
        case CheckState::kSkipped:
            return "skipped";
        case CheckState::kPass:
            return "pass";
        case CheckState::kFail:
            return "FAIL";
    }
    return "unknown";
}

double BuildSample::totalMs() const {
    return createMs + buildMs + statusMs + binaryMs + kernelsMs + kernelInfoMs + releaseMs;
}

bool KernelInfo::sameAttributes(const KernelInfo& other) const {
    return name == other.name && numArgs == other.numArgs && workGroupSize == other.workGroupSize &&
           preferredMultiple == other.preferredMultiple && localMemBytes == other.localMemBytes &&
           privateMemBytes == other.privateMemBytes &&
           compileWorkGroupSize[0] == other.compileWorkGroupSize[0] &&
           compileWorkGroupSize[1] == other.compileWorkGroupSize[1] &&
           compileWorkGroupSize[2] == other.compileWorkGroupSize[2];
}

bool Verification::anyFailed() const {
    return CheckState::kFail == binaryRoundTrip || CheckState::kFail == binaryDeterministic ||
           CheckState::kFail == binaryStable || CheckState::kFail == kernelNamesMatch ||
           CheckState::kFail == kernelAttributesMatch;
}

double CacheRestoreReport::totalRestoreMs() const {
    double total = 0.0;
    for (const CacheRestoreSample& sample : samples) {
        if (sample.ok) {
            total += sample.restoreMs();
        }
    }
    return total;
}

int CacheRestoreReport::restored() const {
    int count = 0;
    for (const CacheRestoreSample& sample : samples) {
        count += sample.ok ? 1 : 0;
    }
    return count;
}

double ContentionReport::effectiveParallelism() const {
    if (wallMs <= 0.0) {
        return 0.0;
    }
    return summedBuildMs / wallMs;
}

std::vector<double> ProgramReport::phaseTimings(Phase phase) const {
    std::vector<double> timings;
    for (const BuildSample& sample : samples) {
        if (sample.phase == phase && sample.ok) {
            timings.push_back(sample.buildMs);
        }
    }
    return timings;
}

std::vector<BuildSample> ProgramReport::phaseSamples(Phase phase) const {
    std::vector<BuildSample> selected;
    for (const BuildSample& sample : samples) {
        if (sample.phase == phase) {
            selected.push_back(sample);
        }
    }
    return selected;
}

BuildProfiler::BuildProfiler(const CLApi& api, const CLEnvironment& environment, const ProfilerConfig& config)
    : mApi(api), mEnvironment(environment), mConfig(config), mSalt(0) {
}

unsigned int BuildProfiler::nextSalt() {
    return ++mSalt;
}

std::string BuildProfiler::readBuildLog(cl_program program) {
    size_t size = 0;
    if (CL_SUCCESS != mApi.clGetProgramBuildInfo(program, mEnvironment.device(), CL_PROGRAM_BUILD_LOG, 0, nullptr,
                                                 &size) ||
        size <= 1) {
        return std::string();
    }
    std::vector<char> buffer(size);
    if (CL_SUCCESS !=
        mApi.clGetProgramBuildInfo(program, mEnvironment.device(), CL_PROGRAM_BUILD_LOG, size, buffer.data(),
                                   nullptr)) {
        return std::string();
    }
    return std::string(buffer.data(), strnlen(buffer.data(), size));
}

bool BuildProfiler::finishProgram(cl_program program, bool collectArtifacts, BuildSample& sample,
                                  BuildArtifacts& artifacts, ProgramReport& report) {
    const cl_device_id device = mEnvironment.device();

    // Binary retrieval. The context holds exactly one device, so the binary of index 0
    // is the one that was just built.
    double start = wallMs();
    cl_uint deviceCount = 0;
    cl_int status = mApi.clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(deviceCount), &deviceCount,
                                          nullptr);
    std::vector<unsigned char> binary;
    if (CL_SUCCESS == status && deviceCount > 0) {
        std::vector<size_t> sizes(deviceCount, 0);
        status = mApi.clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * deviceCount,
                                       sizes.data(), nullptr);
        if (CL_SUCCESS == status && sizes[0] > 0) {
            std::vector<std::vector<unsigned char> > buffers(deviceCount);
            std::vector<unsigned char*> pointers(deviceCount, nullptr);
            for (cl_uint i = 0; i < deviceCount; ++i) {
                buffers[i].resize(sizes[i]);
                pointers[i] = buffers[i].data();
            }
            status = mApi.clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * deviceCount,
                                           pointers.data(), nullptr);
            if (CL_SUCCESS == status) {
                binary.swap(buffers[0]);
            }
        }
    }
    sample.binaryMs    = wallMs() - start;
    sample.binaryBytes = binary.size();

    // Kernel objects.
    start                = wallMs();
    cl_uint kernelCount  = 0;
    status               = mApi.clCreateKernelsInProgram(program, 0, nullptr, &kernelCount);
    std::vector<cl_kernel> kernels;
    if (CL_SUCCESS == status && kernelCount > 0) {
        kernels.resize(kernelCount, nullptr);
        status = mApi.clCreateKernelsInProgram(program, kernelCount, kernels.data(), nullptr);
        if (CL_SUCCESS != status) {
            kernels.clear();
        }
    }
    sample.kernelsMs   = wallMs() - start;
    sample.kernelCount = static_cast<int>(kernels.size());
    if (CL_SUCCESS != status) {
        if (report.failure.empty()) {
            report.failure = formatError("clCreateKernelsInProgram", status);
        }
        mApi.clReleaseProgram(program);
        return false;
    }
    // A program with no kernel is not an error: every kernel of, for example,
    // groupnorm_buf sits behind `#if LOCAL_SIZE > 1`. It still built, it just has
    // nothing to expose, which is worth reporting rather than hiding as a failure.

    // Per kernel introspection: this is what MNN queries right after a build.
    start = wallMs();
    std::vector<KernelInfo> infos(kernels.size());
    for (size_t i = 0; i < kernels.size(); ++i) {
        KernelInfo& info = infos[i];
        size_t nameSize  = 0;
        if (CL_SUCCESS == mApi.clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, 0, nullptr, &nameSize) &&
            nameSize > 0) {
            std::vector<char> nameBuffer(nameSize);
            if (CL_SUCCESS ==
                mApi.clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, nameSize, nameBuffer.data(), nullptr)) {
                info.name.assign(nameBuffer.data(), strnlen(nameBuffer.data(), nameSize));
            }
        }
        mApi.clGetKernelInfo(kernels[i], CL_KERNEL_NUM_ARGS, sizeof(info.numArgs), &info.numArgs, nullptr);
        mApi.clGetKernelWorkGroupInfo(kernels[i], device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(info.workGroupSize),
                                      &info.workGroupSize, nullptr);
        mApi.clGetKernelWorkGroupInfo(kernels[i], device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                      sizeof(info.preferredMultiple), &info.preferredMultiple, nullptr);
        mApi.clGetKernelWorkGroupInfo(kernels[i], device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(info.localMemBytes),
                                      &info.localMemBytes, nullptr);
        mApi.clGetKernelWorkGroupInfo(kernels[i], device, CL_KERNEL_PRIVATE_MEM_SIZE,
                                      sizeof(info.privateMemBytes), &info.privateMemBytes, nullptr);
        mApi.clGetKernelWorkGroupInfo(kernels[i], device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                      sizeof(info.compileWorkGroupSize), info.compileWorkGroupSize, nullptr);
    }
    sample.kernelInfoMs = wallMs() - start;

    start = wallMs();
    for (cl_kernel kernel : kernels) {
        mApi.clReleaseKernel(kernel);
    }
    mApi.clReleaseProgram(program);
    sample.releaseMs = wallMs() - start;

    if (collectArtifacts) {
        std::sort(infos.begin(), infos.end(),
                  [](const KernelInfo& lhs, const KernelInfo& rhs) { return lhs.name < rhs.name; });
        artifacts.binary.swap(binary);
        artifacts.kernels.swap(infos);
    }
    sample.ok = true;
    return true;
}

bool BuildProfiler::buildFromSource(const std::string& source, const std::string& options, bool collectArtifacts,
                                    BuildSample& sample, BuildArtifacts& artifacts, ProgramReport& report) {
    const char* text  = source.c_str();
    const size_t size = source.size();
    cl_int status     = CL_SUCCESS;

    double start   = wallMs();
    cl_program program = mApi.clCreateProgramWithSource(mEnvironment.context(), 1, &text, &size, &status);
    sample.createMs    = wallMs() - start;
    if (nullptr == program || CL_SUCCESS != status) {
        report.failure = formatError("clCreateProgramWithSource", status);
        return false;
    }

    const cl_device_id device = mEnvironment.device();
    const double cpuStart     = cpuMs();
    start                     = wallMs();
    status                    = mApi.clBuildProgram(program, 1, &device, options.c_str(), nullptr, nullptr);
    sample.buildMs            = wallMs() - start;
    sample.buildCpuMs         = cpuMs() - cpuStart;

    start                  = wallMs();
    cl_build_status buildStatus = CL_BUILD_NONE;
    mApi.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(buildStatus), &buildStatus,
                               nullptr);
    sample.statusMs = wallMs() - start;

    if (CL_SUCCESS != status || CL_BUILD_SUCCESS != buildStatus) {
        report.failure  = formatError("clBuildProgram", status);
        report.buildLog = readBuildLog(program);
        mApi.clReleaseProgram(program);
        return false;
    }
    return finishProgram(program, collectArtifacts, sample, artifacts, report);
}

bool BuildProfiler::buildFromBinary(const std::vector<unsigned char>& binary, const std::string& options,
                                    BuildSample& sample, BuildArtifacts& artifacts, ProgramReport& report) {
    const cl_device_id device      = mEnvironment.device();
    const unsigned char* data      = binary.data();
    const size_t size              = binary.size();
    cl_int status                  = CL_SUCCESS;
    cl_int binaryStatus            = CL_SUCCESS;

    double start = wallMs();
    cl_program program =
        mApi.clCreateProgramWithBinary(mEnvironment.context(), 1, &device, &size, &data, &binaryStatus, &status);
    sample.createMs = wallMs() - start;
    if (nullptr == program || CL_SUCCESS != status || CL_SUCCESS != binaryStatus) {
        report.failure = formatError("clCreateProgramWithBinary", CL_SUCCESS != status ? status : binaryStatus);
        if (nullptr != program) {
            mApi.clReleaseProgram(program);
        }
        return false;
    }

    const double cpuStart = cpuMs();
    start                 = wallMs();
    status                = mApi.clBuildProgram(program, 1, &device, options.c_str(), nullptr, nullptr);
    sample.buildMs        = wallMs() - start;
    sample.buildCpuMs     = cpuMs() - cpuStart;

    start                       = wallMs();
    cl_build_status buildStatus = CL_BUILD_NONE;
    mApi.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(buildStatus), &buildStatus,
                               nullptr);
    sample.statusMs = wallMs() - start;

    if (CL_SUCCESS != status || CL_BUILD_SUCCESS != buildStatus) {
        report.failure  = formatError("clBuildProgram(binary)", status);
        report.buildLog = readBuildLog(program);
        mApi.clReleaseProgram(program);
        return false;
    }
    return finishProgram(program, true, sample, artifacts, report);
}

void BuildProfiler::measureSplitBuild(const ProgramSource& program, const std::string& options,
                                      ProgramReport& report) {
    if (!mApi.supportsSeparateCompileAndLink()) {
        return;
    }
    report.split.measured = true;

    const std::string source = mConfig.salt ? saltedSource(program.source, nextSalt()) : program.source;
    const char* text         = source.c_str();
    const size_t size        = source.size();
    cl_int status            = CL_SUCCESS;

    cl_program compiled = mApi.clCreateProgramWithSource(mEnvironment.context(), 1, &text, &size, &status);
    if (nullptr == compiled || CL_SUCCESS != status) {
        report.split.failure = formatError("clCreateProgramWithSource", status);
        return;
    }

    const cl_device_id device = mEnvironment.device();
    double start              = wallMs();
    status = mApi.clCompileProgram(compiled, 1, &device, options.c_str(), 0, nullptr, nullptr, nullptr, nullptr);
    report.split.compileMs = wallMs() - start;
    if (CL_SUCCESS != status) {
        report.split.failure = formatError("clCompileProgram", status);
        mApi.clReleaseProgram(compiled);
        return;
    }

    start           = wallMs();
    cl_program linked = mApi.clLinkProgram(mEnvironment.context(), 1, &device, "", 1, &compiled, nullptr, nullptr,
                                           &status);
    report.split.linkMs = wallMs() - start;
    if (nullptr == linked || CL_SUCCESS != status) {
        report.split.failure = formatError("clLinkProgram", status);
        mApi.clReleaseProgram(compiled);
        return;
    }

    report.split.ok = true;
    mApi.clReleaseProgram(linked);
    mApi.clReleaseProgram(compiled);
}

void BuildProfiler::verify(const BuildArtifacts& reference, const BuildArtifacts& fromBinary, bool deterministic,
                           bool deterministicMeasured, Verification& verification) {
    char note[256];

    if (deterministicMeasured) {
        verification.binaryDeterministic = deterministic ? CheckState::kPass : CheckState::kFail;
        if (!deterministic) {
            verification.notes.push_back(
                "two builds of the same source with the same options produced different binaries");
        }
    }

    if (fromBinary.binary.empty() && fromBinary.kernels.empty()) {
        return;  // the binary phase did not run
    }
    verification.binaryRoundTrip = CheckState::kPass;

    verification.binaryStable = (fromBinary.binary == reference.binary) ? CheckState::kPass : CheckState::kFail;
    if (CheckState::kFail == verification.binaryStable) {
        snprintf(note, sizeof(note), "binary changed across a reload: %zu bytes in, %zu bytes out",
                 reference.binary.size(), fromBinary.binary.size());
        verification.notes.push_back(note);
    }

    std::vector<std::string> referenceNames;
    std::vector<std::string> binaryNames;
    for (const KernelInfo& info : reference.kernels) {
        referenceNames.push_back(info.name);
    }
    for (const KernelInfo& info : fromBinary.kernels) {
        binaryNames.push_back(info.name);
    }
    verification.kernelNamesMatch = (referenceNames == binaryNames) ? CheckState::kPass : CheckState::kFail;
    if (CheckState::kFail == verification.kernelNamesMatch) {
        snprintf(note, sizeof(note), "kernel set differs: %zu kernels from source, %zu from binary",
                 referenceNames.size(), binaryNames.size());
        verification.notes.push_back(note);
        return;
    }

    verification.kernelAttributesMatch = CheckState::kPass;
    for (size_t i = 0; i < reference.kernels.size(); ++i) {
        if (!reference.kernels[i].sameAttributes(fromBinary.kernels[i])) {
            verification.kernelAttributesMatch = CheckState::kFail;
            snprintf(note, sizeof(note),
                     "kernel %s: work group %zu vs %zu, preferred multiple %zu vs %zu, local mem %llu vs %llu",
                     reference.kernels[i].name.c_str(), reference.kernels[i].workGroupSize,
                     fromBinary.kernels[i].workGroupSize, reference.kernels[i].preferredMultiple,
                     fromBinary.kernels[i].preferredMultiple,
                     static_cast<unsigned long long>(reference.kernels[i].localMemBytes),
                     static_cast<unsigned long long>(fromBinary.kernels[i].localMemBytes));
            verification.notes.push_back(note);
        }
    }
}

ProgramReport BuildProfiler::profile(const ProgramSource& program, const std::string& options) {
    ProgramReport report;
    report.name         = program.name;
    report.origin       = program.origin;
    report.sourceBytes  = program.bytes();
    report.sourceLines  = program.lines();
    report.buildOptions = options;

    BuildArtifacts scratch;
    for (int i = 0; i < mConfig.warmup; ++i) {
        BuildSample sample;
        const std::string source = mConfig.salt ? saltedSource(program.source, nextSalt()) : program.source;
        if (!buildFromSource(source, options, false, sample, scratch, report)) {
            return report;
        }
    }

    for (int i = 0; i < mConfig.coldRepeat; ++i) {
        BuildSample sample;
        sample.phase             = Phase::kCold;
        sample.iteration         = i;
        const std::string source = mConfig.salt ? saltedSource(program.source, nextSalt()) : program.source;
        const bool ok            = buildFromSource(source, options, false, sample, scratch, report);
        report.samples.push_back(sample);
        if (!ok) {
            return report;
        }
    }

    // The unsalted source is what the driver would see in production; the first build
    // of it fills the driver compiler cache, every later one measures the hit.
    BuildArtifacts reference;
    {
        BuildSample sample;
        sample.phase     = Phase::kCacheFill;
        sample.iteration = 0;
        const bool ok    = buildFromSource(program.source, options, true, sample, reference, report);
        report.samples.push_back(sample);
        if (!ok) {
            return report;
        }
    }
    report.compiled = true;
    report.kernels  = reference.kernels;
    if (mConfig.keepBinary) {
        report.binary = reference.binary;
    }

    bool deterministic         = true;
    bool deterministicMeasured = false;
    for (int i = 0; i < mConfig.warmRepeat; ++i) {
        BuildSample sample;
        sample.phase     = Phase::kWarm;
        sample.iteration = i;
        // Only the first warm pass needs its binary; comparing it against the
        // cache-fill binary is the determinism check.
        const bool collect = mConfig.verify && 0 == i;
        BuildArtifacts warmArtifacts;
        const bool ok = buildFromSource(program.source, options, collect, sample, warmArtifacts, report);
        report.samples.push_back(sample);
        if (!ok) {
            return report;
        }
        if (collect) {
            deterministicMeasured = true;
            deterministic         = (warmArtifacts.binary == reference.binary);
        }
    }

    if (reference.binary.empty()) {
        report.verification.notes.push_back(
            "the driver returned no program binary, so the reload checks cannot run");
    }

    BuildArtifacts fromBinary;
    if (mConfig.binaryPhase && !reference.binary.empty()) {
        for (int i = 0; i < mConfig.binaryRepeat; ++i) {
            BuildSample sample;
            sample.phase     = Phase::kBinary;
            sample.iteration = i;
            BuildArtifacts artifacts;
            const bool ok = buildFromBinary(reference.binary, options, sample, artifacts, report);
            report.samples.push_back(sample);
            if (!ok) {
                report.verification.binaryRoundTrip = CheckState::kFail;
                report.verification.notes.push_back("binary could not be reloaded: " + report.failure);
                report.failure.clear();
                break;
            }
            if (0 == i) {
                fromBinary = artifacts;
            }
        }
    }

    if (mConfig.verify) {
        verify(reference, fromBinary, deterministic, deterministicMeasured, report.verification);
    }

    if (mConfig.splitCompileLink) {
        measureSplitBuild(program, options, report);
    }
    return report;
}

ContentionReport BuildProfiler::profileContention(const std::vector<ProgramSource>& programs,
                                                  const std::string& options, int jobs) {
    ContentionReport contention;
    if (jobs < 2 || programs.empty()) {
        return contention;
    }
    contention.measured     = true;
    contention.jobs         = jobs;
    contention.programCount = static_cast<int>(programs.size());

    std::mutex mutex;
    auto buildOne = [&](const ProgramSource& program, double& buildMs) {
        ProgramReport scratchReport;
        BuildArtifacts scratchArtifacts;
        BuildSample sample;
        const std::string source = saltedSource(program.source, nextSalt());
        const bool ok = buildFromSource(source, options, false, sample, scratchArtifacts, scratchReport);
        buildMs       = sample.buildMs;
        if (!ok) {
            std::lock_guard<std::mutex> guard(mutex);
            ++contention.failures;
        }
    };

    double serialStart = wallMs();
    for (const ProgramSource& program : programs) {
        double buildMs = 0.0;
        buildOne(program, buildMs);
    }
    contention.serialWallMs = wallMs() - serialStart;

    std::atomic<size_t> cursor(0);
    double summed = 0.0;
    const double parallelStart = wallMs();
    std::vector<std::thread> workers;
    for (int i = 0; i < jobs; ++i) {
        workers.emplace_back([&]() {
            double local = 0.0;
            for (;;) {
                const size_t index = cursor++;
                if (index >= programs.size()) {
                    break;
                }
                double buildMs = 0.0;
                buildOne(programs[index], buildMs);
                local += buildMs;
            }
            std::lock_guard<std::mutex> guard(mutex);
            summed += local;
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    contention.wallMs        = wallMs() - parallelStart;
    contention.summedBuildMs = summed;
    return contention;
}

ExecutionCheck BuildProfiler::runExecutionCheck() {
    ExecutionCheck check;
    if (nullptr == mEnvironment.queue()) {
        check.failure = "no command queue available";
        return check;
    }
    if (!mApi.supportsExecution()) {
        check.failure = "the ICD does not export the enqueue entry points";
        return check;
    }
    check.measured = true;

    const int count           = 4096;
    const cl_device_id device = mEnvironment.device();
    const cl_context context  = mEnvironment.context();
    const cl_command_queue queue = mEnvironment.queue();
    cl_int status             = CL_SUCCESS;

    const size_t sourceSize = strlen(kSelfTestSource);
    double start            = wallMs();
    cl_program program = mApi.clCreateProgramWithSource(context, 1, &kSelfTestSource, &sourceSize, &status);
    if (nullptr == program || CL_SUCCESS != status) {
        check.failure = formatError("clCreateProgramWithSource", status);
        return check;
    }
    status         = mApi.clBuildProgram(program, 1, &device, "-cl-mad-enable -w", nullptr, nullptr);
    check.buildMs  = wallMs() - start;
    if (CL_SUCCESS != status) {
        check.failure = formatError("clBuildProgram", status) + ": " + readBuildLog(program);
        mApi.clReleaseProgram(program);
        return check;
    }

    start              = wallMs();
    cl_kernel kernel   = mApi.clCreateKernel(program, "clprof_selftest", &status);
    check.kernelMs     = wallMs() - start;
    if (nullptr == kernel || CL_SUCCESS != status) {
        check.failure = formatError("clCreateKernel", status);
        mApi.clReleaseProgram(program);
        return check;
    }

    std::vector<float> lhs(count);
    std::vector<float> rhs(count);
    std::vector<float> out(count, 0.0f);
    for (int i = 0; i < count; ++i) {
        lhs[i] = static_cast<float>(i);
        rhs[i] = static_cast<float>(i) * 0.5f;
    }

    const size_t bytes = sizeof(float) * count;
    start              = wallMs();
    cl_mem lhsBuffer   = mApi.clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &status);
    cl_mem rhsBuffer   = mApi.clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &status);
    cl_mem outBuffer   = mApi.clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &status);
    if (nullptr != lhsBuffer && nullptr != rhsBuffer) {
        mApi.clEnqueueWriteBuffer(queue, lhsBuffer, CL_TRUE, 0, bytes, lhs.data(), 0, nullptr, nullptr);
        mApi.clEnqueueWriteBuffer(queue, rhsBuffer, CL_TRUE, 0, bytes, rhs.data(), 0, nullptr, nullptr);
    }
    check.bufferMs = wallMs() - start;
    if (nullptr == lhsBuffer || nullptr == rhsBuffer || nullptr == outBuffer) {
        check.failure = formatError("clCreateBuffer", status);
        mApi.clReleaseKernel(kernel);
        mApi.clReleaseProgram(program);
        return check;
    }

    start  = wallMs();
    status = mApi.clSetKernelArg(kernel, 0, sizeof(cl_mem), &lhsBuffer);
    status |= mApi.clSetKernelArg(kernel, 1, sizeof(cl_mem), &rhsBuffer);
    status |= mApi.clSetKernelArg(kernel, 2, sizeof(cl_mem), &outBuffer);
    status |= mApi.clSetKernelArg(kernel, 3, sizeof(int), &count);
    check.argsMs = wallMs() - start;

    if (CL_SUCCESS == status) {
        const size_t globalSize = static_cast<size_t>(count);
        start                   = wallMs();
        status = mApi.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
        check.enqueueMs = wallMs() - start;

        start = wallMs();
        status |= mApi.clFinish(queue);
        check.finishMs = wallMs() - start;

        start = wallMs();
        status |= mApi.clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, bytes, out.data(), 0, nullptr, nullptr);
        check.readMs = wallMs() - start;
    }

    if (CL_SUCCESS != status) {
        check.failure = formatError("kernel execution", status);
    } else {
        for (int i = 0; i < count; ++i) {
            const float expected = lhs[i] * 2.0f + rhs[i];
            if (std::fabs(out[i] - expected) > 1e-3f * std::max(1.0f, std::fabs(expected))) {
                ++check.mismatches;
            }
        }
        check.ok = (0 == check.mismatches);
        if (!check.ok) {
            char message[128];
            snprintf(message, sizeof(message), "%d of %d values are wrong", check.mismatches, count);
            check.failure = message;
        }
    }

    mApi.clReleaseMemObject(lhsBuffer);
    mApi.clReleaseMemObject(rhsBuffer);
    mApi.clReleaseMemObject(outBuffer);
    mApi.clReleaseKernel(kernel);
    mApi.clReleaseProgram(program);
    return check;
}

CacheRestoreReport BuildProfiler::restoreFromCache(const MnnCacheFile& cache) {
    CacheRestoreReport restore;
    restore.measured      = true;
    restore.path          = cache.path;
    restore.fileBytes     = cache.fileBytes;
    restore.entriesInFile = static_cast<int>(cache.entries.size());
    if (!cache.verdict.empty()) {
        restore.notes.push_back(cache.verdict);
    }

    const cl_device_id device = mEnvironment.device();
    for (const CacheEntry& entry : cache.entries) {
        if (!entry.usable()) {
            ++restore.entriesRejected;
            restore.notes.push_back(entry.program + ": " + entry.rejection);
            continue;
        }
        ++restore.entriesUsable;

        CacheRestoreSample sample;
        sample.program     = entry.program;
        sample.binaryBytes = entry.binary.size();

        const unsigned char* data = entry.binary.data();
        const size_t size         = entry.binary.size();
        cl_int status             = CL_SUCCESS;
        cl_int binaryStatus       = CL_SUCCESS;

        double start = wallMs();
        cl_program program = mApi.clCreateProgramWithBinary(mEnvironment.context(), 1, &device, &size, &data,
                                                            &binaryStatus, &status);
        sample.createMs = wallMs() - start;
        if (nullptr == program || CL_SUCCESS != status || CL_SUCCESS != binaryStatus) {
            sample.failure =
                formatError("clCreateProgramWithBinary", CL_SUCCESS != status ? status : binaryStatus);
            if (nullptr != program) {
                mApi.clReleaseProgram(program);
            }
            restore.samples.push_back(sample);
            continue;
        }

        // setCache rebuilds with the option string stored in the file, not with the
        // options this run would pick, so the same string is used here.
        start          = wallMs();
        status         = mApi.clBuildProgram(program, 1, &device, entry.buildInfo.c_str(), nullptr, nullptr);
        sample.buildMs = wallMs() - start;
        if (CL_SUCCESS != status) {
            sample.failure = formatError("clBuildProgram", status) + ": " + readBuildLog(program);
            mApi.clReleaseProgram(program);
            restore.samples.push_back(sample);
            continue;
        }

        start               = wallMs();
        cl_uint kernelCount = 0;
        std::vector<cl_kernel> kernels;
        if (CL_SUCCESS == mApi.clCreateKernelsInProgram(program, 0, nullptr, &kernelCount) && kernelCount > 0) {
            kernels.resize(kernelCount, nullptr);
            if (CL_SUCCESS != mApi.clCreateKernelsInProgram(program, kernelCount, kernels.data(), nullptr)) {
                kernels.clear();
            }
        }
        sample.kernelsMs   = wallMs() - start;
        sample.kernelCount = static_cast<int>(kernels.size());

        start = wallMs();
        for (cl_kernel kernel : kernels) {
            mApi.clReleaseKernel(kernel);
        }
        mApi.clReleaseProgram(program);
        sample.releaseMs = wallMs() - start;
        sample.ok        = true;
        restore.samples.push_back(sample);
    }
    return restore;
}

double BuildProfiler::unloadCompiler() {
    if (nullptr == mApi.clUnloadPlatformCompiler) {
        return -1.0;
    }
    const double start = wallMs();
    mApi.clUnloadPlatformCompiler(mEnvironment.platform());
    return wallMs() - start;
}

}  // namespace clprof
