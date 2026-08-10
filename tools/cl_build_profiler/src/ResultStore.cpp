//
//  ResultStore.cpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ResultStore.hpp"

#include <cstring>
#include <fstream>
#include <sstream>

namespace clprof {

namespace {

const char kEnvironmentRecord  = 'E';
const char kMarkerRecord       = 'M';
const char kProgramRecord      = 'P';
const char kCacheRestoreRecord = 'C';
const char kRunRecord          = 'R';

/** Appends fixed size values and length prefixed strings to a byte buffer. */
class Writer {
public:
    template <typename T>
    void pod(const T& value) {
        const char* bytes = reinterpret_cast<const char*>(&value);
        mBuffer.append(bytes, sizeof(T));
    }

    void text(const std::string& value) {
        pod<uint32_t>(static_cast<uint32_t>(value.size()));
        mBuffer.append(value);
    }

    void bytes(const void* data, size_t size) {
        mBuffer.append(static_cast<const char*>(data), size);
    }

    void texts(const std::vector<std::string>& values) {
        pod<uint32_t>(static_cast<uint32_t>(values.size()));
        for (const std::string& value : values) {
            text(value);
        }
    }

    const std::string& buffer() const {
        return mBuffer;
    }

private:
    std::string mBuffer;
};

/** Reads what Writer produced, refusing to run past the end of the payload. */
class Reader {
public:
    Reader(const char* data, size_t size) : mData(data), mSize(size) {
    }

    template <typename T>
    T pod() {
        T value = T();
        if (mOffset + sizeof(T) > mSize) {
            mOk = false;
            return value;
        }
        memcpy(&value, mData + mOffset, sizeof(T));
        mOffset += sizeof(T);
        return value;
    }

    std::string text() {
        const uint32_t size = pod<uint32_t>();
        if (!mOk || mOffset + size > mSize) {
            mOk = false;
            return std::string();
        }
        std::string value(mData + mOffset, size);
        mOffset += size;
        return value;
    }

    /** Copies `size` raw bytes; returns false and stops the reader when they are missing. */
    bool bytes(void* target, size_t size) {
        if (!mOk || mOffset + size > mSize) {
            mOk = false;
            return false;
        }
        memcpy(target, mData + mOffset, size);
        mOffset += size;
        return true;
    }

    std::vector<std::string> texts() {
        const uint32_t count = pod<uint32_t>();
        std::vector<std::string> values;
        for (uint32_t i = 0; mOk && i < count; ++i) {
            values.push_back(text());
        }
        return values;
    }

    bool ok() const {
        return mOk;
    }

private:
    const char* mData;
    size_t mSize;
    size_t mOffset = 0;
    bool mOk       = true;
};

void writeDevice(Writer& writer, const DeviceDescription& device) {
    writer.text(device.platformName);
    writer.text(device.platformVendor);
    writer.text(device.platformVersion);
    writer.text(device.platformProfile);
    writer.text(device.platformExtensions);
    writer.text(device.deviceName);
    writer.text(device.deviceVendor);
    writer.text(device.deviceVersion);
    writer.text(device.driverVersion);
    writer.text(device.openclCVersion);
    writer.text(device.deviceProfile);
    writer.text(device.deviceExtensions);
    writer.text(device.builtInKernels);
    writer.text(device.deviceTypeName);
    writer.pod(device.computeUnits);
    writer.pod(device.clockMHz);
    writer.pod(device.addressBits);
    writer.pod(device.globalMem);
    writer.pod(device.localMem);
    writer.pod(device.maxAlloc);
    writer.pod(device.globalCache);
    writer.pod(device.maxWorkGroup);
    writer.pod<uint32_t>(static_cast<uint32_t>(device.maxWorkItemSizes.size()));
    for (size_t value : device.maxWorkItemSizes) {
        writer.pod(value);
    }
    writer.pod(device.compilerAvailable);
    writer.pod(device.linkerAvailable);
    writer.pod(device.platformCount);
    writer.pod(device.deviceCount);
    writer.pod(device.platformIndex);
    writer.pod(device.deviceIndex);
}

void readDevice(Reader& reader, DeviceDescription& device) {
    device.platformName       = reader.text();
    device.platformVendor     = reader.text();
    device.platformVersion    = reader.text();
    device.platformProfile    = reader.text();
    device.platformExtensions = reader.text();
    device.deviceName         = reader.text();
    device.deviceVendor       = reader.text();
    device.deviceVersion      = reader.text();
    device.driverVersion      = reader.text();
    device.openclCVersion     = reader.text();
    device.deviceProfile      = reader.text();
    device.deviceExtensions   = reader.text();
    device.builtInKernels     = reader.text();
    device.deviceTypeName     = reader.text();
    device.computeUnits       = reader.pod<cl_uint>();
    device.clockMHz           = reader.pod<cl_uint>();
    device.addressBits        = reader.pod<cl_uint>();
    device.globalMem          = reader.pod<cl_ulong>();
    device.localMem           = reader.pod<cl_ulong>();
    device.maxAlloc           = reader.pod<cl_ulong>();
    device.globalCache        = reader.pod<cl_ulong>();
    device.maxWorkGroup       = reader.pod<size_t>();
    const uint32_t dimensions = reader.pod<uint32_t>();
    device.maxWorkItemSizes.clear();
    for (uint32_t i = 0; reader.ok() && i < dimensions; ++i) {
        device.maxWorkItemSizes.push_back(reader.pod<size_t>());
    }
    device.compilerAvailable = reader.pod<bool>();
    device.linkerAvailable   = reader.pod<bool>();
    device.platformCount     = reader.pod<cl_uint>();
    device.deviceCount       = reader.pod<cl_uint>();
    device.platformIndex     = reader.pod<int>();
    device.deviceIndex       = reader.pod<int>();
}

void writeKernel(Writer& writer, const KernelInfo& kernel) {
    writer.text(kernel.name);
    writer.pod(kernel.numArgs);
    writer.pod(kernel.workGroupSize);
    writer.pod(kernel.preferredMultiple);
    writer.pod(kernel.localMemBytes);
    writer.pod(kernel.privateMemBytes);
    for (int i = 0; i < 3; ++i) {
        writer.pod(kernel.compileWorkGroupSize[i]);
    }
}

void readKernel(Reader& reader, KernelInfo& kernel) {
    kernel.name              = reader.text();
    kernel.numArgs           = reader.pod<cl_uint>();
    kernel.workGroupSize     = reader.pod<size_t>();
    kernel.preferredMultiple = reader.pod<size_t>();
    kernel.localMemBytes     = reader.pod<cl_ulong>();
    kernel.privateMemBytes   = reader.pod<cl_ulong>();
    for (int i = 0; i < 3; ++i) {
        kernel.compileWorkGroupSize[i] = reader.pod<size_t>();
    }
}

}  // namespace

ResultStore::~ResultStore() {
    close();
}

bool ResultStore::openForAppend(const std::string& path, std::string& error) {
    close();
    mFile = fopen(path.c_str(), "ab");
    if (nullptr == mFile) {
        error = "cannot open " + path + " for writing";
        return false;
    }
    return true;
}

void ResultStore::close() {
    if (nullptr != mFile) {
        fclose(mFile);
        mFile = nullptr;
    }
}

bool ResultStore::writeRecord(char tag, const std::string& payload) {
    if (nullptr == mFile) {
        return false;
    }
    const uint32_t size = static_cast<uint32_t>(payload.size());
    if (1 != fwrite(&tag, 1, 1, mFile) || 1 != fwrite(&size, sizeof(size), 1, mFile)) {
        return false;
    }
    if (!payload.empty() && payload.size() != fwrite(payload.data(), 1, payload.size(), mFile)) {
        return false;
    }
    // Flushed immediately: a crash during the next build must not lose this record.
    return 0 == fflush(mFile);
}

bool ResultStore::writeEnvironment(const DeviceDescription& device, const EnvironmentTimings& timings,
                                   const std::string& libraryPath) {
    Writer writer;
    writeDevice(writer, device);
    writer.pod(timings.libraryLoadMs);
    writer.pod(timings.platformIdsMs);
    writer.pod(timings.deviceIdsMs);
    writer.pod(timings.deviceInfoMs);
    writer.pod(timings.contextMs);
    writer.pod(timings.queueMs);
    writer.text(libraryPath);
    return writeRecord(kEnvironmentRecord, writer.buffer());
}

bool ResultStore::writeMarker(const std::string& programName) {
    Writer writer;
    writer.text(programName);
    return writeRecord(kMarkerRecord, writer.buffer());
}

bool ResultStore::writeProgram(const ProgramReport& report) {
    Writer writer;
    writer.text(report.name);
    writer.text(report.origin);
    writer.pod(report.sourceBytes);
    writer.pod(report.sourceLines);
    writer.text(report.buildOptions);
    writer.pod(report.compiled);
    writer.pod(report.crashed);
    writer.text(report.failure);
    writer.text(report.buildLog);

    writer.pod<uint32_t>(static_cast<uint32_t>(report.samples.size()));
    for (const BuildSample& sample : report.samples) {
        writer.pod(sample);
    }

    writer.pod(report.split.measured);
    writer.pod(report.split.ok);
    writer.pod(report.split.compileMs);
    writer.pod(report.split.linkMs);
    writer.text(report.split.failure);

    writer.pod<uint32_t>(static_cast<uint32_t>(report.kernels.size()));
    for (const KernelInfo& kernel : report.kernels) {
        writeKernel(writer, kernel);
    }

    writer.pod(report.verification.binaryRoundTrip);
    writer.pod(report.verification.binaryDeterministic);
    writer.pod(report.verification.binaryStable);
    writer.pod(report.verification.kernelNamesMatch);
    writer.pod(report.verification.kernelAttributesMatch);
    writer.texts(report.verification.notes);

    writer.pod<uint32_t>(static_cast<uint32_t>(report.binary.size()));
    if (!report.binary.empty()) {
        writer.bytes(report.binary.data(), report.binary.size());
    }
    return writeRecord(kProgramRecord, writer.buffer());
}

bool ResultStore::writeCacheRestore(const CacheRestoreReport& restore) {
    Writer writer;
    writer.pod(restore.measured);
    writer.text(restore.path);
    writer.pod(restore.fileBytes);
    writer.pod(restore.entriesInFile);
    writer.pod(restore.entriesUsable);
    writer.pod(restore.entriesRejected);
    writer.pod<uint32_t>(static_cast<uint32_t>(restore.samples.size()));
    for (const CacheRestoreSample& sample : restore.samples) {
        writer.text(sample.program);
        writer.pod(sample.createMs);
        writer.pod(sample.buildMs);
        writer.pod(sample.kernelsMs);
        writer.pod(sample.releaseMs);
        writer.pod(sample.binaryBytes);
        writer.pod(sample.kernelCount);
        writer.pod(sample.ok);
        writer.text(sample.failure);
    }
    writer.texts(restore.notes);
    return writeRecord(kCacheRestoreRecord, writer.buffer());
}

bool ResultStore::writeRun(const ExecutionCheck& execution, const ContentionReport& contention,
                           double unloadCompilerMs, long peakRssKb) {
    Writer writer;
    writer.pod(execution.measured);
    writer.pod(execution.ok);
    writer.pod(execution.buildMs);
    writer.pod(execution.kernelMs);
    writer.pod(execution.bufferMs);
    writer.pod(execution.argsMs);
    writer.pod(execution.enqueueMs);
    writer.pod(execution.finishMs);
    writer.pod(execution.readMs);
    writer.pod(execution.mismatches);
    writer.text(execution.failure);

    writer.pod(contention.measured);
    writer.pod(contention.jobs);
    writer.pod(contention.programCount);
    writer.pod(contention.wallMs);
    writer.pod(contention.summedBuildMs);
    writer.pod(contention.serialWallMs);
    writer.pod(contention.failures);

    writer.pod(unloadCompilerMs);
    writer.pod<int64_t>(peakRssKb);
    return writeRecord(kRunRecord, writer.buffer());
}

bool ResultStore::read(const std::string& path, RunOutcome& outcome, std::string& error) {
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file.is_open()) {
        error = "cannot read " + path;
        return false;
    }
    std::ostringstream content;
    content << file.rdbuf();
    const std::string blob = content.str();

    size_t offset = 0;
    while (offset + 5 <= blob.size()) {
        const char tag = blob[offset];
        uint32_t size  = 0;
        memcpy(&size, blob.data() + offset + 1, sizeof(size));
        if (offset + 5 + size > blob.size()) {
            break;  // truncated by a crash: everything from here on is unusable
        }
        Reader reader(blob.data() + offset + 5, size);
        offset += 5 + size;

        if (kEnvironmentRecord == tag) {
            readDevice(reader, outcome.device);
            outcome.timings.libraryLoadMs = reader.pod<double>();
            outcome.timings.platformIdsMs = reader.pod<double>();
            outcome.timings.deviceIdsMs   = reader.pod<double>();
            outcome.timings.deviceInfoMs  = reader.pod<double>();
            outcome.timings.contextMs     = reader.pod<double>();
            outcome.timings.queueMs       = reader.pod<double>();
            outcome.libraryPath           = reader.text();
        } else if (kMarkerRecord == tag) {
            outcome.inFlight = reader.text();
        } else if (kProgramRecord == tag) {
            ProgramReport report;
            report.name         = reader.text();
            report.origin       = reader.text();
            report.sourceBytes  = reader.pod<size_t>();
            report.sourceLines  = reader.pod<int>();
            report.buildOptions = reader.text();
            report.compiled     = reader.pod<bool>();
            report.crashed      = reader.pod<bool>();
            report.failure      = reader.text();
            report.buildLog     = reader.text();

            const uint32_t sampleCount = reader.pod<uint32_t>();
            for (uint32_t i = 0; reader.ok() && i < sampleCount; ++i) {
                report.samples.push_back(reader.pod<BuildSample>());
            }

            report.split.measured  = reader.pod<bool>();
            report.split.ok        = reader.pod<bool>();
            report.split.compileMs = reader.pod<double>();
            report.split.linkMs    = reader.pod<double>();
            report.split.failure   = reader.text();

            const uint32_t kernelCount = reader.pod<uint32_t>();
            for (uint32_t i = 0; reader.ok() && i < kernelCount; ++i) {
                KernelInfo kernel;
                readKernel(reader, kernel);
                report.kernels.push_back(kernel);
            }

            report.verification.binaryRoundTrip       = reader.pod<CheckState>();
            report.verification.binaryDeterministic   = reader.pod<CheckState>();
            report.verification.binaryStable          = reader.pod<CheckState>();
            report.verification.kernelNamesMatch      = reader.pod<CheckState>();
            report.verification.kernelAttributesMatch = reader.pod<CheckState>();
            report.verification.notes                 = reader.texts();

            const uint32_t binarySize = reader.pod<uint32_t>();
            if (reader.ok() && binarySize > 0) {
                report.binary.resize(binarySize);
                reader.bytes(report.binary.data(), binarySize);
            }

            if (reader.ok()) {
                outcome.reports.push_back(report);
                outcome.inFlight.clear();
            }
        } else if (kCacheRestoreRecord == tag) {
            CacheRestoreReport restore;
            restore.measured        = reader.pod<bool>();
            restore.path            = reader.text();
            restore.fileBytes       = reader.pod<size_t>();
            restore.entriesInFile   = reader.pod<int>();
            restore.entriesUsable   = reader.pod<int>();
            restore.entriesRejected = reader.pod<int>();

            const uint32_t sampleCount = reader.pod<uint32_t>();
            for (uint32_t i = 0; reader.ok() && i < sampleCount; ++i) {
                CacheRestoreSample sample;
                sample.program     = reader.text();
                sample.createMs    = reader.pod<double>();
                sample.buildMs     = reader.pod<double>();
                sample.kernelsMs   = reader.pod<double>();
                sample.releaseMs   = reader.pod<double>();
                sample.binaryBytes = reader.pod<size_t>();
                sample.kernelCount = reader.pod<int>();
                sample.ok          = reader.pod<bool>();
                sample.failure     = reader.text();
                restore.samples.push_back(sample);
            }
            restore.notes = reader.texts();
            if (reader.ok()) {
                outcome.cacheRestore = restore;
            }
        } else if (kRunRecord == tag) {
            ExecutionCheck execution;
            execution.measured   = reader.pod<bool>();
            execution.ok         = reader.pod<bool>();
            execution.buildMs    = reader.pod<double>();
            execution.kernelMs   = reader.pod<double>();
            execution.bufferMs   = reader.pod<double>();
            execution.argsMs     = reader.pod<double>();
            execution.enqueueMs  = reader.pod<double>();
            execution.finishMs   = reader.pod<double>();
            execution.readMs     = reader.pod<double>();
            execution.mismatches = reader.pod<int>();
            execution.failure    = reader.text();

            ContentionReport contention;
            contention.measured      = reader.pod<bool>();
            contention.jobs          = reader.pod<int>();
            contention.programCount  = reader.pod<int>();
            contention.wallMs        = reader.pod<double>();
            contention.summedBuildMs = reader.pod<double>();
            contention.serialWallMs  = reader.pod<double>();
            contention.failures      = reader.pod<int>();

            const double unloadMs = reader.pod<double>();
            const int64_t rss     = reader.pod<int64_t>();
            if (reader.ok()) {
                outcome.execution        = execution;
                outcome.contention       = contention;
                outcome.unloadCompilerMs = unloadMs;
                outcome.peakRssKb        = static_cast<long>(rss);
            }
        }
    }
    return true;
}

}  // namespace clprof
