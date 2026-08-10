//
//  MnnCache.hpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CLPROF_MNN_CACHE_HPP
#define CLPROF_MNN_CACHE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace clprof {

/** One compiled program stored in an MNN OpenCL cache file. */
struct CacheEntry {
    std::string program;
    std::string buildInfo;
    std::string md5;
    std::vector<uint8_t> binary;

    /** Why OpenCLRuntime::setCache would refuse this entry, empty when it accepts it. */
    std::string rejection;

    bool usable() const {
        return rejection.empty();
    }
};

/** What one device contributed to a cache file. */
struct CacheBackend {
    std::string deviceName;
    std::string driverVersion;
    int programCount     = 0;
    int tuningCount      = 0;
    int gemmCount        = 0;
    size_t binaryBytes   = 0;
    bool matchesDevice   = false;  ///< deviceName equals the device being profiled
    bool matchesDriver   = false;  ///< and the driver version does too
};

/** An MNN OpenCL cache file as seen from outside the engine. */
struct MnnCacheFile {
    std::string path;
    size_t fileBytes = 0;
    std::vector<CacheBackend> backends;

    /** Entries of the backend belonging to this device, in file order. */
    std::vector<CacheEntry> entries;

    /**
     * Empty when setCache would accept the file. Otherwise the reason MNN would
     * discard it and recompile everything, which is the failure users notice as
     * "the cache file exists but the first run is still slow".
     */
    std::string verdict;
};

/**
 * Reads `path` and works out what OpenCLRuntime::setCache would make of it on the
 * device described by `deviceName` and `driverVersion`: which backend entry applies,
 * which programs survive the md5 check against the sources compiled into this build,
 * and which are dropped.
 */
bool readMnnCache(const std::string& path, const std::string& deviceName, const std::string& driverVersion,
                  MnnCacheFile& cache, std::string& error);

/** The md5 MNN records for `program`, empty when this build has no such program. */
std::string programMd5(const std::string& programName);

/**
 * Writes a cache file in the format OpenCLRuntime::makeCache produces, so a warm start
 * can be measured on a device without first running a model. Only the program section
 * is written: autotuning and gemm parameters come from running kernels, which this
 * tool does not do.
 */
bool writeMnnCache(const std::string& path, const std::string& deviceName, const std::string& driverVersion,
                   const std::vector<CacheEntry>& entries, std::string& error);

}  // namespace clprof

#endif  // CLPROF_MNN_CACHE_HPP
