//
//  MnnCache.cpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MnnCache.hpp"

#include <fstream>
#include <map>
#include <memory>
#include <sstream>

#include "CLCache_generated.h"
#include "opencl_source_map.hpp"

namespace clprof {

namespace {

/**
 * The md5 the engine expects for `program`, taken from the table generated next to the
 * kernel sources. setCache compares it with the one in the file and drops the entry on
 * a mismatch, which is how a cache written by a different MNN build is rejected.
 */
bool expectedMd5(const std::string& program, std::string& md5) {
    // The md5 table is generated outside namespace MNN, unlike the source table.
    const auto iterator = ::OpenCLProgramMd5Map.find(program);
    if (iterator == ::OpenCLProgramMd5Map.end()) {
        return false;
    }
    md5 = iterator->second;
    return true;
}

}  // namespace

std::string programMd5(const std::string& programName) {
    std::string md5;
    expectedMd5(programName, md5);
    return md5;
}

bool readMnnCache(const std::string& path, const std::string& deviceName, const std::string& driverVersion,
                  MnnCacheFile& cache, std::string& error) {
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file.is_open()) {
        error = "cannot read " + path;
        return false;
    }
    std::ostringstream content;
    content << file.rdbuf();
    const std::string blob = content.str();

    cache.path      = path;
    cache.fileBytes = blob.size();
    if (blob.empty()) {
        error = path + " is empty";
        return false;
    }

    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
    if (!CLCache::VerifyCacheBuffer(verifier)) {
        error = path + " is not a valid MNN OpenCL cache file";
        return false;
    }

    const CLCache::Cache* root = CLCache::GetCache(blob.data());
    if (nullptr == root->backends() || 0 == root->backends()->size()) {
        cache.verdict = "the file holds no backend entry, MNN would recompile everything";
        return true;
    }

    const CLCache::BackendInfo* mine = nullptr;
    for (uint32_t i = 0; i < root->backends()->size(); ++i) {
        const CLCache::BackendInfo* backend = root->backends()->GetAs<CLCache::BackendInfo>(i);

        CacheBackend summary;
        summary.deviceName    = nullptr == backend->deviceName() ? std::string() : backend->deviceName()->str();
        summary.driverVersion = nullptr == backend->driverVersion() ? std::string() : backend->driverVersion()->str();
        summary.programCount  = nullptr == backend->programs() ? 0 : backend->programs()->size();
        summary.tuningCount   = nullptr == backend->tunings() ? 0 : backend->tunings()->size();
        summary.gemmCount     = nullptr == backend->gemm() ? 0 : backend->gemm()->size();
        summary.matchesDevice = summary.deviceName == deviceName;
        summary.matchesDriver = summary.driverVersion == driverVersion;
        if (nullptr != backend->programs()) {
            for (uint32_t p = 0; p < backend->programs()->size(); ++p) {
                const CLCache::Shader* shader = backend->programs()->GetAs<CLCache::Shader>(p);
                summary.binaryBytes += (nullptr == shader->buffer()) ? 0 : shader->buffer()->size();
            }
        }
        cache.backends.push_back(summary);

        if (summary.matchesDevice && nullptr == mine) {
            mine = backend;
        }
    }

    if (nullptr == mine) {
        cache.verdict = "no entry for this device, MNN would recompile everything";
        return true;
    }

    const std::string cachedDriver = nullptr == mine->driverVersion() ? std::string() : mine->driverVersion()->str();
    if (cachedDriver != driverVersion) {
        cache.verdict = "the entry was written by driver '" + cachedDriver + "', this device runs '" +
                        driverVersion + "', so MNN would drop every program";
        return true;
    }
    if (nullptr == mine->programs() || 0 == mine->programs()->size()) {
        cache.verdict = "the entry for this device holds no program, only tuning data";
        return true;
    }

    for (uint32_t i = 0; i < mine->programs()->size(); ++i) {
        const CLCache::Shader* shader = mine->programs()->GetAs<CLCache::Shader>(i);
        CacheEntry entry;
        entry.program   = nullptr == shader->program() ? std::string() : shader->program()->str();
        entry.buildInfo = nullptr == shader->buildInfo() ? std::string() : shader->buildInfo()->str();
        entry.md5       = nullptr == shader->md5() ? std::string() : shader->md5()->str();
        if (nullptr != shader->buffer()) {
            entry.binary.assign(shader->buffer()->begin(), shader->buffer()->end());
        }

        std::string expected;
        if (entry.program.empty() || entry.binary.empty()) {
            entry.rejection = "incomplete entry";
        } else if (!expectedMd5(entry.program, expected)) {
            entry.rejection = "no such program in this build";
        } else if (expected != entry.md5) {
            entry.rejection = "md5 differs from the source in this build";
        }
        cache.entries.push_back(entry);
    }
    return true;
}

bool writeMnnCache(const std::string& path, const std::string& deviceName, const std::string& driverVersion,
                   const std::vector<CacheEntry>& entries, std::string& error) {
    std::unique_ptr<CLCache::CacheT> cache(new CLCache::CacheT);
    std::unique_ptr<CLCache::BackendInfoT> backend(new CLCache::BackendInfoT);
    backend->deviceName    = deviceName;
    backend->driverVersion = driverVersion;

    for (const CacheEntry& entry : entries) {
        if (entry.binary.empty()) {
            continue;
        }
        std::unique_ptr<CLCache::ShaderT> shader(new CLCache::ShaderT);
        shader->program   = entry.program;
        shader->buildInfo = entry.buildInfo;
        shader->md5       = entry.md5;
        shader->buffer.assign(entry.binary.begin(), entry.binary.end());
        backend->programs.emplace_back(std::move(shader));
    }
    cache->backends.emplace_back(std::move(backend));

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CLCache::Cache::Pack(builder, cache.get()));

    std::ofstream file(path.c_str(), std::ios::binary);
    if (!file.is_open()) {
        error = "cannot write " + path;
        return false;
    }
    file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    if (!file.good()) {
        error = "writing " + path + " failed";
        return false;
    }
    return true;
}

}  // namespace clprof
