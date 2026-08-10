//
//  ProgramCatalog.hpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CLPROF_PROGRAM_CATALOG_HPP
#define CLPROF_PROGRAM_CATALOG_HPP

#include <string>
#include <vector>

namespace clprof {

/** One OpenCL program to be profiled. */
struct ProgramSource {
    std::string name;    ///< program name as MNN knows it, or the file stem
    std::string origin;  ///< "embedded" or the path the source was read from
    std::string source;

    size_t bytes() const {
        return source.size();
    }

    int lines() const;
};

/** Collects the programs to profile from the sources MNN ships or from disk. */
namespace catalog {

/**
 * Every OpenCL program compiled into libMNN, in the exact configuration the engine
 * uses: the same source text and, together with mnnBuildOptions, the same build
 * options that OpenCLRuntime::buildKernelWithCache passes to clBuildProgram.
 */
std::vector<ProgramSource> embedded();

/** Every *.cl file directly inside `directory`, sorted by name. */
bool fromDirectory(const std::string& directory, std::vector<ProgramSource>& out, std::string& error);

/** A single source file; the program name is the file stem. */
bool fromFile(const std::string& path, ProgramSource& out, std::string& error);

/** Keeps the programs whose name matches at least one glob pattern ('*' and '?' supported). */
std::vector<ProgramSource> filter(const std::vector<ProgramSource>& programs,
                                  const std::vector<std::string>& patterns);

}  // namespace catalog

/**
 * The build option string OpenCLRuntime hands to clBuildProgram for `precisionLevel`
 * (0: fp16 storage with fp32 math, 1: fp32, 2: fp16), including the input and output
 * type defines it adds for float tensors and the -cl-mad-enable -w suffix.
 * Kept in sync with source/backend/opencl/core/runtime/OpenCLRuntime.cpp.
 */
std::string mnnBuildOptions(int precisionLevel);

/**
 * Options a specific program needs on top of mnnBuildOptions before any of its
 * kernels compile, copied from the Execution that builds it. Without them the
 * program either fails to preprocess (binary and unary need OPERATOR) or every
 * kernel is guarded out and there is nothing left to measure (groupnorm_buf is
 * wrapped in `#if LOCAL_SIZE > 1`). Returns an empty string for programs that
 * build as they are.
 */
std::string programDefines(const std::string& programName);

/**
 * A copy of `source` carrying a uniquely named constant, so the driver's compiler
 * cache cannot serve a previously produced binary and every build does real work.
 */
std::string saltedSource(const std::string& source, unsigned int salt);

/** Matches `text` against a glob pattern supporting '*' and '?'. */
bool globMatch(const std::string& pattern, const std::string& text);

}  // namespace clprof

#endif  // CLPROF_PROGRAM_CATALOG_HPP
