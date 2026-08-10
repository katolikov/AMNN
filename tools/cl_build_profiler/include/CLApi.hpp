//
//  CLApi.hpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CLPROF_CL_API_HPP
#define CLPROF_CL_API_HPP

#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.h"

namespace clprof {

/** Entry points without which no measurement can be taken. */
#define CLPROF_REQUIRED_SYMBOLS(V)            \
    V(clGetPlatformIDs)                       \
    V(clGetPlatformInfo)                      \
    V(clGetDeviceIDs)                         \
    V(clGetDeviceInfo)                        \
    V(clCreateContext)                        \
    V(clReleaseContext)                       \
    V(clCreateProgramWithSource)              \
    V(clCreateProgramWithBinary)              \
    V(clBuildProgram)                         \
    V(clGetProgramInfo)                       \
    V(clGetProgramBuildInfo)                  \
    V(clReleaseProgram)                       \
    V(clCreateKernel)                         \
    V(clCreateKernelsInProgram)               \
    V(clGetKernelInfo)                        \
    V(clGetKernelWorkGroupInfo)               \
    V(clReleaseKernel)

/** Entry points an older or reduced ICD may not export; the affected phase is skipped. */
#define CLPROF_OPTIONAL_SYMBOLS(V)            \
    V(clReleaseDevice)                        \
    V(clCreateCommandQueue)                   \
    V(clCreateCommandQueueWithProperties)     \
    V(clReleaseCommandQueue)                  \
    V(clCreateBuffer)                         \
    V(clReleaseMemObject)                     \
    V(clSetKernelArg)                         \
    V(clEnqueueWriteBuffer)                   \
    V(clEnqueueReadBuffer)                    \
    V(clEnqueueNDRangeKernel)                 \
    V(clFinish)                               \
    V(clCompileProgram)                       \
    V(clLinkProgram)                          \
    V(clUnloadPlatformCompiler)

/**
 * Function pointer table for the OpenCL C API, resolved from the vendor ICD at runtime.
 * Linking against libOpenCL is deliberately avoided: Android devices ship the
 * implementation under vendor specific paths, and the tool must also start on hosts
 * that have no OpenCL at all so it can report that fact instead of failing to load.
 */
class CLApi {
public:
    CLApi() = default;
    ~CLApi();
    CLApi(const CLApi&)            = delete;
    CLApi& operator=(const CLApi&) = delete;

    /**
     * Opens an OpenCL implementation and resolves the symbol table.
     * `preferredPath` is tried first; when empty the platform default list is probed.
     */
    bool load(const std::string& preferredPath, std::string& error);

    const std::string& libraryPath() const {
        return mLibraryPath;
    }

    /** Wall time spent in dlopen plus symbol resolution for the library that was accepted. */
    double loadCostMs() const {
        return mLoadCostMs;
    }

    /** Libraries that were probed before the accepted one, with the reason they were rejected. */
    const std::vector<std::string>& probeLog() const {
        return mProbeLog;
    }

    bool supportsSeparateCompileAndLink() const {
        return nullptr != clCompileProgram && nullptr != clLinkProgram;
    }

    bool supportsExecution() const {
        return nullptr != clCreateBuffer && nullptr != clSetKernelArg && nullptr != clEnqueueNDRangeKernel &&
               nullptr != clEnqueueWriteBuffer && nullptr != clEnqueueReadBuffer && nullptr != clFinish;
    }

#define CLPROF_DECLARE_SYMBOL(name) decltype(&::name) name = nullptr;
    CLPROF_REQUIRED_SYMBOLS(CLPROF_DECLARE_SYMBOL)
    CLPROF_OPTIONAL_SYMBOLS(CLPROF_DECLARE_SYMBOL)
#undef CLPROF_DECLARE_SYMBOL

private:
    bool openAndResolve(const std::string& path, std::string& missingSymbol);

    void* mHandle = nullptr;
    std::string mLibraryPath;
    double mLoadCostMs = 0.0;
    std::vector<std::string> mProbeLog;
};

/** Name of an OpenCL error code, e.g. "CL_BUILD_PROGRAM_FAILURE". */
const char* clErrorName(cl_int error);

/** Library names and paths probed when no explicit path is given. */
const std::vector<std::string>& defaultLibraryPaths();

}  // namespace clprof

#endif  // CLPROF_CL_API_HPP
