//
//  CLApi.cpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CLApi.hpp"

#include "Metrics.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace clprof {

namespace {

void* openLibrary(const std::string& path) {
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* resolveSymbol(void* handle, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

void closeLibrary(void* handle) {
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

}  // namespace

const std::vector<std::string>& defaultLibraryPaths() {
    static const std::vector<std::string> paths = {
#if defined(__APPLE__) || defined(__MACOSX)
        "/System/Library/Frameworks/OpenCL.framework/OpenCL",
        "libOpenCL.so",
#elif defined(__OHOS__)
        "/vendor/lib64/chipsetsdk/libhvgr_v200.so",
        "/vendor/lib64/chipsetsdk/libGLES_mali.so",
        "/system/lib64/libGLES_mali.so",
        "libGLES_mali.so",
#elif defined(__ANDROID__)
        "libOpenCL.so",
        "libGLES_mali.so",
        "libmali.so",
        "libOpenCL-pixel.so",
#if defined(__aarch64__)
        "/system/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib64/egl/libGLES_mali.so",
        "/system/lib64/egl/libGLES_mali.so",
#else
        "/system/vendor/lib/libOpenCL.so",
        "/system/lib/libOpenCL.so",
        "/system/vendor/lib/egl/libGLES_mali.so",
        "/system/lib/egl/libGLES_mali.so",
#endif
#elif defined(__linux__)
        "libOpenCL.so",
        "/usr/lib/libOpenCL.so",
        "/usr/lib64/libOpenCL.so",
        "/usr/local/lib/libOpenCL.so",
        "/usr/local/lib/libpocl.so",
#elif defined(_WIN64)
        "C:/Windows/System32/OpenCL.dll",
        "C:/Windows/SysWOW64/OpenCL.dll",
#elif defined(_WIN32)
        "C:/Windows/SysWOW64/OpenCL.dll",
        "C:/Windows/System32/OpenCL.dll",
#endif
    };
    return paths;
}

CLApi::~CLApi() {
    // The handle is intentionally leaked. Several mobile GPU drivers register
    // atexit hooks and background threads inside the ICD, and unloading them while
    // those are alive crashes the process on the way out.
}

bool CLApi::openAndResolve(const std::string& path, std::string& missingSymbol) {
    void* handle = openLibrary(path);
    if (nullptr == handle) {
        missingSymbol = "cannot open";
        return false;
    }

#define CLPROF_RESOLVE_REQUIRED(name)                                                   \
    name = reinterpret_cast<decltype(&::name)>(resolveSymbol(handle, #name));           \
    if (nullptr == name) {                                                              \
        missingSymbol = #name;                                                          \
        closeLibrary(handle);                                                           \
        return false;                                                                   \
    }
    CLPROF_REQUIRED_SYMBOLS(CLPROF_RESOLVE_REQUIRED)
#undef CLPROF_RESOLVE_REQUIRED

#define CLPROF_RESOLVE_OPTIONAL(name) name = reinterpret_cast<decltype(&::name)>(resolveSymbol(handle, #name));
    CLPROF_OPTIONAL_SYMBOLS(CLPROF_RESOLVE_OPTIONAL)
#undef CLPROF_RESOLVE_OPTIONAL

    mHandle = handle;
    return true;
}

bool CLApi::load(const std::string& preferredPath, std::string& error) {
    std::vector<std::string> candidates;
    if (!preferredPath.empty()) {
        candidates.push_back(preferredPath);
    } else {
        candidates = defaultLibraryPaths();
    }

    for (const std::string& candidate : candidates) {
        const double start = wallMs();
        std::string missingSymbol;
        if (openAndResolve(candidate, missingSymbol)) {
            mLoadCostMs  = wallMs() - start;
            mLibraryPath = candidate;
            return true;
        }
        mProbeLog.push_back(candidate + " (" + missingSymbol + ")");
    }

    error = "no usable OpenCL library found";
    if (!preferredPath.empty()) {
        error += " at " + preferredPath;
    }
    return false;
}

const char* clErrorName(cl_int error) {
    switch (error) {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE:
            return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:
            return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:
            return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:
            return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:
            return "CL_INVALID_PROPERTY";
        case CL_INVALID_COMPILER_OPTIONS:
            return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:
            return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";
        default:
            return "CL_UNKNOWN_ERROR";
    }
}

}  // namespace clprof
