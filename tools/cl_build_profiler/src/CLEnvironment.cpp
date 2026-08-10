//
//  CLEnvironment.cpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CLEnvironment.hpp"

#include <cstdio>
#include <cstring>

#include "Metrics.hpp"

namespace clprof {

namespace {

std::string platformString(const CLApi& api, cl_platform_id platform, cl_platform_info name) {
    size_t size = 0;
    if (CL_SUCCESS != api.clGetPlatformInfo(platform, name, 0, nullptr, &size) || 0 == size) {
        return std::string();
    }
    std::vector<char> buffer(size);
    if (CL_SUCCESS != api.clGetPlatformInfo(platform, name, size, buffer.data(), nullptr)) {
        return std::string();
    }
    return std::string(buffer.data(), strnlen(buffer.data(), size));
}

std::string deviceString(const CLApi& api, cl_device_id device, cl_device_info name) {
    size_t size = 0;
    if (CL_SUCCESS != api.clGetDeviceInfo(device, name, 0, nullptr, &size) || 0 == size) {
        return std::string();
    }
    std::vector<char> buffer(size);
    if (CL_SUCCESS != api.clGetDeviceInfo(device, name, size, buffer.data(), nullptr)) {
        return std::string();
    }
    return std::string(buffer.data(), strnlen(buffer.data(), size));
}

template <typename T>
T devicePod(const CLApi& api, cl_device_id device, cl_device_info name, T fallback) {
    T value = fallback;
    if (CL_SUCCESS != api.clGetDeviceInfo(device, name, sizeof(T), &value, nullptr)) {
        return fallback;
    }
    return value;
}

std::string describeDeviceType(cl_device_type type) {
    std::string text;
    if (type & CL_DEVICE_TYPE_CPU) {
        text += "CPU ";
    }
    if (type & CL_DEVICE_TYPE_GPU) {
        text += "GPU ";
    }
    if (type & CL_DEVICE_TYPE_ACCELERATOR) {
        text += "ACCELERATOR ";
    }
    if (type & CL_DEVICE_TYPE_DEFAULT) {
        text += "DEFAULT ";
    }
    if (text.empty()) {
        return "UNKNOWN";
    }
    text.resize(text.size() - 1);
    return text;
}

}  // namespace

bool parseDeviceType(const std::string& text, cl_device_type& type) {
    if (text == "gpu") {
        type = CL_DEVICE_TYPE_GPU;
    } else if (text == "cpu") {
        type = CL_DEVICE_TYPE_CPU;
    } else if (text == "accelerator") {
        type = CL_DEVICE_TYPE_ACCELERATOR;
    } else if (text == "all") {
        type = CL_DEVICE_TYPE_ALL;
    } else if (text == "default") {
        type = CL_DEVICE_TYPE_DEFAULT;
    } else {
        return false;
    }
    return true;
}

CLEnvironment::~CLEnvironment() {
    if (nullptr == mApi) {
        return;
    }
    if (nullptr != mQueue && nullptr != mApi->clReleaseCommandQueue) {
        mApi->clReleaseCommandQueue(mQueue);
    }
    if (nullptr != mContext) {
        mApi->clReleaseContext(mContext);
    }
    if (nullptr != mDevice && nullptr != mApi->clReleaseDevice) {
        mApi->clReleaseDevice(mDevice);
    }
}

bool CLEnvironment::initialize(const CLApi& api, int platformIndex, int deviceIndex, cl_device_type deviceType,
                               std::string& error) {
    mApi = &api;
    char message[512];

    double start       = wallMs();
    cl_uint numPlatforms = 0;
    cl_int status        = api.clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (CL_SUCCESS != status || 0 == numPlatforms) {
        snprintf(message, sizeof(message), "clGetPlatformIDs failed: %s (%d), platforms=%u",
                 clErrorName(status), status, numPlatforms);
        error = message;
        return false;
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    status                  = api.clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    mTimings.platformIdsMs  = wallMs() - start;
    if (CL_SUCCESS != status) {
        snprintf(message, sizeof(message), "clGetPlatformIDs failed: %s (%d)", clErrorName(status), status);
        error = message;
        return false;
    }
    if (platformIndex < 0 || platformIndex >= static_cast<int>(numPlatforms)) {
        snprintf(message, sizeof(message), "platform index %d out of range, %u available", platformIndex,
                 numPlatforms);
        error = message;
        return false;
    }
    mPlatform = platforms[platformIndex];

    start              = wallMs();
    cl_uint numDevices = 0;
    status             = api.clGetDeviceIDs(mPlatform, deviceType, 0, nullptr, &numDevices);
    if (CL_SUCCESS != status || 0 == numDevices) {
        snprintf(message, sizeof(message), "clGetDeviceIDs failed: %s (%d), devices=%u", clErrorName(status),
                 status, numDevices);
        error = message;
        return false;
    }
    std::vector<cl_device_id> devices(numDevices);
    status               = api.clGetDeviceIDs(mPlatform, deviceType, numDevices, devices.data(), nullptr);
    mTimings.deviceIdsMs = wallMs() - start;
    if (CL_SUCCESS != status) {
        snprintf(message, sizeof(message), "clGetDeviceIDs failed: %s (%d)", clErrorName(status), status);
        error = message;
        return false;
    }
    if (deviceIndex < 0 || deviceIndex >= static_cast<int>(numDevices)) {
        snprintf(message, sizeof(message), "device index %d out of range, %u available", deviceIndex, numDevices);
        error = message;
        return false;
    }
    mDevice = devices[deviceIndex];

    mDescription.platformCount = numPlatforms;
    mDescription.deviceCount   = numDevices;
    mDescription.platformIndex = platformIndex;
    mDescription.deviceIndex   = deviceIndex;

    start = wallMs();
    queryDescription();
    mTimings.deviceInfoMs = wallMs() - start;

    start                          = wallMs();
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(mPlatform), 0};
    cl_int contextStatus           = CL_SUCCESS;
    mContext          = api.clCreateContext(props, 1, &mDevice, nullptr, nullptr, &contextStatus);
    mTimings.contextMs = wallMs() - start;
    if (nullptr == mContext || CL_SUCCESS != contextStatus) {
        snprintf(message, sizeof(message), "clCreateContext failed: %s (%d)", clErrorName(contextStatus),
                 contextStatus);
        error = message;
        return false;
    }

    createQueue();
    return true;
}

void CLEnvironment::queryDescription() {
    const CLApi& api = *mApi;

    mDescription.platformName       = platformString(api, mPlatform, CL_PLATFORM_NAME);
    mDescription.platformVendor     = platformString(api, mPlatform, CL_PLATFORM_VENDOR);
    mDescription.platformVersion    = platformString(api, mPlatform, CL_PLATFORM_VERSION);
    mDescription.platformProfile    = platformString(api, mPlatform, CL_PLATFORM_PROFILE);
    mDescription.platformExtensions = platformString(api, mPlatform, CL_PLATFORM_EXTENSIONS);

    mDescription.deviceName       = deviceString(api, mDevice, CL_DEVICE_NAME);
    mDescription.deviceVendor     = deviceString(api, mDevice, CL_DEVICE_VENDOR);
    mDescription.deviceVersion    = deviceString(api, mDevice, CL_DEVICE_VERSION);
    mDescription.driverVersion    = deviceString(api, mDevice, CL_DRIVER_VERSION);
    mDescription.openclCVersion   = deviceString(api, mDevice, CL_DEVICE_OPENCL_C_VERSION);
    mDescription.deviceProfile    = deviceString(api, mDevice, CL_DEVICE_PROFILE);
    mDescription.deviceExtensions = deviceString(api, mDevice, CL_DEVICE_EXTENSIONS);
    mDescription.builtInKernels   = deviceString(api, mDevice, CL_DEVICE_BUILT_IN_KERNELS);

    mDescription.deviceTypeName =
        describeDeviceType(devicePod<cl_device_type>(api, mDevice, CL_DEVICE_TYPE, CL_DEVICE_TYPE_DEFAULT));
    mDescription.computeUnits = devicePod<cl_uint>(api, mDevice, CL_DEVICE_MAX_COMPUTE_UNITS, 0);
    mDescription.clockMHz     = devicePod<cl_uint>(api, mDevice, CL_DEVICE_MAX_CLOCK_FREQUENCY, 0);
    mDescription.addressBits  = devicePod<cl_uint>(api, mDevice, CL_DEVICE_ADDRESS_BITS, 0);
    mDescription.globalMem    = devicePod<cl_ulong>(api, mDevice, CL_DEVICE_GLOBAL_MEM_SIZE, 0);
    mDescription.localMem     = devicePod<cl_ulong>(api, mDevice, CL_DEVICE_LOCAL_MEM_SIZE, 0);
    mDescription.maxAlloc     = devicePod<cl_ulong>(api, mDevice, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 0);
    mDescription.globalCache  = devicePod<cl_ulong>(api, mDevice, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, 0);
    mDescription.maxWorkGroup = devicePod<size_t>(api, mDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0);
    mDescription.compilerAvailable =
        CL_TRUE == devicePod<cl_bool>(api, mDevice, CL_DEVICE_COMPILER_AVAILABLE, CL_FALSE);
    mDescription.linkerAvailable =
        CL_TRUE == devicePod<cl_bool>(api, mDevice, CL_DEVICE_LINKER_AVAILABLE, CL_FALSE);

    const cl_uint dimensions = devicePod<cl_uint>(api, mDevice, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 0);
    if (dimensions > 0) {
        mDescription.maxWorkItemSizes.resize(dimensions);
        if (CL_SUCCESS != api.clGetDeviceInfo(mDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                              sizeof(size_t) * dimensions,
                                              mDescription.maxWorkItemSizes.data(), nullptr)) {
            mDescription.maxWorkItemSizes.clear();
        }
    }
}

void CLEnvironment::createQueue() {
    const CLApi& api = *mApi;
    const double start = wallMs();
    cl_int status      = CL_SUCCESS;

    // clCreateCommandQueueWithProperties is the OpenCL 2.0 entry point; drivers that
    // only expose 1.x still need the deprecated call, so both are attempted.
    if (nullptr != api.clCreateCommandQueueWithProperties) {
        const cl_queue_properties properties[] = {0};
        mQueue = api.clCreateCommandQueueWithProperties(mContext, mDevice, properties, &status);
    }
    if (nullptr == mQueue && nullptr != api.clCreateCommandQueue) {
        mQueue = api.clCreateCommandQueue(mContext, mDevice, 0, &status);
    }
    mTimings.queueMs = (nullptr == mQueue) ? -1.0 : wallMs() - start;
}

}  // namespace clprof
