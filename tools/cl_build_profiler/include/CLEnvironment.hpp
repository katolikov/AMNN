//
//  CLEnvironment.hpp
//  MNN
//
//  Created by MNN on 2026/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CLPROF_CL_ENVIRONMENT_HPP
#define CLPROF_CL_ENVIRONMENT_HPP

#include <string>
#include <vector>

#include "CLApi.hpp"

namespace clprof {

/** Everything the report needs to identify the machine a measurement was taken on. */
struct DeviceDescription {
    std::string platformName;
    std::string platformVendor;
    std::string platformVersion;
    std::string platformProfile;
    std::string platformExtensions;

    std::string deviceName;
    std::string deviceVendor;
    std::string deviceVersion;
    std::string driverVersion;
    std::string openclCVersion;
    std::string deviceProfile;
    std::string deviceExtensions;
    std::string builtInKernels;
    std::string deviceTypeName;

    cl_uint computeUnits  = 0;
    cl_uint clockMHz      = 0;
    cl_uint addressBits   = 0;
    cl_ulong globalMem    = 0;
    cl_ulong localMem     = 0;
    cl_ulong maxAlloc     = 0;
    cl_ulong globalCache  = 0;
    size_t maxWorkGroup   = 0;
    std::vector<size_t> maxWorkItemSizes;
    bool compilerAvailable = false;
    bool linkerAvailable   = false;

    /** Number of platforms and devices the ICD reported, for context on the selection. */
    cl_uint platformCount = 0;
    cl_uint deviceCount   = 0;
    int platformIndex     = 0;
    int deviceIndex       = 0;
};

/** Wall time of every one-off call made before the first program is compiled. */
struct EnvironmentTimings {
    double libraryLoadMs = 0.0;
    double platformIdsMs = 0.0;
    double deviceIdsMs   = 0.0;
    double deviceInfoMs  = 0.0;
    double contextMs     = 0.0;
    double queueMs       = -1.0;  ///< negative when no command queue could be created
};

/**
 * Owns the platform, device, context and command queue used for all measurements.
 * A single device is selected on purpose: a multi device context would make
 * clBuildProgram compile for every device and blur the numbers being collected.
 */
class CLEnvironment {
public:
    CLEnvironment() = default;
    ~CLEnvironment();
    CLEnvironment(const CLEnvironment&)            = delete;
    CLEnvironment& operator=(const CLEnvironment&) = delete;

    bool initialize(const CLApi& api, int platformIndex, int deviceIndex, cl_device_type deviceType,
                    std::string& error);

    cl_device_id device() const {
        return mDevice;
    }

    cl_context context() const {
        return mContext;
    }

    /** May be null: the queue is only needed by the execution check. */
    cl_command_queue queue() const {
        return mQueue;
    }

    cl_platform_id platform() const {
        return mPlatform;
    }

    const DeviceDescription& description() const {
        return mDescription;
    }

    const EnvironmentTimings& timings() const {
        return mTimings;
    }

private:
    void queryDescription();
    void createQueue();

    const CLApi* mApi        = nullptr;
    cl_platform_id mPlatform = nullptr;
    cl_device_id mDevice     = nullptr;
    cl_context mContext      = nullptr;
    cl_command_queue mQueue  = nullptr;
    DeviceDescription mDescription;
    EnvironmentTimings mTimings;
};

/** Parses "gpu", "cpu", "accelerator", "all" or "default" into a cl_device_type. */
bool parseDeviceType(const std::string& text, cl_device_type& type);

}  // namespace clprof

#endif  // CLPROF_CL_ENVIRONMENT_HPP
