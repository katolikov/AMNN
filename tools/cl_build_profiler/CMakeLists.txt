cmake_minimum_required(VERSION 3.10)
project(cl_build_profiler CXX)

# Builds standalone: nothing links against libMNN. The only pieces taken from the
# engine are the generated kernel sources, so the tool compiles exactly the OpenCL
# programs the OpenCL backend ships.
get_filename_component(MNN_ROOT_DEFAULT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(MNN_ROOT "${MNN_ROOT_DEFAULT}" CACHE PATH "Root of the MNN source tree")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CL_KERNEL_DIR "${MNN_ROOT}/source/backend/opencl/execution/cl")
file(GLOB MNN_CL_SOURCES "${CL_KERNEL_DIR}/*_mnn_cl.cpp")
if(NOT MNN_CL_SOURCES)
    message(FATAL_ERROR "No *_mnn_cl.cpp found in ${CL_KERNEL_DIR}; set -DMNN_ROOT=<path to MNN>")
endif()

set(PROFILER_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/src/BuildProfiler.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/CLApi.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/CLEnvironment.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/Metrics.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/MnnCache.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/ProgramCatalog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/Report.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/ResultStore.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/main.cpp"
)

add_executable(cl_build_profiler ${PROFILER_SOURCES} ${MNN_CL_SOURCES})

target_include_directories(cl_build_profiler PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/include"
    "${MNN_ROOT}/3rd_party/OpenCLHeaders"
    "${MNN_ROOT}/3rd_party/flatbuffers/include"
    "${MNN_ROOT}/source/backend/opencl/schema/current"
    "${CL_KERNEL_DIR}"
)

# The subgroup kernels are Intel only and MNN excludes them on every other target;
# leaving MNN_SUPPORT_INTEL_SUBGROUP undefined keeps the program set identical to
# the one an Android or macOS build of libMNN contains.
target_compile_options(cl_build_profiler PRIVATE -Wall -Wextra -Wno-unused-parameter)

find_package(Threads REQUIRED)
target_link_libraries(cl_build_profiler PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

install(TARGETS cl_build_profiler RUNTIME DESTINATION bin)
