# ─────────────────────────────────────────────────────────────
#  MNN LLM/VLM Benchmark Suite — CMakeLists.txt
#
#  This file is designed to be invoked from WITHIN the MNN
#  source tree (or by pointing MNN_ROOT at it) so that all
#  MNN library targets are available.
#
#  Typical invocation (via the Python orchestrator):
#
#    cmake -S <MNN_ROOT> -B build_android_bench          \
#      -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
#      -DANDROID_ABI=arm64-v8a                           \
#      -DANDROID_NATIVE_API_LEVEL=28                     \
#      -DANDROID_STL=c++_static                          \
#      -DCMAKE_BUILD_TYPE=Release                        \
#      -DMNN_BUILD_LLM=ON                               \
#      -DMNN_BUILD_LLM_OMNI=ON                          \
#      -DMNN_LOW_MEMORY=ON                              \
#      -DMNN_SUPPORT_TRANSFORMER_FUSE=ON                \
#      -DMNN_BUILD_BENCHMARK=OFF                        \
#      -DMNN_BUILD_TEST=OFF                             \
#      -DMNN_OPENCL=ON                                  \
#      -DMNN_ARM82=ON                                   \
#      -DMNN_USE_LOGCAT=OFF                             \
#      -DMNN_BUILD_FOR_ANDROID_COMMAND=true             \
#      -DMNN_BUILD_BENCH_SUITE=ON                       \
#      -DBENCH_SUITE_DIR=<path/to/benchmark_suite>
#
#  Then:  cmake --build build_android_bench --target llm_benchmark -j8
#
# ─────────────────────────────────────────────────────────────

# This file is meant to be included from the top-level MNN CMakeLists.txt
# via an option gate, OR sourced standalone after MNN targets are defined.

option(MNN_BUILD_BENCH_SUITE "Build the LLM/VLM benchmark binary" OFF)

if(MNN_BUILD_BENCH_SUITE)

    # Allow the orchestrator to point at the benchmark_suite directory
    if(NOT DEFINED BENCH_SUITE_DIR)
        set(BENCH_SUITE_DIR "${CMAKE_CURRENT_LIST_DIR}")
    endif()

    # ── Collect link dependencies ──
    # When MNN_SEP_BUILD=OFF, the llm OBJECT lib is linked into libMNN.so
    # via MNN_OBJECTS_TO_LINK, so MNN_DEPS (which contains MNN) is sufficient.
    # When MNN_SEP_BUILD=ON, llm is a separate static/shared lib.
    set(BENCH_DEPS ${MNN_DEPS})

    if(MNN_SEP_BUILD)
        list(APPEND BENCH_DEPS llm)
        if(MNN_BUILD_OPENCV)
            list(APPEND BENCH_DEPS MNNOpenCV)
        endif()
        if(MNN_BUILD_AUDIO)
            list(APPEND BENCH_DEPS MNNAudio)
        endif()
    endif()

    # ── Benchmark executable ──
    add_executable(llm_benchmark
        "${BENCH_SUITE_DIR}/llm_benchmark.cpp"
        "${BENCH_SUITE_DIR}/src/enum_maps.cpp"
        "${BENCH_SUITE_DIR}/src/bench_args.cpp"
        "${BENCH_SUITE_DIR}/src/round_runner.cpp"
        "${CMAKE_SOURCE_DIR}/tools/cpp/Profiler.cpp"
    )

    target_include_directories(llm_benchmark PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/transformers/llm/engine/include
        ${CMAKE_SOURCE_DIR}/tools/cv/include
        ${CMAKE_SOURCE_DIR}/tools/audio/include
        ${CMAKE_SOURCE_DIR}/tools/cpp               # Profiler.hpp
        ${BENCH_SUITE_DIR}                           # src/ headers
    )

    target_link_libraries(llm_benchmark ${BENCH_DEPS})
    # Ensure OpenCV/Audio symbols are resolved (link order matters on Android)
    if(MNN_BUILD_OPENCV AND TARGET MNNOpenCV)
        target_link_libraries(llm_benchmark MNNOpenCV)
    endif()
    if(MNN_BUILD_AUDIO AND TARGET MNNAudio)
        target_link_libraries(llm_benchmark MNNAudio)
    endif()

    # Propagate vision / audio compile definitions
    if(MNN_BUILD_OPENCV)
        target_compile_definitions(llm_benchmark PRIVATE LLM_SUPPORT_VISION)
    endif()
    if(MNN_BUILD_AUDIO)
        target_compile_definitions(llm_benchmark PRIVATE LLM_SUPPORT_AUDIO)
    endif()

    # MNN is compiled with -fno-exceptions; match that
    target_compile_definitions(llm_benchmark PRIVATE CPPHTTPLIB_NO_EXCEPTIONS)

    # ── Also build MNNConvert for on-device ONNX→MNN conversion ──
    # This is always built so the Python orchestrator can push it to the device
    # for on-device model conversion when no host MNNConvert is available.
    if(MNN_BUILD_CONVERTER)
        message(STATUS "LLM_Benchmark: MNNConvert will be built (MNN_BUILD_CONVERTER=ON)")
    else()
        message(STATUS "LLM_Benchmark: MNNConvert not built. Add -DMNN_BUILD_CONVERTER=ON for on-device conversion.")
    endif()

endif()
