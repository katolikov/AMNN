cmake_minimum_required(VERSION 3.22.1)
project("mnnbench")

add_library(${CMAKE_PROJECT_NAME} SHARED
    bench_jni.cpp
    bench_session.cpp
)

# 16KB page size support for Android 15+
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,max-page-size=16384")

# MNN source root (relative to this CMakeLists.txt location)
# Layout: LLM_Benchmark/android_app/app/src/main/cpp/ -> 6 levels up = MNN/
set(MNN_SOURCE_ROOT "${CMAKE_SOURCE_DIR}/../../../../../..")

# Include MNN headers
include_directories("${MNN_SOURCE_ROOT}/include/")
include_directories("${MNN_SOURCE_ROOT}/transformers/llm/engine/include")
include_directories("${MNN_SOURCE_ROOT}/3rd_party/")

# Prebuilt MNN libraries from jniLibs (injected by AppBuilder)
set(LIB_PATH "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN.so")

add_library(llm SHARED IMPORTED)
set_target_properties(llm PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libllm.so")

add_library(mnn_Express SHARED IMPORTED)
set_target_properties(mnn_Express PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN_Express.so")

# Runtime dependencies of libllm.so — must be declared so the Gradle build
# packages them into the APK.  Without these the app crashes with
# UnsatisfiedLinkError at model-load time.
if(EXISTS "${LIB_PATH}/libMNNOpenCV.so")
    add_library(MNNOpenCV SHARED IMPORTED)
    set_target_properties(MNNOpenCV PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNNOpenCV.so")
endif()

if(EXISTS "${LIB_PATH}/libMNNAudio.so")
    add_library(MNNAudio SHARED IMPORTED)
    set_target_properties(MNNAudio PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNNAudio.so")
endif()

if(EXISTS "${LIB_PATH}/libMNN_CL.so")
    add_library(MNN_CL SHARED IMPORTED)
    set_target_properties(MNN_CL PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN_CL.so")
endif()

target_link_libraries(${CMAKE_PROJECT_NAME}
    android
    log
    MNN
    llm
    mnn_Express
    $<$<TARGET_EXISTS:MNNOpenCV>:MNNOpenCV>
    $<$<TARGET_EXISTS:MNNAudio>:MNNAudio>
    $<$<TARGET_EXISTS:MNN_CL>:MNN_CL>
)

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_17)
