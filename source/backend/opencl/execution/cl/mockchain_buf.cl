#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// MockChain: a TEMPLATE op demonstrating how one custom OpType dispatches
// several OpenCL kernels sequentially (a fan-out / fan-in DAG), instead of the
// usual single kernel. The five kernels below are deliberately trivial
// elementwise arithmetic -- the value of this file is the dispatch/sync
// skeleton in MockChainBufExecution.cpp, not the math. To adapt it for a real
// op, replace each kernel body (and the host-side global/local work sizes and
// scratch-tensor shapes) with your own; the wiring stays the same.
//
// All buffers are read/written through the precision-dependent FLOAT macro, so
// the same kernels work in fp16 buffer mode (half storage) and high precision
// (float storage). Every kernel is a plain 1-D elementwise map over `size`
// elements with a bounds guard, so a single global-work-size = size (rounded up
// to the local size) drives all of them.

// func1(x) = x + 1
__kernel void mockchain_func1_buf(
    __global const FLOAT *input,
    __global FLOAT *output,
    __private const int size) {
    const int i = get_global_id(0);
    if (i >= size) {
        return;
    }
    output[i] = input[i] + (FLOAT)1;
}

// func2(x) = x * 2
__kernel void mockchain_func2_buf(
    __global const FLOAT *input,
    __global FLOAT *output,
    __private const int size) {
    const int i = get_global_id(0);
    if (i >= size) {
        return;
    }
    output[i] = input[i] * (FLOAT)2;
}

// func3(a, b) = a + b   (fan-in of the two branch results)
__kernel void mockchain_func3_buf(
    __global const FLOAT *inputA,
    __global const FLOAT *inputB,
    __global FLOAT *output,
    __private const int size) {
    const int i = get_global_id(0);
    if (i >= size) {
        return;
    }
    output[i] = inputA[i] + inputB[i];
}

// func4(x, k) = x + k   (k = per-run offset, passed as a private arg)
__kernel void mockchain_func4_buf(
    __global const FLOAT *input,
    __global FLOAT *output,
    __private const float offset,
    __private const int size) {
    const int i = get_global_id(0);
    if (i >= size) {
        return;
    }
    output[i] = input[i] + (FLOAT)offset;
}

// func5(x) = x * 3
__kernel void mockchain_func5_buf(
    __global const FLOAT *input,
    __global FLOAT *output,
    __private const int size) {
    const int i = get_global_id(0);
    if (i >= size) {
        return;
    }
    output[i] = input[i] * (FLOAT)3;
}
