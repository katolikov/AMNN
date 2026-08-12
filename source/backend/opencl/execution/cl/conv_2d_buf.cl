#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

#ifdef CONV_LOCAL_SIZE
__kernel
void conv_2d_1x1_local(__private const int out_w_blocks,
                          __global const FLOAT *input,
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          __global FLOAT *output,
                          __private const int in_c_block,
                          __private const int batch,
                          __private const int out_h,
                          __private const int out_w,
                          __private const int out_c_block,
                          __private const int out_c_pack
                          #ifdef PRELU
                          ,__global const FLOAT *slope_ptr
                          #endif
) {

    const int lid = get_local_id(0);
    const int out_c_w_idx = get_global_id(1); //c/4 w
    const int out_b_h_idx  = get_global_id(2); //b h
    
    COMPUTE_FLOAT4 local sum_mnn[CONV_LOCAL_SIZE];
    
    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h; // equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h; // equal to in_h_idx

    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias_ptr));
    COMPUTE_FLOAT4 out0 = (COMPUTE_FLOAT4)0;

    int offset = out_c_idx*4;
    int inp_offset = ((out_b_idx*out_h + out_h_idx)* out_w + out_w_idx) << 2;
    
    const int inp_add = batch*out_h*out_w*4;
    for (ushort in_channel_block_idx = lid; in_channel_block_idx < in_c_block; in_channel_block_idx+=CONV_LOCAL_SIZE) {
        
        int offset = mad24(in_channel_block_idx*4, out_c_pack, out_c_idx*4);

        COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset+in_channel_block_idx*inp_add));
        COMPUTE_FLOAT4 weights0 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset));
        COMPUTE_FLOAT4 weights1 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack));
        COMPUTE_FLOAT4 weights2 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack));
        COMPUTE_FLOAT4 weights3 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack + out_c_pack));

        out0 = mad(in0.x, weights0, out0);
        out0 = mad(in0.y, weights1, out0);
        out0 = mad(in0.z, weights2, out0);
        out0 = mad(in0.w, weights3, out0);
    }
    
    sum_mnn[lid] = out0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = CONV_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    out0 = sum_mnn[0] + bias0;
    if(lid == 0){
#ifdef RELU
        out0 = fmax(out0, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
        out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
        COMPUTE_FLOAT4 slope_in = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
        out0 = select(out0 * slope_in, out0, out0 >= 0);
#endif

        const int out_offset = (((out_b_idx + out_c_idx*batch)*out_h + out_h_idx)* out_w + out_w_idx)*4;
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
}
#endif

__kernel
void conv_2d_1x1_c4h1w4(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks,
                          __global const FLOAT *input,
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          __global FLOAT *output,
                          __private const int in_c_block,
                          __private const int out_h,
                          __private const int out_w,
                          __private const int out_b,
                          __private const int out_c_block,
                          __private const int out_c_pack
                          #ifdef PRELU
                          ,__global const FLOAT *slope_ptr
                          #endif
) {

    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h; // equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h; // equal to in_h_idx

    const int out_w4_idx = mul24(out_w_idx, 4);
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias_ptr));
    COMPUTE_FLOAT4 out1 = out0;
    COMPUTE_FLOAT4 out2 = out0;
    COMPUTE_FLOAT4 out3 = out0;

    const int intput_width_idx0 = out_w4_idx;
    int inp_offset = ((out_b_idx * out_h + out_h_idx)* out_w + intput_width_idx0) << 2;
    int offset = out_c_idx*4;
    const int inp_add = out_b*out_h*out_w*4;
    for (ushort in_channel_block_idx = 0; in_channel_block_idx < in_c_block; ++in_channel_block_idx) {
        

        COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset));
        COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4(vload4(1, input+inp_offset));
        COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4(vload4(2, input+inp_offset));
        COMPUTE_FLOAT4 in3 = CONVERT_COMPUTE_FLOAT4(vload4(3, input+inp_offset));
        COMPUTE_FLOAT4 weights0 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset));
        COMPUTE_FLOAT4 weights1 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack));
        COMPUTE_FLOAT4 weights2 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack));
        COMPUTE_FLOAT4 weights3 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack + out_c_pack));

        out0 = mad(in0.x, weights0, out0);
        out0 = mad(in0.y, weights1, out0);
        out0 = mad(in0.z, weights2, out0);
        out0 = mad(in0.w, weights3, out0);
        
        out1 = mad(in1.x, weights0, out1);
        out1 = mad(in1.y, weights1, out1);
        out1 = mad(in1.z, weights2, out1);
        out1 = mad(in1.w, weights3, out1);
        
        out2 = mad(in2.x, weights0, out2);
        out2 = mad(in2.y, weights1, out2);
        out2 = mad(in2.z, weights2, out2);
        out2 = mad(in2.w, weights3, out2);
        
        out3 = mad(in3.x, weights0, out3);
        out3 = mad(in3.y, weights1, out3);
        out3 = mad(in3.z, weights2, out3);
        out3 = mad(in3.w, weights3, out3);
        
        offset += 4 * out_c_pack;
        inp_offset += inp_add;
    }

#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
    out0 = select(out0 * slope_in, out0, out0 >= 0);
    out1 = select(out1 * slope_in, out1, out1 >= 0);
    out2 = select(out2 * slope_in, out2, out2 >= 0);
    out3 = select(out3 * slope_in, out3, out3 >= 0);
#endif

    const int out_offset = (((out_b_idx + out_c_idx * out_b)*out_h + out_h_idx)* out_w + out_w4_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_w - out_w4_idx;
    if (remain >= 4) {
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, output+out_offset);
    } else if (remain == 3) {
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2, output+out_offset);
    } else if (remain == 2) {
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
    } else if (remain == 1) {
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
#else
    vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, output+out_offset);
#endif
}


__kernel
void conv_2d_1x1_c8h1w4(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks,
                          __global const FLOAT *input,
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          __global FLOAT *output,
                          __private const int in_c_block,
                          __private const int out_h,
                          __private const int out_w,
                          __private const int out_b,
                          __private const int out_c_block,
                          __private const int out_c_pack
                          #ifdef PRELU
                          ,__global const FLOAT *slope_ptr
                          #endif
) {

    const int out_c_w_idx = get_global_id(0); //c/8 w/4
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx_0 = (out_c_w_idx / out_w_blocks) << 1;
    const int out_c_idx_1 = out_c_idx_0 + 1;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h;//equal to in_h_idx

    const int out_w4_idx = mul24(out_w_idx, 4);
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, bias_ptr));
    COMPUTE_FLOAT4 out1 = out0;
    COMPUTE_FLOAT4 out2 = out0;
    COMPUTE_FLOAT4 out3 = out0;
    
    #ifdef CHANNEL_BOUNDARY_PROTECT
    COMPUTE_FLOAT4 out4 = out_c_idx_1 >= out_c_block ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias_ptr));
    COMPUTE_FLOAT4 out5 = out4;
    COMPUTE_FLOAT4 out6 = out4;
    COMPUTE_FLOAT4 out7 = out4;
    #else
    COMPUTE_FLOAT4 out4 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias_ptr));
    COMPUTE_FLOAT4 out5 = out4;
    COMPUTE_FLOAT4 out6 = out4;
    COMPUTE_FLOAT4 out7 = out4;
    #endif

    const int intput_width_idx0 = out_w4_idx;
    int inp_offset = ((out_b_idx * out_h + out_h_idx)* out_w + intput_width_idx0)<<2;
    int offset = out_c_idx_0*4;
    const int inp_add = out_b*out_h*out_w*4;

    for (int in_channel_block_idx = 0; in_channel_block_idx < in_c_block; ++in_channel_block_idx) {

        
        COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset));
        COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4(vload4(1, input+inp_offset));
        COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4(vload4(2, input+inp_offset));
        COMPUTE_FLOAT4 in3 = CONVERT_COMPUTE_FLOAT4(vload4(3, input+inp_offset));
        
        // output_channel at least pack to 8, no need boundry protect
        COMPUTE_FLOAT4 weights0 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset));
        COMPUTE_FLOAT4 weights1 = CONVERT_COMPUTE_FLOAT4(vload4(1, kernel_ptr + offset));
        COMPUTE_FLOAT4 weights2 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack));
        COMPUTE_FLOAT4 weights3 = CONVERT_COMPUTE_FLOAT4(vload4(1, kernel_ptr + offset + out_c_pack));
        COMPUTE_FLOAT4 weights4 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack));
        COMPUTE_FLOAT4 weights5 = CONVERT_COMPUTE_FLOAT4(vload4(1, kernel_ptr + offset + out_c_pack + out_c_pack));
        COMPUTE_FLOAT4 weights6 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack + out_c_pack));
        COMPUTE_FLOAT4 weights7 = CONVERT_COMPUTE_FLOAT4(vload4(1, kernel_ptr + offset + out_c_pack + out_c_pack + out_c_pack));

        out0 = mad(in0.x, weights0, out0);
        out0 = mad(in0.y, weights2, out0);
        out0 = mad(in0.z, weights4, out0);
        out0 = mad(in0.w, weights6, out0);
        
        out1 = mad(in1.x, weights0, out1);
        out1 = mad(in1.y, weights2, out1);
        out1 = mad(in1.z, weights4, out1);
        out1 = mad(in1.w, weights6, out1);
        
        out2 = mad(in2.x, weights0, out2);
        out2 = mad(in2.y, weights2, out2);
        out2 = mad(in2.z, weights4, out2);
        out2 = mad(in2.w, weights6, out2);
        
        out3 = mad(in3.x, weights0, out3);
        out3 = mad(in3.y, weights2, out3);
        out3 = mad(in3.z, weights4, out3);
        out3 = mad(in3.w, weights6, out3);
        
        out4 = mad(in0.x, weights1, out4);
        out4 = mad(in0.y, weights3, out4);
        out4 = mad(in0.z, weights5, out4);
        out4 = mad(in0.w, weights7, out4);
        
        out5 = mad(in1.x, weights1, out5);
        out5 = mad(in1.y, weights3, out5);
        out5 = mad(in1.z, weights5, out5);
        out5 = mad(in1.w, weights7, out5);
        
        out6 = mad(in2.x, weights1, out6);
        out6 = mad(in2.y, weights3, out6);
        out6 = mad(in2.z, weights5, out6);
        out6 = mad(in2.w, weights7, out6);
        
        out7 = mad(in3.x, weights1, out7);
        out7 = mad(in3.y, weights3, out7);
        out7 = mad(in3.z, weights5, out7);
        out7 = mad(in3.w, weights7, out7);
        
        offset += 4 * out_c_pack;
        inp_offset += inp_add;
    }

#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
    
    out4 = fmax(out4, (COMPUTE_FLOAT4)0);
    out5 = fmax(out5, (COMPUTE_FLOAT4)0);
    out6 = fmax(out6, (COMPUTE_FLOAT4)0);
    out7 = fmax(out7, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    
    out4 = clamp(out4, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out5 = clamp(out5, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out6 = clamp(out6, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out7 = clamp(out7, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, slope_ptr));
    COMPUTE_FLOAT4 slope_in1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, slope_ptr));
    out0 = select(out0 * slope_in0, out0, out0 >= 0);
    out1 = select(out1 * slope_in0, out1, out1 >= 0);
    out2 = select(out2 * slope_in0, out2, out2 >= 0);
    out3 = select(out3 * slope_in0, out3, out3 >= 0);
    out4 = select(out4 * slope_in1, out4, out4 >= 0);
    out5 = select(out5 * slope_in1, out5, out5 >= 0);
    out6 = select(out6 * slope_in1, out6, out6 >= 0);
    out7 = select(out7 * slope_in1, out7, out7 >= 0);
#endif

    const int out_offset = (((out_b_idx + out_c_idx_0*out_b)*out_h + out_h_idx)* out_w + out_w4_idx)*4;

    __global FLOAT * _tempoutput = output + out_offset;
    __global FLOAT * _tempoutput1 = _tempoutput + 4*out_h*out_w*out_b;

#ifdef BLOCK_LEAVE
    const int remain = out_w - out_w4_idx;
    if (remain >= 4) {
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, _tempoutput);
    } else if (remain == 3) {
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, _tempoutput);
        vstore4(CONVERT_FLOAT4(out2), 2, _tempoutput);
    } else if (remain == 2) {
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, _tempoutput);
    } else if (remain == 1) {
        vstore4(CONVERT_FLOAT4(out0), 0, _tempoutput);
    }
#ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_block) {
        return;
    }
#endif
    if (remain >= 4) {
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out4, out5, out6, out7)), 0, _tempoutput1);
    } else if (remain == 3) {
        vstore8(CONVERT_FLOAT8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4, out5))), 0, _tempoutput1);
        vstore4(CONVERT_FLOAT4(out6), 2, _tempoutput1);
    } else if (remain == 2) {
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4, out5)), 0, _tempoutput1);
    } else if (remain == 1) {
        vstore4(CONVERT_FLOAT4(out4), 0, _tempoutput1);
    }
#else
    vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, _tempoutput);
#ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_block) {
        return;
    }
#endif
    vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out4, out5, out6, out7)), 0, _tempoutput1);
#endif
}


__kernel
void conv_2d_1x1_c8h1w2(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks,
                          __global const FLOAT *input,
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          __global FLOAT *output,
                          __private const int in_c_block,
                          __private const int out_h,
                          __private const int out_w,
                          __private const int out_b,
                          __private const int out_c_block,
                          __private const int out_c_pack
                          #ifdef PRELU
                          ,__global const FLOAT *slope_ptr
                          #endif
) {

    const int out_c_w_idx = get_global_id(0); //c/8 w/4
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx_0 = (out_c_w_idx / out_w_blocks) << 1;
    const int out_c_idx_1 = out_c_idx_0 + 1;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h;//equal to in_h_idx
    
    const int out_w2_idx = mul24(out_w_idx, 2);
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, bias_ptr));
    COMPUTE_FLOAT4 out1 = out0;
    
    #ifdef CHANNEL_BOUNDARY_PROTECT
    COMPUTE_FLOAT4 out4 = out_c_idx_1 >= out_c_block ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias_ptr));
    #else
    COMPUTE_FLOAT4 out4 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias_ptr));
    #endif
    COMPUTE_FLOAT4 out5 = out4;

    const int intput_width_idx0 = out_w2_idx;
    int inp_offset = ((out_b_idx * out_h + out_h_idx)* out_w + intput_width_idx0)<<2;
    int offset = out_c_idx_0*4;
    const int inp_add = out_b*out_h*out_w*4;
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_c_block; ++in_channel_block_idx) {
        
        COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset));
        COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4(vload4(1, input+inp_offset));
        COMPUTE_FLOAT4 weights0 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset));
        COMPUTE_FLOAT4 weights1 = CONVERT_COMPUTE_FLOAT4(vload4(1, kernel_ptr + offset));
        COMPUTE_FLOAT4 weights2 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack));
        COMPUTE_FLOAT4 weights3 = CONVERT_COMPUTE_FLOAT4(vload4(1, kernel_ptr + offset + out_c_pack));
        COMPUTE_FLOAT4 weights4 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack));
        COMPUTE_FLOAT4 weights5 = CONVERT_COMPUTE_FLOAT4(vload4(1, kernel_ptr + offset + out_c_pack + out_c_pack));
        COMPUTE_FLOAT4 weights6 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack + out_c_pack));
        COMPUTE_FLOAT4 weights7 = CONVERT_COMPUTE_FLOAT4(vload4(1, kernel_ptr + offset + out_c_pack + out_c_pack + out_c_pack));

        out0 = mad(in0.x, weights0, out0);
        out0 = mad(in0.y, weights2, out0);
        out0 = mad(in0.z, weights4, out0);
        out0 = mad(in0.w, weights6, out0);
        
        out1 = mad(in1.x, weights0, out1);
        out1 = mad(in1.y, weights2, out1);
        out1 = mad(in1.z, weights4, out1);
        out1 = mad(in1.w, weights6, out1);
        
        out4 = mad(in0.x, weights1, out4);
        out4 = mad(in0.y, weights3, out4);
        out4 = mad(in0.z, weights5, out4);
        out4 = mad(in0.w, weights7, out4);
        
        out5 = mad(in1.x, weights1, out5);
        out5 = mad(in1.y, weights3, out5);
        out5 = mad(in1.z, weights5, out5);
        out5 = mad(in1.w, weights7, out5);
        
        offset += 4 * out_c_pack;
        inp_offset += inp_add;
    }

#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);

    out4 = fmax(out4, (COMPUTE_FLOAT4)0);
    out5 = fmax(out5, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);

    out4 = clamp(out4, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out5 = clamp(out5, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, slope_ptr));
    COMPUTE_FLOAT4 slope_in1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, slope_ptr));
    out0 = select(out0 * slope_in0, out0, out0 >= 0);
    out1 = select(out1 * slope_in0, out1, out1 >= 0);
    out4 = select(out4 * slope_in1, out4, out4 >= 0);
    out5 = select(out5 * slope_in1, out5, out5 >= 0);
#endif

    const int out_offset = (((out_b_idx + out_c_idx_0*out_b)*out_h + out_h_idx)* out_w + out_w2_idx)*4;


    __global FLOAT * _tempoutput = output + out_offset;
    __global FLOAT * _tempoutput1 = _tempoutput + 4*out_h*out_w*out_b;

#ifdef BLOCK_LEAVE
    const int remain = out_w - out_w2_idx;
    if (remain >= 2) {
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, _tempoutput);
    } else if (remain == 1) {
        vstore4(CONVERT_FLOAT4(out0), 0, _tempoutput);
    }
#ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_block) {
        return;
    }
#endif
    if (remain >= 2) {
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4, out5)), 0, _tempoutput1);
    } else if (remain == 1) {
        vstore4(CONVERT_FLOAT4(out4), 0, _tempoutput1);
    }
#else
    vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, _tempoutput);
#ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_block) {
        return;
    }
#endif
    vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4, out5)), 0, _tempoutput1);
#endif
}

__kernel
void conv_2d_1x1_c4h1w1(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks,
                          __global const FLOAT *input,
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          __global FLOAT *output,
                          __private const int in_c_block,
                          __private const int out_h,
                          __private const int out_w,
                          __private const int out_b,
                          __private const int out_c_block,
                          __private const int out_c_pack
                          #ifdef PRELU
                          ,__global const FLOAT *slope_ptr
                          #endif
) {

    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w;
    const int out_w_idx = out_c_w_idx % out_w;
    const int out_b_idx = out_b_h_idx / out_h;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h;//equal to in_h_idx

    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias_ptr));
    const int intput_width_idx0 = out_w_idx;
    int offset = out_c_idx*4;
    int inp_offset = ((out_b_idx * out_h + out_h_idx) * out_w + intput_width_idx0)*4;
    const int inp_add = out_b*out_h*out_w*4;
    
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_c_block; ++in_channel_block_idx) {
        
        
        COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset));
        COMPUTE_FLOAT4 weights0 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset));
        COMPUTE_FLOAT4 weights1 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack));
        COMPUTE_FLOAT4 weights2 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack));
        COMPUTE_FLOAT4 weights3 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack + out_c_pack));

        out0 = mad(in0.x, weights0, out0);
        out0 = mad(in0.y, weights1, out0);
        out0 = mad(in0.z, weights2, out0);
        out0 = mad(in0.w, weights3, out0);
        
        offset += 4 * out_c_pack;
        inp_offset += inp_add;
    }

#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
    out0 = select(out0 * slope_in, out0, out0 >= 0);
#endif

    const int out_offset = (((out_b_idx + out_c_idx*out_b)*out_h + out_h_idx)* out_w + out_w_idx)*4;

    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
}


__kernel
void conv_2d_1x1_c4h1w2(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks,
                          __global const FLOAT *input,
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          __global FLOAT *output,
                          __private const int in_c_block,
                          __private const int out_h,
                          __private const int out_w,
                          __private const int out_b,
                          __private const int out_c_block,
                          __private const int out_c_pack
                          #ifdef PRELU
                          ,__global const FLOAT *slope_ptr
                          #endif
) {

    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h;//equal to in_h_idx

    const int out_w2_idx = mul24(out_w_idx, 2);

    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias_ptr));
    COMPUTE_FLOAT4 out1 = out0;

    const int intput_width_idx0 = out_w2_idx;
    int offset = out_c_idx*4;
    int inp_offset = ((out_b_idx*out_h + out_h_idx)* out_w + intput_width_idx0)*4;
    const int inp_add = out_b*out_h*out_w*4;
    
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_c_block; ++in_channel_block_idx) {
        
        COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset));
        COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4(vload4(1, input+inp_offset));

        COMPUTE_FLOAT4 weights0 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset));
        COMPUTE_FLOAT4 weights1 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack));
        COMPUTE_FLOAT4 weights2 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack));
        COMPUTE_FLOAT4 weights3 = CONVERT_COMPUTE_FLOAT4(vload4(0, kernel_ptr + offset + out_c_pack + out_c_pack + out_c_pack));

        out0 = mad(in0.x, weights0, out0);
        out0 = mad(in0.y, weights1, out0);
        out0 = mad(in0.z, weights2, out0);
        out0 = mad(in0.w, weights3, out0);
        
        out1 = mad(in1.x, weights0, out1);
        out1 = mad(in1.y, weights1, out1);
        out1 = mad(in1.z, weights2, out1);
        out1 = mad(in1.w, weights3, out1);
        
        offset += 4 * out_c_pack;
        inp_offset += inp_add;
    }

#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
    out0 = select(out0 * slope_in, out0, out0 >= 0);
    out1 = select(out1 * slope_in, out1, out1 >= 0);
#endif

    const int out_offset = (((out_b_idx + out_c_idx*out_b)*out_h + out_h_idx)* out_w + out_w2_idx)*4;

#ifdef BLOCK_LEAVE
    const int remain = out_w - out_w2_idx;

    if (remain >= 2) {
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
    } else if (remain == 1) {
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
#else
    vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
#endif
}

__kernel
void conv_2d_c4h1w1(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int batch,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_hw.y + out_c_base_index;
    if(out_c_idx >= out_c_blocks) return;
    const int out_w_idx = out_c_w_idx % out_hw.y;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    
    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);

    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);
    
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + kh_start)*filter_hw.y + kw_start) * 4;
        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            for(int ix = in_w_idx_start; ix < in_w_idx_end; ix += dilate_hw.y) {
                int inp_offset = (((out_b_idx + in_c_idx * batch) * in_hw.x + iy) * in_hw.y + ix) * 4;
                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset));
                
                const int filter_w_inc = (ix-in_w_idx_start)/dilate_hw.y;

                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(vload4(filter_w_inc, weight+weight_offset));
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(vload4(filter_w_inc, weight+weight_offset+weight_oc_offset));
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(vload4(filter_w_inc, weight+weight_offset+weight_oc_offset*2));
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(vload4(filter_w_inc, weight+weight_offset+weight_oc_offset*3));

                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);

            }
            weight_offset += 4*filter_hw.y;
        }
    }
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
    out0 = select(out0 * slope_in, out0, out0 >= 0);
#endif
    const int out_offset = (((out_b_idx + out_c_idx*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
 
}

__kernel
void conv_2d_c4h1w2(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int batch,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,//generate width's num
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks + out_c_base_index;
    if(out_c_idx >= out_c_blocks) return;
    const int out_w_idx = (out_c_w_idx % out_w_blocks) << 1;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out1 = out0;
    
    const int in_w0_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_w1_idx_base = in_w0_idx_base + stride_hw.y;

    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + kh_start)*filter_hw.y + 0) * 4;

        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            const int inp_offset_base = (((out_b_idx + in_c_idx*batch) * in_hw.x + iy) * in_hw.y + 0) * 4;

            for(int fw = 0; fw < filter_hw.y; fw++) {
                const int in_w0_idx = fw * dilate_hw.y + in_w0_idx_base;
                const int in_w1_idx = fw * dilate_hw.y + in_w1_idx_base;

                COMPUTE_FLOAT4 in0 = (in_w0_idx < 0 || in_w0_idx >= in_hw.y) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w0_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = (in_w1_idx < 0 || in_w1_idx >= in_hw.y) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w1_idx, input+inp_offset_base));
                
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset));
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset*2));
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset*3));

                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                weight_offset += 4;
            }
        }
    }
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
    out0 = select(out0 * slope_in, out0, out0 >= 0);
    out1 = select(out1 * slope_in, out1, out1 >= 0);
#endif

    const int out_offset = (((out_b_idx + out_c_idx*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    if(out_w_idx + 1 >= out_hw.y) return;
    vstore4(CONVERT_FLOAT4(out1), 1, output+out_offset);
#else
    vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
#endif
}

__kernel
void conv_2d_c4h1w4(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int batch,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks + out_c_base_index;
    if(out_c_idx >= out_c_blocks) return;
    const int out_w_idx = (out_c_w_idx % out_w_blocks) << 2;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;

    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out1 = out0;
    COMPUTE_FLOAT4 out2 = out0;
    COMPUTE_FLOAT4 out3 = out0;

    const int in_w0_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_w1_idx_base = in_w0_idx_base + stride_hw.y;
    const int in_w2_idx_base = in_w1_idx_base + stride_hw.y;
    const int in_w3_idx_base = in_w2_idx_base + stride_hw.y;

    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + kh_start)*filter_hw.y + 0) * 4;

        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            const int inp_offset_base = (((out_b_idx + in_c_idx*batch) * in_hw.x + iy) * in_hw.y + 0) * 4;

            for(int fw = 0; fw < filter_hw.y; fw++) {
                const int in_w0_idx = fw * dilate_hw.y + in_w0_idx_base;
                const int in_w1_idx = fw * dilate_hw.y + in_w1_idx_base;
                const int in_w2_idx = fw * dilate_hw.y + in_w2_idx_base;
                const int in_w3_idx = fw * dilate_hw.y + in_w3_idx_base;

                COMPUTE_FLOAT4 in0 = (in_w0_idx < 0 || in_w0_idx >= in_hw.y) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w0_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = (in_w1_idx < 0 || in_w1_idx >= in_hw.y) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w1_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = (in_w2_idx < 0 || in_w2_idx >= in_hw.y) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w2_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = (in_w3_idx < 0 || in_w3_idx >= in_hw.y) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w3_idx, input+inp_offset_base));

                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset));
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset*2));
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset*3));

                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                out2 = mad(in2.x, weight0, out2);
                out2 = mad(in2.y, weight1, out2);
                out2 = mad(in2.z, weight2, out2);
                out2 = mad(in2.w, weight3, out2);
                
                out3 = mad(in3.x, weight0, out3);
                out3 = mad(in3.y, weight1, out3);
                out3 = mad(in3.z, weight2, out3);
                out3 = mad(in3.w, weight3, out3);
                
                weight_offset += 4;
            }
        }
    }
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
    out0 = select(out0 * slope_in, out0, out0 >= 0);
    out1 = select(out1 * slope_in, out1, out1 >= 0);
    out2 = select(out2 * slope_in, out2, out2 >= 0);
    out3 = select(out3 * slope_in, out3, out3 >= 0);
#endif

    const int out_offset = (((out_b_idx + out_c_idx*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.y - out_w_idx;

    if (remain >= 4) {
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, output+out_offset);
    }else if(remain == 3){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2, output+out_offset);
    }else if(remain == 2){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
#else
    vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, output+out_offset);
#endif
}

__kernel
void conv_2d_c4h4w1(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int batch,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks + out_c_base_index;
    if(out_c_idx >= out_c_blocks) return;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h_blocks;//equal to in_b_idx
    const int out_h_idx = (out_b_h_idx % out_h_blocks) << 2;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out1 = out0;
    COMPUTE_FLOAT4 out2 = out0;
    COMPUTE_FLOAT4 out3 = out0;

    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);

    const int in_h0_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    const int in_h1_idx_base = in_h0_idx_base + stride_hw.x;
    const int in_h2_idx_base = in_h1_idx_base + stride_hw.x;
    const int in_h3_idx_base = in_h2_idx_base + stride_hw.x;
    
    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    const int in_hw_size = in_hw.x * in_hw.y;
#ifdef CONV_SPEC_UNROLL
    __attribute__((opencl_unroll_hint))
#endif
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        const int inp_offset_base = (out_b_idx + in_c_idx*batch) * in_hw.x * in_hw.y * 4;

        for(int iy = 0; iy < filter_hw.x; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + iy)*filter_hw.y + kw_start) * 4;
            const int in_h0_idx = (iy * dilate_hw.x + in_h0_idx_base) * in_hw.y;
            const int in_h1_idx = (iy * dilate_hw.x + in_h1_idx_base) * in_hw.y;
            const int in_h2_idx = (iy * dilate_hw.x + in_h2_idx_base) * in_hw.y;
            const int in_h3_idx = (iy * dilate_hw.x + in_h3_idx_base) * in_hw.y;

            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += dilate_hw.y) {
                COMPUTE_FLOAT4 in0 = (in_h0_idx < 0 || in_h0_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h0_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = (in_h1_idx < 0 || in_h1_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h1_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = (in_h2_idx < 0 || in_h2_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h2_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = (in_h3_idx < 0 || in_h3_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h3_idx + fw, input+inp_offset_base));

                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset));
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset*2));
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset*3));
                
                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                out2 = mad(in2.x, weight0, out2);
                out2 = mad(in2.y, weight1, out2);
                out2 = mad(in2.z, weight2, out2);
                out2 = mad(in2.w, weight3, out2);
                
                out3 = mad(in3.x, weight0, out3);
                out3 = mad(in3.y, weight1, out3);
                out3 = mad(in3.z, weight2, out3);
                out3 = mad(in3.w, weight3, out3);
                
                weight_offset += 4;
            }
        }
    }
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
    out0 = select(out0 * slope_in, out0, out0 >= 0);
    out1 = select(out1 * slope_in, out1, out1 >= 0);
    out2 = select(out2 * slope_in, out2, out2 >= 0);
    out3 = select(out3 * slope_in, out3, out3 >= 0);
#endif

    const int out_offset = (((out_b_idx + out_c_idx*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.x - out_h_idx;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out3), 3 * out_hw.y, output+out_offset);
    }else if(remain == 3){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
    }else if(remain == 2){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), 3 * out_hw.y, output+out_offset);
#endif
}

__kernel
void conv_2d_c8h4w1(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int batch,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx_0 = ((out_c_w_idx / out_w_blocks + out_c_base_index) << 1);
    if(out_c_idx_0 >= out_c_blocks) return;
    const int out_c_idx_1 = out_c_idx_0 + 1;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h_blocks;//equal to in_b_idx
    const int out_h_idx = (out_b_h_idx % out_h_blocks) << 2;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, bias));
    COMPUTE_FLOAT4 out1 = out0;
    COMPUTE_FLOAT4 out2 = out0;
    COMPUTE_FLOAT4 out3 = out0;
    #ifdef CHANNEL_BOUNDARY_PROTECT
    COMPUTE_FLOAT4 out4 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #else
    COMPUTE_FLOAT4 out4 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #endif
    COMPUTE_FLOAT4 out5 = out4;
    COMPUTE_FLOAT4 out6 = out4;
    COMPUTE_FLOAT4 out7 = out4;

    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);

    const int in_h0_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    const int in_h1_idx_base = in_h0_idx_base + stride_hw.x;
    const int in_h2_idx_base = in_h1_idx_base + stride_hw.x;
    const int in_h3_idx_base = in_h2_idx_base + stride_hw.x;
    
    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);
    
    const int weight_oc_offset = filter_hw.x * filter_hw.y * 4;
    const int weight_ic_offset = out_c_blocks * weight_oc_offset;
    const int in_hw_size = in_hw.x * in_hw.y;
#ifdef CONV_SPEC_UNROLL
    __attribute__((opencl_unroll_hint))
#endif
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        //weights  NC4HW4   [ic/4, ic_4, oc/4, kh*kw, oc_4]
        //index:   [0, 4*in_c_idx, out_c_idx_0*kh*kw + kh_start*kw + kw_start, 0]
        const int inp_offset_base = (out_b_idx + in_c_idx * batch) * in_hw.x * in_hw.y * 4;

        for(int iy = 0; iy < filter_hw.x; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx_0) *filter_hw.x + iy)*filter_hw.y + kw_start) * 4;
            const int in_h0_idx = (iy * dilate_hw.x + in_h0_idx_base) * in_hw.y;
            const int in_h1_idx = (iy * dilate_hw.x + in_h1_idx_base) * in_hw.y;
            const int in_h2_idx = (iy * dilate_hw.x + in_h2_idx_base) * in_hw.y;
            const int in_h3_idx = (iy * dilate_hw.x + in_h3_idx_base) * in_hw.y;

            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += dilate_hw.y) {
                COMPUTE_FLOAT4 in0 = (in_h0_idx < 0 || in_h0_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h0_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = (in_h1_idx < 0 || in_h1_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h1_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = (in_h2_idx < 0 || in_h2_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h2_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = (in_h3_idx < 0 || in_h3_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h3_idx + fw, input+inp_offset_base));

                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset));
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset));
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*2));
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*3));
                
                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                out2 = mad(in2.x, weight0, out2);
                out2 = mad(in2.y, weight1, out2);
                out2 = mad(in2.z, weight2, out2);
                out2 = mad(in2.w, weight3, out2);
                
                out3 = mad(in3.x, weight0, out3);
                out3 = mad(in3.y, weight1, out3);
                out3 = mad(in3.z, weight2, out3);
                out3 = mad(in3.w, weight3, out3);

                // weight: [ic/4, ic_4, oc/4, kh*kw, oc_4]
                #ifdef CHANNEL_BOUNDARY_PROTECT
                weight0 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                weight1 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                weight2 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                weight3 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
                #else
                weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
                #endif
                out4 = mad(in0.x, weight0, out4);
                out4 = mad(in0.y, weight1, out4);
                out4 = mad(in0.z, weight2, out4);
                out4 = mad(in0.w, weight3, out4);
                
                out5 = mad(in1.x, weight0, out5);
                out5 = mad(in1.y, weight1, out5);
                out5 = mad(in1.z, weight2, out5);
                out5 = mad(in1.w, weight3, out5);
                
                out6 = mad(in2.x, weight0, out6);
                out6 = mad(in2.y, weight1, out6);
                out6 = mad(in2.z, weight2, out6);
                out6 = mad(in2.w, weight3, out6);
                
                out7 = mad(in3.x, weight0, out7);
                out7 = mad(in3.y, weight1, out7);
                out7 = mad(in3.z, weight2, out7);
                out7 = mad(in3.w, weight3, out7);
                
                weight_offset += 4;
            }
        }
    }
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
    out4 = fmax(out4, (COMPUTE_FLOAT4)0);
    out5 = fmax(out5, (COMPUTE_FLOAT4)0);
    out6 = fmax(out6, (COMPUTE_FLOAT4)0);
    out7 = fmax(out7, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out4 = clamp(out4, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out5 = clamp(out5, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out6 = clamp(out6, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out7 = clamp(out7, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, slope_ptr));
    COMPUTE_FLOAT4 slope_in1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, slope_ptr));
    out0 = select(out0 * slope_in0, out0, out0 >= 0);
    out1 = select(out1 * slope_in0, out1, out1 >= 0);
    out2 = select(out2 * slope_in0, out2, out2 >= 0);
    out3 = select(out3 * slope_in0, out3, out3 >= 0);
    out4 = select(out4 * slope_in1, out4, out4 >= 0);
    out5 = select(out5 * slope_in1, out5, out5 >= 0);
    out6 = select(out6 * slope_in1, out6, out6 >= 0);
    out7 = select(out7 * slope_in1, out7, out7 >= 0);
#endif

    int out_offset = (((out_b_idx + out_c_idx_0*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.x - out_h_idx;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out3), 3 * out_hw.y, output+out_offset);
    }else if(remain == 3){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
    }else if(remain == 2){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_blocks){
        return;
    }
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out5), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out6), 2 * out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out7), 3 * out_hw.y, output+out_offset);
    }else if(remain == 3){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out5), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out6), 2 * out_hw.y, output+out_offset);
    }else if(remain == 2){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out5), out_hw.y, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
    }
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), 3 * out_hw.y, output+out_offset);
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_blocks){
        return;
    }
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out5), out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out6), 2 * out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out7), 3 * out_hw.y, output+out_offset);
#endif
}

__kernel
void conv_2d_c8h2w1(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int batch,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx_0 = (out_c_w_idx / out_w_blocks + out_c_base_index) << 1;
    if(out_c_idx_0 >= out_c_blocks) return;
    const int out_c_idx_1 = out_c_idx_0 + 1;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h_blocks;//equal to in_b_idx
    const int out_h_idx = (out_b_h_idx % out_h_blocks) << 1;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, bias));
    COMPUTE_FLOAT4 out1 = out0;
    #ifdef CHANNEL_BOUNDARY_PROTECT
    COMPUTE_FLOAT4 out2 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #else
    COMPUTE_FLOAT4 out2 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #endif
    COMPUTE_FLOAT4 out3 = out2;
    
    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);

    const int in_h0_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    const int in_h1_idx_base = in_h0_idx_base + stride_hw.x;
    
    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);
    
    const int weight_oc_offset = filter_hw.x * filter_hw.y * 4;
    const int weight_ic_offset = out_c_blocks * weight_oc_offset;
    const int in_hw_size = in_hw.x * in_hw.y;
    // weight: [ic/4, oc, 4], loop: ic/4
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx_0*kh*kw + kh_start*kw + kw_start, 0]
        const int inp_offset_base = (out_b_idx + in_c_idx*batch) * in_hw.x * in_hw.y * 4;

        for(int iy = 0; iy < filter_hw.x; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx_0) *filter_hw.x + iy)*filter_hw.y + kw_start) * 4;
            const int in_h0_idx = (iy * dilate_hw.x + in_h0_idx_base) * in_hw.y;
            const int in_h1_idx = (iy * dilate_hw.x + in_h1_idx_base) * in_hw.y;

            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += dilate_hw.y) {
                COMPUTE_FLOAT4 in0 = (in_h0_idx < 0 || in_h0_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h0_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = (in_h1_idx < 0 || in_h1_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h1_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset));
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset));
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*2));
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*3));
                
                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                #ifdef CHANNEL_BOUNDARY_PROTECT
                weight0 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                weight1 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                weight2 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                weight3 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
                #else
                weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
                #endif
                out2 = mad(in0.x, weight0, out2);
                out2 = mad(in0.y, weight1, out2);
                out2 = mad(in0.z, weight2, out2);
                out2 = mad(in0.w, weight3, out2);
                
                out3 = mad(in1.x, weight0, out3);
                out3 = mad(in1.y, weight1, out3);
                out3 = mad(in1.z, weight2, out3);
                out3 = mad(in1.w, weight3, out3);
                
                weight_offset += 4;
            }
        }
    }
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, slope_ptr));
    COMPUTE_FLOAT4 slope_in1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, slope_ptr));
    out0 = select(out0 * slope_in0, out0, out0 >= 0);
    out1 = select(out1 * slope_in0, out1, out1 >= 0);
    out2 = select(out2 * slope_in1, out2, out2 >= 0);
    out3 = select(out3 * slope_in1, out3, out3 >= 0);
#endif

    int out_offset = (((out_b_idx + out_c_idx_0*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.x - out_h_idx;
    if(remain >= 2){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_blocks){
        return;
    }
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    if(remain >= 2){
        vstore4(CONVERT_FLOAT4(out2), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out3), out_hw.y, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out2), 0, output+out_offset);
    }
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_blocks){
        return;
    }
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out2), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), out_hw.y, output+out_offset);
#endif
}

__kernel
void conv_2d_c8h1w4(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int batch,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx_0 = (out_c_w_idx / out_w_blocks + out_c_base_index) << 1;
    if(out_c_idx_0 >= out_c_blocks) return;
    const int out_c_idx_1 = out_c_idx_0 + 1;
    const int out_w_idx = (out_c_w_idx % out_w_blocks) << 2;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, bias));
    COMPUTE_FLOAT4 out1 = out0;
    COMPUTE_FLOAT4 out2 = out0;
    COMPUTE_FLOAT4 out3 = out0;
    #ifdef CHANNEL_BOUNDARY_PROTECT
    COMPUTE_FLOAT4 out4 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #else
    COMPUTE_FLOAT4 out4 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #endif
    COMPUTE_FLOAT4 out5 = out4;
    COMPUTE_FLOAT4 out6 = out4;
    COMPUTE_FLOAT4 out7 = out4;

    const int in_w0_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_w1_idx_base = in_w0_idx_base + stride_hw.y;
    const int in_w2_idx_base = in_w1_idx_base + stride_hw.y;
    const int in_w3_idx_base = in_w2_idx_base + stride_hw.y;

    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = filter_hw.x * filter_hw.y * 4;
    const int weight_ic_offset = out_c_blocks * weight_oc_offset;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx_0*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx_0) *filter_hw.x + kh_start)*filter_hw.y + 0) * 4;

        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            const int inp_offset_base = (((out_b_idx + in_c_idx * batch) * in_hw.x + iy) * in_hw.y + 0) * 4;

            for(int fw = 0; fw < filter_hw.y; fw++) {
                const int in_w0_idx = fw * dilate_hw.y + in_w0_idx_base;
                const int in_w1_idx = fw * dilate_hw.y + in_w1_idx_base;
                const int in_w2_idx = fw * dilate_hw.y + in_w2_idx_base;
                const int in_w3_idx = fw * dilate_hw.y + in_w3_idx_base;

                COMPUTE_FLOAT4 in0 = (in_w0_idx < 0 || in_w0_idx >= in_hw.y) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w0_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = (in_w1_idx < 0 || in_w1_idx >= in_hw.y) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w1_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = (in_w2_idx < 0 || in_w2_idx >= in_hw.y) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w2_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = (in_w3_idx < 0 || in_w3_idx >= in_hw.y) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w3_idx, input+inp_offset_base));

                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset));
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset));
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*2));
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*3));

                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                out2 = mad(in2.x, weight0, out2);
                out2 = mad(in2.y, weight1, out2);
                out2 = mad(in2.z, weight2, out2);
                out2 = mad(in2.w, weight3, out2);
                
                out3 = mad(in3.x, weight0, out3);
                out3 = mad(in3.y, weight1, out3);
                out3 = mad(in3.z, weight2, out3);
                out3 = mad(in3.w, weight3, out3);
                
                #ifdef CHANNEL_BOUNDARY_PROTECT
                weight0 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                weight1 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                weight2 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                weight3 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
                #else
                weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
                #endif
                out4 = mad(in0.x, weight0, out4);
                out4 = mad(in0.y, weight1, out4);
                out4 = mad(in0.z, weight2, out4);
                out4 = mad(in0.w, weight3, out4);
                
                out5 = mad(in1.x, weight0, out5);
                out5 = mad(in1.y, weight1, out5);
                out5 = mad(in1.z, weight2, out5);
                out5 = mad(in1.w, weight3, out5);
                
                out6 = mad(in2.x, weight0, out6);
                out6 = mad(in2.y, weight1, out6);
                out6 = mad(in2.z, weight2, out6);
                out6 = mad(in2.w, weight3, out6);
                
                out7 = mad(in3.x, weight0, out7);
                out7 = mad(in3.y, weight1, out7);
                out7 = mad(in3.z, weight2, out7);
                out7 = mad(in3.w, weight3, out7);
                
                weight_offset += 4;
            }
        }
    }
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
    out4 = fmax(out4, (COMPUTE_FLOAT4)0);
    out5 = fmax(out5, (COMPUTE_FLOAT4)0);
    out6 = fmax(out6, (COMPUTE_FLOAT4)0);
    out7 = fmax(out7, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out4 = clamp(out4, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out5 = clamp(out5, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out6 = clamp(out6, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out7 = clamp(out7, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

#ifdef PRELU
    COMPUTE_FLOAT4 slope_in0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, slope_ptr));
    COMPUTE_FLOAT4 slope_in1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, slope_ptr));
    out0 = select(out0 * slope_in0, out0, out0 >= 0);
    out1 = select(out1 * slope_in0, out1, out1 >= 0);
    out2 = select(out2 * slope_in0, out2, out2 >= 0);
    out3 = select(out3 * slope_in0, out3, out3 >= 0);
    out4 = select(out4 * slope_in1, out4, out4 >= 0);
    out5 = select(out5 * slope_in1, out5, out5 >= 0);
    out6 = select(out6 * slope_in1, out6, out6 >= 0);
    out7 = select(out7 * slope_in1, out7, out7 >= 0);
#endif

    int out_offset = (((out_b_idx + out_c_idx_0*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.y - out_w_idx;
    if(remain >= 4){
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, output+out_offset);
    }else if(remain == 3){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2, output+out_offset);
    }else if(remain == 2){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_blocks)return;
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    if(remain >= 4){
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out4, out5, out6, out7)), 0, output+out_offset);
    }else if(remain == 3){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4, out5)), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out6), 2, output+out_offset);
    }else if(remain == 2){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4, out5)), 0, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
    }
#else
    vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, output+out_offset);
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_blocks)return;
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out4, out5, out6, out7)), 0, output+out_offset);
#endif
}

// LDS input-halo tiled 3x3 stride-1 conv (specialization for the occupancy-starved
// small-channel cores). One workgroup = TILE_W x TILE_H output tile for one output
// channel-block; the (TILE_W+2)x(TILE_H+2) input halo for each input channel-block is
// staged in __local ONCE and reused across the 9 taps + all neighbouring outputs,
// killing the 9x redundant global input reads of the general conv_2d_c* path.
// gws = {out_c_blocks*out_w, batch*out_h}, lws = {TILE_W, TILE_H} (exact multiples =>
// no partial workgroups, so no early return before the barriers). Index math mirrors
// conv_2d_c4h1w1 exactly. Requires: 3x3, stride 1, dilation 1, pad 1, out_w%TILE_W==0,
// out_h%TILE_H==0.
#ifndef TILE_W
#define TILE_W 16
#endif
#ifndef TILE_H
#define TILE_H 4
#endif
__kernel
void conv_2d_3x3s1_lds(GLOBAL_SIZE_2_DIMS
                       __global const FLOAT *input,
                       __global const FLOAT *weight,
                       __global const FLOAT *bias,
                       __global FLOAT *output,
                       __private const int2 in_hw,
                       __private const int inChannel,
                       __private const int in_c_blocks,
                       __private const int batch,
                       __private const int2 out_hw,
                       __private const int2 filter_hw,
                       __private const int2 stride_hw,
                       __private const int2 pad_hw,
                       __private const int2 dilate_hw,
                       __private const int out_w_blocks,
                       __private const int out_c_blocks,
                       __private const int out_h_blocks,
                       __private const int out_c_base_index
                       #ifdef PRELU
                       ,__global const FLOAT *slope_ptr
                       #endif
) {
    const int LDS_W = TILE_W + 2;
    const int LDS_H = TILE_H + 2;
    __local COMPUTE_FLOAT4 lds[(TILE_H + 2) * (TILE_W + 2)];

    const int gx = get_global_id(0);   // out_c_idx * out_w + out_x
    const int gy = get_global_id(1);   // out_b_idx * out_h + out_y
    const int lx = get_local_id(0);    // 0..TILE_W-1
    const int ly = get_local_id(1);    // 0..TILE_H-1

    const int out_w = out_hw.y;
    const int out_h = out_hw.x;
    const int in_w  = in_hw.y;
    const int in_h  = in_hw.x;

    const int out_c_idx = gx / out_w;
    const int out_x     = gx % out_w;
    const int out_b_idx = gy / out_h;
    const int out_y     = gy % out_h;

    // top-left output pixel of this workgroup, and the input halo origin (stride 1)
    const int in_base_x = (out_x - lx) - pad_hw.y;
    const int in_base_y = (out_y - ly) - pad_hw.x;

    COMPUTE_FLOAT4 acc = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));

    const int weight_oc_offset = out_c_blocks * 9 * 4;   // stride between the 4 input sub-channels
    const int lid       = ly * TILE_W + lx;
    const int nthreads  = TILE_W * TILE_H;
    const int tile_elems = LDS_W * LDS_H;

    for (int ic = 0; ic < in_c_blocks; ic++) {
        // cooperative halo load into LDS for this input channel-block (zero-pad OOB)
        for (int e = lid; e < tile_elems; e += nthreads) {
            const int ty = e / LDS_W;
            const int tx = e % LDS_W;
            const int iy = in_base_y + ty;
            const int ix = in_base_x + tx;
            COMPUTE_FLOAT4 v = (COMPUTE_FLOAT4)0;
            if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                const int inp_offset = (((ic * batch + out_b_idx) * in_h + iy) * in_w + ix) * 4;
                v = CONVERT_COMPUTE_FLOAT4(vload4(0, input + inp_offset));
            }
            lds[e] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int w_base = (((4 * ic) * out_c_blocks + out_c_idx) * 9) * 4;   // (s=0,kh=0,kw=0)
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                COMPUTE_FLOAT4 in0 = lds[(ly + kh) * LDS_W + (lx + kw)];
                const int w_off = w_base + (kh * 3 + kw) * 4;
                COMPUTE_FLOAT4 w0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_off));
                COMPUTE_FLOAT4 w1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_off + weight_oc_offset));
                COMPUTE_FLOAT4 w2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_off + weight_oc_offset * 2));
                COMPUTE_FLOAT4 w3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_off + weight_oc_offset * 3));
                acc = mad(in0.x, w0, acc);
                acc = mad(in0.y, w1, acc);
                acc = mad(in0.z, w2, acc);
                acc = mad(in0.w, w3, acc);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#ifdef RELU
    acc = fmax(acc, (COMPUTE_FLOAT4)0);
#endif
#ifdef RELU6
    acc = clamp(acc, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif
#ifdef PRELU
    COMPUTE_FLOAT4 slope_in = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
    acc = select(acc * slope_in, acc, acc >= 0);
#endif

    const int out_offset = (((out_b_idx + out_c_idx * batch) * out_h + out_y) * out_w + out_x) * 4;
    vstore4(CONVERT_FLOAT4(acc), 0, output + out_offset);
}

// c8h8w1: register-blocked 3x3 variant, 8 output channels x 8 rows/thread = 16 accumulators
// (2x the tile height of c8h4w1). More work-in-flight + more input-row reuse; tests whether the
// autotuner's 8-accumulator optimum is the true optimum. Same NC4HW4 index math as c8h4w1.
__kernel
void conv_2d_c8h8w1(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int batch,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0);
    const int out_b_h_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx_0 = ((out_c_w_idx / out_w_blocks + out_c_base_index) << 1);
    if(out_c_idx_0 >= out_c_blocks) return;
    const int out_c_idx_1 = out_c_idx_0 + 1;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h_blocks;
    const int out_h_idx = (out_b_h_idx % out_h_blocks) << 3;

    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, bias));
    COMPUTE_FLOAT4 out1 = out0; COMPUTE_FLOAT4 out2 = out0; COMPUTE_FLOAT4 out3 = out0;
    COMPUTE_FLOAT4 out4 = out0; COMPUTE_FLOAT4 out5 = out0; COMPUTE_FLOAT4 out6 = out0; COMPUTE_FLOAT4 out7 = out0;
    #ifdef CHANNEL_BOUNDARY_PROTECT
    COMPUTE_FLOAT4 out8 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #else
    COMPUTE_FLOAT4 out8 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #endif
    COMPUTE_FLOAT4 out9 = out8; COMPUTE_FLOAT4 out10 = out8; COMPUTE_FLOAT4 out11 = out8;
    COMPUTE_FLOAT4 out12 = out8; COMPUTE_FLOAT4 out13 = out8; COMPUTE_FLOAT4 out14 = out8; COMPUTE_FLOAT4 out15 = out8;

    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_h0_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    const int in_h1_idx_base = in_h0_idx_base + stride_hw.x;
    const int in_h2_idx_base = in_h1_idx_base + stride_hw.x;
    const int in_h3_idx_base = in_h2_idx_base + stride_hw.x;
    const int in_h4_idx_base = in_h3_idx_base + stride_hw.x;
    const int in_h5_idx_base = in_h4_idx_base + stride_hw.x;
    const int in_h6_idx_base = in_h5_idx_base + stride_hw.x;
    const int in_h7_idx_base = in_h6_idx_base + stride_hw.x;

    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);

    const int weight_oc_offset = filter_hw.x * filter_hw.y * 4;
    const int weight_ic_offset = out_c_blocks * weight_oc_offset;
    const int in_hw_size = in_hw.x * in_hw.y;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        const int inp_offset_base = (out_b_idx + in_c_idx * batch) * in_hw.x * in_hw.y * 4;
        for(int iy = 0; iy < filter_hw.x; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx_0) *filter_hw.x + iy)*filter_hw.y + kw_start) * 4;
            const int in_h0_idx = (iy * dilate_hw.x + in_h0_idx_base) * in_hw.y;
            const int in_h1_idx = (iy * dilate_hw.x + in_h1_idx_base) * in_hw.y;
            const int in_h2_idx = (iy * dilate_hw.x + in_h2_idx_base) * in_hw.y;
            const int in_h3_idx = (iy * dilate_hw.x + in_h3_idx_base) * in_hw.y;
            const int in_h4_idx = (iy * dilate_hw.x + in_h4_idx_base) * in_hw.y;
            const int in_h5_idx = (iy * dilate_hw.x + in_h5_idx_base) * in_hw.y;
            const int in_h6_idx = (iy * dilate_hw.x + in_h6_idx_base) * in_hw.y;
            const int in_h7_idx = (iy * dilate_hw.x + in_h7_idx_base) * in_hw.y;
            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += dilate_hw.y) {
                COMPUTE_FLOAT4 in0 = (in_h0_idx < 0 || in_h0_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h0_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = (in_h1_idx < 0 || in_h1_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h1_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = (in_h2_idx < 0 || in_h2_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h2_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = (in_h3_idx < 0 || in_h3_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h3_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in4 = (in_h4_idx < 0 || in_h4_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h4_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in5 = (in_h5_idx < 0 || in_h5_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h5_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in6 = (in_h6_idx < 0 || in_h6_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h6_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in7 = (in_h7_idx < 0 || in_h7_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h7_idx + fw, input+inp_offset_base));

                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset));
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset));
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*2));
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*3));

                out0 = mad(in0.x, weight0, out0); out0 = mad(in0.y, weight1, out0); out0 = mad(in0.z, weight2, out0); out0 = mad(in0.w, weight3, out0);
                out1 = mad(in1.x, weight0, out1); out1 = mad(in1.y, weight1, out1); out1 = mad(in1.z, weight2, out1); out1 = mad(in1.w, weight3, out1);
                out2 = mad(in2.x, weight0, out2); out2 = mad(in2.y, weight1, out2); out2 = mad(in2.z, weight2, out2); out2 = mad(in2.w, weight3, out2);
                out3 = mad(in3.x, weight0, out3); out3 = mad(in3.y, weight1, out3); out3 = mad(in3.z, weight2, out3); out3 = mad(in3.w, weight3, out3);
                out4 = mad(in4.x, weight0, out4); out4 = mad(in4.y, weight1, out4); out4 = mad(in4.z, weight2, out4); out4 = mad(in4.w, weight3, out4);
                out5 = mad(in5.x, weight0, out5); out5 = mad(in5.y, weight1, out5); out5 = mad(in5.z, weight2, out5); out5 = mad(in5.w, weight3, out5);
                out6 = mad(in6.x, weight0, out6); out6 = mad(in6.y, weight1, out6); out6 = mad(in6.z, weight2, out6); out6 = mad(in6.w, weight3, out6);
                out7 = mad(in7.x, weight0, out7); out7 = mad(in7.y, weight1, out7); out7 = mad(in7.z, weight2, out7); out7 = mad(in7.w, weight3, out7);

                #ifdef CHANNEL_BOUNDARY_PROTECT
                weight0 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                weight1 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                weight2 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                weight3 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
                #else
                weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
                #endif
                out8  = mad(in0.x, weight0, out8);  out8  = mad(in0.y, weight1, out8);  out8  = mad(in0.z, weight2, out8);  out8  = mad(in0.w, weight3, out8);
                out9  = mad(in1.x, weight0, out9);  out9  = mad(in1.y, weight1, out9);  out9  = mad(in1.z, weight2, out9);  out9  = mad(in1.w, weight3, out9);
                out10 = mad(in2.x, weight0, out10); out10 = mad(in2.y, weight1, out10); out10 = mad(in2.z, weight2, out10); out10 = mad(in2.w, weight3, out10);
                out11 = mad(in3.x, weight0, out11); out11 = mad(in3.y, weight1, out11); out11 = mad(in3.z, weight2, out11); out11 = mad(in3.w, weight3, out11);
                out12 = mad(in4.x, weight0, out12); out12 = mad(in4.y, weight1, out12); out12 = mad(in4.z, weight2, out12); out12 = mad(in4.w, weight3, out12);
                out13 = mad(in5.x, weight0, out13); out13 = mad(in5.y, weight1, out13); out13 = mad(in5.z, weight2, out13); out13 = mad(in5.w, weight3, out13);
                out14 = mad(in6.x, weight0, out14); out14 = mad(in6.y, weight1, out14); out14 = mad(in6.z, weight2, out14); out14 = mad(in6.w, weight3, out14);
                out15 = mad(in7.x, weight0, out15); out15 = mad(in7.y, weight1, out15); out15 = mad(in7.z, weight2, out15); out15 = mad(in7.w, weight3, out15);

                weight_offset += 4;
            }
        }
    }
#ifdef RELU
    out0=fmax(out0,(COMPUTE_FLOAT4)0);out1=fmax(out1,(COMPUTE_FLOAT4)0);out2=fmax(out2,(COMPUTE_FLOAT4)0);out3=fmax(out3,(COMPUTE_FLOAT4)0);
    out4=fmax(out4,(COMPUTE_FLOAT4)0);out5=fmax(out5,(COMPUTE_FLOAT4)0);out6=fmax(out6,(COMPUTE_FLOAT4)0);out7=fmax(out7,(COMPUTE_FLOAT4)0);
    out8=fmax(out8,(COMPUTE_FLOAT4)0);out9=fmax(out9,(COMPUTE_FLOAT4)0);out10=fmax(out10,(COMPUTE_FLOAT4)0);out11=fmax(out11,(COMPUTE_FLOAT4)0);
    out12=fmax(out12,(COMPUTE_FLOAT4)0);out13=fmax(out13,(COMPUTE_FLOAT4)0);out14=fmax(out14,(COMPUTE_FLOAT4)0);out15=fmax(out15,(COMPUTE_FLOAT4)0);
#endif
#ifdef RELU6
    out0=clamp(out0,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out1=clamp(out1,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out2=clamp(out2,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out3=clamp(out3,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    out4=clamp(out4,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out5=clamp(out5,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out6=clamp(out6,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out7=clamp(out7,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    out8=clamp(out8,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out9=clamp(out9,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out10=clamp(out10,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out11=clamp(out11,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    out12=clamp(out12,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out13=clamp(out13,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out14=clamp(out14,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out15=clamp(out15,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
#endif
#ifdef PRELU
    COMPUTE_FLOAT4 slope0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, slope_ptr));
    COMPUTE_FLOAT4 slope1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, slope_ptr));
    out0=select(out0*slope0,out0,out0>=0);out1=select(out1*slope0,out1,out1>=0);out2=select(out2*slope0,out2,out2>=0);out3=select(out3*slope0,out3,out3>=0);
    out4=select(out4*slope0,out4,out4>=0);out5=select(out5*slope0,out5,out5>=0);out6=select(out6*slope0,out6,out6>=0);out7=select(out7*slope0,out7,out7>=0);
    out8=select(out8*slope1,out8,out8>=0);out9=select(out9*slope1,out9,out9>=0);out10=select(out10*slope1,out10,out10>=0);out11=select(out11*slope1,out11,out11>=0);
    out12=select(out12*slope1,out12,out12>=0);out13=select(out13*slope1,out13,out13>=0);out14=select(out14*slope1,out14,out14>=0);out15=select(out15*slope1,out15,out15>=0);
#endif
    int out_offset = (((out_b_idx + out_c_idx_0*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.x - out_h_idx;
                        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    if(remain > 1)      vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    if(remain > 2)      vstore4(CONVERT_FLOAT4(out2), 2*out_hw.y, output+out_offset);
    if(remain > 3)      vstore4(CONVERT_FLOAT4(out3), 3*out_hw.y, output+out_offset);
    if(remain > 4)      vstore4(CONVERT_FLOAT4(out4), 4*out_hw.y, output+out_offset);
    if(remain > 5)      vstore4(CONVERT_FLOAT4(out5), 5*out_hw.y, output+out_offset);
    if(remain > 6)      vstore4(CONVERT_FLOAT4(out6), 6*out_hw.y, output+out_offset);
    if(remain > 7)      vstore4(CONVERT_FLOAT4(out7), 7*out_hw.y, output+out_offset);
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_blocks) return;
    #endif
    out_offset = (((out_b_idx + out_c_idx_1*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
                        vstore4(CONVERT_FLOAT4(out8), 0, output+out_offset);
    if(remain > 1)      vstore4(CONVERT_FLOAT4(out9), out_hw.y, output+out_offset);
    if(remain > 2)      vstore4(CONVERT_FLOAT4(out10), 2*out_hw.y, output+out_offset);
    if(remain > 3)      vstore4(CONVERT_FLOAT4(out11), 3*out_hw.y, output+out_offset);
    if(remain > 4)      vstore4(CONVERT_FLOAT4(out12), 4*out_hw.y, output+out_offset);
    if(remain > 5)      vstore4(CONVERT_FLOAT4(out13), 5*out_hw.y, output+out_offset);
    if(remain > 6)      vstore4(CONVERT_FLOAT4(out14), 6*out_hw.y, output+out_offset);
    if(remain > 7)      vstore4(CONVERT_FLOAT4(out15), 7*out_hw.y, output+out_offset);
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out2), 2*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), 3*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out4), 4*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out5), 5*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out6), 6*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out7), 7*out_hw.y, output+out_offset);
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_blocks) return;
    #endif
    out_offset = (((out_b_idx + out_c_idx_1*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out8), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out9), out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out10), 2*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out11), 3*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out12), 4*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out13), 5*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out14), 6*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out15), 7*out_hw.y, output+out_offset);
#endif
}

#ifdef SUBGROUP_PROBE_BCAST
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel void subgroup_probe_bcast(__global FLOAT* out){
    const uint lid = get_sub_group_local_id();
    FLOAT v = (FLOAT)(lid);
    out[get_global_id(0)] = sub_group_broadcast(v, 0);
}
#endif
#ifdef SUBGROUP_PROBE_SHUFFLE
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
__kernel void subgroup_probe_shuffle(__global FLOAT* out){
    const uint sz = get_sub_group_size();
    const uint lid = get_sub_group_local_id();
    FLOAT v = (FLOAT)(lid);
    out[get_global_id(0)] = sub_group_shuffle(v, (lid + 1) % sz);
}
#endif

// c8h4w1_pa: identical blocking to c8h4w1 (8 out-ch x 4 rows/thread) but each output's
// accumulation is split into TWO partial sums (a: taps x,z ; b: taps y,w) -> two independent
// FMA dependency chains for ILP, summed before activation. Tests idea #2 (break FMA chain).
// Costs 2x accumulator registers (occupancy test). Same dispatch/args as c8h4w1.
__kernel
void conv_2d_c8h4w1_pa(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int batch,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0);
    const int out_b_h_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx_0 = ((out_c_w_idx / out_w_blocks + out_c_base_index) << 1);
    if(out_c_idx_0 >= out_c_blocks) return;
    const int out_c_idx_1 = out_c_idx_0 + 1;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h_blocks;
    const int out_h_idx = (out_b_h_idx % out_h_blocks) << 2;

    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, bias));
    COMPUTE_FLOAT4 out1 = out0, out2 = out0, out3 = out0;
    #ifdef CHANNEL_BOUNDARY_PROTECT
    COMPUTE_FLOAT4 out4 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #else
    COMPUTE_FLOAT4 out4 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #endif
    COMPUTE_FLOAT4 out5 = out4, out6 = out4, out7 = out4;
    // partial-sum accumulators (chain b), start at 0
    COMPUTE_FLOAT4 p0=(COMPUTE_FLOAT4)0,p1=(COMPUTE_FLOAT4)0,p2=(COMPUTE_FLOAT4)0,p3=(COMPUTE_FLOAT4)0;
    COMPUTE_FLOAT4 p4=(COMPUTE_FLOAT4)0,p5=(COMPUTE_FLOAT4)0,p6=(COMPUTE_FLOAT4)0,p7=(COMPUTE_FLOAT4)0;

    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_h0_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    const int in_h1_idx_base = in_h0_idx_base + stride_hw.x;
    const int in_h2_idx_base = in_h1_idx_base + stride_hw.x;
    const int in_h3_idx_base = in_h2_idx_base + stride_hw.x;
    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);
    const int weight_oc_offset = filter_hw.x * filter_hw.y * 4;
    const int weight_ic_offset = out_c_blocks * weight_oc_offset;
    const int in_hw_size = in_hw.x * in_hw.y;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        const int inp_offset_base = (out_b_idx + in_c_idx * batch) * in_hw.x * in_hw.y * 4;
        for(int iy = 0; iy < filter_hw.x; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx_0) *filter_hw.x + iy)*filter_hw.y + kw_start) * 4;
            const int in_h0_idx = (iy * dilate_hw.x + in_h0_idx_base) * in_hw.y;
            const int in_h1_idx = (iy * dilate_hw.x + in_h1_idx_base) * in_hw.y;
            const int in_h2_idx = (iy * dilate_hw.x + in_h2_idx_base) * in_hw.y;
            const int in_h3_idx = (iy * dilate_hw.x + in_h3_idx_base) * in_hw.y;
            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += dilate_hw.y) {
                COMPUTE_FLOAT4 in0 = (in_h0_idx < 0 || in_h0_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h0_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = (in_h1_idx < 0 || in_h1_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h1_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = (in_h2_idx < 0 || in_h2_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h2_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = (in_h3_idx < 0 || in_h3_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h3_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 w0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset));
                COMPUTE_FLOAT4 w1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset));
                COMPUTE_FLOAT4 w2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*2));
                COMPUTE_FLOAT4 w3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*3));
                // split: chain a gets .x,.z ; chain b (p*) gets .y,.w -> two independent chains
                out0 = mad(in0.x,w0,out0); p0 = mad(in0.y,w1,p0); out0 = mad(in0.z,w2,out0); p0 = mad(in0.w,w3,p0);
                out1 = mad(in1.x,w0,out1); p1 = mad(in1.y,w1,p1); out1 = mad(in1.z,w2,out1); p1 = mad(in1.w,w3,p1);
                out2 = mad(in2.x,w0,out2); p2 = mad(in2.y,w1,p2); out2 = mad(in2.z,w2,out2); p2 = mad(in2.w,w3,p2);
                out3 = mad(in3.x,w0,out3); p3 = mad(in3.y,w1,p3); out3 = mad(in3.z,w2,out3); p3 = mad(in3.w,w3,p3);
                #ifdef CHANNEL_BOUNDARY_PROTECT
                w0 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                w1 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                w2 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                w3 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
                #else
                w0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                w1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                w2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                w3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
                #endif
                out4 = mad(in0.x,w0,out4); p4 = mad(in0.y,w1,p4); out4 = mad(in0.z,w2,out4); p4 = mad(in0.w,w3,p4);
                out5 = mad(in1.x,w0,out5); p5 = mad(in1.y,w1,p5); out5 = mad(in1.z,w2,out5); p5 = mad(in1.w,w3,p5);
                out6 = mad(in2.x,w0,out6); p6 = mad(in2.y,w1,p6); out6 = mad(in2.z,w2,out6); p6 = mad(in2.w,w3,p6);
                out7 = mad(in3.x,w0,out7); p7 = mad(in3.y,w1,p7); out7 = mad(in3.z,w2,out7); p7 = mad(in3.w,w3,p7);
                weight_offset += 4;
            }
        }
    }
    out0+=p0; out1+=p1; out2+=p2; out3+=p3; out4+=p4; out5+=p5; out6+=p6; out7+=p7;
#ifdef RELU
    out0=fmax(out0,(COMPUTE_FLOAT4)0);out1=fmax(out1,(COMPUTE_FLOAT4)0);out2=fmax(out2,(COMPUTE_FLOAT4)0);out3=fmax(out3,(COMPUTE_FLOAT4)0);
    out4=fmax(out4,(COMPUTE_FLOAT4)0);out5=fmax(out5,(COMPUTE_FLOAT4)0);out6=fmax(out6,(COMPUTE_FLOAT4)0);out7=fmax(out7,(COMPUTE_FLOAT4)0);
#endif
#ifdef RELU6
    out0=clamp(out0,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out1=clamp(out1,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out2=clamp(out2,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out3=clamp(out3,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    out4=clamp(out4,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out5=clamp(out5,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out6=clamp(out6,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);out7=clamp(out7,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
#endif
#ifdef PRELU
    COMPUTE_FLOAT4 s0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, slope_ptr));
    COMPUTE_FLOAT4 s1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, slope_ptr));
    out0=select(out0*s0,out0,out0>=0);out1=select(out1*s0,out1,out1>=0);out2=select(out2*s0,out2,out2>=0);out3=select(out3*s0,out3,out3>=0);
    out4=select(out4*s1,out4,out4>=0);out5=select(out5*s1,out5,out5>=0);out6=select(out6*s1,out6,out6>=0);out7=select(out7*s1,out7,out7>=0);
#endif
    int out_offset = (((out_b_idx + out_c_idx_0*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.x - out_h_idx;
                   vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    if(remain>1) vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    if(remain>2) vstore4(CONVERT_FLOAT4(out2), 2*out_hw.y, output+out_offset);
    if(remain>3) vstore4(CONVERT_FLOAT4(out3), 3*out_hw.y, output+out_offset);
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_blocks) return;
    #endif
    out_offset = (((out_b_idx + out_c_idx_1*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
                   vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
    if(remain>1) vstore4(CONVERT_FLOAT4(out5), out_hw.y, output+out_offset);
    if(remain>2) vstore4(CONVERT_FLOAT4(out6), 2*out_hw.y, output+out_offset);
    if(remain>3) vstore4(CONVERT_FLOAT4(out7), 3*out_hw.y, output+out_offset);
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out2), 2*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), 3*out_hw.y, output+out_offset);
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_blocks) return;
    #endif
    out_offset = (((out_b_idx + out_c_idx_1*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out5), out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out6), 2*out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out7), 3*out_hw.y, output+out_offset);
#endif
}

// im2col for 3x3 stride-1 pad-1 conv: NC4HW4 [Cin,H,W] -> NC4HW4 [Cin*9,H,W] (Cin mult of 4).
// Output channel c_out = tap*Cin + c_in (tap=kh*3+kw); im2col[c_out,y,x] = input[c_in, y-1+kh, x-1+kw].
// Because Cin is a multiple of 4, tap boundaries align with NC4HW4 channel-blocks, so this is a pure
// float4 gather. Feeds a 1x1 conv (GEMM) that reduces over the Cin*9 channels. Env: MNN_CONV_IM2COL.
__kernel
void im2col_3x3s1(GLOBAL_SIZE_2_DIMS
                  __global const FLOAT *input,
                  __global FLOAT *output,
                  __private const int2 in_hw,      // (H, W)
                  __private const int in_c_blocks,  // Cin/4
                  __private const int batch,
                  __private const int out_w) {      // = W (stride 1)
    const int gid0 = get_global_id(0);   // out_cblock * out_w + x
    const int gid1 = get_global_id(1);   // b * H + y
    DEAL_NON_UNIFORM_DIM2(gid0, gid1);
    const int H = in_hw.x, W = in_hw.y;
    const int x = gid0 % out_w;
    const int out_cblock = gid0 / out_w;
    const int b = gid1 / H;
    const int y = gid1 % H;
    const int tap = out_cblock / in_c_blocks;    // 0..8
    const int cb_in = out_cblock % in_c_blocks;
    const int kh = tap / 3, kw = tap % 3;
    const int iy = y - 1 + kh, ix = x - 1 + kw;
    COMPUTE_FLOAT4 v = (COMPUTE_FLOAT4)0;
    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
        const int in_off = (((cb_in * batch + b) * H + iy) * W + ix) * 4;
        v = CONVERT_COMPUTE_FLOAT4(vload4(0, input + in_off));
    }
    const int out_off = (((out_cblock * batch + b) * H + y) * out_w + x) * 4;
    vstore4(CONVERT_FLOAT4(v), 0, output + out_off);
}

// Fused 2-layer 3x3 s1 pad1 conv (C->C, SAME weights both layers): out = conv(conv(x,W,b),W,b),
// one megakernel, intermediate kept in LDS (never written to global). Tests the fused GEMM-arch
// megakernel hypothesis: one workgroup computes an FT_W x FT_H output tile; input halo (FT+4) and
// intermediate (FT+2) staged in __local. Env MNN_CONV_FUSED2. Reduction matches conv_2d_c4h1w1.
#ifndef FT_W
#define FT_W 6
#endif
#ifndef FT_H
#define FT_H 6
#endif
#ifndef C_BLK
#define C_BLK 8
#endif
__kernel
void conv_2d_3x3s1_fused2(GLOBAL_SIZE_2_DIMS
                          __global const FLOAT *input,
                          __global const FLOAT *weight,
                          __global const FLOAT *bias,
                          __global FLOAT *output,
                          __private const int2 in_hw,    // (H, W)
                          __private const int c_blocks,   // C/4 (== C_BLK)
                          __private const int batch) {
    const int LIN_W = FT_W + 4, LIN_H = FT_H + 4;
    const int LMID_W = FT_W + 2, LMID_H = FT_H + 2;
    __local COMPUTE_FLOAT4 lds_in[(FT_H + 4) * (FT_W + 4) * C_BLK];
    __local COMPUTE_FLOAT4 lds_mid[(FT_H + 2) * (FT_W + 2) * C_BLK];

    const int gx = get_global_id(0);   // ox
    const int gy = get_global_id(1);   // b*H + oy
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int H = in_hw.x, W = in_hw.y;
    const int ox = gx;
    const int b  = gy / H;
    const int oy = gy % H;
    const int oy0 = oy - ly, ox0 = ox - lx;   // workgroup output origin
    const int iy_base = oy0 - 2, ix_base = ox0 - 2;
    const int lid = ly * FT_W + lx;
    const int nthreads = FT_W * FT_H;
    const int woc = c_blocks * 9 * 4;

    // Phase 0: load input halo tile into LDS (zero-pad OOB)
    for (int e = lid; e < LIN_W * LIN_H * C_BLK; e += nthreads) {
        const int cb = e % C_BLK, t = e / C_BLK;
        const int tx = t % LIN_W, ty = t / LIN_W;
        const int iy = iy_base + ty, ix = ix_base + tx;
        COMPUTE_FLOAT4 v = (COMPUTE_FLOAT4)0;
        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
            v = CONVERT_COMPUTE_FLOAT4(vload4(0, input + ((((cb * batch + b) * H + iy) * W + ix) * 4)));
        }
        lds_in[e] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 1: conv1 -> intermediate (LMID tile) in LDS
    for (int e = lid; e < LMID_W * LMID_H * c_blocks; e += nthreads) {
        const int oc = e % c_blocks, t = e / c_blocks;
        const int mx = t % LMID_W, my = t / LMID_W;
        COMPUTE_FLOAT4 acc = CONVERT_COMPUTE_FLOAT4(vload4(oc, bias));
        for (int ic = 0; ic < c_blocks; ic++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    COMPUTE_FLOAT4 in0 = lds_in[((my + kh) * LIN_W + (mx + kw)) * C_BLK + ic];
                    const int wb = (((4 * ic) * c_blocks + oc) * 9 + (kh * 3 + kw)) * 4;
                    acc = mad(in0.x, CONVERT_COMPUTE_FLOAT4(vload4(0, weight + wb)), acc);
                    acc = mad(in0.y, CONVERT_COMPUTE_FLOAT4(vload4(0, weight + wb + woc)), acc);
                    acc = mad(in0.z, CONVERT_COMPUTE_FLOAT4(vload4(0, weight + wb + woc * 2)), acc);
                    acc = mad(in0.w, CONVERT_COMPUTE_FLOAT4(vload4(0, weight + wb + woc * 3)), acc);
                }
            }
        }
        lds_mid[(my * LMID_W + mx) * C_BLK + oc] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: conv2 -> output (this thread's pixel, all oc blocks)
    for (int oc = 0; oc < c_blocks; oc++) {
        COMPUTE_FLOAT4 acc = CONVERT_COMPUTE_FLOAT4(vload4(oc, bias));
        for (int ic = 0; ic < c_blocks; ic++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    COMPUTE_FLOAT4 in0 = lds_mid[((ly + kh) * LMID_W + (lx + kw)) * C_BLK + ic];
                    const int wb = (((4 * ic) * c_blocks + oc) * 9 + (kh * 3 + kw)) * 4;
                    acc = mad(in0.x, CONVERT_COMPUTE_FLOAT4(vload4(0, weight + wb)), acc);
                    acc = mad(in0.y, CONVERT_COMPUTE_FLOAT4(vload4(0, weight + wb + woc)), acc);
                    acc = mad(in0.z, CONVERT_COMPUTE_FLOAT4(vload4(0, weight + wb + woc * 2)), acc);
                    acc = mad(in0.w, CONVERT_COMPUTE_FLOAT4(vload4(0, weight + wb + woc * 3)), acc);
                }
            }
        }
        vstore4(CONVERT_FLOAT4(acc), 0, output + ((((oc * batch + b) * H + oy) * W + ox) * 4));
    }
}
