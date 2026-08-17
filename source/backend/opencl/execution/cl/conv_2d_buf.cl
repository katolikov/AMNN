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

// conv_2d_c4h8w1 (env MNN_CONV_SPEC): best weight-load amortization PER REGISTER in the set.
// The c8h1w1/c4h1w1 measurements showed the kernel is bound by WEIGHT loads (4 weight float4 per
// input float4), which only h/w-blocking amortizes -- and h-blocking keeps the w-coalescing that
// w-blocking destroys. c8h8w1 reached amort 8 but paid 16 accumulators and lost on occupancy;
// this reaches the same amort 8 with only 8 accumulators (one oc-block, 8 rows) = the same
// register class as the c8h4w1 winner at double its amortization.
__kernel
void conv_2d_c4h8w1(GLOBAL_SIZE_2_DIMS
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
    const int out_h_idx = (out_b_h_idx % out_h_blocks) << 3;

    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out1 = out0;
    COMPUTE_FLOAT4 out2 = out0;
    COMPUTE_FLOAT4 out3 = out0;
    COMPUTE_FLOAT4 out4 = out0;
    COMPUTE_FLOAT4 out5 = out0;
    COMPUTE_FLOAT4 out6 = out0;
    COMPUTE_FLOAT4 out7 = out0;

    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_h0_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);

    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);

    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    const int in_hw_size = in_hw.x * in_hw.y;
    const int row_stride = stride_hw.x * in_hw.y;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        const int inp_offset_base = (out_b_idx + in_c_idx*batch) * in_hw.x * in_hw.y * 4;

        for(int iy = 0; iy < filter_hw.x; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + iy)*filter_hw.y + kw_start) * 4;
            const int in_h0_idx = (iy * dilate_hw.x + in_h0_idx_base) * in_hw.y;

            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += dilate_hw.y) {
                const int ib = in_h0_idx + fw;
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset));
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset*2));
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset*3));

                int hidx = ib;
                COMPUTE_FLOAT4 inr = (hidx < 0 || hidx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(hidx, input+inp_offset_base));
                out0 = mad(inr.x, weight0, out0);
                out0 = mad(inr.y, weight1, out0);
                out0 = mad(inr.z, weight2, out0);
                out0 = mad(inr.w, weight3, out0);

                hidx += row_stride;
                inr = (hidx < 0 || hidx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(hidx, input+inp_offset_base));
                out1 = mad(inr.x, weight0, out1);
                out1 = mad(inr.y, weight1, out1);
                out1 = mad(inr.z, weight2, out1);
                out1 = mad(inr.w, weight3, out1);

                hidx += row_stride;
                inr = (hidx < 0 || hidx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(hidx, input+inp_offset_base));
                out2 = mad(inr.x, weight0, out2);
                out2 = mad(inr.y, weight1, out2);
                out2 = mad(inr.z, weight2, out2);
                out2 = mad(inr.w, weight3, out2);

                hidx += row_stride;
                inr = (hidx < 0 || hidx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(hidx, input+inp_offset_base));
                out3 = mad(inr.x, weight0, out3);
                out3 = mad(inr.y, weight1, out3);
                out3 = mad(inr.z, weight2, out3);
                out3 = mad(inr.w, weight3, out3);

                hidx += row_stride;
                inr = (hidx < 0 || hidx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(hidx, input+inp_offset_base));
                out4 = mad(inr.x, weight0, out4);
                out4 = mad(inr.y, weight1, out4);
                out4 = mad(inr.z, weight2, out4);
                out4 = mad(inr.w, weight3, out4);

                hidx += row_stride;
                inr = (hidx < 0 || hidx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(hidx, input+inp_offset_base));
                out5 = mad(inr.x, weight0, out5);
                out5 = mad(inr.y, weight1, out5);
                out5 = mad(inr.z, weight2, out5);
                out5 = mad(inr.w, weight3, out5);

                hidx += row_stride;
                inr = (hidx < 0 || hidx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(hidx, input+inp_offset_base));
                out6 = mad(inr.x, weight0, out6);
                out6 = mad(inr.y, weight1, out6);
                out6 = mad(inr.z, weight2, out6);
                out6 = mad(inr.w, weight3, out6);

                hidx += row_stride;
                inr = (hidx < 0 || hidx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(hidx, input+inp_offset_base));
                out7 = mad(inr.x, weight0, out7);
                out7 = mad(inr.y, weight1, out7);
                out7 = mad(inr.z, weight2, out7);
                out7 = mad(inr.w, weight3, out7);

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
    COMPUTE_FLOAT4 slope_in = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
    out0 = select(out0 * slope_in, out0, out0 >= 0);
    out1 = select(out1 * slope_in, out1, out1 >= 0);
    out2 = select(out2 * slope_in, out2, out2 >= 0);
    out3 = select(out3 * slope_in, out3, out3 >= 0);
    out4 = select(out4 * slope_in, out4, out4 >= 0);
    out5 = select(out5 * slope_in, out5, out5 >= 0);
    out6 = select(out6 * slope_in, out6, out6 >= 0);
    out7 = select(out7 * slope_in, out7, out7 >= 0);
#endif

    const int out_offset = (((out_b_idx + out_c_idx*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.x - out_h_idx;
    if(remain > 0) vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    if(remain > 1) vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    if(remain > 2) vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
    if(remain > 3) vstore4(CONVERT_FLOAT4(out3), 3 * out_hw.y, output+out_offset);
    if(remain > 4) vstore4(CONVERT_FLOAT4(out4), 4 * out_hw.y, output+out_offset);
    if(remain > 5) vstore4(CONVERT_FLOAT4(out5), 5 * out_hw.y, output+out_offset);
    if(remain > 6) vstore4(CONVERT_FLOAT4(out6), 6 * out_hw.y, output+out_offset);
    if(remain > 7) vstore4(CONVERT_FLOAT4(out7), 7 * out_hw.y, output+out_offset);
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), 3 * out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out4), 4 * out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out5), 5 * out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out6), 6 * out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out7), 7 * out_hw.y, output+out_offset);
#endif
}

// conv_2d_c8h1w1 (env MNN_CONV_SPEC): the naive-but-max-parallel point that was missing from the
// candidate set. 2 accumulators (same register class as c4h1w2 / lowest of the c8 family) but the
// HIGHEST thread count of any 2-acc variant, and the best input-load-per-output ratio: each input
// float4 is loaded once and reused across BOTH output channel blocks with zero halo growth
// (4.5 loads/out-float4 vs 9 for c4h1w1 and 6 for c4h1w2). Reduction order matches conv_2d_c8h2w1.
__kernel
void conv_2d_c8h1w1(GLOBAL_SIZE_2_DIMS
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
    const int out_c_w_idx = get_global_id(0); //c/8 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx_0 = (out_c_w_idx / out_w_blocks + out_c_base_index) << 1;
    if(out_c_idx_0 >= out_c_blocks) return;
    const int out_c_idx_1 = out_c_idx_0 + 1;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h_blocks;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h_blocks;

    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, bias));
    #ifdef CHANNEL_BOUNDARY_PROTECT
    COMPUTE_FLOAT4 out1 = out_c_idx_1 >= out_c_blocks ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #else
    COMPUTE_FLOAT4 out1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #endif

    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);

    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);

    const int weight_oc_offset = filter_hw.x * filter_hw.y * 4;
    const int weight_ic_offset = out_c_blocks * weight_oc_offset;
    const int in_hw_size = in_hw.x * in_hw.y;
    // weight: [ic/4, oc, 4], loop: ic/4
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        const int inp_offset_base = (out_b_idx + in_c_idx*batch) * in_hw.x * in_hw.y * 4;

        for(int iy = 0; iy < filter_hw.x; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx_0) *filter_hw.x + iy)*filter_hw.y + kw_start) * 4;
            const int in_h_idx = (iy * dilate_hw.x + in_h_idx_base) * in_hw.y;

            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += dilate_hw.y) {
                COMPUTE_FLOAT4 in0 = (in_h_idx < 0 || in_h_idx >= in_hw_size) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_h_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset));
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset));
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*2));
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_ic_offset*3));

                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);

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
                out1 = mad(in0.x, weight0, out1);
                out1 = mad(in0.y, weight1, out1);
                out1 = mad(in0.z, weight2, out1);
                out1 = mad(in0.w, weight3, out1);

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
    COMPUTE_FLOAT4 slope_in0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, slope_ptr));
    COMPUTE_FLOAT4 slope_in1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, slope_ptr));
    out0 = select(out0 * slope_in0, out0, out0 >= 0);
    out1 = select(out1 * slope_in1, out1, out1 >= 0);
#endif

    int out_offset = (((out_b_idx + out_c_idx_0*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
#ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= out_c_blocks){
        return;
    }
#endif
    out_offset = (((out_b_idx + out_c_idx_1*batch)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out1), 0, output+out_offset);
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

// ---------------------------------------------------------------------------------------------
// conv_2d_3x3s1_lds_w2 — LDS staging at c4h1w2 blocking (env MNN_CONV_LDS=w2).
//
// The plain conv_2d_3x3s1_lds above is 1 output/thread and lost by 1.78x. That left one
// confound open: it changed TWO things at once against the stock kernels (added LDS, and
// dropped the register blocking from ~8 outputs/thread to 1). This kernel isolates LDS at
// CONSTANT blocking - it is conv_2d_c4h1w2 (4 out-channels x 2 adjacent columns, same 2
// accumulators, same reduction order) with the input read from LDS instead of global.
// Comparing it against conv_2d_c4h1w2 leaves LDS as the only difference.
//
// One workgroup = (2*TILE_W) x TILE_H outputs for one output channel-block; the
// (2*TILE_W+2) x (TILE_H+2) input halo is staged once per input channel-block.
// Requires 3x3, stride 1, dilation 1, pad 1, out_w % (2*TILE_W) == 0, out_h % TILE_H == 0.
#define LDSW2_W (2 * TILE_W + 2)
#define LDSW2_H (TILE_H + 2)
__kernel
void conv_2d_3x3s1_lds_w2(GLOBAL_SIZE_2_DIMS
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
    __local COMPUTE_FLOAT4 lds[LDSW2_H * LDSW2_W];

    const int gx = get_global_id(0);   // out_c_idx * (out_w/2) + w_block
    const int gy = get_global_id(1);   // out_b_idx * out_h + out_y
    const int lx = get_local_id(0);    // 0..TILE_W-1
    const int ly = get_local_id(1);    // 0..TILE_H-1

    const int out_w = out_hw.y;
    const int out_h = out_hw.x;
    const int in_w  = in_hw.y;
    const int in_h  = in_hw.x;

    const int w_blocks  = out_w >> 1;
    const int out_c_idx = gx / w_blocks;
    const int out_x     = (gx % w_blocks) << 1;   // first of the two columns
    const int out_b_idx = gy / out_h;
    const int out_y     = gy % out_h;

    // input halo origin for this workgroup (stride 1): the thread at (lx,ly) owns output
    // columns out_x, out_x+1, so the group starts 2*lx columns to the left.
    const int in_base_x = (out_x - (lx << 1)) - pad_hw.y;
    const int in_base_y = (out_y - ly) - pad_hw.x;

    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out1 = out0;

    const int weight_oc_offset = out_c_blocks * 9 * 4;
    const int lid        = ly * TILE_W + lx;
    const int nthreads   = TILE_W * TILE_H;
    const int tile_elems = LDSW2_W * LDSW2_H;

    for (int ic = 0; ic < in_c_blocks; ic++) {
        for (int e = lid; e < tile_elems; e += nthreads) {
            const int ty = e / LDSW2_W;
            const int tx = e % LDSW2_W;
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

        const int w_base = (((4 * ic) * out_c_blocks + out_c_idx) * 9) * 4;
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                const int lrow = (ly + kh) * LDSW2_W + (lx << 1) + kw;
                COMPUTE_FLOAT4 in0 = lds[lrow];
                COMPUTE_FLOAT4 in1 = lds[lrow + 1];
                const int w_off = w_base + (kh * 3 + kw) * 4;
                COMPUTE_FLOAT4 w0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_off));
                COMPUTE_FLOAT4 w1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_off + weight_oc_offset));
                COMPUTE_FLOAT4 w2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_off + weight_oc_offset * 2));
                COMPUTE_FLOAT4 w3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_off + weight_oc_offset * 3));
                out0 = mad(in0.x, w0, out0);
                out0 = mad(in0.y, w1, out0);
                out0 = mad(in0.z, w2, out0);
                out0 = mad(in0.w, w3, out0);
                out1 = mad(in1.x, w0, out1);
                out1 = mad(in1.y, w1, out1);
                out1 = mad(in1.z, w2, out1);
                out1 = mad(in1.w, w3, out1);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
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

    const int out_offset = (((out_b_idx + out_c_idx * batch) * out_h + out_y) * out_w + out_x) * 4;
    vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output + out_offset);
}

// c8h8w1: register-blocked 3x3 variant, 8 output channels x 8 rows/thread = 16 accumulators
// (2x the tile height of c8h4w1). More work-in-flight + more input-row reuse; tests whether the
// autotuner's 8-accumulator optimum is the true optimum. Same NC4HW4 index math as c8h4w1.

// ---------------------------------------------------------------------------------------------
// conv_2d_3x3s2_lds — LDS-staged 3x3 STRIDE-2 conv (env MNN_CONV_LDS on a stride-2 conv).
//
// Why this exists when the stride-1 LDS kernel was falsified: at stride 1 neighbouring threads
// read neighbouring input pixels, so global access is already fully coalesced and LDS bought
// nothing (the reuse was L2-served). At STRIDE 2 neighbouring threads read pixels TWO apart, so
// every cache line is only half used and the request count per useful byte doubles. Staging a
// CONTIGUOUS input tile in LDS restores full coalescing — a problem that only exists at stride 2.
//
// One workgroup = TILE_W x TILE_H outputs for one output channel-block; the (2*TILE_W+1) x
// (2*TILE_H+1) input patch is loaded once per input channel-block. Reduction order matches
// conv_2d_c4h1w1. Requires 3x3, stride 2, dilation 1, pad 1, out_w%TILE_W==0, out_h%TILE_H==0.
#define LDS2_W (2 * TILE_W + 1)
#define LDS2_H (2 * TILE_H + 1)
__kernel __attribute__((reqd_work_group_size(TILE_W, TILE_H, 1)))
void conv_2d_3x3s2_lds(GLOBAL_SIZE_2_DIMS
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
    __local COMPUTE_FLOAT4 lds[LDS2_H * LDS2_W];

    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const int out_w = out_hw.y;
    const int out_h = out_hw.x;
    const int in_w  = in_hw.y;
    const int in_h  = in_hw.x;

    const int out_c_idx = gx / out_w;
    const int out_x     = gx % out_w;
    const int out_b_idx = gy / out_h;
    const int out_y     = gy % out_h;

    // input origin of this workgroup's halo (stride 2)
    const int in_base_x = 2 * (out_x - lx) - pad_hw.y;
    const int in_base_y = 2 * (out_y - ly) - pad_hw.x;

    COMPUTE_FLOAT4 acc = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));

    const int weight_oc_offset = out_c_blocks * 9 * 4;
    const int lid        = ly * TILE_W + lx;
    const int nthreads   = TILE_W * TILE_H;
    const int tile_elems = LDS2_W * LDS2_H;

    for (int ic = 0; ic < in_c_blocks; ic++) {
        // cooperative, CONTIGUOUS load -> full cache-line utilisation (the whole point)
        for (int e = lid; e < tile_elems; e += nthreads) {
            const int ty = e / LDS2_W;
            const int tx = e % LDS2_W;
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

        const int w_base = (((4 * ic) * out_c_blocks + out_c_idx) * 9) * 4;
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                COMPUTE_FLOAT4 in0 = lds[(2 * ly + kh) * LDS2_W + (2 * lx + kw)];
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
    {
        COMPUTE_FLOAT4 slope_in = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
        acc = select(acc * slope_in, acc, acc >= 0);
    }
#endif
    const int out_offset = (((out_b_idx + out_c_idx * batch) * out_h + out_y) * out_w + out_x) * 4;
    vstore4(CONVERT_FLOAT4(acc), 0, output + out_offset);
}


// ---- shape access: runtime args by default, compile-time constants under MNN_CONV_HARD ----
//
// The HOST supplies every HC* name as a -D build option (ConvBufExecution's hcPut): the runtime
// expression by default, a literal under MNN_CONV_HARD=1. The fallbacks below are ONLY for kernels
// that are built outside that path (conv_2d_3x3s1_fused2 assembles its own build options), so each
// one must be guarded -- an unguarded #define here silently REDEFINES the host's -D and wins,
// because the .cl text is processed after the command-line macros.
//
// That is exactly the bug that made MNN_CONV_HARD a no-op for the four 2-D tile kernels between
// §H.26 and §H.34: the guard used to be `#ifdef HC_IN_H`, but §H.26 renamed the host's defines to
// HCINH/HCINW/... (no underscores), so the guard never fired, the #else branch ran, and it
// overwrote the literals with the runtime expressions. Cost: the -18.7% win read as -4%.
#ifndef HCINH
  #define HCINH   in_hw.x
#endif
#ifndef HCINW
  #define HCINW   in_hw.y
#endif
#ifndef HCOUTH
  #define HCOUTH  out_hw.x
#endif
#ifndef HCOUTW
  #define HCOUTW  out_hw.y
#endif
#ifndef HCICB
  #define HCICB   in_c_blocks
#endif
#ifndef HCOCB
  #define HCOCB   out_c_blocks
#endif
#ifndef HCBATCH
  #define HCBATCH batch
#endif
#ifndef HCWB
  #define HCWB    out_w_blocks
#endif
#ifndef HCHB
  #define HCHB    out_h_blocks
#endif
#ifdef HC_UNROLL_IC
  #define HCUNROLL __attribute__((opencl_unroll_hint))
#else
  #define HCUNROLL
#endif

// conv_2d_c4h4w2 (env MNN_CONV_SPEC, stride-1 only): 2-D register tile, 4x2 outputs.
// With MNN_CONV_HARD=1 the host also passes -DHC_* so every shape value becomes a COMPILE-TIME
// constant: the channel loop unrolls, all index arithmetic constant-folds, and the halo bounds
// checks collapse wherever the tile is provably interior. Costs one program build per shape.
__kernel
void conv_2d_c4h4w2(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input, __global const FLOAT *weight,
                      __global const FLOAT *bias, __global FLOAT *output,
                      __private const int2 in_hw, __private const int inChannel,
                      __private const int in_c_blocks, __private const int batch,
                      __private const int2 out_hw, __private const int2 filter_hw,
                      __private const int2 stride_hw, __private const int2 pad_hw,
                      __private const int2 dilate_hw, __private const int out_w_blocks,
                      __private const int out_c_blocks, __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0);
    const int out_b_h_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);
    const int out_c_idx = out_c_w_idx / HCWB + out_c_base_index;
    if(out_c_idx >= HCOCB) return;
    const int out_w_idx = (out_c_w_idx % HCWB) * 2;
    const int out_b_idx = out_b_h_idx / HCHB;
    const int out_h_idx = (out_b_h_idx % HCHB) * 4;
    COMPUTE_FLOAT4 bv = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 o0_0=bv, o0_1=bv, o1_0=bv, o1_1=bv, o2_0=bv, o2_1=bv, o3_0=bv, o3_1=bv;
    const int in_x0 = out_w_idx - pad_hw.y;
    const int in_y0 = out_h_idx - pad_hw.x;
    const int weight_oc_offset = HCOCB * 9 * 4;
    HCUNROLL
    for(ushort ic = 0; ic < HCICB; ic++) {
        const int inp_base = (out_b_idx + ic * HCBATCH) * HCINH * HCINW * 4;
        const int w_base = (((4 * ic) * HCOCB + out_c_idx) * 9) * 4;
        { const int iy = in_y0 + 0;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
          }
        }
        { const int iy = in_y0 + 1;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
          }
        }
        { const int iy = in_y0 + 2;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v0.x, k0, o2_0);
            o2_0 = mad(v0.y, k1, o2_0);
            o2_0 = mad(v0.z, k2, o2_0);
            o2_0 = mad(v0.w, k3, o2_0);
            o2_1 = mad(v1.x, k0, o2_1);
            o2_1 = mad(v1.y, k1, o2_1);
            o2_1 = mad(v1.z, k2, o2_1);
            o2_1 = mad(v1.w, k3, o2_1);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v1.x, k0, o2_0);
            o2_0 = mad(v1.y, k1, o2_0);
            o2_0 = mad(v1.z, k2, o2_0);
            o2_0 = mad(v1.w, k3, o2_0);
            o2_1 = mad(v2.x, k0, o2_1);
            o2_1 = mad(v2.y, k1, o2_1);
            o2_1 = mad(v2.z, k2, o2_1);
            o2_1 = mad(v2.w, k3, o2_1);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v2.x, k0, o2_0);
            o2_0 = mad(v2.y, k1, o2_0);
            o2_0 = mad(v2.z, k2, o2_0);
            o2_0 = mad(v2.w, k3, o2_0);
            o2_1 = mad(v3.x, k0, o2_1);
            o2_1 = mad(v3.y, k1, o2_1);
            o2_1 = mad(v3.z, k2, o2_1);
            o2_1 = mad(v3.w, k3, o2_1);
          }
        }
        { const int iy = in_y0 + 3;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v0.x, k0, o2_0);
            o2_0 = mad(v0.y, k1, o2_0);
            o2_0 = mad(v0.z, k2, o2_0);
            o2_0 = mad(v0.w, k3, o2_0);
            o2_1 = mad(v1.x, k0, o2_1);
            o2_1 = mad(v1.y, k1, o2_1);
            o2_1 = mad(v1.z, k2, o2_1);
            o2_1 = mad(v1.w, k3, o2_1);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v1.x, k0, o2_0);
            o2_0 = mad(v1.y, k1, o2_0);
            o2_0 = mad(v1.z, k2, o2_0);
            o2_0 = mad(v1.w, k3, o2_0);
            o2_1 = mad(v2.x, k0, o2_1);
            o2_1 = mad(v2.y, k1, o2_1);
            o2_1 = mad(v2.z, k2, o2_1);
            o2_1 = mad(v2.w, k3, o2_1);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v2.x, k0, o2_0);
            o2_0 = mad(v2.y, k1, o2_0);
            o2_0 = mad(v2.z, k2, o2_0);
            o2_0 = mad(v2.w, k3, o2_0);
            o2_1 = mad(v3.x, k0, o2_1);
            o2_1 = mad(v3.y, k1, o2_1);
            o2_1 = mad(v3.z, k2, o2_1);
            o2_1 = mad(v3.w, k3, o2_1);
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v0.x, k0, o3_0);
            o3_0 = mad(v0.y, k1, o3_0);
            o3_0 = mad(v0.z, k2, o3_0);
            o3_0 = mad(v0.w, k3, o3_0);
            o3_1 = mad(v1.x, k0, o3_1);
            o3_1 = mad(v1.y, k1, o3_1);
            o3_1 = mad(v1.z, k2, o3_1);
            o3_1 = mad(v1.w, k3, o3_1);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v1.x, k0, o3_0);
            o3_0 = mad(v1.y, k1, o3_0);
            o3_0 = mad(v1.z, k2, o3_0);
            o3_0 = mad(v1.w, k3, o3_0);
            o3_1 = mad(v2.x, k0, o3_1);
            o3_1 = mad(v2.y, k1, o3_1);
            o3_1 = mad(v2.z, k2, o3_1);
            o3_1 = mad(v2.w, k3, o3_1);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v2.x, k0, o3_0);
            o3_0 = mad(v2.y, k1, o3_0);
            o3_0 = mad(v2.z, k2, o3_0);
            o3_0 = mad(v2.w, k3, o3_0);
            o3_1 = mad(v3.x, k0, o3_1);
            o3_1 = mad(v3.y, k1, o3_1);
            o3_1 = mad(v3.z, k2, o3_1);
            o3_1 = mad(v3.w, k3, o3_1);
          }
        }
        { const int iy = in_y0 + 4;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v0.x, k0, o2_0);
            o2_0 = mad(v0.y, k1, o2_0);
            o2_0 = mad(v0.z, k2, o2_0);
            o2_0 = mad(v0.w, k3, o2_0);
            o2_1 = mad(v1.x, k0, o2_1);
            o2_1 = mad(v1.y, k1, o2_1);
            o2_1 = mad(v1.z, k2, o2_1);
            o2_1 = mad(v1.w, k3, o2_1);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v1.x, k0, o2_0);
            o2_0 = mad(v1.y, k1, o2_0);
            o2_0 = mad(v1.z, k2, o2_0);
            o2_0 = mad(v1.w, k3, o2_0);
            o2_1 = mad(v2.x, k0, o2_1);
            o2_1 = mad(v2.y, k1, o2_1);
            o2_1 = mad(v2.z, k2, o2_1);
            o2_1 = mad(v2.w, k3, o2_1);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v2.x, k0, o2_0);
            o2_0 = mad(v2.y, k1, o2_0);
            o2_0 = mad(v2.z, k2, o2_0);
            o2_0 = mad(v2.w, k3, o2_0);
            o2_1 = mad(v3.x, k0, o2_1);
            o2_1 = mad(v3.y, k1, o2_1);
            o2_1 = mad(v3.z, k2, o2_1);
            o2_1 = mad(v3.w, k3, o2_1);
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v0.x, k0, o3_0);
            o3_0 = mad(v0.y, k1, o3_0);
            o3_0 = mad(v0.z, k2, o3_0);
            o3_0 = mad(v0.w, k3, o3_0);
            o3_1 = mad(v1.x, k0, o3_1);
            o3_1 = mad(v1.y, k1, o3_1);
            o3_1 = mad(v1.z, k2, o3_1);
            o3_1 = mad(v1.w, k3, o3_1);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v1.x, k0, o3_0);
            o3_0 = mad(v1.y, k1, o3_0);
            o3_0 = mad(v1.z, k2, o3_0);
            o3_0 = mad(v1.w, k3, o3_0);
            o3_1 = mad(v2.x, k0, o3_1);
            o3_1 = mad(v2.y, k1, o3_1);
            o3_1 = mad(v2.z, k2, o3_1);
            o3_1 = mad(v2.w, k3, o3_1);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v2.x, k0, o3_0);
            o3_0 = mad(v2.y, k1, o3_0);
            o3_0 = mad(v2.z, k2, o3_0);
            o3_0 = mad(v2.w, k3, o3_0);
            o3_1 = mad(v3.x, k0, o3_1);
            o3_1 = mad(v3.y, k1, o3_1);
            o3_1 = mad(v3.z, k2, o3_1);
            o3_1 = mad(v3.w, k3, o3_1);
          }
        }
        { const int iy = in_y0 + 5;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v0.x, k0, o3_0);
            o3_0 = mad(v0.y, k1, o3_0);
            o3_0 = mad(v0.z, k2, o3_0);
            o3_0 = mad(v0.w, k3, o3_0);
            o3_1 = mad(v1.x, k0, o3_1);
            o3_1 = mad(v1.y, k1, o3_1);
            o3_1 = mad(v1.z, k2, o3_1);
            o3_1 = mad(v1.w, k3, o3_1);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v1.x, k0, o3_0);
            o3_0 = mad(v1.y, k1, o3_0);
            o3_0 = mad(v1.z, k2, o3_0);
            o3_0 = mad(v1.w, k3, o3_0);
            o3_1 = mad(v2.x, k0, o3_1);
            o3_1 = mad(v2.y, k1, o3_1);
            o3_1 = mad(v2.z, k2, o3_1);
            o3_1 = mad(v2.w, k3, o3_1);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v2.x, k0, o3_0);
            o3_0 = mad(v2.y, k1, o3_0);
            o3_0 = mad(v2.z, k2, o3_0);
            o3_0 = mad(v2.w, k3, o3_0);
            o3_1 = mad(v3.x, k0, o3_1);
            o3_1 = mad(v3.y, k1, o3_1);
            o3_1 = mad(v3.z, k2, o3_1);
            o3_1 = mad(v3.w, k3, o3_1);
          }
        }
    }
#ifdef RELU
    o0_0 = fmax(o0_0,(COMPUTE_FLOAT4)0);
    o0_1 = fmax(o0_1,(COMPUTE_FLOAT4)0);
    o1_0 = fmax(o1_0,(COMPUTE_FLOAT4)0);
    o1_1 = fmax(o1_1,(COMPUTE_FLOAT4)0);
    o2_0 = fmax(o2_0,(COMPUTE_FLOAT4)0);
    o2_1 = fmax(o2_1,(COMPUTE_FLOAT4)0);
    o3_0 = fmax(o3_0,(COMPUTE_FLOAT4)0);
    o3_1 = fmax(o3_1,(COMPUTE_FLOAT4)0);
#endif
#ifdef PRELU
    { COMPUTE_FLOAT4 sl = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
      o0_0 = select(o0_0*sl,o0_0,o0_0>=0);
      o0_1 = select(o0_1*sl,o0_1,o0_1>=0);
      o1_0 = select(o1_0*sl,o1_0,o1_0>=0);
      o1_1 = select(o1_1*sl,o1_1,o1_1>=0);
      o2_0 = select(o2_0*sl,o2_0,o2_0>=0);
      o2_1 = select(o2_1*sl,o2_1,o2_1>=0);
      o3_0 = select(o3_0*sl,o3_0,o3_0>=0);
      o3_1 = select(o3_1*sl,o3_1,o3_1>=0);
    }
#endif
    const int base = (((out_b_idx + out_c_idx * HCBATCH) * HCOUTH + out_h_idx) * HCOUTW + out_w_idx) * 4;
    const int rh = HCOUTH - out_h_idx; const int rw = HCOUTW - out_w_idx;
    if(0 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o0_0), 0, output + base + (0*HCOUTW + 0)*4);
    if(0 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o0_1), 0, output + base + (0*HCOUTW + 1)*4);
    if(1 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o1_0), 0, output + base + (1*HCOUTW + 0)*4);
    if(1 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o1_1), 0, output + base + (1*HCOUTW + 1)*4);
    if(2 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o2_0), 0, output + base + (2*HCOUTW + 0)*4);
    if(2 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o2_1), 0, output + base + (2*HCOUTW + 1)*4);
    if(3 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o3_0), 0, output + base + (3*HCOUTW + 0)*4);
    if(3 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o3_1), 0, output + base + (3*HCOUTW + 1)*4);
}

// conv_2d_c4h2w2 (env MNN_CONV_SPEC, stride-1 only): 2-D register tile, 2x2 outputs.
// With MNN_CONV_HARD=1 the host also passes -DHC_* so every shape value becomes a COMPILE-TIME
// constant: the channel loop unrolls, all index arithmetic constant-folds, and the halo bounds
// checks collapse wherever the tile is provably interior. Costs one program build per shape.
__kernel
void conv_2d_c4h2w2(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input, __global const FLOAT *weight,
                      __global const FLOAT *bias, __global FLOAT *output,
                      __private const int2 in_hw, __private const int inChannel,
                      __private const int in_c_blocks, __private const int batch,
                      __private const int2 out_hw, __private const int2 filter_hw,
                      __private const int2 stride_hw, __private const int2 pad_hw,
                      __private const int2 dilate_hw, __private const int out_w_blocks,
                      __private const int out_c_blocks, __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0);
    const int out_b_h_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);
    const int out_c_idx = out_c_w_idx / HCWB + out_c_base_index;
    if(out_c_idx >= HCOCB) return;
    const int out_w_idx = (out_c_w_idx % HCWB) * 2;
    const int out_b_idx = out_b_h_idx / HCHB;
    const int out_h_idx = (out_b_h_idx % HCHB) * 2;
    COMPUTE_FLOAT4 bv = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 o0_0=bv, o0_1=bv, o1_0=bv, o1_1=bv;
    const int in_x0 = out_w_idx - pad_hw.y;
    const int in_y0 = out_h_idx - pad_hw.x;
    const int weight_oc_offset = HCOCB * 9 * 4;
    HCUNROLL
    for(ushort ic = 0; ic < HCICB; ic++) {
        const int inp_base = (out_b_idx + ic * HCBATCH) * HCINH * HCINW * 4;
        const int w_base = (((4 * ic) * HCOCB + out_c_idx) * 9) * 4;
        { const int iy = in_y0 + 0;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
          }
        }
        { const int iy = in_y0 + 1;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
          }
        }
        { const int iy = in_y0 + 2;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
          }
        }
        { const int iy = in_y0 + 3;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
          }
        }
    }
#ifdef RELU
    o0_0 = fmax(o0_0,(COMPUTE_FLOAT4)0);
    o0_1 = fmax(o0_1,(COMPUTE_FLOAT4)0);
    o1_0 = fmax(o1_0,(COMPUTE_FLOAT4)0);
    o1_1 = fmax(o1_1,(COMPUTE_FLOAT4)0);
#endif
#ifdef PRELU
    { COMPUTE_FLOAT4 sl = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
      o0_0 = select(o0_0*sl,o0_0,o0_0>=0);
      o0_1 = select(o0_1*sl,o0_1,o0_1>=0);
      o1_0 = select(o1_0*sl,o1_0,o1_0>=0);
      o1_1 = select(o1_1*sl,o1_1,o1_1>=0);
    }
#endif
    const int base = (((out_b_idx + out_c_idx * HCBATCH) * HCOUTH + out_h_idx) * HCOUTW + out_w_idx) * 4;
    const int rh = HCOUTH - out_h_idx; const int rw = HCOUTW - out_w_idx;
    if(0 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o0_0), 0, output + base + (0*HCOUTW + 0)*4);
    if(0 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o0_1), 0, output + base + (0*HCOUTW + 1)*4);
    if(1 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o1_0), 0, output + base + (1*HCOUTW + 0)*4);
    if(1 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o1_1), 0, output + base + (1*HCOUTW + 1)*4);
}

// conv_2d_c4h2w4 (env MNN_CONV_SPEC, stride-1 only): 2-D register tile, 2x4 outputs.
// With MNN_CONV_HARD=1 the host also passes -DHC_* so every shape value becomes a COMPILE-TIME
// constant: the channel loop unrolls, all index arithmetic constant-folds, and the halo bounds
// checks collapse wherever the tile is provably interior. Costs one program build per shape.
__kernel
void conv_2d_c4h2w4(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input, __global const FLOAT *weight,
                      __global const FLOAT *bias, __global FLOAT *output,
                      __private const int2 in_hw, __private const int inChannel,
                      __private const int in_c_blocks, __private const int batch,
                      __private const int2 out_hw, __private const int2 filter_hw,
                      __private const int2 stride_hw, __private const int2 pad_hw,
                      __private const int2 dilate_hw, __private const int out_w_blocks,
                      __private const int out_c_blocks, __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0);
    const int out_b_h_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);
    const int out_c_idx = out_c_w_idx / HCWB + out_c_base_index;
    if(out_c_idx >= HCOCB) return;
    const int out_w_idx = (out_c_w_idx % HCWB) * 4;
    const int out_b_idx = out_b_h_idx / HCHB;
    const int out_h_idx = (out_b_h_idx % HCHB) * 2;
    COMPUTE_FLOAT4 bv = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 o0_0=bv, o0_1=bv, o0_2=bv, o0_3=bv, o1_0=bv, o1_1=bv, o1_2=bv, o1_3=bv;
    const int in_x0 = out_w_idx - pad_hw.y;
    const int in_y0 = out_h_idx - pad_hw.x;
    const int weight_oc_offset = HCOCB * 9 * 4;
    HCUNROLL
    for(ushort ic = 0; ic < HCICB; ic++) {
        const int inp_base = (out_b_idx + ic * HCBATCH) * HCINH * HCINW * 4;
        const int w_base = (((4 * ic) * HCOCB + out_c_idx) * 9) * 4;
        { const int iy = in_y0 + 0;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0, v4=(COMPUTE_FLOAT4)0, v5=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
            if(in_x0+4 >= 0 && in_x0+4 < HCINW) v4 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+4)*4));
            if(in_x0+5 >= 0 && in_x0+5 < HCINW) v5 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+5)*4));
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
            o0_2 = mad(v2.x, k0, o0_2);
            o0_2 = mad(v2.y, k1, o0_2);
            o0_2 = mad(v2.z, k2, o0_2);
            o0_2 = mad(v2.w, k3, o0_2);
            o0_3 = mad(v3.x, k0, o0_3);
            o0_3 = mad(v3.y, k1, o0_3);
            o0_3 = mad(v3.z, k2, o0_3);
            o0_3 = mad(v3.w, k3, o0_3);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
            o0_2 = mad(v3.x, k0, o0_2);
            o0_2 = mad(v3.y, k1, o0_2);
            o0_2 = mad(v3.z, k2, o0_2);
            o0_2 = mad(v3.w, k3, o0_2);
            o0_3 = mad(v4.x, k0, o0_3);
            o0_3 = mad(v4.y, k1, o0_3);
            o0_3 = mad(v4.z, k2, o0_3);
            o0_3 = mad(v4.w, k3, o0_3);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
            o0_2 = mad(v4.x, k0, o0_2);
            o0_2 = mad(v4.y, k1, o0_2);
            o0_2 = mad(v4.z, k2, o0_2);
            o0_2 = mad(v4.w, k3, o0_2);
            o0_3 = mad(v5.x, k0, o0_3);
            o0_3 = mad(v5.y, k1, o0_3);
            o0_3 = mad(v5.z, k2, o0_3);
            o0_3 = mad(v5.w, k3, o0_3);
          }
        }
        { const int iy = in_y0 + 1;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0, v4=(COMPUTE_FLOAT4)0, v5=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
            if(in_x0+4 >= 0 && in_x0+4 < HCINW) v4 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+4)*4));
            if(in_x0+5 >= 0 && in_x0+5 < HCINW) v5 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+5)*4));
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
            o0_2 = mad(v2.x, k0, o0_2);
            o0_2 = mad(v2.y, k1, o0_2);
            o0_2 = mad(v2.z, k2, o0_2);
            o0_2 = mad(v2.w, k3, o0_2);
            o0_3 = mad(v3.x, k0, o0_3);
            o0_3 = mad(v3.y, k1, o0_3);
            o0_3 = mad(v3.z, k2, o0_3);
            o0_3 = mad(v3.w, k3, o0_3);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
            o0_2 = mad(v3.x, k0, o0_2);
            o0_2 = mad(v3.y, k1, o0_2);
            o0_2 = mad(v3.z, k2, o0_2);
            o0_2 = mad(v3.w, k3, o0_2);
            o0_3 = mad(v4.x, k0, o0_3);
            o0_3 = mad(v4.y, k1, o0_3);
            o0_3 = mad(v4.z, k2, o0_3);
            o0_3 = mad(v4.w, k3, o0_3);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
            o0_2 = mad(v4.x, k0, o0_2);
            o0_2 = mad(v4.y, k1, o0_2);
            o0_2 = mad(v4.z, k2, o0_2);
            o0_2 = mad(v4.w, k3, o0_2);
            o0_3 = mad(v5.x, k0, o0_3);
            o0_3 = mad(v5.y, k1, o0_3);
            o0_3 = mad(v5.z, k2, o0_3);
            o0_3 = mad(v5.w, k3, o0_3);
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
            o1_2 = mad(v2.x, k0, o1_2);
            o1_2 = mad(v2.y, k1, o1_2);
            o1_2 = mad(v2.z, k2, o1_2);
            o1_2 = mad(v2.w, k3, o1_2);
            o1_3 = mad(v3.x, k0, o1_3);
            o1_3 = mad(v3.y, k1, o1_3);
            o1_3 = mad(v3.z, k2, o1_3);
            o1_3 = mad(v3.w, k3, o1_3);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
            o1_2 = mad(v3.x, k0, o1_2);
            o1_2 = mad(v3.y, k1, o1_2);
            o1_2 = mad(v3.z, k2, o1_2);
            o1_2 = mad(v3.w, k3, o1_2);
            o1_3 = mad(v4.x, k0, o1_3);
            o1_3 = mad(v4.y, k1, o1_3);
            o1_3 = mad(v4.z, k2, o1_3);
            o1_3 = mad(v4.w, k3, o1_3);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
            o1_2 = mad(v4.x, k0, o1_2);
            o1_2 = mad(v4.y, k1, o1_2);
            o1_2 = mad(v4.z, k2, o1_2);
            o1_2 = mad(v4.w, k3, o1_2);
            o1_3 = mad(v5.x, k0, o1_3);
            o1_3 = mad(v5.y, k1, o1_3);
            o1_3 = mad(v5.z, k2, o1_3);
            o1_3 = mad(v5.w, k3, o1_3);
          }
        }
        { const int iy = in_y0 + 2;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0, v4=(COMPUTE_FLOAT4)0, v5=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
            if(in_x0+4 >= 0 && in_x0+4 < HCINW) v4 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+4)*4));
            if(in_x0+5 >= 0 && in_x0+5 < HCINW) v5 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+5)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
            o0_2 = mad(v2.x, k0, o0_2);
            o0_2 = mad(v2.y, k1, o0_2);
            o0_2 = mad(v2.z, k2, o0_2);
            o0_2 = mad(v2.w, k3, o0_2);
            o0_3 = mad(v3.x, k0, o0_3);
            o0_3 = mad(v3.y, k1, o0_3);
            o0_3 = mad(v3.z, k2, o0_3);
            o0_3 = mad(v3.w, k3, o0_3);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
            o0_2 = mad(v3.x, k0, o0_2);
            o0_2 = mad(v3.y, k1, o0_2);
            o0_2 = mad(v3.z, k2, o0_2);
            o0_2 = mad(v3.w, k3, o0_2);
            o0_3 = mad(v4.x, k0, o0_3);
            o0_3 = mad(v4.y, k1, o0_3);
            o0_3 = mad(v4.z, k2, o0_3);
            o0_3 = mad(v4.w, k3, o0_3);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
            o0_2 = mad(v4.x, k0, o0_2);
            o0_2 = mad(v4.y, k1, o0_2);
            o0_2 = mad(v4.z, k2, o0_2);
            o0_2 = mad(v4.w, k3, o0_2);
            o0_3 = mad(v5.x, k0, o0_3);
            o0_3 = mad(v5.y, k1, o0_3);
            o0_3 = mad(v5.z, k2, o0_3);
            o0_3 = mad(v5.w, k3, o0_3);
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
            o1_2 = mad(v2.x, k0, o1_2);
            o1_2 = mad(v2.y, k1, o1_2);
            o1_2 = mad(v2.z, k2, o1_2);
            o1_2 = mad(v2.w, k3, o1_2);
            o1_3 = mad(v3.x, k0, o1_3);
            o1_3 = mad(v3.y, k1, o1_3);
            o1_3 = mad(v3.z, k2, o1_3);
            o1_3 = mad(v3.w, k3, o1_3);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
            o1_2 = mad(v3.x, k0, o1_2);
            o1_2 = mad(v3.y, k1, o1_2);
            o1_2 = mad(v3.z, k2, o1_2);
            o1_2 = mad(v3.w, k3, o1_2);
            o1_3 = mad(v4.x, k0, o1_3);
            o1_3 = mad(v4.y, k1, o1_3);
            o1_3 = mad(v4.z, k2, o1_3);
            o1_3 = mad(v4.w, k3, o1_3);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
            o1_2 = mad(v4.x, k0, o1_2);
            o1_2 = mad(v4.y, k1, o1_2);
            o1_2 = mad(v4.z, k2, o1_2);
            o1_2 = mad(v4.w, k3, o1_2);
            o1_3 = mad(v5.x, k0, o1_3);
            o1_3 = mad(v5.y, k1, o1_3);
            o1_3 = mad(v5.z, k2, o1_3);
            o1_3 = mad(v5.w, k3, o1_3);
          }
        }
        { const int iy = in_y0 + 3;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0, v4=(COMPUTE_FLOAT4)0, v5=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
            if(in_x0+4 >= 0 && in_x0+4 < HCINW) v4 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+4)*4));
            if(in_x0+5 >= 0 && in_x0+5 < HCINW) v5 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+5)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
            o1_2 = mad(v2.x, k0, o1_2);
            o1_2 = mad(v2.y, k1, o1_2);
            o1_2 = mad(v2.z, k2, o1_2);
            o1_2 = mad(v2.w, k3, o1_2);
            o1_3 = mad(v3.x, k0, o1_3);
            o1_3 = mad(v3.y, k1, o1_3);
            o1_3 = mad(v3.z, k2, o1_3);
            o1_3 = mad(v3.w, k3, o1_3);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
            o1_2 = mad(v3.x, k0, o1_2);
            o1_2 = mad(v3.y, k1, o1_2);
            o1_2 = mad(v3.z, k2, o1_2);
            o1_2 = mad(v3.w, k3, o1_2);
            o1_3 = mad(v4.x, k0, o1_3);
            o1_3 = mad(v4.y, k1, o1_3);
            o1_3 = mad(v4.z, k2, o1_3);
            o1_3 = mad(v4.w, k3, o1_3);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
            o1_2 = mad(v4.x, k0, o1_2);
            o1_2 = mad(v4.y, k1, o1_2);
            o1_2 = mad(v4.z, k2, o1_2);
            o1_2 = mad(v4.w, k3, o1_2);
            o1_3 = mad(v5.x, k0, o1_3);
            o1_3 = mad(v5.y, k1, o1_3);
            o1_3 = mad(v5.z, k2, o1_3);
            o1_3 = mad(v5.w, k3, o1_3);
          }
        }
    }
#ifdef RELU
    o0_0 = fmax(o0_0,(COMPUTE_FLOAT4)0);
    o0_1 = fmax(o0_1,(COMPUTE_FLOAT4)0);
    o0_2 = fmax(o0_2,(COMPUTE_FLOAT4)0);
    o0_3 = fmax(o0_3,(COMPUTE_FLOAT4)0);
    o1_0 = fmax(o1_0,(COMPUTE_FLOAT4)0);
    o1_1 = fmax(o1_1,(COMPUTE_FLOAT4)0);
    o1_2 = fmax(o1_2,(COMPUTE_FLOAT4)0);
    o1_3 = fmax(o1_3,(COMPUTE_FLOAT4)0);
#endif
#ifdef PRELU
    { COMPUTE_FLOAT4 sl = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
      o0_0 = select(o0_0*sl,o0_0,o0_0>=0);
      o0_1 = select(o0_1*sl,o0_1,o0_1>=0);
      o0_2 = select(o0_2*sl,o0_2,o0_2>=0);
      o0_3 = select(o0_3*sl,o0_3,o0_3>=0);
      o1_0 = select(o1_0*sl,o1_0,o1_0>=0);
      o1_1 = select(o1_1*sl,o1_1,o1_1>=0);
      o1_2 = select(o1_2*sl,o1_2,o1_2>=0);
      o1_3 = select(o1_3*sl,o1_3,o1_3>=0);
    }
#endif
    const int base = (((out_b_idx + out_c_idx * HCBATCH) * HCOUTH + out_h_idx) * HCOUTW + out_w_idx) * 4;
    const int rh = HCOUTH - out_h_idx; const int rw = HCOUTW - out_w_idx;
    if(0 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o0_0), 0, output + base + (0*HCOUTW + 0)*4);
    if(0 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o0_1), 0, output + base + (0*HCOUTW + 1)*4);
    if(0 < rh && 2 < rw) vstore4(CONVERT_FLOAT4(o0_2), 0, output + base + (0*HCOUTW + 2)*4);
    if(0 < rh && 3 < rw) vstore4(CONVERT_FLOAT4(o0_3), 0, output + base + (0*HCOUTW + 3)*4);
    if(1 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o1_0), 0, output + base + (1*HCOUTW + 0)*4);
    if(1 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o1_1), 0, output + base + (1*HCOUTW + 1)*4);
    if(1 < rh && 2 < rw) vstore4(CONVERT_FLOAT4(o1_2), 0, output + base + (1*HCOUTW + 2)*4);
    if(1 < rh && 3 < rw) vstore4(CONVERT_FLOAT4(o1_3), 0, output + base + (1*HCOUTW + 3)*4);
}

// conv_2d_c4h4w4 (env MNN_CONV_SPEC, stride-1 only): 2-D register tile, 4x4 outputs.
// With MNN_CONV_HARD=1 the host also passes -DHC_* so every shape value becomes a COMPILE-TIME
// constant: the channel loop unrolls, all index arithmetic constant-folds, and the halo bounds
// checks collapse wherever the tile is provably interior. Costs one program build per shape.
__kernel
void conv_2d_c4h4w4(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input, __global const FLOAT *weight,
                      __global const FLOAT *bias, __global FLOAT *output,
                      __private const int2 in_hw, __private const int inChannel,
                      __private const int in_c_blocks, __private const int batch,
                      __private const int2 out_hw, __private const int2 filter_hw,
                      __private const int2 stride_hw, __private const int2 pad_hw,
                      __private const int2 dilate_hw, __private const int out_w_blocks,
                      __private const int out_c_blocks, __private const int out_h_blocks,
                      __private const int out_c_base_index
                      #ifdef PRELU
                      ,__global const FLOAT *slope_ptr
                      #endif
) {
    const int out_c_w_idx = get_global_id(0);
    const int out_b_h_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);
    const int out_c_idx = out_c_w_idx / HCWB + out_c_base_index;
    if(out_c_idx >= HCOCB) return;
    const int out_w_idx = (out_c_w_idx % HCWB) * 4;
    const int out_b_idx = out_b_h_idx / HCHB;
    const int out_h_idx = (out_b_h_idx % HCHB) * 4;
    COMPUTE_FLOAT4 bv = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 o0_0=bv, o0_1=bv, o0_2=bv, o0_3=bv, o1_0=bv, o1_1=bv, o1_2=bv, o1_3=bv, o2_0=bv, o2_1=bv, o2_2=bv, o2_3=bv, o3_0=bv, o3_1=bv, o3_2=bv, o3_3=bv;
    const int in_x0 = out_w_idx - pad_hw.y;
    const int in_y0 = out_h_idx - pad_hw.x;
    const int weight_oc_offset = HCOCB * 9 * 4;
    HCUNROLL
    for(ushort ic = 0; ic < HCICB; ic++) {
        const int inp_base = (out_b_idx + ic * HCBATCH) * HCINH * HCINW * 4;
        const int w_base = (((4 * ic) * HCOCB + out_c_idx) * 9) * 4;
        { const int iy = in_y0 + 0;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0, v4=(COMPUTE_FLOAT4)0, v5=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
            if(in_x0+4 >= 0 && in_x0+4 < HCINW) v4 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+4)*4));
            if(in_x0+5 >= 0 && in_x0+5 < HCINW) v5 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+5)*4));
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
            o0_2 = mad(v2.x, k0, o0_2);
            o0_2 = mad(v2.y, k1, o0_2);
            o0_2 = mad(v2.z, k2, o0_2);
            o0_2 = mad(v2.w, k3, o0_2);
            o0_3 = mad(v3.x, k0, o0_3);
            o0_3 = mad(v3.y, k1, o0_3);
            o0_3 = mad(v3.z, k2, o0_3);
            o0_3 = mad(v3.w, k3, o0_3);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
            o0_2 = mad(v3.x, k0, o0_2);
            o0_2 = mad(v3.y, k1, o0_2);
            o0_2 = mad(v3.z, k2, o0_2);
            o0_2 = mad(v3.w, k3, o0_2);
            o0_3 = mad(v4.x, k0, o0_3);
            o0_3 = mad(v4.y, k1, o0_3);
            o0_3 = mad(v4.z, k2, o0_3);
            o0_3 = mad(v4.w, k3, o0_3);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
            o0_2 = mad(v4.x, k0, o0_2);
            o0_2 = mad(v4.y, k1, o0_2);
            o0_2 = mad(v4.z, k2, o0_2);
            o0_2 = mad(v4.w, k3, o0_2);
            o0_3 = mad(v5.x, k0, o0_3);
            o0_3 = mad(v5.y, k1, o0_3);
            o0_3 = mad(v5.z, k2, o0_3);
            o0_3 = mad(v5.w, k3, o0_3);
          }
        }
        { const int iy = in_y0 + 1;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0, v4=(COMPUTE_FLOAT4)0, v5=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
            if(in_x0+4 >= 0 && in_x0+4 < HCINW) v4 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+4)*4));
            if(in_x0+5 >= 0 && in_x0+5 < HCINW) v5 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+5)*4));
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
            o0_2 = mad(v2.x, k0, o0_2);
            o0_2 = mad(v2.y, k1, o0_2);
            o0_2 = mad(v2.z, k2, o0_2);
            o0_2 = mad(v2.w, k3, o0_2);
            o0_3 = mad(v3.x, k0, o0_3);
            o0_3 = mad(v3.y, k1, o0_3);
            o0_3 = mad(v3.z, k2, o0_3);
            o0_3 = mad(v3.w, k3, o0_3);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
            o0_2 = mad(v3.x, k0, o0_2);
            o0_2 = mad(v3.y, k1, o0_2);
            o0_2 = mad(v3.z, k2, o0_2);
            o0_2 = mad(v3.w, k3, o0_2);
            o0_3 = mad(v4.x, k0, o0_3);
            o0_3 = mad(v4.y, k1, o0_3);
            o0_3 = mad(v4.z, k2, o0_3);
            o0_3 = mad(v4.w, k3, o0_3);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
            o0_2 = mad(v4.x, k0, o0_2);
            o0_2 = mad(v4.y, k1, o0_2);
            o0_2 = mad(v4.z, k2, o0_2);
            o0_2 = mad(v4.w, k3, o0_2);
            o0_3 = mad(v5.x, k0, o0_3);
            o0_3 = mad(v5.y, k1, o0_3);
            o0_3 = mad(v5.z, k2, o0_3);
            o0_3 = mad(v5.w, k3, o0_3);
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
            o1_2 = mad(v2.x, k0, o1_2);
            o1_2 = mad(v2.y, k1, o1_2);
            o1_2 = mad(v2.z, k2, o1_2);
            o1_2 = mad(v2.w, k3, o1_2);
            o1_3 = mad(v3.x, k0, o1_3);
            o1_3 = mad(v3.y, k1, o1_3);
            o1_3 = mad(v3.z, k2, o1_3);
            o1_3 = mad(v3.w, k3, o1_3);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
            o1_2 = mad(v3.x, k0, o1_2);
            o1_2 = mad(v3.y, k1, o1_2);
            o1_2 = mad(v3.z, k2, o1_2);
            o1_2 = mad(v3.w, k3, o1_2);
            o1_3 = mad(v4.x, k0, o1_3);
            o1_3 = mad(v4.y, k1, o1_3);
            o1_3 = mad(v4.z, k2, o1_3);
            o1_3 = mad(v4.w, k3, o1_3);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
            o1_2 = mad(v4.x, k0, o1_2);
            o1_2 = mad(v4.y, k1, o1_2);
            o1_2 = mad(v4.z, k2, o1_2);
            o1_2 = mad(v4.w, k3, o1_2);
            o1_3 = mad(v5.x, k0, o1_3);
            o1_3 = mad(v5.y, k1, o1_3);
            o1_3 = mad(v5.z, k2, o1_3);
            o1_3 = mad(v5.w, k3, o1_3);
          }
        }
        { const int iy = in_y0 + 2;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0, v4=(COMPUTE_FLOAT4)0, v5=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
            if(in_x0+4 >= 0 && in_x0+4 < HCINW) v4 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+4)*4));
            if(in_x0+5 >= 0 && in_x0+5 < HCINW) v5 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+5)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v0.x, k0, o0_0);
            o0_0 = mad(v0.y, k1, o0_0);
            o0_0 = mad(v0.z, k2, o0_0);
            o0_0 = mad(v0.w, k3, o0_0);
            o0_1 = mad(v1.x, k0, o0_1);
            o0_1 = mad(v1.y, k1, o0_1);
            o0_1 = mad(v1.z, k2, o0_1);
            o0_1 = mad(v1.w, k3, o0_1);
            o0_2 = mad(v2.x, k0, o0_2);
            o0_2 = mad(v2.y, k1, o0_2);
            o0_2 = mad(v2.z, k2, o0_2);
            o0_2 = mad(v2.w, k3, o0_2);
            o0_3 = mad(v3.x, k0, o0_3);
            o0_3 = mad(v3.y, k1, o0_3);
            o0_3 = mad(v3.z, k2, o0_3);
            o0_3 = mad(v3.w, k3, o0_3);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v1.x, k0, o0_0);
            o0_0 = mad(v1.y, k1, o0_0);
            o0_0 = mad(v1.z, k2, o0_0);
            o0_0 = mad(v1.w, k3, o0_0);
            o0_1 = mad(v2.x, k0, o0_1);
            o0_1 = mad(v2.y, k1, o0_1);
            o0_1 = mad(v2.z, k2, o0_1);
            o0_1 = mad(v2.w, k3, o0_1);
            o0_2 = mad(v3.x, k0, o0_2);
            o0_2 = mad(v3.y, k1, o0_2);
            o0_2 = mad(v3.z, k2, o0_2);
            o0_2 = mad(v3.w, k3, o0_2);
            o0_3 = mad(v4.x, k0, o0_3);
            o0_3 = mad(v4.y, k1, o0_3);
            o0_3 = mad(v4.z, k2, o0_3);
            o0_3 = mad(v4.w, k3, o0_3);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o0_0 = mad(v2.x, k0, o0_0);
            o0_0 = mad(v2.y, k1, o0_0);
            o0_0 = mad(v2.z, k2, o0_0);
            o0_0 = mad(v2.w, k3, o0_0);
            o0_1 = mad(v3.x, k0, o0_1);
            o0_1 = mad(v3.y, k1, o0_1);
            o0_1 = mad(v3.z, k2, o0_1);
            o0_1 = mad(v3.w, k3, o0_1);
            o0_2 = mad(v4.x, k0, o0_2);
            o0_2 = mad(v4.y, k1, o0_2);
            o0_2 = mad(v4.z, k2, o0_2);
            o0_2 = mad(v4.w, k3, o0_2);
            o0_3 = mad(v5.x, k0, o0_3);
            o0_3 = mad(v5.y, k1, o0_3);
            o0_3 = mad(v5.z, k2, o0_3);
            o0_3 = mad(v5.w, k3, o0_3);
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
            o1_2 = mad(v2.x, k0, o1_2);
            o1_2 = mad(v2.y, k1, o1_2);
            o1_2 = mad(v2.z, k2, o1_2);
            o1_2 = mad(v2.w, k3, o1_2);
            o1_3 = mad(v3.x, k0, o1_3);
            o1_3 = mad(v3.y, k1, o1_3);
            o1_3 = mad(v3.z, k2, o1_3);
            o1_3 = mad(v3.w, k3, o1_3);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
            o1_2 = mad(v3.x, k0, o1_2);
            o1_2 = mad(v3.y, k1, o1_2);
            o1_2 = mad(v3.z, k2, o1_2);
            o1_2 = mad(v3.w, k3, o1_2);
            o1_3 = mad(v4.x, k0, o1_3);
            o1_3 = mad(v4.y, k1, o1_3);
            o1_3 = mad(v4.z, k2, o1_3);
            o1_3 = mad(v4.w, k3, o1_3);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
            o1_2 = mad(v4.x, k0, o1_2);
            o1_2 = mad(v4.y, k1, o1_2);
            o1_2 = mad(v4.z, k2, o1_2);
            o1_2 = mad(v4.w, k3, o1_2);
            o1_3 = mad(v5.x, k0, o1_3);
            o1_3 = mad(v5.y, k1, o1_3);
            o1_3 = mad(v5.z, k2, o1_3);
            o1_3 = mad(v5.w, k3, o1_3);
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v0.x, k0, o2_0);
            o2_0 = mad(v0.y, k1, o2_0);
            o2_0 = mad(v0.z, k2, o2_0);
            o2_0 = mad(v0.w, k3, o2_0);
            o2_1 = mad(v1.x, k0, o2_1);
            o2_1 = mad(v1.y, k1, o2_1);
            o2_1 = mad(v1.z, k2, o2_1);
            o2_1 = mad(v1.w, k3, o2_1);
            o2_2 = mad(v2.x, k0, o2_2);
            o2_2 = mad(v2.y, k1, o2_2);
            o2_2 = mad(v2.z, k2, o2_2);
            o2_2 = mad(v2.w, k3, o2_2);
            o2_3 = mad(v3.x, k0, o2_3);
            o2_3 = mad(v3.y, k1, o2_3);
            o2_3 = mad(v3.z, k2, o2_3);
            o2_3 = mad(v3.w, k3, o2_3);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v1.x, k0, o2_0);
            o2_0 = mad(v1.y, k1, o2_0);
            o2_0 = mad(v1.z, k2, o2_0);
            o2_0 = mad(v1.w, k3, o2_0);
            o2_1 = mad(v2.x, k0, o2_1);
            o2_1 = mad(v2.y, k1, o2_1);
            o2_1 = mad(v2.z, k2, o2_1);
            o2_1 = mad(v2.w, k3, o2_1);
            o2_2 = mad(v3.x, k0, o2_2);
            o2_2 = mad(v3.y, k1, o2_2);
            o2_2 = mad(v3.z, k2, o2_2);
            o2_2 = mad(v3.w, k3, o2_2);
            o2_3 = mad(v4.x, k0, o2_3);
            o2_3 = mad(v4.y, k1, o2_3);
            o2_3 = mad(v4.z, k2, o2_3);
            o2_3 = mad(v4.w, k3, o2_3);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v2.x, k0, o2_0);
            o2_0 = mad(v2.y, k1, o2_0);
            o2_0 = mad(v2.z, k2, o2_0);
            o2_0 = mad(v2.w, k3, o2_0);
            o2_1 = mad(v3.x, k0, o2_1);
            o2_1 = mad(v3.y, k1, o2_1);
            o2_1 = mad(v3.z, k2, o2_1);
            o2_1 = mad(v3.w, k3, o2_1);
            o2_2 = mad(v4.x, k0, o2_2);
            o2_2 = mad(v4.y, k1, o2_2);
            o2_2 = mad(v4.z, k2, o2_2);
            o2_2 = mad(v4.w, k3, o2_2);
            o2_3 = mad(v5.x, k0, o2_3);
            o2_3 = mad(v5.y, k1, o2_3);
            o2_3 = mad(v5.z, k2, o2_3);
            o2_3 = mad(v5.w, k3, o2_3);
          }
        }
        { const int iy = in_y0 + 3;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0, v4=(COMPUTE_FLOAT4)0, v5=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
            if(in_x0+4 >= 0 && in_x0+4 < HCINW) v4 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+4)*4));
            if(in_x0+5 >= 0 && in_x0+5 < HCINW) v5 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+5)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v0.x, k0, o1_0);
            o1_0 = mad(v0.y, k1, o1_0);
            o1_0 = mad(v0.z, k2, o1_0);
            o1_0 = mad(v0.w, k3, o1_0);
            o1_1 = mad(v1.x, k0, o1_1);
            o1_1 = mad(v1.y, k1, o1_1);
            o1_1 = mad(v1.z, k2, o1_1);
            o1_1 = mad(v1.w, k3, o1_1);
            o1_2 = mad(v2.x, k0, o1_2);
            o1_2 = mad(v2.y, k1, o1_2);
            o1_2 = mad(v2.z, k2, o1_2);
            o1_2 = mad(v2.w, k3, o1_2);
            o1_3 = mad(v3.x, k0, o1_3);
            o1_3 = mad(v3.y, k1, o1_3);
            o1_3 = mad(v3.z, k2, o1_3);
            o1_3 = mad(v3.w, k3, o1_3);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v1.x, k0, o1_0);
            o1_0 = mad(v1.y, k1, o1_0);
            o1_0 = mad(v1.z, k2, o1_0);
            o1_0 = mad(v1.w, k3, o1_0);
            o1_1 = mad(v2.x, k0, o1_1);
            o1_1 = mad(v2.y, k1, o1_1);
            o1_1 = mad(v2.z, k2, o1_1);
            o1_1 = mad(v2.w, k3, o1_1);
            o1_2 = mad(v3.x, k0, o1_2);
            o1_2 = mad(v3.y, k1, o1_2);
            o1_2 = mad(v3.z, k2, o1_2);
            o1_2 = mad(v3.w, k3, o1_2);
            o1_3 = mad(v4.x, k0, o1_3);
            o1_3 = mad(v4.y, k1, o1_3);
            o1_3 = mad(v4.z, k2, o1_3);
            o1_3 = mad(v4.w, k3, o1_3);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o1_0 = mad(v2.x, k0, o1_0);
            o1_0 = mad(v2.y, k1, o1_0);
            o1_0 = mad(v2.z, k2, o1_0);
            o1_0 = mad(v2.w, k3, o1_0);
            o1_1 = mad(v3.x, k0, o1_1);
            o1_1 = mad(v3.y, k1, o1_1);
            o1_1 = mad(v3.z, k2, o1_1);
            o1_1 = mad(v3.w, k3, o1_1);
            o1_2 = mad(v4.x, k0, o1_2);
            o1_2 = mad(v4.y, k1, o1_2);
            o1_2 = mad(v4.z, k2, o1_2);
            o1_2 = mad(v4.w, k3, o1_2);
            o1_3 = mad(v5.x, k0, o1_3);
            o1_3 = mad(v5.y, k1, o1_3);
            o1_3 = mad(v5.z, k2, o1_3);
            o1_3 = mad(v5.w, k3, o1_3);
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v0.x, k0, o2_0);
            o2_0 = mad(v0.y, k1, o2_0);
            o2_0 = mad(v0.z, k2, o2_0);
            o2_0 = mad(v0.w, k3, o2_0);
            o2_1 = mad(v1.x, k0, o2_1);
            o2_1 = mad(v1.y, k1, o2_1);
            o2_1 = mad(v1.z, k2, o2_1);
            o2_1 = mad(v1.w, k3, o2_1);
            o2_2 = mad(v2.x, k0, o2_2);
            o2_2 = mad(v2.y, k1, o2_2);
            o2_2 = mad(v2.z, k2, o2_2);
            o2_2 = mad(v2.w, k3, o2_2);
            o2_3 = mad(v3.x, k0, o2_3);
            o2_3 = mad(v3.y, k1, o2_3);
            o2_3 = mad(v3.z, k2, o2_3);
            o2_3 = mad(v3.w, k3, o2_3);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v1.x, k0, o2_0);
            o2_0 = mad(v1.y, k1, o2_0);
            o2_0 = mad(v1.z, k2, o2_0);
            o2_0 = mad(v1.w, k3, o2_0);
            o2_1 = mad(v2.x, k0, o2_1);
            o2_1 = mad(v2.y, k1, o2_1);
            o2_1 = mad(v2.z, k2, o2_1);
            o2_1 = mad(v2.w, k3, o2_1);
            o2_2 = mad(v3.x, k0, o2_2);
            o2_2 = mad(v3.y, k1, o2_2);
            o2_2 = mad(v3.z, k2, o2_2);
            o2_2 = mad(v3.w, k3, o2_2);
            o2_3 = mad(v4.x, k0, o2_3);
            o2_3 = mad(v4.y, k1, o2_3);
            o2_3 = mad(v4.z, k2, o2_3);
            o2_3 = mad(v4.w, k3, o2_3);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v2.x, k0, o2_0);
            o2_0 = mad(v2.y, k1, o2_0);
            o2_0 = mad(v2.z, k2, o2_0);
            o2_0 = mad(v2.w, k3, o2_0);
            o2_1 = mad(v3.x, k0, o2_1);
            o2_1 = mad(v3.y, k1, o2_1);
            o2_1 = mad(v3.z, k2, o2_1);
            o2_1 = mad(v3.w, k3, o2_1);
            o2_2 = mad(v4.x, k0, o2_2);
            o2_2 = mad(v4.y, k1, o2_2);
            o2_2 = mad(v4.z, k2, o2_2);
            o2_2 = mad(v4.w, k3, o2_2);
            o2_3 = mad(v5.x, k0, o2_3);
            o2_3 = mad(v5.y, k1, o2_3);
            o2_3 = mad(v5.z, k2, o2_3);
            o2_3 = mad(v5.w, k3, o2_3);
          }
          { const int wo = w_base + (0*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v0.x, k0, o3_0);
            o3_0 = mad(v0.y, k1, o3_0);
            o3_0 = mad(v0.z, k2, o3_0);
            o3_0 = mad(v0.w, k3, o3_0);
            o3_1 = mad(v1.x, k0, o3_1);
            o3_1 = mad(v1.y, k1, o3_1);
            o3_1 = mad(v1.z, k2, o3_1);
            o3_1 = mad(v1.w, k3, o3_1);
            o3_2 = mad(v2.x, k0, o3_2);
            o3_2 = mad(v2.y, k1, o3_2);
            o3_2 = mad(v2.z, k2, o3_2);
            o3_2 = mad(v2.w, k3, o3_2);
            o3_3 = mad(v3.x, k0, o3_3);
            o3_3 = mad(v3.y, k1, o3_3);
            o3_3 = mad(v3.z, k2, o3_3);
            o3_3 = mad(v3.w, k3, o3_3);
          }
          { const int wo = w_base + (0*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v1.x, k0, o3_0);
            o3_0 = mad(v1.y, k1, o3_0);
            o3_0 = mad(v1.z, k2, o3_0);
            o3_0 = mad(v1.w, k3, o3_0);
            o3_1 = mad(v2.x, k0, o3_1);
            o3_1 = mad(v2.y, k1, o3_1);
            o3_1 = mad(v2.z, k2, o3_1);
            o3_1 = mad(v2.w, k3, o3_1);
            o3_2 = mad(v3.x, k0, o3_2);
            o3_2 = mad(v3.y, k1, o3_2);
            o3_2 = mad(v3.z, k2, o3_2);
            o3_2 = mad(v3.w, k3, o3_2);
            o3_3 = mad(v4.x, k0, o3_3);
            o3_3 = mad(v4.y, k1, o3_3);
            o3_3 = mad(v4.z, k2, o3_3);
            o3_3 = mad(v4.w, k3, o3_3);
          }
          { const int wo = w_base + (0*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v2.x, k0, o3_0);
            o3_0 = mad(v2.y, k1, o3_0);
            o3_0 = mad(v2.z, k2, o3_0);
            o3_0 = mad(v2.w, k3, o3_0);
            o3_1 = mad(v3.x, k0, o3_1);
            o3_1 = mad(v3.y, k1, o3_1);
            o3_1 = mad(v3.z, k2, o3_1);
            o3_1 = mad(v3.w, k3, o3_1);
            o3_2 = mad(v4.x, k0, o3_2);
            o3_2 = mad(v4.y, k1, o3_2);
            o3_2 = mad(v4.z, k2, o3_2);
            o3_2 = mad(v4.w, k3, o3_2);
            o3_3 = mad(v5.x, k0, o3_3);
            o3_3 = mad(v5.y, k1, o3_3);
            o3_3 = mad(v5.z, k2, o3_3);
            o3_3 = mad(v5.w, k3, o3_3);
          }
        }
        { const int iy = in_y0 + 4;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0, v4=(COMPUTE_FLOAT4)0, v5=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
            if(in_x0+4 >= 0 && in_x0+4 < HCINW) v4 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+4)*4));
            if(in_x0+5 >= 0 && in_x0+5 < HCINW) v5 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+5)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v0.x, k0, o2_0);
            o2_0 = mad(v0.y, k1, o2_0);
            o2_0 = mad(v0.z, k2, o2_0);
            o2_0 = mad(v0.w, k3, o2_0);
            o2_1 = mad(v1.x, k0, o2_1);
            o2_1 = mad(v1.y, k1, o2_1);
            o2_1 = mad(v1.z, k2, o2_1);
            o2_1 = mad(v1.w, k3, o2_1);
            o2_2 = mad(v2.x, k0, o2_2);
            o2_2 = mad(v2.y, k1, o2_2);
            o2_2 = mad(v2.z, k2, o2_2);
            o2_2 = mad(v2.w, k3, o2_2);
            o2_3 = mad(v3.x, k0, o2_3);
            o2_3 = mad(v3.y, k1, o2_3);
            o2_3 = mad(v3.z, k2, o2_3);
            o2_3 = mad(v3.w, k3, o2_3);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v1.x, k0, o2_0);
            o2_0 = mad(v1.y, k1, o2_0);
            o2_0 = mad(v1.z, k2, o2_0);
            o2_0 = mad(v1.w, k3, o2_0);
            o2_1 = mad(v2.x, k0, o2_1);
            o2_1 = mad(v2.y, k1, o2_1);
            o2_1 = mad(v2.z, k2, o2_1);
            o2_1 = mad(v2.w, k3, o2_1);
            o2_2 = mad(v3.x, k0, o2_2);
            o2_2 = mad(v3.y, k1, o2_2);
            o2_2 = mad(v3.z, k2, o2_2);
            o2_2 = mad(v3.w, k3, o2_2);
            o2_3 = mad(v4.x, k0, o2_3);
            o2_3 = mad(v4.y, k1, o2_3);
            o2_3 = mad(v4.z, k2, o2_3);
            o2_3 = mad(v4.w, k3, o2_3);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o2_0 = mad(v2.x, k0, o2_0);
            o2_0 = mad(v2.y, k1, o2_0);
            o2_0 = mad(v2.z, k2, o2_0);
            o2_0 = mad(v2.w, k3, o2_0);
            o2_1 = mad(v3.x, k0, o2_1);
            o2_1 = mad(v3.y, k1, o2_1);
            o2_1 = mad(v3.z, k2, o2_1);
            o2_1 = mad(v3.w, k3, o2_1);
            o2_2 = mad(v4.x, k0, o2_2);
            o2_2 = mad(v4.y, k1, o2_2);
            o2_2 = mad(v4.z, k2, o2_2);
            o2_2 = mad(v4.w, k3, o2_2);
            o2_3 = mad(v5.x, k0, o2_3);
            o2_3 = mad(v5.y, k1, o2_3);
            o2_3 = mad(v5.z, k2, o2_3);
            o2_3 = mad(v5.w, k3, o2_3);
          }
          { const int wo = w_base + (1*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v0.x, k0, o3_0);
            o3_0 = mad(v0.y, k1, o3_0);
            o3_0 = mad(v0.z, k2, o3_0);
            o3_0 = mad(v0.w, k3, o3_0);
            o3_1 = mad(v1.x, k0, o3_1);
            o3_1 = mad(v1.y, k1, o3_1);
            o3_1 = mad(v1.z, k2, o3_1);
            o3_1 = mad(v1.w, k3, o3_1);
            o3_2 = mad(v2.x, k0, o3_2);
            o3_2 = mad(v2.y, k1, o3_2);
            o3_2 = mad(v2.z, k2, o3_2);
            o3_2 = mad(v2.w, k3, o3_2);
            o3_3 = mad(v3.x, k0, o3_3);
            o3_3 = mad(v3.y, k1, o3_3);
            o3_3 = mad(v3.z, k2, o3_3);
            o3_3 = mad(v3.w, k3, o3_3);
          }
          { const int wo = w_base + (1*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v1.x, k0, o3_0);
            o3_0 = mad(v1.y, k1, o3_0);
            o3_0 = mad(v1.z, k2, o3_0);
            o3_0 = mad(v1.w, k3, o3_0);
            o3_1 = mad(v2.x, k0, o3_1);
            o3_1 = mad(v2.y, k1, o3_1);
            o3_1 = mad(v2.z, k2, o3_1);
            o3_1 = mad(v2.w, k3, o3_1);
            o3_2 = mad(v3.x, k0, o3_2);
            o3_2 = mad(v3.y, k1, o3_2);
            o3_2 = mad(v3.z, k2, o3_2);
            o3_2 = mad(v3.w, k3, o3_2);
            o3_3 = mad(v4.x, k0, o3_3);
            o3_3 = mad(v4.y, k1, o3_3);
            o3_3 = mad(v4.z, k2, o3_3);
            o3_3 = mad(v4.w, k3, o3_3);
          }
          { const int wo = w_base + (1*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v2.x, k0, o3_0);
            o3_0 = mad(v2.y, k1, o3_0);
            o3_0 = mad(v2.z, k2, o3_0);
            o3_0 = mad(v2.w, k3, o3_0);
            o3_1 = mad(v3.x, k0, o3_1);
            o3_1 = mad(v3.y, k1, o3_1);
            o3_1 = mad(v3.z, k2, o3_1);
            o3_1 = mad(v3.w, k3, o3_1);
            o3_2 = mad(v4.x, k0, o3_2);
            o3_2 = mad(v4.y, k1, o3_2);
            o3_2 = mad(v4.z, k2, o3_2);
            o3_2 = mad(v4.w, k3, o3_2);
            o3_3 = mad(v5.x, k0, o3_3);
            o3_3 = mad(v5.y, k1, o3_3);
            o3_3 = mad(v5.z, k2, o3_3);
            o3_3 = mad(v5.w, k3, o3_3);
          }
        }
        { const int iy = in_y0 + 5;
          COMPUTE_FLOAT4 v0=(COMPUTE_FLOAT4)0, v1=(COMPUTE_FLOAT4)0, v2=(COMPUTE_FLOAT4)0, v3=(COMPUTE_FLOAT4)0, v4=(COMPUTE_FLOAT4)0, v5=(COMPUTE_FLOAT4)0;
          if(iy >= 0 && iy < HCINH) { const int row = inp_base + iy * HCINW * 4;
            if(in_x0+0 >= 0 && in_x0+0 < HCINW) v0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+0)*4));
            if(in_x0+1 >= 0 && in_x0+1 < HCINW) v1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+1)*4));
            if(in_x0+2 >= 0 && in_x0+2 < HCINW) v2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+2)*4));
            if(in_x0+3 >= 0 && in_x0+3 < HCINW) v3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+3)*4));
            if(in_x0+4 >= 0 && in_x0+4 < HCINW) v4 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+4)*4));
            if(in_x0+5 >= 0 && in_x0+5 < HCINW) v5 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + row + (in_x0+5)*4));
          }
          { const int wo = w_base + (2*3+0)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v0.x, k0, o3_0);
            o3_0 = mad(v0.y, k1, o3_0);
            o3_0 = mad(v0.z, k2, o3_0);
            o3_0 = mad(v0.w, k3, o3_0);
            o3_1 = mad(v1.x, k0, o3_1);
            o3_1 = mad(v1.y, k1, o3_1);
            o3_1 = mad(v1.z, k2, o3_1);
            o3_1 = mad(v1.w, k3, o3_1);
            o3_2 = mad(v2.x, k0, o3_2);
            o3_2 = mad(v2.y, k1, o3_2);
            o3_2 = mad(v2.z, k2, o3_2);
            o3_2 = mad(v2.w, k3, o3_2);
            o3_3 = mad(v3.x, k0, o3_3);
            o3_3 = mad(v3.y, k1, o3_3);
            o3_3 = mad(v3.z, k2, o3_3);
            o3_3 = mad(v3.w, k3, o3_3);
          }
          { const int wo = w_base + (2*3+1)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v1.x, k0, o3_0);
            o3_0 = mad(v1.y, k1, o3_0);
            o3_0 = mad(v1.z, k2, o3_0);
            o3_0 = mad(v1.w, k3, o3_0);
            o3_1 = mad(v2.x, k0, o3_1);
            o3_1 = mad(v2.y, k1, o3_1);
            o3_1 = mad(v2.z, k2, o3_1);
            o3_1 = mad(v2.w, k3, o3_1);
            o3_2 = mad(v3.x, k0, o3_2);
            o3_2 = mad(v3.y, k1, o3_2);
            o3_2 = mad(v3.z, k2, o3_2);
            o3_2 = mad(v3.w, k3, o3_2);
            o3_3 = mad(v4.x, k0, o3_3);
            o3_3 = mad(v4.y, k1, o3_3);
            o3_3 = mad(v4.z, k2, o3_3);
            o3_3 = mad(v4.w, k3, o3_3);
          }
          { const int wo = w_base + (2*3+2)*4;
            COMPUTE_FLOAT4 k0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo));
            COMPUTE_FLOAT4 k1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset));
            COMPUTE_FLOAT4 k2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*2));
            COMPUTE_FLOAT4 k3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight+wo+weight_oc_offset*3));
            o3_0 = mad(v2.x, k0, o3_0);
            o3_0 = mad(v2.y, k1, o3_0);
            o3_0 = mad(v2.z, k2, o3_0);
            o3_0 = mad(v2.w, k3, o3_0);
            o3_1 = mad(v3.x, k0, o3_1);
            o3_1 = mad(v3.y, k1, o3_1);
            o3_1 = mad(v3.z, k2, o3_1);
            o3_1 = mad(v3.w, k3, o3_1);
            o3_2 = mad(v4.x, k0, o3_2);
            o3_2 = mad(v4.y, k1, o3_2);
            o3_2 = mad(v4.z, k2, o3_2);
            o3_2 = mad(v4.w, k3, o3_2);
            o3_3 = mad(v5.x, k0, o3_3);
            o3_3 = mad(v5.y, k1, o3_3);
            o3_3 = mad(v5.z, k2, o3_3);
            o3_3 = mad(v5.w, k3, o3_3);
          }
        }
    }
#ifdef RELU
    o0_0 = fmax(o0_0,(COMPUTE_FLOAT4)0);
    o0_1 = fmax(o0_1,(COMPUTE_FLOAT4)0);
    o0_2 = fmax(o0_2,(COMPUTE_FLOAT4)0);
    o0_3 = fmax(o0_3,(COMPUTE_FLOAT4)0);
    o1_0 = fmax(o1_0,(COMPUTE_FLOAT4)0);
    o1_1 = fmax(o1_1,(COMPUTE_FLOAT4)0);
    o1_2 = fmax(o1_2,(COMPUTE_FLOAT4)0);
    o1_3 = fmax(o1_3,(COMPUTE_FLOAT4)0);
    o2_0 = fmax(o2_0,(COMPUTE_FLOAT4)0);
    o2_1 = fmax(o2_1,(COMPUTE_FLOAT4)0);
    o2_2 = fmax(o2_2,(COMPUTE_FLOAT4)0);
    o2_3 = fmax(o2_3,(COMPUTE_FLOAT4)0);
    o3_0 = fmax(o3_0,(COMPUTE_FLOAT4)0);
    o3_1 = fmax(o3_1,(COMPUTE_FLOAT4)0);
    o3_2 = fmax(o3_2,(COMPUTE_FLOAT4)0);
    o3_3 = fmax(o3_3,(COMPUTE_FLOAT4)0);
#endif
#ifdef PRELU
    { COMPUTE_FLOAT4 sl = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, slope_ptr));
      o0_0 = select(o0_0*sl,o0_0,o0_0>=0);
      o0_1 = select(o0_1*sl,o0_1,o0_1>=0);
      o0_2 = select(o0_2*sl,o0_2,o0_2>=0);
      o0_3 = select(o0_3*sl,o0_3,o0_3>=0);
      o1_0 = select(o1_0*sl,o1_0,o1_0>=0);
      o1_1 = select(o1_1*sl,o1_1,o1_1>=0);
      o1_2 = select(o1_2*sl,o1_2,o1_2>=0);
      o1_3 = select(o1_3*sl,o1_3,o1_3>=0);
      o2_0 = select(o2_0*sl,o2_0,o2_0>=0);
      o2_1 = select(o2_1*sl,o2_1,o2_1>=0);
      o2_2 = select(o2_2*sl,o2_2,o2_2>=0);
      o2_3 = select(o2_3*sl,o2_3,o2_3>=0);
      o3_0 = select(o3_0*sl,o3_0,o3_0>=0);
      o3_1 = select(o3_1*sl,o3_1,o3_1>=0);
      o3_2 = select(o3_2*sl,o3_2,o3_2>=0);
      o3_3 = select(o3_3*sl,o3_3,o3_3>=0);
    }
#endif
    const int base = (((out_b_idx + out_c_idx * HCBATCH) * HCOUTH + out_h_idx) * HCOUTW + out_w_idx) * 4;
    const int rh = HCOUTH - out_h_idx; const int rw = HCOUTW - out_w_idx;
    if(0 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o0_0), 0, output + base + (0*HCOUTW + 0)*4);
    if(0 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o0_1), 0, output + base + (0*HCOUTW + 1)*4);
    if(0 < rh && 2 < rw) vstore4(CONVERT_FLOAT4(o0_2), 0, output + base + (0*HCOUTW + 2)*4);
    if(0 < rh && 3 < rw) vstore4(CONVERT_FLOAT4(o0_3), 0, output + base + (0*HCOUTW + 3)*4);
    if(1 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o1_0), 0, output + base + (1*HCOUTW + 0)*4);
    if(1 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o1_1), 0, output + base + (1*HCOUTW + 1)*4);
    if(1 < rh && 2 < rw) vstore4(CONVERT_FLOAT4(o1_2), 0, output + base + (1*HCOUTW + 2)*4);
    if(1 < rh && 3 < rw) vstore4(CONVERT_FLOAT4(o1_3), 0, output + base + (1*HCOUTW + 3)*4);
    if(2 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o2_0), 0, output + base + (2*HCOUTW + 0)*4);
    if(2 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o2_1), 0, output + base + (2*HCOUTW + 1)*4);
    if(2 < rh && 2 < rw) vstore4(CONVERT_FLOAT4(o2_2), 0, output + base + (2*HCOUTW + 2)*4);
    if(2 < rh && 3 < rw) vstore4(CONVERT_FLOAT4(o2_3), 0, output + base + (2*HCOUTW + 3)*4);
    if(3 < rh && 0 < rw) vstore4(CONVERT_FLOAT4(o3_0), 0, output + base + (3*HCOUTW + 0)*4);
    if(3 < rh && 1 < rw) vstore4(CONVERT_FLOAT4(o3_1), 0, output + base + (3*HCOUTW + 1)*4);
    if(3 < rh && 2 < rw) vstore4(CONVERT_FLOAT4(o3_2), 0, output + base + (3*HCOUTW + 2)*4);
    if(3 < rh && 3 < rw) vstore4(CONVERT_FLOAT4(o3_3), 0, output + base + (3*HCOUTW + 3)*4);
}

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
    const int H = HCINH, W = HCINW;
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
            v = CONVERT_COMPUTE_FLOAT4(vload4(0, input + ((((cb * HCBATCH + b) * H + iy) * W + ix) * 4)));
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
        vstore4(CONVERT_FLOAT4(acc), 0, output + ((((oc * HCBATCH + b) * H + oy) * W + ox) * 4));
    }
}

// ---------------------------------------------------------------------------------------------
// Split-K over input channels (env MNN_CONV_SPLITK=<2|4|8>).
//
// Why: the stride-2 heads are OUTPUT-starved, not arithmetic-bound. 64->96@36x48 s2 emits only
// 18*24*24 = 10368 output float4s, which is too few waves to fill 8 CUs -- measured directly by
// batch scaling (FINDINGS §H.31): batch 2 costs only 1.33x batch 1, so a third of the machine is
// idle at batch 1. Splitting the Cin reduction across SPLITK workgroups multiplies the thread
// count by SPLITK without touching the output shape, which is the one lever that attacks that
// starvation head-on.
//
// Two passes, no atomics (fp16 atomics do not exist here): the partial kernel writes SPLITK
// un-reduced partials to a scratch buffer, the reduce kernel sums them and applies bias +
// activation. Blocking is c4h1w1 -- the max-parallel point, which is what an occupancy fix wants.
//
// partial layout: [SPLITK][out_c_blocks][batch][out_h][out_w][4]
__kernel
void conv_2d_c4h1w1_splitk(GLOBAL_SIZE_2_DIMS
                           __global const FLOAT *input,
                           __global const FLOAT *weight,
                           __global FLOAT *partial,
                           __private const int2 in_hw,
                           __private const int in_c_blocks,
                           __private const int batch,
                           __private const int2 out_hw,
                           __private const int2 stride_hw,
                           __private const int2 pad_hw,
                           __private const int out_c_blocks,
                           __private const int split_k
) {
    const int out_c_w_idx = get_global_id(0);   // out_c_idx * out_w + out_w_idx
    const int bhs_idx     = get_global_id(1);   // (s * batch * out_h) + (b * out_h + h)

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, bhs_idx);

    const int out_w   = out_hw.y;
    const int out_h   = out_hw.x;
    const int bh_span = batch * out_h;

    const int out_c_idx = out_c_w_idx / out_w;
    const int out_w_idx = out_c_w_idx % out_w;
    const int s         = bhs_idx / bh_span;
    const int bh        = bhs_idx % bh_span;
    const int out_b_idx = bh / out_h;
    const int out_h_idx = bh % out_h;

    // this workgroup's slice of the input-channel reduction
    const int ic_per   = (in_c_blocks + split_k - 1) / split_k;
    const int ic_start = s * ic_per;
    const int ic_end   = min(ic_start + ic_per, in_c_blocks);

    COMPUTE_FLOAT4 out0 = (COMPUTE_FLOAT4)0;

    const int in_w_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_h_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    const int weight_oc_offset = out_c_blocks * 9 * 4;

    for (int ic = ic_start; ic < ic_end; ic++) {
        int weight_offset = ((((4 * ic) * out_c_blocks + out_c_idx) * 3) * 3) * 4;
        for (int fh = 0; fh < 3; fh++) {
            const int iy = in_h_base + fh;
            if (iy < 0 || iy >= in_hw.x) { weight_offset += 3 * 4; continue; }
            const int inp_offset_base = (((out_b_idx + ic * batch) * in_hw.x + iy) * in_hw.y) * 4;
            for (int fw = 0; fw < 3; fw++) {
                const int ix = in_w_base + fw;
                COMPUTE_FLOAT4 in0 = (ix < 0 || ix >= in_hw.y) ? (COMPUTE_FLOAT4)0
                                     : CONVERT_COMPUTE_FLOAT4(vload4(ix, input + inp_offset_base));
                COMPUTE_FLOAT4 w0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + weight_offset));
                COMPUTE_FLOAT4 w1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + weight_offset + weight_oc_offset));
                COMPUTE_FLOAT4 w2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + weight_offset + weight_oc_offset * 2));
                COMPUTE_FLOAT4 w3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + weight_offset + weight_oc_offset * 3));
                out0 = mad(in0.x, w0, out0);
                out0 = mad(in0.y, w1, out0);
                out0 = mad(in0.z, w2, out0);
                out0 = mad(in0.w, w3, out0);
                weight_offset += 4;
            }
        }
    }

    const int plane      = out_c_blocks * batch * out_h * out_w;
    const int out_offset = (((out_b_idx + out_c_idx * batch) * out_h + out_h_idx) * out_w + out_w_idx) * 4;
    vstore4(CONVERT_FLOAT4(out0), 0, partial + s * plane * 4 + out_offset);
}

// Reduce the SPLITK partials, add bias, apply the fused activation, write the real output.
__kernel
void conv_2d_splitk_reduce(GLOBAL_SIZE_2_DIMS
                           __global const FLOAT *partial,
                           __global const FLOAT *bias,
                           __global FLOAT *output,
                           __private const int batch,
                           __private const int2 out_hw,
                           __private const int out_c_blocks,
                           __private const int split_k
                           #ifdef PRELU
                           ,__global const FLOAT *slope_ptr
                           #endif
) {
    const int out_c_w_idx = get_global_id(0);
    const int bh_idx      = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, bh_idx);

    const int out_w = out_hw.y;
    const int out_h = out_hw.x;

    const int out_c_idx = out_c_w_idx / out_w;
    const int out_w_idx = out_c_w_idx % out_w;
    const int out_b_idx = bh_idx / out_h;
    const int out_h_idx = bh_idx % out_h;

    const int plane      = out_c_blocks * batch * out_h * out_w;
    const int out_offset = (((out_b_idx + out_c_idx * batch) * out_h + out_h_idx) * out_w + out_w_idx) * 4;

    COMPUTE_FLOAT4 acc = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    for (int s = 0; s < split_k; s++) {
        acc += CONVERT_COMPUTE_FLOAT4(vload4(0, partial + s * plane * 4 + out_offset));
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

    vstore4(CONVERT_FLOAT4(acc), 0, output + out_offset);
}

// ---------------------------------------------------------------------------------------------
// Plain-NCHW convolution path (env MNN_CONV_NCHW=1).
//
// Every stock MNN buffer conv works in NC4HW4: a float4 load fetches 4 CHANNELS of one pixel, and
// the inner product is 4 mads broadcasting those 4 components against 4 weight vectors. NCHW
// instead makes a float4 load fetch 4 adjacent PIXELS of one channel, so a weight value becomes a
// scalar multiplier applied to a whole w-vector. Same arithmetic intensity, completely different
// register/broadcast structure -- and untested here, because MNN never offers the choice.
//
// To keep this an honest comparison rather than a timing hack, the conv is made CORRECT: the
// NC4HW4 input is converted into a scratch NCHW buffer first, the conv runs entirely in NCHW, and
// the result is converted back. The conversion kernels are timed separately so the conv kernel can
// be compared like for like (the caller was explicit that the surrounding layout cost is not the
// question being asked).
//
// Blocking is 4 output channels x 8 adjacent pixels = 32 accumulator scalars = 8 float4, which is
// deliberately the same register class as conv_2d_c4h4w2, the best NC4HW4 kernel found (§H.22).
// Requires 3x3, stride 1, dilation 1, pad 1.
__kernel
void cvt_nc4hw4_to_nchw(GLOBAL_SIZE_2_DIMS
                        __global const FLOAT *src, __global FLOAT *dst,
                        __private const int channels, __private const int batch,
                        __private const int height, __private const int width) {
    const int cx_idx = get_global_id(0);   // c * width + x
    const int bh_idx = get_global_id(1);   // b * height + y
    DEAL_NON_UNIFORM_DIM2(cx_idx, bh_idx);
    const int c = cx_idx / width;
    const int x = cx_idx % width;
    const int b = bh_idx / height;
    const int y = bh_idx % height;
    const int cb = c >> 2, lane = c & 3;
    const int s = ((((cb * batch + b) * height + y) * width + x) << 2) + lane;
    // cstride = channels rounded up to 4. The pad planes are zero-filled so the conv can write a
    // full float4 of output channels without a bounds check and without running off the buffer.
    const int cstride = (channels + 3) & ~3;
    if (c >= cstride) return;
    dst[((b * cstride + c) * height + y) * width + x] = (c < channels) ? src[s] : (FLOAT)0;
}

__kernel
void cvt_nchw_to_nc4hw4(GLOBAL_SIZE_2_DIMS
                        __global const FLOAT *src, __global FLOAT *dst,
                        __private const int channels, __private const int batch,
                        __private const int height, __private const int width) {
    const int cx_idx = get_global_id(0);
    const int bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cx_idx, bh_idx);
    const int cb = cx_idx / width;          // destination channel BLOCK
    const int x  = cx_idx % width;
    const int b  = bh_idx / height;
    const int y  = bh_idx % height;
    const int d = ((((cb * batch + b) * height + y) * width + x) << 2);
    for (int lane = 0; lane < 4; lane++) {
        const int c = (cb << 2) + lane;
        // zero-fill the padding lanes when channels % 4 != 0, so the NC4HW4 consumer sees clean data
        const int cstride = (channels + 3) & ~3;
        dst[d + lane] = (c < channels) ? src[((b * cstride + c) * height + y) * width + x]
                                       : (FLOAT)0;
    }
}

#define NCHW_WT 8
// ---- NCHW shape access: runtime args by default, compile-time constants under MNN_CONV_HARD ----
// The host emits every NC_* name as a -D (the runtime expression, or a literal under
// MNN_CONV_HARD=1). Guarded exactly like the HC_* set -- an unguarded #define here would shadow
// the host's -D and silently disable hardcoding, which is the §H.34 bug.
#ifndef NC_IC
  #define NC_IC    in_channels
#endif
#ifndef NC_OC
  #define NC_OC    out_channels
#endif
#ifndef NC_BATCH
  #define NC_BATCH batch
#endif
#ifndef NC_INH
  #define NC_INH   in_h
#endif
#ifndef NC_INW
  #define NC_INW   in_w
#endif
#ifndef NC_OUTH
  #define NC_OUTH  out_h
#endif
#ifndef NC_OUTW
  #define NC_OUTW  out_w
#endif
// Fair-fight NCHW kernel, v2. v1 (indexed acc[4][8] + scalar strided weight loads) measured
// 2-4x slower, which is exactly the shape of the false negative that the c4h4w2 2-D tile produced
// once already -- so it was rewritten with every known trap removed:
//   * 8 EXPLICIT float4 accumulators (a0..a7), one per pixel holding its 4 output channels; no
//     array indexing anywhere, so nothing can spill to scratch;
//   * weights REPACKED to [ocb][ic][kh][kw][4oc] so one tap is one ALIGNED float4 load serving all
//     4 output channels -- v1 issued 4 scalar loads strided by ic*9, which is cache-hostile;
//   * input read with 4 ALIGNED float4 loads per (channel,row) instead of 10 scalar loads.
// Weight layout is free to differ: NCHW here is a claim about the ACTIVATION layout only.
// Fast path requires out_channels % 4 == 0 and width % 8 == 0 (all target shapes qualify).
__kernel
void conv_2d_nchw_c4w8(GLOBAL_SIZE_2_DIMS
                       __global const FLOAT *input,    // NCHW
                       __global const FLOAT *weight,   // [ocb][ic][kh][kw][4]
                       __global const FLOAT *bias,
                       __global FLOAT *output,         // NCHW
                       __private const int in_channels,
                       __private const int out_channels,
                       __private const int batch,
                       __private const int in_h,
                       __private const int in_w,
                       __private const int out_h,
                       __private const int out_w
                       #ifdef PRELU
                       ,__global const FLOAT *slope_ptr
                       #endif
) {
    const int ocw_idx = get_global_id(0);
    const int bh_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(ocw_idx, bh_idx);

    const int w_blocks = NC_OUTW >> 3;
    const int ocb = ocw_idx / w_blocks;
    const int x0  = (ocw_idx % w_blocks) << 3;
    const int b   = bh_idx / NC_OUTH;
    const int y   = bh_idx % NC_OUTH;
    const int oc0 = ocb << 2;
    if (oc0 >= NC_OC) return;

    COMPUTE_FLOAT4 bv = CONVERT_COMPUTE_FLOAT4(vload4(ocb, bias));
    COMPUTE_FLOAT4 a0=bv,a1=bv,a2=bv,a3=bv,a4=bv,a5=bv,a6=bv,a7=bv;

    const int wbase_ocb = ocb * NC_IC * 9;

    for (int ic = 0; ic < NC_IC; ic++) {
        const int wbase_ic = wbase_ocb + ic * 9;
        for (int kh = 0; kh < 3; kh++) {
            const int iy = y + kh - 1;
            if (iy < 0 || iy >= NC_INH) continue;
            const int in_cstride = (NC_IC + 3) & ~3;
            __global const FLOAT *row = input + ((b * in_cstride + ic) * NC_INH + iy) * NC_INW;

            // 4 ALIGNED float4 loads spanning x0-4 .. x0+11; only r[-1..8] are used.
            COMPUTE_FLOAT4 vm = (x0 >= 4) ? CONVERT_COMPUTE_FLOAT4(vload4((x0 >> 2) - 1, row))
                                          : (COMPUTE_FLOAT4)0;
            COMPUTE_FLOAT4 v0 = CONVERT_COMPUTE_FLOAT4(vload4((x0 >> 2),     row));
            COMPUTE_FLOAT4 v1 = CONVERT_COMPUTE_FLOAT4(vload4((x0 >> 2) + 1, row));
            COMPUTE_FLOAT4 v2 = (x0 + 8 < NC_INW) ? CONVERT_COMPUTE_FLOAT4(vload4((x0 >> 2) + 2, row))
                                                 : (COMPUTE_FLOAT4)0;
            // left halo column is 0 at the image edge, not a wrapped neighbour
            const COMPUTE_FLOAT rm1 = (x0 >= 1) ? vm.w : (COMPUTE_FLOAT)0;
            const COMPUTE_FLOAT r8  = (x0 + 8 < NC_INW) ? v2.x : (COMPUTE_FLOAT)0;

            #define NCHW_TAP(KW, e0,e1,e2,e3,e4,e5,e6,e7)                              \
                { COMPUTE_FLOAT4 wv = CONVERT_COMPUTE_FLOAT4(vload4(wbase_ic + kh*3 + (KW), weight)); \
                  a0 = mad(wv, (COMPUTE_FLOAT4)(e0), a0); a1 = mad(wv, (COMPUTE_FLOAT4)(e1), a1); \
                  a2 = mad(wv, (COMPUTE_FLOAT4)(e2), a2); a3 = mad(wv, (COMPUTE_FLOAT4)(e3), a3); \
                  a4 = mad(wv, (COMPUTE_FLOAT4)(e4), a4); a5 = mad(wv, (COMPUTE_FLOAT4)(e5), a5); \
                  a6 = mad(wv, (COMPUTE_FLOAT4)(e6), a6); a7 = mad(wv, (COMPUTE_FLOAT4)(e7), a7); }

            NCHW_TAP(0, rm1,  v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z)
            NCHW_TAP(1, v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w)
            NCHW_TAP(2, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w, r8)
            #undef NCHW_TAP
        }
    }

#ifdef RELU
    a0=fmax(a0,(COMPUTE_FLOAT4)0); a1=fmax(a1,(COMPUTE_FLOAT4)0);
    a2=fmax(a2,(COMPUTE_FLOAT4)0); a3=fmax(a3,(COMPUTE_FLOAT4)0);
    a4=fmax(a4,(COMPUTE_FLOAT4)0); a5=fmax(a5,(COMPUTE_FLOAT4)0);
    a6=fmax(a6,(COMPUTE_FLOAT4)0); a7=fmax(a7,(COMPUTE_FLOAT4)0);
#endif
#ifdef RELU6
    a0=clamp(a0,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a1=clamp(a1,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    a2=clamp(a2,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a3=clamp(a3,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    a4=clamp(a4,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a5=clamp(a5,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    a6=clamp(a6,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a7=clamp(a7,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
#endif
#ifdef PRELU
    { COMPUTE_FLOAT4 sv = CONVERT_COMPUTE_FLOAT4(vload4(ocb, slope_ptr));
      a0=select(a0*sv,a0,a0>=0); a1=select(a1*sv,a1,a1>=0);
      a2=select(a2*sv,a2,a2>=0); a3=select(a3*sv,a3,a3>=0);
      a4=select(a4*sv,a4,a4>=0); a5=select(a5*sv,a5,a5>=0);
      a6=select(a6*sv,a6,a6>=0); a7=select(a7*sv,a7,a7>=0);
    }
#endif

    // scatter back: each accumulator holds 4 CHANNELS of one pixel, and channels are far apart
    // in NCHW -- this transpose-on-store is the structural cost of the layout.
    const int plane = NC_OUTH * NC_OUTW;
    // padded stride: writing the full float4 of channels is then always in bounds, even when
    // out_channels is not a multiple of 4 (C=18/34/50 in the real model's heads)
    const int out_cstride = (NC_OC + 3) & ~3;
    __global FLOAT *o = output + ((b * out_cstride + oc0) * NC_OUTH + y) * NC_OUTW + x0;
    #define NCHW_ST(I, A) { o[(I)] = (FLOAT)(A).x; o[(I)+plane] = (FLOAT)(A).y; \
                            o[(I)+2*plane] = (FLOAT)(A).z; o[(I)+3*plane] = (FLOAT)(A).w; }
    NCHW_ST(0,a0) NCHW_ST(1,a1) NCHW_ST(2,a2) NCHW_ST(3,a3)
    NCHW_ST(4,a4) NCHW_ST(5,a5) NCHW_ST(6,a6) NCHW_ST(7,a7)
    #undef NCHW_ST
}

// ---------------------------------------------------------------------------------------------
// im2col + GEMM in NCHW (env MNN_CONV_IMGEMM=1).
//
// Retried because the reason it was rejected has been falsified. §H.13 killed im2col partly on the
// cost of materialising the column matrix; §H.35 then measured that memory traffic on this device
// costs approximately nothing at any size (a 6.9 MB intermediate round-trip is free). And §H.36
// showed NCHW's efficiency improves with channel count, which is exactly the regime a GEMM wants.
//
// Under NCHW the shapes line up with no contortion: col is [K = C*9, M = B*H*W], the GEMM is
// [OC, K] x [K, M], and its output [OC, M] IS the NCHW output -- no post-transform of the result,
// only the final NC4HW4 repack that the surrounding engine requires.
//
// Requires 3x3 s1 p1 d1, W % 4 == 0, (H*W) % 8 == 0.

// im2col: reads the NC4HW4 input DIRECTLY (no separate layout pass) and writes the NCHW column
// matrix. One thread owns (channel block, tap, 4 consecutive output pixels): it reads 4 whole
// float4s -- so the NC4HW4 side stays fully coalesced -- and scatters them into the 4 column rows
// belonging to that block's 4 channels.
__kernel
void im2col_nchw(GLOBAL_SIZE_2_DIMS
                 __global const FLOAT *input,   // NC4HW4
                 __global FLOAT *col,           // [K = C*9][M]
                 __private const int in_channels,
                 __private const int batch,
                 __private const int height,
                 __private const int width) {
    const int mq_idx  = get_global_id(0);   // 4 output pixels per thread
    const int cbt_idx = get_global_id(1);   // channel_block * 9 + tap
    DEAL_NON_UNIFORM_DIM2(mq_idx, cbt_idx);

    const int plane = height * width;
    const int M = batch * plane;
    const int m0 = mq_idx << 2;
    if (m0 >= M) return;

    const int cb  = cbt_idx / 9;
    const int tap = cbt_idx % 9;
    const int kh = tap / 3, kw = tap % 3;

    const int b   = m0 / plane;
    const int rem = m0 % plane;
    const int y   = rem / width;
    const int x0  = rem % width;

    const int iy = y + kh - 1;
    const int K  = in_channels * 9;

    COMPUTE_FLOAT4 v[4];
    if (iy < 0 || iy >= height) {
        v[0] = v[1] = v[2] = v[3] = (COMPUTE_FLOAT4)0;
    } else {
        const int rowbase = (((cb * batch + b) * height + iy) * width) << 2;
        for (int i = 0; i < 4; i++) {
            const int ix = x0 + i + kw - 1;
            v[i] = (ix >= 0 && ix < width) ? CONVERT_COMPUTE_FLOAT4(vload4(0, input + rowbase + (ix << 2)))
                                           : (COMPUTE_FLOAT4)0;
        }
    }
    // scatter: lane L of the float4 is channel cb*4+L, i.e. column row (cb*4+L)*9 + tap
    for (int lane = 0; lane < 4; lane++) {
        const int c = (cb << 2) + lane;
        if (c >= in_channels) break;
        const int k = c * 9 + tap;
        if (k >= K) break;
        __global FLOAT *dst = col + (size_t)k * M + m0;
        COMPUTE_FLOAT4 out4 = (COMPUTE_FLOAT4)(
            (lane == 0 ? v[0].x : lane == 1 ? v[0].y : lane == 2 ? v[0].z : v[0].w),
            (lane == 0 ? v[1].x : lane == 1 ? v[1].y : lane == 2 ? v[1].z : v[1].w),
            (lane == 0 ? v[2].x : lane == 1 ? v[2].y : lane == 2 ? v[2].z : v[2].w),
            (lane == 0 ? v[3].x : lane == 1 ? v[3].y : lane == 2 ? v[3].z : v[3].w));
        vstore4(CONVERT_FLOAT4(out4), 0, dst);
    }
}

// GEMM: [OC, K] x [K, M] -> [OC, M] == NCHW output. 4 output channels x 8 pixels per thread = 8
// float4 accumulators, deliberately the same register class as conv_2d_c4h4w2 (§H.22's optimum).
// Weights are pre-packed [ocb][k][4oc] so one k is one aligned float4 load covering 4 channels.
__kernel
void gemm_nchw_c4m8(GLOBAL_SIZE_2_DIMS
                    __global const FLOAT *col,      // [K][M]
                    __global const FLOAT *weight,   // [ocb][k][4]
                    __global const FLOAT *bias,
                    __global FLOAT *output,         // NCHW, channel stride padded to 4
                    __private const int K,
                    __private const int out_channels,
                    __private const int batch,
                    __private const int height,
                    __private const int width
                    #ifdef PRELU
                    ,__global const FLOAT *slope_ptr
                    #endif
) {
    const int mq_idx = get_global_id(0);   // 8 pixels per thread
    const int ocb    = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(mq_idx, ocb);

    const int plane = height * width;
    const int M = batch * plane;
    const int m0 = mq_idx << 3;
    if (m0 >= M) return;
    const int oc0 = ocb << 2;
    if (oc0 >= out_channels) return;

    COMPUTE_FLOAT4 bv = CONVERT_COMPUTE_FLOAT4(vload4(ocb, bias));
    COMPUTE_FLOAT4 a0=bv,a1=bv,a2=bv,a3=bv,a4=bv,a5=bv,a6=bv,a7=bv;

    const int wbase = ocb * K;
    for (int k = 0; k < K; k++) {
        COMPUTE_FLOAT4 wv = CONVERT_COMPUTE_FLOAT4(vload4(wbase + k, weight));
        __global const FLOAT *c = col + (size_t)k * M + m0;
        COMPUTE_FLOAT4 c0 = CONVERT_COMPUTE_FLOAT4(vload4(0, c));
        COMPUTE_FLOAT4 c1 = CONVERT_COMPUTE_FLOAT4(vload4(0, c + 4));
        a0 = mad(wv, (COMPUTE_FLOAT4)(c0.x), a0);
        a1 = mad(wv, (COMPUTE_FLOAT4)(c0.y), a1);
        a2 = mad(wv, (COMPUTE_FLOAT4)(c0.z), a2);
        a3 = mad(wv, (COMPUTE_FLOAT4)(c0.w), a3);
        a4 = mad(wv, (COMPUTE_FLOAT4)(c1.x), a4);
        a5 = mad(wv, (COMPUTE_FLOAT4)(c1.y), a5);
        a6 = mad(wv, (COMPUTE_FLOAT4)(c1.z), a6);
        a7 = mad(wv, (COMPUTE_FLOAT4)(c1.w), a7);
    }

#ifdef RELU
    a0=fmax(a0,(COMPUTE_FLOAT4)0); a1=fmax(a1,(COMPUTE_FLOAT4)0);
    a2=fmax(a2,(COMPUTE_FLOAT4)0); a3=fmax(a3,(COMPUTE_FLOAT4)0);
    a4=fmax(a4,(COMPUTE_FLOAT4)0); a5=fmax(a5,(COMPUTE_FLOAT4)0);
    a6=fmax(a6,(COMPUTE_FLOAT4)0); a7=fmax(a7,(COMPUTE_FLOAT4)0);
#endif
#ifdef RELU6
    a0=clamp(a0,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a1=clamp(a1,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    a2=clamp(a2,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a3=clamp(a3,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    a4=clamp(a4,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a5=clamp(a5,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    a6=clamp(a6,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a7=clamp(a7,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
#endif
#ifdef PRELU
    { COMPUTE_FLOAT4 sv = CONVERT_COMPUTE_FLOAT4(vload4(ocb, slope_ptr));
      a0=select(a0*sv,a0,a0>=0); a1=select(a1*sv,a1,a1>=0);
      a2=select(a2*sv,a2,a2>=0); a3=select(a3*sv,a3,a3>=0);
      a4=select(a4*sv,a4,a4>=0); a5=select(a5*sv,a5,a5>=0);
      a6=select(a6*sv,a6,a6>=0); a7=select(a7*sv,a7,a7>=0);
    }
#endif

    const int b    = m0 / plane;
    const int rem  = m0 % plane;
    const int ocst = (out_channels + 3) & ~3;
    __global FLOAT *o = output + ((size_t)b * ocst + oc0) * plane + rem;
    #define GEMM_ST(I, A) { o[(I)] = (FLOAT)(A).x; o[(I)+plane] = (FLOAT)(A).y; \
                            o[(I)+2*plane] = (FLOAT)(A).z; o[(I)+3*plane] = (FLOAT)(A).w; }
    GEMM_ST(0,a0) GEMM_ST(1,a1) GEMM_ST(2,a2) GEMM_ST(3,a3)
    GEMM_ST(4,a4) GEMM_ST(5,a5) GEMM_ST(6,a6) GEMM_ST(7,a7)
    #undef GEMM_ST
}

// ---------------------------------------------------------------------------------------------
// conv_2d_nchw_s2_c4w8 — NCHW 3x3 STRIDE-2 convolution (env MNN_CONV_NCHW=1 on a stride-2 conv).
//
// This is the case the model actually cares about: its stride-2 heads are 18->16 and 34->32, i.e.
// UNALIGNED input channel counts, which is exactly where NC4HW4 pays its padding tax (Cin 18 is
// computed as 20, Cin 34 as 36). The stride-1 sweep measured NCHW ahead by 4.7% at C=18 and 26.9%
// at C=34, so the heads are the shapes where NCHW should look best -- and until now the NCHW path
// was stride-1 only, so it had never been tried on them.
//
// One thread owns 4 output channels x 8 adjacent OUTPUT pixels = 8 float4 accumulators (same
// register class as the stride-1 variant and as conv_2d_c4h4w2). Output pixel ox needs input
// columns 2*ox-1 .. 2*ox+1, so 8 outputs span input columns 2*x0-1 .. 2*x0+15 -- covered by 5
// ALIGNED float4 loads (x0 is a multiple of 8, so 2*x0 is a multiple of 16).
// Requires 3x3, stride 2, dilation 1, pad 1, out_w % 8 == 0.
__kernel
void conv_2d_nchw_s2_c4w8(GLOBAL_SIZE_2_DIMS
                          __global const FLOAT *input,    // NCHW
                          __global const FLOAT *weight,   // [ocb][ic][kh][kw][4]
                          __global const FLOAT *bias,
                          __global FLOAT *output,         // NCHW
                          __private const int in_channels,
                          __private const int out_channels,
                          __private const int batch,
                          __private const int in_h,
                          __private const int in_w,
                          __private const int out_h,
                          __private const int out_w
                          #ifdef PRELU
                          ,__global const FLOAT *slope_ptr
                          #endif
) {
    const int ocw_idx = get_global_id(0);
    const int bh_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(ocw_idx, bh_idx);

    const int w_blocks = NC_OUTW >> 3;
    const int ocb = ocw_idx / w_blocks;
    const int x0  = (ocw_idx % w_blocks) << 3;      // first OUTPUT column
    const int b   = bh_idx / NC_OUTH;
    const int y   = bh_idx % NC_OUTH;               // OUTPUT row
    const int oc0 = ocb << 2;
    if (oc0 >= NC_OC) return;

    COMPUTE_FLOAT4 bv = CONVERT_COMPUTE_FLOAT4(vload4(ocb, bias));
    COMPUTE_FLOAT4 a0=bv,a1=bv,a2=bv,a3=bv,a4=bv,a5=bv,a6=bv,a7=bv;

    const int ix0 = x0 << 1;                        // input column of output column x0
    const int in_cstride = (NC_IC + 3) & ~3;
    const int wbase_ocb = ocb * NC_IC * 9;

    for (int ic = 0; ic < NC_IC; ic++) {
        const int wbase_ic = wbase_ocb + ic * 9;
        for (int kh = 0; kh < 3; kh++) {
            const int iy = (y << 1) + kh - 1;
            if (iy < 0 || iy >= NC_INH) continue;
            __global const FLOAT *row = input + ((b * in_cstride + ic) * NC_INH + iy) * NC_INW;

            // aligned window ix0-4 .. ix0+15; only ix0-1 .. ix0+15 are consumed
            COMPUTE_FLOAT4 vm = (ix0 >= 4) ? CONVERT_COMPUTE_FLOAT4(vload4((ix0 >> 2) - 1, row))
                                           : (COMPUTE_FLOAT4)0;
            COMPUTE_FLOAT4 v0 = CONVERT_COMPUTE_FLOAT4(vload4((ix0 >> 2),     row));
            COMPUTE_FLOAT4 v1 = CONVERT_COMPUTE_FLOAT4(vload4((ix0 >> 2) + 1, row));
            COMPUTE_FLOAT4 v2 = (ix0 +  8 < NC_INW) ? CONVERT_COMPUTE_FLOAT4(vload4((ix0 >> 2) + 2, row)) : (COMPUTE_FLOAT4)0;
            COMPUTE_FLOAT4 v3 = (ix0 + 12 < NC_INW) ? CONVERT_COMPUTE_FLOAT4(vload4((ix0 >> 2) + 3, row)) : (COMPUTE_FLOAT4)0;
            const COMPUTE_FLOAT rm1 = (ix0 >= 1) ? vm.w : (COMPUTE_FLOAT)0;

            // offset o (from ix0) -> value; o runs -1..15 across the three taps
            #define NCS2_V(O) ((O) < 0 ? rm1 : (O) < 4 ? ((O)==0?v0.x:(O)==1?v0.y:(O)==2?v0.z:v0.w) \
                                             : (O) < 8 ? ((O)==4?v1.x:(O)==5?v1.y:(O)==6?v1.z:v1.w) \
                                             : (O) <12 ? ((O)==8?v2.x:(O)==9?v2.y:(O)==10?v2.z:v2.w) \
                                                       : ((O)==12?v3.x:(O)==13?v3.y:(O)==14?v3.z:v3.w))
            #define NCS2_TAP(KW)                                                                  \
                { COMPUTE_FLOAT4 wv = CONVERT_COMPUTE_FLOAT4(vload4(wbase_ic + kh*3 + (KW), weight)); \
                  a0 = mad(wv, (COMPUTE_FLOAT4)(NCS2_V( 0 + (KW) - 1)), a0);                      \
                  a1 = mad(wv, (COMPUTE_FLOAT4)(NCS2_V( 2 + (KW) - 1)), a1);                      \
                  a2 = mad(wv, (COMPUTE_FLOAT4)(NCS2_V( 4 + (KW) - 1)), a2);                      \
                  a3 = mad(wv, (COMPUTE_FLOAT4)(NCS2_V( 6 + (KW) - 1)), a3);                      \
                  a4 = mad(wv, (COMPUTE_FLOAT4)(NCS2_V( 8 + (KW) - 1)), a4);                      \
                  a5 = mad(wv, (COMPUTE_FLOAT4)(NCS2_V(10 + (KW) - 1)), a5);                      \
                  a6 = mad(wv, (COMPUTE_FLOAT4)(NCS2_V(12 + (KW) - 1)), a6);                      \
                  a7 = mad(wv, (COMPUTE_FLOAT4)(NCS2_V(14 + (KW) - 1)), a7); }
            NCS2_TAP(0)
            NCS2_TAP(1)
            NCS2_TAP(2)
            #undef NCS2_TAP
            #undef NCS2_V
        }
    }

#ifdef RELU
    a0=fmax(a0,(COMPUTE_FLOAT4)0); a1=fmax(a1,(COMPUTE_FLOAT4)0);
    a2=fmax(a2,(COMPUTE_FLOAT4)0); a3=fmax(a3,(COMPUTE_FLOAT4)0);
    a4=fmax(a4,(COMPUTE_FLOAT4)0); a5=fmax(a5,(COMPUTE_FLOAT4)0);
    a6=fmax(a6,(COMPUTE_FLOAT4)0); a7=fmax(a7,(COMPUTE_FLOAT4)0);
#endif
#ifdef RELU6
    a0=clamp(a0,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a1=clamp(a1,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    a2=clamp(a2,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a3=clamp(a3,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    a4=clamp(a4,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a5=clamp(a5,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    a6=clamp(a6,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); a7=clamp(a7,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
#endif
#ifdef PRELU
    { COMPUTE_FLOAT4 sv = CONVERT_COMPUTE_FLOAT4(vload4(ocb, slope_ptr));
      a0=select(a0*sv,a0,a0>=0); a1=select(a1*sv,a1,a1>=0);
      a2=select(a2*sv,a2,a2>=0); a3=select(a3*sv,a3,a3>=0);
      a4=select(a4*sv,a4,a4>=0); a5=select(a5*sv,a5,a5>=0);
      a6=select(a6*sv,a6,a6>=0); a7=select(a7*sv,a7,a7>=0);
    }
#endif

    const int plane = NC_OUTH * NC_OUTW;
    const int out_cstride = (NC_OC + 3) & ~3;
    __global FLOAT *o = output + ((b * out_cstride + oc0) * NC_OUTH + y) * NC_OUTW + x0;
    #define NCS2_ST(I, A) { o[(I)] = (FLOAT)(A).x; o[(I)+plane] = (FLOAT)(A).y; \
                            o[(I)+2*plane] = (FLOAT)(A).z; o[(I)+3*plane] = (FLOAT)(A).w; }
    NCS2_ST(0,a0) NCS2_ST(1,a1) NCS2_ST(2,a2) NCS2_ST(3,a3)
    NCS2_ST(4,a4) NCS2_ST(5,a5) NCS2_ST(6,a6) NCS2_ST(7,a7)
    #undef NCS2_ST
}

// ---------------------------------------------------------------------------------------------
// conv_2d_nchw_fused2 — TWO chained 3x3 convs in one NCHW kernel (env MNN_CONV_NCHW_FUSE2=1).
//
// The NCHW counterpart of conv_2d_3x3s1_fused2, built to close by MEASUREMENT what had only been
// closed by argument: whether fusing consecutive convs behaves differently once the activations
// are NCHW. Like the NC4HW4 original it applies the SAME weights twice, so it fits inside one conv
// Execution and is validated against a numpy conv(conv(x)) reference.
//
// One workgroup owns a FUSE_T x FUSE_T output tile. conv2 over that tile needs conv1's output over
// a (FUSE_T+2)^2 halo for ALL channels, which is staged in __local. Note the halo is the reason
// fusion costs arithmetic: the workgroup computes (T+2)^2 mid-layer pixels to produce T^2 outputs,
// so conv1 is recomputed ((T+2)/T)^2 times -- 1.78x at T=6.
//
// Register/LDS note: the intermediate needs (T+2)^2 * C values in EITHER layout (NC4HW4 stores
// them as (T+2)^2 * C/4 float4, the same scalar count), so NCHW does not relax the capacity
// bound that §H.25 identified -- only the addressing changes.
#ifndef FUSE_T
#define FUSE_T 6
#endif
#ifndef FUSE_C
#define FUSE_C 32
#endif
#define FT2 (FUSE_T + 2)
#define FUSE_CS ((FUSE_C + 3) & ~3)          /* padded channel stride of the NCHW scratch planes */
#define FW(OC, IC, T) weight[((((OC) >> 2) * FUSE_C + (IC)) * 9 + (T)) * 4 + ((OC) & 3)]

__kernel __attribute__((reqd_work_group_size(FUSE_T, FUSE_T, 1)))
void conv_2d_nchw_fused2(GLOBAL_SIZE_2_DIMS
                         __global const FLOAT *input,    // NCHW
                         __global const FLOAT *weight,   // [ocb][ic][9][4]
                         __global const FLOAT *bias,
                         __global FLOAT *output,         // NCHW
                         __private const int batch,
                         __private const int height,
                         __private const int width) {
    __local COMPUTE_FLOAT mid[FUSE_C * FT2 * FT2];

    const int out_x = get_global_id(0);
    const int bh    = get_global_id(1);
    const int lx = get_local_id(0), ly = get_local_id(1);
    // gws is an exact multiple of lws, so there are no partial workgroups and no early return --
    // an early return here would leave threads out of the barrier below and hang the group.
    const int b     = bh / height;
    const int out_y = bh % height;

    const int mx0 = (out_x - lx) - 1;         // halo origin of the intermediate tile
    const int my0 = (out_y - ly) - 1;
    const int lid = ly * FUSE_T + lx, nthreads = FUSE_T * FUSE_T;
    const int plane = height * width;

    // ---- phase 1: conv1 over the (T+2)^2 halo, every channel, into LDS
    for (int p = lid; p < FT2 * FT2; p += nthreads) {
        const int ty = p / FT2, tx = p % FT2;
        const int my = my0 + ty, mx = mx0 + tx;
        // outside the image the intermediate is the zero pad conv2 will read
        const bool inside = (my >= 0 && my < height && mx >= 0 && mx < width);
        for (int oc0 = 0; oc0 < FUSE_C; oc0 += 4) {
            COMPUTE_FLOAT a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            if (inside) {
                a0 = (COMPUTE_FLOAT)bias[oc0 + 0]; a1 = (COMPUTE_FLOAT)bias[oc0 + 1];
                a2 = (COMPUTE_FLOAT)bias[oc0 + 2]; a3 = (COMPUTE_FLOAT)bias[oc0 + 3];
                for (int ic = 0; ic < FUSE_C; ic++) {
                    for (int kh = 0; kh < 3; kh++) {
                        const int iy = my + kh - 1;
                        if (iy < 0 || iy >= height) continue;
                        __global const FLOAT *row = input + ((b * FUSE_CS + ic) * height + iy) * width;
                        for (int kw = 0; kw < 3; kw++) {
                            const int ix = mx + kw - 1;
                            if (ix < 0 || ix >= width) continue;
                            const COMPUTE_FLOAT v = (COMPUTE_FLOAT)row[ix];
                            const int t = kh * 3 + kw;
                            a0 = mad(v, (COMPUTE_FLOAT)FW(oc0 + 0, ic, t), a0);
                            a1 = mad(v, (COMPUTE_FLOAT)FW(oc0 + 1, ic, t), a1);
                            a2 = mad(v, (COMPUTE_FLOAT)FW(oc0 + 2, ic, t), a2);
                            a3 = mad(v, (COMPUTE_FLOAT)FW(oc0 + 3, ic, t), a3);
                        }
                    }
                }
            }
            mid[((oc0 + 0) * FT2 + ty) * FT2 + tx] = a0;
            mid[((oc0 + 1) * FT2 + ty) * FT2 + tx] = a1;
            mid[((oc0 + 2) * FT2 + ty) * FT2 + tx] = a2;
            mid[((oc0 + 3) * FT2 + ty) * FT2 + tx] = a3;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- phase 2: conv2 for this thread's pixel, reading the intermediate from LDS
    for (int oc0 = 0; oc0 < FUSE_C; oc0 += 4) {
        COMPUTE_FLOAT a0 = (COMPUTE_FLOAT)bias[oc0 + 0], a1 = (COMPUTE_FLOAT)bias[oc0 + 1];
        COMPUTE_FLOAT a2 = (COMPUTE_FLOAT)bias[oc0 + 2], a3 = (COMPUTE_FLOAT)bias[oc0 + 3];
        for (int mc = 0; mc < FUSE_C; mc++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    const COMPUTE_FLOAT v = mid[(mc * FT2 + ly + kh) * FT2 + lx + kw];
                    const int t = kh * 3 + kw;
                    a0 = mad(v, (COMPUTE_FLOAT)FW(oc0 + 0, mc, t), a0);
                    a1 = mad(v, (COMPUTE_FLOAT)FW(oc0 + 1, mc, t), a1);
                    a2 = mad(v, (COMPUTE_FLOAT)FW(oc0 + 2, mc, t), a2);
                    a3 = mad(v, (COMPUTE_FLOAT)FW(oc0 + 3, mc, t), a3);
                }
            }
        }
        __global FLOAT *o = output + ((b * FUSE_CS + oc0) * height + out_y) * width + out_x;
        o[0]         = (FLOAT)a0;
        o[plane]     = (FLOAT)a1;
        o[2 * plane] = (FLOAT)a2;
        o[3 * plane] = (FLOAT)a3;
    }
}

// ---------------------------------------------------------------------------------------------
// conv_2d_implicit_gemm — LDS-tiled IMPLICIT GEMM 3x3 stride-1 conv.
//   env MNN_CONV_IGEMM=1        -> NC4HW4 input/output (MNN's native layout)
//   env MNN_CONV_IGEMM=nchw     -> NCHW input/output (via the existing conversion kernels)
//
// The GEMM is C[oc][m] = sum_k W[oc][k] * Im2Col[k][m] with k = (ic, kh, kw) and m = (b, y, x).
// IMPLICIT means the Im2Col matrix is never materialised: the A-tile is gathered straight out of
// the input while it is being staged into __local. That is the whole point -- §H.38's explicit
// im2col+GEMM lost largely because it streamed a 9x-inflated operand out of global memory.
//
// The other thing this tests, which no previous kernel here does: a GEMM tile REUSES each gathered
// column across all IG_NT output channels in the workgroup. Every direct conv_2d_c* kernel handles
// only 4 output channels per thread, so it re-reads the input once per output-channel block
// (8x for C=32, 12x for C=48). This is the untested LDS-tiled variant §H.38 called out.
//
// Layout is a compile-time switch and the ONLY thing that differs between the two variants, so the
// NC4HW4-vs-NCHW comparison is exact: same tiling, same math, same register budget.
//
// Tile: 64 pixels x 32 output channels per workgroup, 64 threads, each owning 8 pixels x 4 channels
// = 8 float4 accumulators -- deliberately the register class §H.22 measured as optimal.
// K-step is 9, i.e. exactly one input channel's taps, which keeps the gather index math cheap.
#define IG_MT 64          /* pixels per workgroup            */
#define IG_NT 32          /* output channels per workgroup   */
#define IG_TM 8           /* pixels per thread               */
#define IG_TN 4           /* output channels per thread      */
#define IG_KS 9           /* k-step = one input channel      */

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void conv_2d_implicit_gemm(GLOBAL_SIZE_2_DIMS
                           __global const FLOAT *input,
                           __global const FLOAT *weight,   /* [ocb][ic][9][4] */
                           __global const FLOAT *bias,
                           __global FLOAT *output,
                           __private const int in_channels,
                           __private const int out_channels,
                           __private const int batch,
                           __private const int height,
                           __private const int width
                           #ifdef PRELU
                           ,__global const FLOAT *slope_ptr
                           #endif
) {
    __local COMPUTE_FLOAT lA[IG_KS * IG_MT];
    __local COMPUTE_FLOAT lB[IG_KS * IG_NT];

    const int lid   = get_local_id(1) * 8 + get_local_id(0);   /* 0..63 */
    const int tid_m = lid & 7;          /* which 8-pixel strip  */
    const int tid_n = lid >> 3;         /* which 4-channel strip */

    const int m0  = get_group_id(0) * IG_MT;    /* first pixel  of this tile */
    const int oc0 = get_group_id(1) * IG_NT;    /* first out-ch of this tile */

    const int plane = height * width;
    const int M     = batch * plane;
    const int ocst  = (out_channels + 3) & ~3;

    COMPUTE_FLOAT4 acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,acc5=0,acc6=0,acc7=0;

    // Loop-invariant pixel decomposition, hoisted out of both the tap loop and the channel loop.
    // Thread `lid` stages the 9 taps of pixel (m0+lid), so this costs TWO integer divisions for
    // the whole kernel instead of two per element per channel step. Integer division has no
    // hardware instruction here, so leaving it in the inner loop is a large, purely artificial cost.
    const int mg_s  = m0 + lid;
    const int ok_s  = (mg_s < M);
    const int b_s   = ok_s ? (mg_s / plane) : 0;
    const int rem_s = mg_s - b_s * plane;
    const int y_s   = ok_s ? (rem_s / width) : 0;
    const int x_s   = rem_s - y_s * width;
    // B staging: 9x32 elements over 64 threads. n is a power-of-two split, so no division either.
    const int bn_s = lid & 31;
    const int bt_s = lid >> 5;          /* 0 or 1 -> two taps per pass, 9 taps in ceil(9/2) passes */

    for (int ic = 0; ic < in_channels; ic++) {
        // ---- stage A: gather this input channel's 9 taps for this thread's pixel, straight from
        //      the input. This is the implicit im2col -- no expanded buffer anywhere.
        for (int t = 0; t < IG_KS; t++) {
            COMPUTE_FLOAT v = 0;
            if (ok_s) {
                const int iy = y_s + (t / 3) - 1;
                const int ix = x_s + (t - (t / 3) * 3) - 1;
                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
#ifdef IGEMM_NCHW
                    const int cs = (in_channels + 3) & ~3;
                    v = (COMPUTE_FLOAT)input[((b_s * cs + ic) * height + iy) * width + ix];
#else
                    /* NC4HW4: channels packed by 4, so this is a strided scalar read */
                    v = (COMPUTE_FLOAT)input[((((ic >> 2) * batch + b_s) * height + iy) * width + ix) * 4
                                             + (ic & 3)];
#endif
                }
            }
            lA[t * IG_MT + lid] = v;
        }
        // ---- stage B: the matching 9x32 weight block
        for (int t = bt_s; t < IG_KS; t += 2) {
            const int oc = oc0 + bn_s;
            lB[t * IG_NT + bn_s] = (oc < out_channels)
                  ? (COMPUTE_FLOAT)weight[((((oc >> 2) * in_channels + ic) * 9 + t) * 4) + (oc & 3)]
                  : (COMPUTE_FLOAT)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- multiply: each gathered column is reused across all 32 output channels
        for (int t = 0; t < IG_KS; t++) {
            const COMPUTE_FLOAT4 wv = vload4(t * (IG_NT / 4) + tid_n, lB);
            __local const COMPUTE_FLOAT *a = lA + t * IG_MT + tid_m * IG_TM;
            acc0 = mad(wv, (COMPUTE_FLOAT4)(a[0]), acc0);
            acc1 = mad(wv, (COMPUTE_FLOAT4)(a[1]), acc1);
            acc2 = mad(wv, (COMPUTE_FLOAT4)(a[2]), acc2);
            acc3 = mad(wv, (COMPUTE_FLOAT4)(a[3]), acc3);
            acc4 = mad(wv, (COMPUTE_FLOAT4)(a[4]), acc4);
            acc5 = mad(wv, (COMPUTE_FLOAT4)(a[5]), acc5);
            acc6 = mad(wv, (COMPUTE_FLOAT4)(a[6]), acc6);
            acc7 = mad(wv, (COMPUTE_FLOAT4)(a[7]), acc7);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int oc = oc0 + tid_n * IG_TN;
    if (oc >= out_channels) return;
    COMPUTE_FLOAT4 bv = CONVERT_COMPUTE_FLOAT4(vload4(oc >> 2, bias));
    acc0 += bv; acc1 += bv; acc2 += bv; acc3 += bv;
    acc4 += bv; acc5 += bv; acc6 += bv; acc7 += bv;
#ifdef RELU
    acc0=fmax(acc0,(COMPUTE_FLOAT4)0); acc1=fmax(acc1,(COMPUTE_FLOAT4)0);
    acc2=fmax(acc2,(COMPUTE_FLOAT4)0); acc3=fmax(acc3,(COMPUTE_FLOAT4)0);
    acc4=fmax(acc4,(COMPUTE_FLOAT4)0); acc5=fmax(acc5,(COMPUTE_FLOAT4)0);
    acc6=fmax(acc6,(COMPUTE_FLOAT4)0); acc7=fmax(acc7,(COMPUTE_FLOAT4)0);
#endif
#ifdef RELU6
    acc0=clamp(acc0,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); acc1=clamp(acc1,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    acc2=clamp(acc2,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); acc3=clamp(acc3,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    acc4=clamp(acc4,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); acc5=clamp(acc5,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
    acc6=clamp(acc6,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6); acc7=clamp(acc7,(COMPUTE_FLOAT4)0,(COMPUTE_FLOAT4)6);
#endif
#ifdef PRELU
    { COMPUTE_FLOAT4 sv = CONVERT_COMPUTE_FLOAT4(vload4(oc >> 2, slope_ptr));
      acc0=select(acc0*sv,acc0,acc0>=0); acc1=select(acc1*sv,acc1,acc1>=0);
      acc2=select(acc2*sv,acc2,acc2>=0); acc3=select(acc3*sv,acc3,acc3>=0);
      acc4=select(acc4*sv,acc4,acc4>=0); acc5=select(acc5*sv,acc5,acc5>=0);
      acc6=select(acc6*sv,acc6,acc6>=0); acc7=select(acc7*sv,acc7,acc7>=0);
    }
#endif

    // ---- store: 8 pixels, 4 consecutive output channels each
    /* one decomposition for the thread's 8 consecutive pixels, then walk x/y forward */
    const int mg_o  = m0 + tid_m * IG_TM;
    const int b_o   = (mg_o < M) ? (mg_o / plane) : 0;
    const int rem_o = mg_o - b_o * plane;
    const int y_o   = (mg_o < M) ? (rem_o / width) : 0;
    const int x_o   = rem_o - y_o * width;
    #define IG_ST(I, A)                                                                           \
    { const int mg = mg_o + (I);                                                                  \
      if (mg < M) {                                                                               \
        int xx = x_o + (I), yy = y_o, bb = b_o;                                                   \
        while (xx >= width)  { xx -= width;  yy++; }                                              \
        while (yy >= height) { yy -= height; bb++; }                                              \
        _IG_WRITE(bb, yy, xx, A)                                                                  \
      } }
#ifdef IGEMM_NCHW
    #define _IG_WRITE(B, Y, X, A)                                                                 \
        { __global FLOAT *o = output + (((B) * ocst + oc) * height + (Y)) * width + (X);           \
          o[0] = (FLOAT)(A).x; o[plane] = (FLOAT)(A).y;                                            \
          o[2*plane] = (FLOAT)(A).z; o[3*plane] = (FLOAT)(A).w; }
#else
    #define _IG_WRITE(B, Y, X, A)                                                                 \
        vstore4(CONVERT_FLOAT4(A), 0,                                                              \
                output + (((((oc >> 2) * batch + (B)) * height + (Y)) * width + (X)) * 4));
#endif
    IG_ST(0,acc0) IG_ST(1,acc1) IG_ST(2,acc2) IG_ST(3,acc3)
    IG_ST(4,acc4) IG_ST(5,acc5) IG_ST(6,acc6) IG_ST(7,acc7)
    #undef IG_ST
    #undef _IG_WRITE
}
