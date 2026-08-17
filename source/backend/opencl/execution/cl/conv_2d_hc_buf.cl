#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

// =============================================================================================
// Shape-specialised copies of MNN's stock conv kernels, in their OWN program so conv_2d_buf.cl
// is never touched. Every shape value (HCINH/HCOUTW/HCICB/HCSH/...) arrives as a -D build
// option: the runtime expression by default, a literal under MNN_CONV_HARD=1. With literals the
// compiler folds all index arithmetic, bounds the channel/filter loops and drops halo branches.
// Deliberate duplication -- the originals stay byte-identical.
// =============================================================================================

__kernel
void conv_2d_c4h1w1_hc(GLOBAL_SIZE_2_DIMS
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

    const int out_c_idx = out_c_w_idx / HCOUTW + out_c_base_index;
    if(out_c_idx >= HCOCB) return;
    const int out_w_idx = out_c_w_idx % HCOUTW;
    const int out_b_idx = out_b_h_idx / HCOUTH;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % HCOUTH;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    
    const int in_w_idx_base = mad24(out_w_idx, HCSW, -HCPW);
    const int in_h_idx_base = mad24(out_h_idx, HCSH, -HCPH);
    
    const int kw_start = select(0, (-in_w_idx_base + HCDW - 1) / HCDW, in_w_idx_base < 0);
    const int kh_start = select(0, (-in_h_idx_base + HCDH - 1) / HCDH, in_h_idx_base < 0);

    const int in_w_idx_start = mad24(kw_start, HCDW, in_w_idx_base);
    const int in_w_idx_end = min(mad24(HCFW, HCDW, in_w_idx_base), HCINW);
    
    const int in_h_idx_start = mad24(kh_start, HCDH, in_h_idx_base);
    const int in_h_idx_end = min(mad24(HCFH, HCDH, in_h_idx_base), HCINH);
    
    const int weight_oc_offset = HCOCB * HCFH * HCFW * 4;
    for(ushort in_c_idx = 0; in_c_idx < HCICB; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* HCOCB + out_c_idx) *HCFH + kh_start)*HCFW + kw_start) * 4;
        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += HCDH) {
            for(int ix = in_w_idx_start; ix < in_w_idx_end; ix += HCDW) {
                int inp_offset = (((out_b_idx + in_c_idx * HCBATCH) * HCINH + iy) * HCINW + ix) * 4;
                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset));
                
                const int filter_w_inc = (ix-in_w_idx_start)/HCDW;

                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(vload4(filter_w_inc, weight+weight_offset));
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(vload4(filter_w_inc, weight+weight_offset+weight_oc_offset));
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(vload4(filter_w_inc, weight+weight_offset+weight_oc_offset*2));
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(vload4(filter_w_inc, weight+weight_offset+weight_oc_offset*3));

                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);

            }
            weight_offset += 4*HCFW;
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
    const int out_offset = (((out_b_idx + out_c_idx*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
 
}

__kernel
void conv_2d_c4h1w2_hc(GLOBAL_SIZE_2_DIMS
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

    const int out_c_idx = out_c_w_idx / HCWB + out_c_base_index;
    if(out_c_idx >= HCOCB) return;
    const int out_w_idx = (out_c_w_idx % HCWB) << 1;
    const int out_b_idx = out_b_h_idx / HCOUTH;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % HCOUTH;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out1 = out0;
    
    const int in_w0_idx_base = mad24(out_w_idx, HCSW, -HCPW);
    const int in_w1_idx_base = in_w0_idx_base + HCSW;

    const int in_h_idx_base = mad24(out_h_idx, HCSH, -HCPH);
    
    const int kh_start = select(0, (-in_h_idx_base + HCDH - 1) / HCDH, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, HCDH, in_h_idx_base);
    const int in_h_idx_end = min(mad24(HCFH, HCDH, in_h_idx_base), HCINH);
    
    const int weight_oc_offset = HCOCB * HCFH * HCFW * 4;
    for(ushort in_c_idx = 0; in_c_idx < HCICB; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* HCOCB + out_c_idx) *HCFH + kh_start)*HCFW + 0) * 4;

        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += HCDH) {
            const int inp_offset_base = (((out_b_idx + in_c_idx*HCBATCH) * HCINH + iy) * HCINW + 0) * 4;

            for(int fw = 0; fw < HCFW; fw++) {
                const int in_w0_idx = fw * HCDW + in_w0_idx_base;
                const int in_w1_idx = fw * HCDW + in_w1_idx_base;

                COMPUTE_FLOAT4 in0 = (in_w0_idx < 0 || in_w0_idx >= HCINW) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w0_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = (in_w1_idx < 0 || in_w1_idx >= HCINW) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w1_idx, input+inp_offset_base));
                
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

    const int out_offset = (((out_b_idx + out_c_idx*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    if(out_w_idx + 1 >= HCOUTW) return;
    vstore4(CONVERT_FLOAT4(out1), 1, output+out_offset);
#else
    vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
#endif
}

__kernel
void conv_2d_c4h1w4_hc(GLOBAL_SIZE_2_DIMS
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

    const int out_c_idx = out_c_w_idx / HCWB + out_c_base_index;
    if(out_c_idx >= HCOCB) return;
    const int out_w_idx = (out_c_w_idx % HCWB) << 2;
    const int out_b_idx = out_b_h_idx / HCOUTH;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % HCOUTH;

    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out1 = out0;
    COMPUTE_FLOAT4 out2 = out0;
    COMPUTE_FLOAT4 out3 = out0;

    const int in_w0_idx_base = mad24(out_w_idx, HCSW, -HCPW);
    const int in_w1_idx_base = in_w0_idx_base + HCSW;
    const int in_w2_idx_base = in_w1_idx_base + HCSW;
    const int in_w3_idx_base = in_w2_idx_base + HCSW;

    const int in_h_idx_base = mad24(out_h_idx, HCSH, -HCPH);
    
    const int kh_start = select(0, (-in_h_idx_base + HCDH - 1) / HCDH, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, HCDH, in_h_idx_base);
    const int in_h_idx_end = min(mad24(HCFH, HCDH, in_h_idx_base), HCINH);
    
    const int weight_oc_offset = HCOCB * HCFH * HCFW * 4;
    for(ushort in_c_idx = 0; in_c_idx < HCICB; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* HCOCB + out_c_idx) *HCFH + kh_start)*HCFW + 0) * 4;

        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += HCDH) {
            const int inp_offset_base = (((out_b_idx + in_c_idx*HCBATCH) * HCINH + iy) * HCINW + 0) * 4;

            for(int fw = 0; fw < HCFW; fw++) {
                const int in_w0_idx = fw * HCDW + in_w0_idx_base;
                const int in_w1_idx = fw * HCDW + in_w1_idx_base;
                const int in_w2_idx = fw * HCDW + in_w2_idx_base;
                const int in_w3_idx = fw * HCDW + in_w3_idx_base;

                COMPUTE_FLOAT4 in0 = (in_w0_idx < 0 || in_w0_idx >= HCINW) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w0_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = (in_w1_idx < 0 || in_w1_idx >= HCINW) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w1_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = (in_w2_idx < 0 || in_w2_idx >= HCINW) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w2_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = (in_w3_idx < 0 || in_w3_idx >= HCINW) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w3_idx, input+inp_offset_base));

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

    const int out_offset = (((out_b_idx + out_c_idx*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = HCOUTW - out_w_idx;

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
void conv_2d_c4h4w1_hc(GLOBAL_SIZE_2_DIMS
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

    const int out_c_idx = out_c_w_idx / HCWB + out_c_base_index;
    if(out_c_idx >= HCOCB) return;
    const int out_w_idx = out_c_w_idx % HCWB;
    const int out_b_idx = out_b_h_idx / HCHB;//equal to in_b_idx
    const int out_h_idx = (out_b_h_idx % HCHB) << 2;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out1 = out0;
    COMPUTE_FLOAT4 out2 = out0;
    COMPUTE_FLOAT4 out3 = out0;

    const int in_w_idx_base = mad24(out_w_idx, HCSW, -HCPW);

    const int in_h0_idx_base = mad24(out_h_idx, HCSH, -HCPH);
    const int in_h1_idx_base = in_h0_idx_base + HCSH;
    const int in_h2_idx_base = in_h1_idx_base + HCSH;
    const int in_h3_idx_base = in_h2_idx_base + HCSH;
    
    const int kw_start = select(0, (-in_w_idx_base + HCDW - 1) / HCDW, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, HCDW, in_w_idx_base);
    const int in_w_idx_end = min(mad24(HCFW, HCDW, in_w_idx_base), HCINW);
    
    const int weight_oc_offset = HCOCB * HCFH * HCFW * 4;
    const int in_hw_size = HCINH * HCINW;
#ifdef CONV_SPEC_UNROLL
    __attribute__((opencl_unroll_hint))
#endif
    for(ushort in_c_idx = 0; in_c_idx < HCICB; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        const int inp_offset_base = (out_b_idx + in_c_idx*HCBATCH) * HCINH * HCINW * 4;

        for(int iy = 0; iy < HCFH; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* HCOCB + out_c_idx) *HCFH + iy)*HCFW + kw_start) * 4;
            const int in_h0_idx = (iy * HCDH + in_h0_idx_base) * HCINW;
            const int in_h1_idx = (iy * HCDH + in_h1_idx_base) * HCINW;
            const int in_h2_idx = (iy * HCDH + in_h2_idx_base) * HCINW;
            const int in_h3_idx = (iy * HCDH + in_h3_idx_base) * HCINW;

            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += HCDW) {
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

    const int out_offset = (((out_b_idx + out_c_idx*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = HCOUTH - out_h_idx;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), HCOUTW, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * HCOUTW, output+out_offset);
        vstore4(CONVERT_FLOAT4(out3), 3 * HCOUTW, output+out_offset);
    }else if(remain == 3){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), HCOUTW, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * HCOUTW, output+out_offset);
    }else if(remain == 2){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), HCOUTW, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), HCOUTW, output+out_offset);
    vstore4(CONVERT_FLOAT4(out2), 2 * HCOUTW, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), 3 * HCOUTW, output+out_offset);
#endif
}

__kernel
void conv_2d_c8h4w1_hc(GLOBAL_SIZE_2_DIMS
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

    const int out_c_idx_0 = ((out_c_w_idx / HCWB + out_c_base_index) << 1);
    if(out_c_idx_0 >= HCOCB) return;
    const int out_c_idx_1 = out_c_idx_0 + 1;
    const int out_w_idx = out_c_w_idx % HCWB;
    const int out_b_idx = out_b_h_idx / HCHB;//equal to in_b_idx
    const int out_h_idx = (out_b_h_idx % HCHB) << 2;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, bias));
    COMPUTE_FLOAT4 out1 = out0;
    COMPUTE_FLOAT4 out2 = out0;
    COMPUTE_FLOAT4 out3 = out0;
    #ifdef CHANNEL_BOUNDARY_PROTECT
    COMPUTE_FLOAT4 out4 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #else
    COMPUTE_FLOAT4 out4 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #endif
    COMPUTE_FLOAT4 out5 = out4;
    COMPUTE_FLOAT4 out6 = out4;
    COMPUTE_FLOAT4 out7 = out4;

    const int in_w_idx_base = mad24(out_w_idx, HCSW, -HCPW);

    const int in_h0_idx_base = mad24(out_h_idx, HCSH, -HCPH);
    const int in_h1_idx_base = in_h0_idx_base + HCSH;
    const int in_h2_idx_base = in_h1_idx_base + HCSH;
    const int in_h3_idx_base = in_h2_idx_base + HCSH;
    
    const int kw_start = select(0, (-in_w_idx_base + HCDW - 1) / HCDW, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, HCDW, in_w_idx_base);
    const int in_w_idx_end = min(mad24(HCFW, HCDW, in_w_idx_base), HCINW);
    
    const int weight_oc_offset = HCFH * HCFW * 4;
    const int weight_ic_offset = HCOCB * weight_oc_offset;
    const int in_hw_size = HCINH * HCINW;
#ifdef CONV_SPEC_UNROLL
    __attribute__((opencl_unroll_hint))
#endif
    for(ushort in_c_idx = 0; in_c_idx < HCICB; in_c_idx++) {
        //weights  NC4HW4   [ic/4, ic_4, oc/4, kh*kw, oc_4]
        //index:   [0, 4*in_c_idx, out_c_idx_0*kh*kw + kh_start*kw + kw_start, 0]
        const int inp_offset_base = (out_b_idx + in_c_idx * HCBATCH) * HCINH * HCINW * 4;

        for(int iy = 0; iy < HCFH; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* HCOCB + out_c_idx_0) *HCFH + iy)*HCFW + kw_start) * 4;
            const int in_h0_idx = (iy * HCDH + in_h0_idx_base) * HCINW;
            const int in_h1_idx = (iy * HCDH + in_h1_idx_base) * HCINW;
            const int in_h2_idx = (iy * HCDH + in_h2_idx_base) * HCINW;
            const int in_h3_idx = (iy * HCDH + in_h3_idx_base) * HCINW;

            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += HCDW) {
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
                weight0 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                weight1 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                weight2 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                weight3 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
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

    int out_offset = (((out_b_idx + out_c_idx_0*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = HCOUTH - out_h_idx;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), HCOUTW, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * HCOUTW, output+out_offset);
        vstore4(CONVERT_FLOAT4(out3), 3 * HCOUTW, output+out_offset);
    }else if(remain == 3){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), HCOUTW, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * HCOUTW, output+out_offset);
    }else if(remain == 2){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), HCOUTW, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= HCOCB){
        return;
    }
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out5), HCOUTW, output+out_offset);
        vstore4(CONVERT_FLOAT4(out6), 2 * HCOUTW, output+out_offset);
        vstore4(CONVERT_FLOAT4(out7), 3 * HCOUTW, output+out_offset);
    }else if(remain == 3){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out5), HCOUTW, output+out_offset);
        vstore4(CONVERT_FLOAT4(out6), 2 * HCOUTW, output+out_offset);
    }else if(remain == 2){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out5), HCOUTW, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
    }
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), HCOUTW, output+out_offset);
    vstore4(CONVERT_FLOAT4(out2), 2 * HCOUTW, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), 3 * HCOUTW, output+out_offset);
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= HCOCB){
        return;
    }
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out5), HCOUTW, output+out_offset);
    vstore4(CONVERT_FLOAT4(out6), 2 * HCOUTW, output+out_offset);
    vstore4(CONVERT_FLOAT4(out7), 3 * HCOUTW, output+out_offset);
#endif
}

__kernel
void conv_2d_c8h2w1_hc(GLOBAL_SIZE_2_DIMS
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

    const int out_c_idx_0 = (out_c_w_idx / HCWB + out_c_base_index) << 1;
    if(out_c_idx_0 >= HCOCB) return;
    const int out_c_idx_1 = out_c_idx_0 + 1;
    const int out_w_idx = out_c_w_idx % HCWB;
    const int out_b_idx = out_b_h_idx / HCHB;//equal to in_b_idx
    const int out_h_idx = (out_b_h_idx % HCHB) << 1;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, bias));
    COMPUTE_FLOAT4 out1 = out0;
    #ifdef CHANNEL_BOUNDARY_PROTECT
    COMPUTE_FLOAT4 out2 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #else
    COMPUTE_FLOAT4 out2 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #endif
    COMPUTE_FLOAT4 out3 = out2;
    
    const int in_w_idx_base = mad24(out_w_idx, HCSW, -HCPW);

    const int in_h0_idx_base = mad24(out_h_idx, HCSH, -HCPH);
    const int in_h1_idx_base = in_h0_idx_base + HCSH;
    
    const int kw_start = select(0, (-in_w_idx_base + HCDW - 1) / HCDW, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, HCDW, in_w_idx_base);
    const int in_w_idx_end = min(mad24(HCFW, HCDW, in_w_idx_base), HCINW);
    
    const int weight_oc_offset = HCFH * HCFW * 4;
    const int weight_ic_offset = HCOCB * weight_oc_offset;
    const int in_hw_size = HCINH * HCINW;
    // weight: [ic/4, oc, 4], loop: ic/4
    for(ushort in_c_idx = 0; in_c_idx < HCICB; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx_0*kh*kw + kh_start*kw + kw_start, 0]
        const int inp_offset_base = (out_b_idx + in_c_idx*HCBATCH) * HCINH * HCINW * 4;

        for(int iy = 0; iy < HCFH; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* HCOCB + out_c_idx_0) *HCFH + iy)*HCFW + kw_start) * 4;
            const int in_h0_idx = (iy * HCDH + in_h0_idx_base) * HCINW;
            const int in_h1_idx = (iy * HCDH + in_h1_idx_base) * HCINW;

            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += HCDW) {
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
                weight0 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                weight1 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                weight2 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                weight3 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
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

    int out_offset = (((out_b_idx + out_c_idx_0*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = HCOUTH - out_h_idx;
    if(remain >= 2){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), HCOUTW, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= HCOCB){
        return;
    }
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
    if(remain >= 2){
        vstore4(CONVERT_FLOAT4(out2), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out3), HCOUTW, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out2), 0, output+out_offset);
    }
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), HCOUTW, output+out_offset);
    #ifdef CHANNEL_BOUNDARY_PROTECT
    if(out_c_idx_1 >= HCOCB){
        return;
    }
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out2), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), HCOUTW, output+out_offset);
#endif
}

__kernel
void conv_2d_c8h1w4_hc(GLOBAL_SIZE_2_DIMS
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

    const int out_c_idx_0 = (out_c_w_idx / HCWB + out_c_base_index) << 1;
    if(out_c_idx_0 >= HCOCB) return;
    const int out_c_idx_1 = out_c_idx_0 + 1;
    const int out_w_idx = (out_c_w_idx % HCWB) << 2;
    const int out_b_idx = out_b_h_idx / HCOUTH;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % HCOUTH;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_0, bias));
    COMPUTE_FLOAT4 out1 = out0;
    COMPUTE_FLOAT4 out2 = out0;
    COMPUTE_FLOAT4 out3 = out0;
    #ifdef CHANNEL_BOUNDARY_PROTECT
    COMPUTE_FLOAT4 out4 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #else
    COMPUTE_FLOAT4 out4 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx_1, bias));
    #endif
    COMPUTE_FLOAT4 out5 = out4;
    COMPUTE_FLOAT4 out6 = out4;
    COMPUTE_FLOAT4 out7 = out4;

    const int in_w0_idx_base = mad24(out_w_idx, HCSW, -HCPW);
    const int in_w1_idx_base = in_w0_idx_base + HCSW;
    const int in_w2_idx_base = in_w1_idx_base + HCSW;
    const int in_w3_idx_base = in_w2_idx_base + HCSW;

    const int in_h_idx_base = mad24(out_h_idx, HCSH, -HCPH);
    
    const int kh_start = select(0, (-in_h_idx_base + HCDH - 1) / HCDH, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, HCDH, in_h_idx_base);
    const int in_h_idx_end = min(mad24(HCFH, HCDH, in_h_idx_base), HCINH);
    
    const int weight_oc_offset = HCFH * HCFW * 4;
    const int weight_ic_offset = HCOCB * weight_oc_offset;
    for(ushort in_c_idx = 0; in_c_idx < HCICB; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx_0*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* HCOCB + out_c_idx_0) *HCFH + kh_start)*HCFW + 0) * 4;

        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += HCDH) {
            const int inp_offset_base = (((out_b_idx + in_c_idx * HCBATCH) * HCINH + iy) * HCINW + 0) * 4;

            for(int fw = 0; fw < HCFW; fw++) {
                const int in_w0_idx = fw * HCDW + in_w0_idx_base;
                const int in_w1_idx = fw * HCDW + in_w1_idx_base;
                const int in_w2_idx = fw * HCDW + in_w2_idx_base;
                const int in_w3_idx = fw * HCDW + in_w3_idx_base;

                COMPUTE_FLOAT4 in0 = (in_w0_idx < 0 || in_w0_idx >= HCINW) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w0_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = (in_w1_idx < 0 || in_w1_idx >= HCINW) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w1_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = (in_w2_idx < 0 || in_w2_idx >= HCINW) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w2_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = (in_w3_idx < 0 || in_w3_idx >= HCINW) ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(in_w3_idx, input+inp_offset_base));

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
                weight0 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset));
                weight1 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset));
                weight2 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2));
                weight3 = out_c_idx_1 >= HCOCB ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3));
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

    int out_offset = (((out_b_idx + out_c_idx_0*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = HCOUTW - out_w_idx;
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
    if(out_c_idx_1 >= HCOCB)return;
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
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
    if(out_c_idx_1 >= HCOCB)return;
    #endif
    out_offset = (((out_b_idx + (out_c_idx_1)*HCBATCH)*HCOUTH + out_h_idx)*HCOUTW + out_w_idx)*4;
    vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out4, out5, out6, out7)), 0, output+out_offset);
#endif
}
