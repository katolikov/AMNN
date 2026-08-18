#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

// [dstChannel, srcChannel, 3, 3] -> [4x4, srcChannelPad, dstChannelpad] (N, Kpad, Npad)
__kernel void winoTransWeightBuf2_3_1(GLOBAL_SIZE_DIM2
                              __global const float* input, // 0
                              __global FLOAT* output,
                              __private const int srcChannel, // 3
                              __private const int dstChannel,
                              __private const int srcChannelPad, // 6
                              __private const int dstChannelPad
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);
    
    const int src_c = pos.x;
    const int dst_c = pos.y;
    
    const int out_offset = (0 * srcChannelPad + src_c) * dstChannelPad + dst_c;
    const int out_offset_add = srcChannelPad * dstChannelPad;
    if(src_c >= srcChannel || dst_c >= dstChannel) {
        for(int i = 0; i < 16; i++) {
            output[out_offset + i * out_offset_add] = (FLOAT)0;
        }
        return;
    }
    
    const int in_offset = (dst_c * srcChannel + src_c) * 9;
    FLOAT8 in = CONVERT_FLOAT8(vload8(0, input + in_offset));
    FLOAT in8 = input[in_offset+8];
    
    FLOAT GB_00 = in.s0;
    FLOAT GB_01 = in.s1;
    FLOAT GB_02 = in.s2;
    FLOAT GB_10 = in.s0 + in.s3 + in.s6;
    FLOAT GB_11 = in.s1 + in.s4 + in.s7;
    FLOAT GB_12 = in.s2 + in.s5 + in8;
    FLOAT GB_20 = in.s0 - in.s3 + in.s6;
    FLOAT GB_21 = in.s1 - in.s4 + in.s7;
    FLOAT GB_22 = in.s2 - in.s5 + in8;
    FLOAT GB_30 = in.s6;
    FLOAT GB_31 = in.s7;
    FLOAT GB_32 = in8;
    
    FLOAT GBGT_00 = GB_00;
    FLOAT GBGT_01 = GB_00 + GB_01  + GB_02;
    FLOAT GBGT_02 = GB_00 - GB_01  + GB_02;
    FLOAT GBGT_03 = GB_02;
    
    FLOAT GBGT_10 = GB_10;
    FLOAT GBGT_11 = GB_10 + GB_11  + GB_12;
    FLOAT GBGT_12 = GB_10 - GB_11  + GB_12;
    FLOAT GBGT_13 = GB_12;
    
    FLOAT GBGT_20 = GB_20;
    FLOAT GBGT_21 = GB_20 + GB_21  + GB_22;
    FLOAT GBGT_22 = GB_20 - GB_21  + GB_22;
    FLOAT GBGT_23 = GB_22;
    
    FLOAT GBGT_30 = GB_30;
    FLOAT GBGT_31 = GB_30 + GB_31  + GB_32;
    FLOAT GBGT_32 = GB_30 - GB_31  + GB_32;
    FLOAT GBGT_33 = GB_32;

    output[out_offset + 0 * out_offset_add] = GBGT_00;
    output[out_offset + 1 * out_offset_add] = GBGT_01;
    output[out_offset + 2 * out_offset_add] = GBGT_02;
    output[out_offset + 3 * out_offset_add] = GBGT_03;
    output[out_offset + 4 * out_offset_add] = GBGT_10;
    output[out_offset + 5 * out_offset_add] = GBGT_11;
    output[out_offset + 6 * out_offset_add] = GBGT_12;
    output[out_offset + 7 * out_offset_add] = GBGT_13;
    output[out_offset + 8 * out_offset_add] = GBGT_20;
    output[out_offset + 9 * out_offset_add] = GBGT_21;
    output[out_offset + 10 * out_offset_add] = GBGT_22;
    output[out_offset + 11 * out_offset_add] = GBGT_23;
    output[out_offset + 12 * out_offset_add] = GBGT_30;
    output[out_offset + 13 * out_offset_add] = GBGT_31;
    output[out_offset + 14 * out_offset_add] = GBGT_32;
    output[out_offset + 15 * out_offset_add] = GBGT_33;
}

__kernel void winoTransSrcBuf2_3_1(GLOBAL_SIZE_DIM2
                                      __global const FLOAT* uInput, // 0
                                      __global FLOAT* uOutput, __private const int unitWidth,
                                      __private const int unitHeight, // 3
                                      __private const int padX, __private const int padY,
                                      __private const int srcWidth, // 6
                                      __private const int srcHeight, __private const int srcChannelC4,
                                      __private const int dstHeightPad, __private const int srcChannelPad,
                                      __private const int batch,
                                      __private const int batchOffset) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); 
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);
    
    if(pos.x >= unitWidth * unitHeight || pos.y >= srcChannelC4) {
        return;
    }
    int unitWidth_idx = pos.x % unitWidth;
    int unitHeight_idx = pos.x / unitWidth;
    int2 realPos   = (int2)(unitWidth_idx, unitHeight_idx);
    int dstXOrigin = pos.y;
    int batchIndex = pos.y / srcChannelC4;
    int srcZ       = pos.y % srcChannelC4;
    int dstYOrigin = unitWidth * unitHeight_idx + unitWidth_idx;

    batchIndex = batchOffset;
    {
        int sxStart = (realPos.x) * 2 - padX;
        int syStart = (realPos.y) * 2 - padY;
        FLOAT4 S00;
        FLOAT4 S10;
        FLOAT4 S20;
        FLOAT4 S30;
        FLOAT4 S01;
        FLOAT4 S11;
        FLOAT4 S21;
        FLOAT4 S31;
        FLOAT4 S02;
        FLOAT4 S12;
        FLOAT4 S22;
        FLOAT4 S32;
        FLOAT4 S03;
        FLOAT4 S13;
        FLOAT4 S23;
        FLOAT4 S33;
        
        int inp_offset = (((batchIndex + srcZ * batch) * srcHeight + syStart) * srcWidth + sxStart) * 4;
        {
            int sx      = 0 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S00         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset);
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S10         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4);
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S20         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8);
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 0 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S30         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12);
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S01         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4*srcWidth);
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S11         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4*srcWidth+4);
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S21         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4*srcWidth+8);
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S31         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4*srcWidth+12);
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S02         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8*srcWidth);
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S12         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8*srcWidth+4);
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S22         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8*srcWidth+8);
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S32         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8*srcWidth+12);
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 3 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S03         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12*srcWidth);
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 3 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S13         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12*srcWidth+4);
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 3 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S23         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12*srcWidth+8);
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 3 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S33         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12*srcWidth+12);
        }
        FLOAT4 m00 = +S00 - S02;
        FLOAT4 m10 = +S10 - S12;
        FLOAT4 m20 = +S20 - S22;
        FLOAT4 m30 = +S30 - S32;
        FLOAT4 m01 = +(FLOAT)0.5f * S01 + (FLOAT)0.5f * S02;
        FLOAT4 m11 = +(FLOAT)0.5f * S11 + (FLOAT)0.5f * S12;
        FLOAT4 m21 = +(FLOAT)0.5f * S21 + (FLOAT)0.5f * S22;
        FLOAT4 m31 = +(FLOAT)0.5f * S31 + (FLOAT)0.5f * S32;
        FLOAT4 m02 = -(FLOAT)0.5f * S01 + (FLOAT)0.5f * S02;
        FLOAT4 m12 = -(FLOAT)0.5f * S11 + (FLOAT)0.5f * S12;
        FLOAT4 m22 = -(FLOAT)0.5f * S21 + (FLOAT)0.5f * S22;
        FLOAT4 m32 = -(FLOAT)0.5f * S31 + (FLOAT)0.5f * S32;
        FLOAT4 m03 = -S01 + S03;
        FLOAT4 m13 = -S11 + S13;
        FLOAT4 m23 = -S21 + S23;
        FLOAT4 m33 = -S31 + S33;
        
        //NC4HW4 [alpha*alpha, srcChannelPad, dstHeightPad]
        //index: [0,           dstXOrigin,   dstY,      dstYOrigin % 4]

        int out_offset = (0*srcChannelPad + 4*dstXOrigin) * dstHeightPad + dstYOrigin;
        int batch_offset = srcChannelPad*dstHeightPad;
        
        FLOAT4 res = (+m00 - m20);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;

        out_offset += batch_offset;
        res = (+(FLOAT)0.5f * m10 + (FLOAT)0.5f * m20);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-(FLOAT)0.5f * m10 + (FLOAT)0.5f * m20);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-m10 + m30);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        
        out_offset += batch_offset;
        res = (+m01 - m21);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (+(FLOAT)0.5f * m11 + (FLOAT)0.5f * m21);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-(FLOAT)0.5f * m11 + (FLOAT)0.5f * m21);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-m11 + m31);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (+m02 - m22);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (+(FLOAT)0.5f * m12 + (FLOAT)0.5f * m22);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-(FLOAT)0.5f * m12 + (FLOAT)0.5f * m22);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-m12 + m32);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (+m03 - m23);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (+(FLOAT)0.5f * m13 + (FLOAT)0.5f * m23);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-(FLOAT)0.5f * m13 + (FLOAT)0.5f * m23);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-m13 + m33);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
    }
}


__kernel void winoTransDstBuf2_3_1(GLOBAL_SIZE_DIM2
                                    __global const FLOAT* uInput,
                                    __global const FLOAT* uBias,
#ifdef PRELU
                                    __global const FLOAT* uSlope,
#endif
                                    __global FLOAT* uOutput,
                                    __private const int unitWidth, //wUnit
                                    __private const int unitHeight, //hUnit
                                    __private const int dstWidth,
                                    __private const int dstHeight,
                                    __private const int dstChannelC4,
                                    __private const int srcWidthPad,
                                    __private const int dstChannelPad,
                                    __private const int batch,
                                    __private const int batchOffset) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    int unitWidth_idx = pos.x % unitWidth;
    int unitHeight_idx = pos.x / unitWidth;
    int2 realPos   = (int2)(unitWidth_idx, unitHeight_idx);
    int dstXOrigin = unitWidth * unitHeight_idx + unitWidth_idx;
    int oz         = pos.y % dstChannelC4;
    
    FLOAT4 bias    = vload4(0, uBias+oz*4);
#ifdef PRELU
    FLOAT4 slope   = vload4(0, uSlope+oz*4);
#endif
    int batchIndex = pos.y / dstChannelC4;

    batchIndex = batchOffset;
    {
        int oyStart = realPos.y * 2;
        int oxStart = realPos.x * 2;
        
        // [alpha2, srcWidthPad, dstChannelPad]
        //index: [0, dstXOrigin, 4*oz]

        const int inp_offset = (0 * srcWidthPad + dstXOrigin) * dstChannelPad + 4*oz;
        const int b_offset = dstChannelPad*srcWidthPad;

        FLOAT4 S00  = vload4(0, uInput+inp_offset+b_offset*0);
        FLOAT4 S10  = vload4(0, uInput+inp_offset+b_offset*1);
        FLOAT4 S20  = vload4(0, uInput+inp_offset+b_offset*2);
        FLOAT4 S30  = vload4(0, uInput+inp_offset+b_offset*3);
        FLOAT4 S01  = vload4(0, uInput+inp_offset+b_offset*4);
        FLOAT4 S11  = vload4(0, uInput+inp_offset+b_offset*5);
        FLOAT4 S21  = vload4(0, uInput+inp_offset+b_offset*6);
        FLOAT4 S31  = vload4(0, uInput+inp_offset+b_offset*7);
        FLOAT4 S02  = vload4(0, uInput+inp_offset+b_offset*8);
        FLOAT4 S12  = vload4(0, uInput+inp_offset+b_offset*9);
        FLOAT4 S22  = vload4(0, uInput+inp_offset+b_offset*10);
        FLOAT4 S32  = vload4(0, uInput+inp_offset+b_offset*11);
        FLOAT4 S03  = vload4(0, uInput+inp_offset+b_offset*12);
        FLOAT4 S13  = vload4(0, uInput+inp_offset+b_offset*13);
        FLOAT4 S23  = vload4(0, uInput+inp_offset+b_offset*14);
        FLOAT4 S33  = vload4(0, uInput+inp_offset+b_offset*15);

        FLOAT4 m00  = +S00 + S01 + S02;
        FLOAT4 m10  = +S10 + S11 + S12;
        FLOAT4 m20  = +S20 + S21 + S22;
        FLOAT4 m30  = +S30 + S31 + S32;
        FLOAT4 m01  = +S01 - S02 + S03;
        FLOAT4 m11  = +S11 - S12 + S13;
        FLOAT4 m21  = +S21 - S22 + S23;
        FLOAT4 m31  = +S31 - S32 + S33;
        
        //NC4HW4 [batch, dstChannelC4, dstHeight, dstWidth]
        //index: [batchIndex, oz,      oyStart,   oxStart]
        int out_offset = (((batchIndex + oz * batch) * dstHeight + oyStart) * dstWidth + oxStart)*4;
        {
            int ox = oxStart + 0;
            int oy = oyStart + 0;
            if (ox < dstWidth && oy < dstHeight) {
                FLOAT4 res  = bias + m00 + m10 + m20;
#ifdef RELU
                res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
#ifdef PRELU
                res = fmax(res, (FLOAT4)(0)) + slope * fmin(res, (FLOAT4)(0));
#endif
                vstore4(res, 0, uOutput+out_offset);
            }
        }
        {
            int ox = oxStart + 1;
            int oy = oyStart + 0;
            if (ox < dstWidth && oy < dstHeight) {
                FLOAT4 res  = bias + m10 - m20 + m30;
#ifdef RELU
                res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
#ifdef PRELU
                res = fmax(res, (FLOAT4)(0)) + slope * fmin(res, (FLOAT4)(0));
#endif
                vstore4(res, 0, uOutput+out_offset+4);
            }
        }
        {
            int ox = oxStart + 0;
            int oy = oyStart + 1;
            if (ox < dstWidth && oy < dstHeight) {
                FLOAT4 res  = bias + m01 + m11 + m21;
#ifdef RELU
                res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
#ifdef PRELU
                res = fmax(res, (FLOAT4)(0)) + slope * fmin(res, (FLOAT4)(0));
#endif
                vstore4(res, 0, uOutput+out_offset+4*dstWidth);
            }
        }
        {
            int ox = oxStart + 1;
            int oy = oyStart + 1;
            if (ox < dstWidth && oy < dstHeight) {
                FLOAT4 res  = bias + m11 - m21 + m31;
#ifdef RELU
                res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
#ifdef PRELU
                res = fmax(res, (FLOAT4)(0)) + slope * fmin(res, (FLOAT4)(0));
#endif
                vstore4(res, 0, uOutput+out_offset+4*dstWidth+4);
            }
        }
    }
}


__kernel void winoTransSrcBuf2_3_1_w2(GLOBAL_SIZE_DIM2
                                      __global const FLOAT* uInput, // 0
                                      __global FLOAT* uOutput, __private const int unitWidth,
                                      __private const int unitHeight, // 3
                                      __private const int padX, __private const int padY,
                                      __private const int srcWidth, // 6
                                      __private const int srcHeight, __private const int srcChannelC4,
                                      __private const int dstHeightPad, __private const int srcChannelPad,
                                      __private const int batch,
                                      __private const int batchOffset) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    const int wPair = (unitWidth + 1) >> 1;   // tile PAIRS per unit row
    if(pos.x >= wPair * unitHeight || pos.y >= srcChannelC4) {
        return;
    }
    const int pairIdx        = pos.x % wPair;
    const int unitHeight_idx = pos.x / wPair;
    const int uw0 = pairIdx << 1;
    const int uw1 = uw0 + 1;
    const bool hasSecond = (uw1 < unitWidth);

    const int dstXOrigin = pos.y;
    const int srcZ       = pos.y % srcChannelC4;
    const int batchIndex = batchOffset;
    const int dstYOrigin = unitWidth * unitHeight_idx + uw0;

    const int sxStart = uw0 * 2 - padX;   // first of SIX columns covering both tiles
    const int syStart = unitHeight_idx * 2 - padY;
    const int inp_offset = (((batchIndex + srcZ * batch) * srcHeight + syStart) * srcWidth + sxStart) * 4;

    FLOAT4 C00, C01, C02, C03;
    FLOAT4 C10, C11, C12, C13;
    FLOAT4 C20, C21, C22, C23;
    FLOAT4 C30, C31, C32, C33;
    FLOAT4 C40, C41, C42, C43;
    FLOAT4 C50, C51, C52, C53;

    // Interior fast path: the whole 6x4 footprint is inside, so every bounds test is provably
    // false. This is the majority of tiles -- only the border ring takes the checked path.
    if(sxStart >= 0 && sxStart + 5 < srcWidth && syStart >= 0 && syStart + 3 < srcHeight) {
        C00 = vload4(0, uInput+inp_offset+0+0*srcWidth);
        C10 = vload4(0, uInput+inp_offset+4+0*srcWidth);
        C20 = vload4(0, uInput+inp_offset+8+0*srcWidth);
        C30 = vload4(0, uInput+inp_offset+12+0*srcWidth);
        C40 = vload4(0, uInput+inp_offset+16+0*srcWidth);
        C50 = vload4(0, uInput+inp_offset+20+0*srcWidth);
        C01 = vload4(0, uInput+inp_offset+0+4*srcWidth);
        C11 = vload4(0, uInput+inp_offset+4+4*srcWidth);
        C21 = vload4(0, uInput+inp_offset+8+4*srcWidth);
        C31 = vload4(0, uInput+inp_offset+12+4*srcWidth);
        C41 = vload4(0, uInput+inp_offset+16+4*srcWidth);
        C51 = vload4(0, uInput+inp_offset+20+4*srcWidth);
        C02 = vload4(0, uInput+inp_offset+0+8*srcWidth);
        C12 = vload4(0, uInput+inp_offset+4+8*srcWidth);
        C22 = vload4(0, uInput+inp_offset+8+8*srcWidth);
        C32 = vload4(0, uInput+inp_offset+12+8*srcWidth);
        C42 = vload4(0, uInput+inp_offset+16+8*srcWidth);
        C52 = vload4(0, uInput+inp_offset+20+8*srcWidth);
        C03 = vload4(0, uInput+inp_offset+0+12*srcWidth);
        C13 = vload4(0, uInput+inp_offset+4+12*srcWidth);
        C23 = vload4(0, uInput+inp_offset+8+12*srcWidth);
        C33 = vload4(0, uInput+inp_offset+12+12*srcWidth);
        C43 = vload4(0, uInput+inp_offset+16+12*srcWidth);
        C53 = vload4(0, uInput+inp_offset+20+12*srcWidth);
    } else {
        C00 = (sxStart+0 < 0 || sxStart+0 >= srcWidth || syStart+0 < 0 || syStart+0 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+0+0*srcWidth);
        C10 = (sxStart+1 < 0 || sxStart+1 >= srcWidth || syStart+0 < 0 || syStart+0 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4+0*srcWidth);
        C20 = (sxStart+2 < 0 || sxStart+2 >= srcWidth || syStart+0 < 0 || syStart+0 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8+0*srcWidth);
        C30 = (sxStart+3 < 0 || sxStart+3 >= srcWidth || syStart+0 < 0 || syStart+0 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12+0*srcWidth);
        C40 = (sxStart+4 < 0 || sxStart+4 >= srcWidth || syStart+0 < 0 || syStart+0 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+16+0*srcWidth);
        C50 = (sxStart+5 < 0 || sxStart+5 >= srcWidth || syStart+0 < 0 || syStart+0 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+20+0*srcWidth);
        C01 = (sxStart+0 < 0 || sxStart+0 >= srcWidth || syStart+1 < 0 || syStart+1 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+0+4*srcWidth);
        C11 = (sxStart+1 < 0 || sxStart+1 >= srcWidth || syStart+1 < 0 || syStart+1 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4+4*srcWidth);
        C21 = (sxStart+2 < 0 || sxStart+2 >= srcWidth || syStart+1 < 0 || syStart+1 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8+4*srcWidth);
        C31 = (sxStart+3 < 0 || sxStart+3 >= srcWidth || syStart+1 < 0 || syStart+1 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12+4*srcWidth);
        C41 = (sxStart+4 < 0 || sxStart+4 >= srcWidth || syStart+1 < 0 || syStart+1 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+16+4*srcWidth);
        C51 = (sxStart+5 < 0 || sxStart+5 >= srcWidth || syStart+1 < 0 || syStart+1 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+20+4*srcWidth);
        C02 = (sxStart+0 < 0 || sxStart+0 >= srcWidth || syStart+2 < 0 || syStart+2 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+0+8*srcWidth);
        C12 = (sxStart+1 < 0 || sxStart+1 >= srcWidth || syStart+2 < 0 || syStart+2 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4+8*srcWidth);
        C22 = (sxStart+2 < 0 || sxStart+2 >= srcWidth || syStart+2 < 0 || syStart+2 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8+8*srcWidth);
        C32 = (sxStart+3 < 0 || sxStart+3 >= srcWidth || syStart+2 < 0 || syStart+2 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12+8*srcWidth);
        C42 = (sxStart+4 < 0 || sxStart+4 >= srcWidth || syStart+2 < 0 || syStart+2 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+16+8*srcWidth);
        C52 = (sxStart+5 < 0 || sxStart+5 >= srcWidth || syStart+2 < 0 || syStart+2 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+20+8*srcWidth);
        C03 = (sxStart+0 < 0 || sxStart+0 >= srcWidth || syStart+3 < 0 || syStart+3 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+0+12*srcWidth);
        C13 = (sxStart+1 < 0 || sxStart+1 >= srcWidth || syStart+3 < 0 || syStart+3 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4+12*srcWidth);
        C23 = (sxStart+2 < 0 || sxStart+2 >= srcWidth || syStart+3 < 0 || syStart+3 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8+12*srcWidth);
        C33 = (sxStart+3 < 0 || sxStart+3 >= srcWidth || syStart+3 < 0 || syStart+3 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12+12*srcWidth);
        C43 = (sxStart+4 < 0 || sxStart+4 >= srcWidth || syStart+3 < 0 || syStart+3 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+16+12*srcWidth);
        C53 = (sxStart+5 < 0 || sxStart+5 >= srcWidth || syStart+3 < 0 || syStart+3 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+20+12*srcWidth);
    }

    int out_offset = (0 * srcChannelPad + 4 * dstXOrigin) * dstHeightPad + dstYOrigin;
    const int batch_offset = srcChannelPad * dstHeightPad;

    // ---- tile a: input columns 0..3
    FLOAT4 am00 = C00 - C02;
    FLOAT4 am01 = (FLOAT)0.5f * C01 + (FLOAT)0.5f * C02;
    FLOAT4 am02 = -(FLOAT)0.5f * C01 + (FLOAT)0.5f * C02;
    FLOAT4 am03 = -C01 + C03;
    FLOAT4 am10 = C10 - C12;
    FLOAT4 am11 = (FLOAT)0.5f * C11 + (FLOAT)0.5f * C12;
    FLOAT4 am12 = -(FLOAT)0.5f * C11 + (FLOAT)0.5f * C12;
    FLOAT4 am13 = -C11 + C13;
    FLOAT4 am20 = C20 - C22;
    FLOAT4 am21 = (FLOAT)0.5f * C21 + (FLOAT)0.5f * C22;
    FLOAT4 am22 = -(FLOAT)0.5f * C21 + (FLOAT)0.5f * C22;
    FLOAT4 am23 = -C21 + C23;
    FLOAT4 am30 = C30 - C32;
    FLOAT4 am31 = (FLOAT)0.5f * C31 + (FLOAT)0.5f * C32;
    FLOAT4 am32 = -(FLOAT)0.5f * C31 + (FLOAT)0.5f * C32;
    FLOAT4 am33 = -C31 + C33;

    // ---- tile b: input columns 2..5
    FLOAT4 bm00 = C20 - C22;
    FLOAT4 bm01 = (FLOAT)0.5f * C21 + (FLOAT)0.5f * C22;
    FLOAT4 bm02 = -(FLOAT)0.5f * C21 + (FLOAT)0.5f * C22;
    FLOAT4 bm03 = -C21 + C23;
    FLOAT4 bm10 = C30 - C32;
    FLOAT4 bm11 = (FLOAT)0.5f * C31 + (FLOAT)0.5f * C32;
    FLOAT4 bm12 = -(FLOAT)0.5f * C31 + (FLOAT)0.5f * C32;
    FLOAT4 bm13 = -C31 + C33;
    FLOAT4 bm20 = C40 - C42;
    FLOAT4 bm21 = (FLOAT)0.5f * C41 + (FLOAT)0.5f * C42;
    FLOAT4 bm22 = -(FLOAT)0.5f * C41 + (FLOAT)0.5f * C42;
    FLOAT4 bm23 = -C41 + C43;
    FLOAT4 bm30 = C50 - C52;
    FLOAT4 bm31 = (FLOAT)0.5f * C51 + (FLOAT)0.5f * C52;
    FLOAT4 bm32 = -(FLOAT)0.5f * C51 + (FLOAT)0.5f * C52;
    FLOAT4 bm33 = -C51 + C53;

    // Row-direction transform + store. The two tiles land on CONSECUTIVE dstYOrigin slots, so
    // each component pair is one vstore2 instead of two scalar stores.
    FLOAT4 ra, rb;
    FLOAT2 p;
    ra = am00 - am20;
    rb = bm00 - bm20;
    {
        int o = out_offset + 0 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = (FLOAT)0.5f * am10 + (FLOAT)0.5f * am20;
    rb = (FLOAT)0.5f * bm10 + (FLOAT)0.5f * bm20;
    {
        int o = out_offset + 1 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = -(FLOAT)0.5f * am10 + (FLOAT)0.5f * am20;
    rb = -(FLOAT)0.5f * bm10 + (FLOAT)0.5f * bm20;
    {
        int o = out_offset + 2 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = -am10 + am30;
    rb = -bm10 + bm30;
    {
        int o = out_offset + 3 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = am01 - am21;
    rb = bm01 - bm21;
    {
        int o = out_offset + 4 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = (FLOAT)0.5f * am11 + (FLOAT)0.5f * am21;
    rb = (FLOAT)0.5f * bm11 + (FLOAT)0.5f * bm21;
    {
        int o = out_offset + 5 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = -(FLOAT)0.5f * am11 + (FLOAT)0.5f * am21;
    rb = -(FLOAT)0.5f * bm11 + (FLOAT)0.5f * bm21;
    {
        int o = out_offset + 6 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = -am11 + am31;
    rb = -bm11 + bm31;
    {
        int o = out_offset + 7 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = am02 - am22;
    rb = bm02 - bm22;
    {
        int o = out_offset + 8 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = (FLOAT)0.5f * am12 + (FLOAT)0.5f * am22;
    rb = (FLOAT)0.5f * bm12 + (FLOAT)0.5f * bm22;
    {
        int o = out_offset + 9 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = -(FLOAT)0.5f * am12 + (FLOAT)0.5f * am22;
    rb = -(FLOAT)0.5f * bm12 + (FLOAT)0.5f * bm22;
    {
        int o = out_offset + 10 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = -am12 + am32;
    rb = -bm12 + bm32;
    {
        int o = out_offset + 11 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = am03 - am23;
    rb = bm03 - bm23;
    {
        int o = out_offset + 12 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = (FLOAT)0.5f * am13 + (FLOAT)0.5f * am23;
    rb = (FLOAT)0.5f * bm13 + (FLOAT)0.5f * bm23;
    {
        int o = out_offset + 13 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = -(FLOAT)0.5f * am13 + (FLOAT)0.5f * am23;
    rb = -(FLOAT)0.5f * bm13 + (FLOAT)0.5f * bm23;
    {
        int o = out_offset + 14 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
    ra = -am13 + am33;
    rb = -bm13 + bm33;
    {
        int o = out_offset + 15 * batch_offset;
        if(hasSecond) {
            p = (FLOAT2)(ra.x, rb.x); vstore2(p, 0, uOutput+o+0*dstHeightPad);
            p = (FLOAT2)(ra.y, rb.y); vstore2(p, 0, uOutput+o+1*dstHeightPad);
            p = (FLOAT2)(ra.z, rb.z); vstore2(p, 0, uOutput+o+2*dstHeightPad);
            p = (FLOAT2)(ra.w, rb.w); vstore2(p, 0, uOutput+o+3*dstHeightPad);
        } else {
            uOutput[o+0*dstHeightPad] = ra.x;
            uOutput[o+1*dstHeightPad] = ra.y;
            uOutput[o+2*dstHeightPad] = ra.z;
            uOutput[o+3*dstHeightPad] = ra.w;
        }
    }
}


__kernel void winoTransSrcBuf2_3_1_fast(GLOBAL_SIZE_DIM2
                                      __global const FLOAT* uInput, // 0
                                      __global FLOAT* uOutput, __private const int unitWidth,
                                      __private const int unitHeight, // 3
                                      __private const int padX, __private const int padY,
                                      __private const int srcWidth, // 6
                                      __private const int srcHeight, __private const int srcChannelC4,
                                      __private const int dstHeightPad, __private const int srcChannelPad,
                                      __private const int batch,
                                      __private const int batchOffset) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);
    if(pos.x >= unitWidth * unitHeight || pos.y >= srcChannelC4) {
        return;
    }
    int unitWidth_idx = pos.x % unitWidth;
    int unitHeight_idx = pos.x / unitWidth;
    int dstXOrigin = pos.y;
    int srcZ       = pos.y % srcChannelC4;
    int dstYOrigin = unitWidth * unitHeight_idx + unitWidth_idx;
    int batchIndex = batchOffset;
    int sxStart = unitWidth_idx * 2 - padX;
    int syStart = unitHeight_idx * 2 - padY;
    int inp_offset = (((batchIndex + srcZ * batch) * srcHeight + syStart) * srcWidth + sxStart) * 4;
    FLOAT4 S00, S10, S20, S30;
    FLOAT4 S01, S11, S21, S31;
    FLOAT4 S02, S12, S22, S32;
    FLOAT4 S03, S13, S23, S33;
    if(sxStart >= 0 && sxStart + 3 < srcWidth && syStart >= 0 && syStart + 3 < srcHeight) {
        S00 = vload4(0, uInput+inp_offset+0+0*srcWidth);
        S10 = vload4(0, uInput+inp_offset+4+0*srcWidth);
        S20 = vload4(0, uInput+inp_offset+8+0*srcWidth);
        S30 = vload4(0, uInput+inp_offset+12+0*srcWidth);
        S01 = vload4(0, uInput+inp_offset+0+4*srcWidth);
        S11 = vload4(0, uInput+inp_offset+4+4*srcWidth);
        S21 = vload4(0, uInput+inp_offset+8+4*srcWidth);
        S31 = vload4(0, uInput+inp_offset+12+4*srcWidth);
        S02 = vload4(0, uInput+inp_offset+0+8*srcWidth);
        S12 = vload4(0, uInput+inp_offset+4+8*srcWidth);
        S22 = vload4(0, uInput+inp_offset+8+8*srcWidth);
        S32 = vload4(0, uInput+inp_offset+12+8*srcWidth);
        S03 = vload4(0, uInput+inp_offset+0+12*srcWidth);
        S13 = vload4(0, uInput+inp_offset+4+12*srcWidth);
        S23 = vload4(0, uInput+inp_offset+8+12*srcWidth);
        S33 = vload4(0, uInput+inp_offset+12+12*srcWidth);
    } else {
        S00 = (sxStart+0 < 0 || sxStart+0 >= srcWidth || syStart+0 < 0 || syStart+0 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+0+0*srcWidth);
        S10 = (sxStart+1 < 0 || sxStart+1 >= srcWidth || syStart+0 < 0 || syStart+0 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4+0*srcWidth);
        S20 = (sxStart+2 < 0 || sxStart+2 >= srcWidth || syStart+0 < 0 || syStart+0 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8+0*srcWidth);
        S30 = (sxStart+3 < 0 || sxStart+3 >= srcWidth || syStart+0 < 0 || syStart+0 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12+0*srcWidth);
        S01 = (sxStart+0 < 0 || sxStart+0 >= srcWidth || syStart+1 < 0 || syStart+1 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+0+4*srcWidth);
        S11 = (sxStart+1 < 0 || sxStart+1 >= srcWidth || syStart+1 < 0 || syStart+1 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4+4*srcWidth);
        S21 = (sxStart+2 < 0 || sxStart+2 >= srcWidth || syStart+1 < 0 || syStart+1 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8+4*srcWidth);
        S31 = (sxStart+3 < 0 || sxStart+3 >= srcWidth || syStart+1 < 0 || syStart+1 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12+4*srcWidth);
        S02 = (sxStart+0 < 0 || sxStart+0 >= srcWidth || syStart+2 < 0 || syStart+2 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+0+8*srcWidth);
        S12 = (sxStart+1 < 0 || sxStart+1 >= srcWidth || syStart+2 < 0 || syStart+2 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4+8*srcWidth);
        S22 = (sxStart+2 < 0 || sxStart+2 >= srcWidth || syStart+2 < 0 || syStart+2 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8+8*srcWidth);
        S32 = (sxStart+3 < 0 || sxStart+3 >= srcWidth || syStart+2 < 0 || syStart+2 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12+8*srcWidth);
        S03 = (sxStart+0 < 0 || sxStart+0 >= srcWidth || syStart+3 < 0 || syStart+3 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+0+12*srcWidth);
        S13 = (sxStart+1 < 0 || sxStart+1 >= srcWidth || syStart+3 < 0 || syStart+3 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4+12*srcWidth);
        S23 = (sxStart+2 < 0 || sxStart+2 >= srcWidth || syStart+3 < 0 || syStart+3 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8+12*srcWidth);
        S33 = (sxStart+3 < 0 || sxStart+3 >= srcWidth || syStart+3 < 0 || syStart+3 >= srcHeight)
                ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12+12*srcWidth);
    }
    FLOAT4 m00 = S00 - S02;
    FLOAT4 m01 = (FLOAT)0.5f * S01 + (FLOAT)0.5f * S02;
    FLOAT4 m02 = -(FLOAT)0.5f * S01 + (FLOAT)0.5f * S02;
    FLOAT4 m03 = -S01 + S03;
    FLOAT4 m10 = S10 - S12;
    FLOAT4 m11 = (FLOAT)0.5f * S11 + (FLOAT)0.5f * S12;
    FLOAT4 m12 = -(FLOAT)0.5f * S11 + (FLOAT)0.5f * S12;
    FLOAT4 m13 = -S11 + S13;
    FLOAT4 m20 = S20 - S22;
    FLOAT4 m21 = (FLOAT)0.5f * S21 + (FLOAT)0.5f * S22;
    FLOAT4 m22 = -(FLOAT)0.5f * S21 + (FLOAT)0.5f * S22;
    FLOAT4 m23 = -S21 + S23;
    FLOAT4 m30 = S30 - S32;
    FLOAT4 m31 = (FLOAT)0.5f * S31 + (FLOAT)0.5f * S32;
    FLOAT4 m32 = -(FLOAT)0.5f * S31 + (FLOAT)0.5f * S32;
    FLOAT4 m33 = -S31 + S33;
    int out_offset = (0 * srcChannelPad + 4 * dstXOrigin) * dstHeightPad + dstYOrigin;
    const int batch_offset = srcChannelPad * dstHeightPad;
    FLOAT4 res;
    res = m00 - m20;
    uOutput[out_offset + 0 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 0 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 0 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 0 * batch_offset + 3 * dstHeightPad] = res.w;
    res = (FLOAT)0.5f * m10 + (FLOAT)0.5f * m20;
    uOutput[out_offset + 1 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 1 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 1 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 1 * batch_offset + 3 * dstHeightPad] = res.w;
    res = -(FLOAT)0.5f * m10 + (FLOAT)0.5f * m20;
    uOutput[out_offset + 2 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 2 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 2 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 2 * batch_offset + 3 * dstHeightPad] = res.w;
    res = -m10 + m30;
    uOutput[out_offset + 3 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 3 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 3 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 3 * batch_offset + 3 * dstHeightPad] = res.w;
    res = m01 - m21;
    uOutput[out_offset + 4 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 4 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 4 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 4 * batch_offset + 3 * dstHeightPad] = res.w;
    res = (FLOAT)0.5f * m11 + (FLOAT)0.5f * m21;
    uOutput[out_offset + 5 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 5 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 5 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 5 * batch_offset + 3 * dstHeightPad] = res.w;
    res = -(FLOAT)0.5f * m11 + (FLOAT)0.5f * m21;
    uOutput[out_offset + 6 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 6 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 6 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 6 * batch_offset + 3 * dstHeightPad] = res.w;
    res = -m11 + m31;
    uOutput[out_offset + 7 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 7 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 7 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 7 * batch_offset + 3 * dstHeightPad] = res.w;
    res = m02 - m22;
    uOutput[out_offset + 8 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 8 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 8 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 8 * batch_offset + 3 * dstHeightPad] = res.w;
    res = (FLOAT)0.5f * m12 + (FLOAT)0.5f * m22;
    uOutput[out_offset + 9 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 9 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 9 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 9 * batch_offset + 3 * dstHeightPad] = res.w;
    res = -(FLOAT)0.5f * m12 + (FLOAT)0.5f * m22;
    uOutput[out_offset + 10 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 10 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 10 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 10 * batch_offset + 3 * dstHeightPad] = res.w;
    res = -m12 + m32;
    uOutput[out_offset + 11 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 11 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 11 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 11 * batch_offset + 3 * dstHeightPad] = res.w;
    res = m03 - m23;
    uOutput[out_offset + 12 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 12 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 12 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 12 * batch_offset + 3 * dstHeightPad] = res.w;
    res = (FLOAT)0.5f * m13 + (FLOAT)0.5f * m23;
    uOutput[out_offset + 13 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 13 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 13 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 13 * batch_offset + 3 * dstHeightPad] = res.w;
    res = -(FLOAT)0.5f * m13 + (FLOAT)0.5f * m23;
    uOutput[out_offset + 14 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 14 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 14 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 14 * batch_offset + 3 * dstHeightPad] = res.w;
    res = -m13 + m33;
    uOutput[out_offset + 15 * batch_offset + 0 * dstHeightPad] = res.x;
    uOutput[out_offset + 15 * batch_offset + 1 * dstHeightPad] = res.y;
    uOutput[out_offset + 15 * batch_offset + 2 * dstHeightPad] = res.z;
    uOutput[out_offset + 15 * batch_offset + 3 * dstHeightPad] = res.w;
}
