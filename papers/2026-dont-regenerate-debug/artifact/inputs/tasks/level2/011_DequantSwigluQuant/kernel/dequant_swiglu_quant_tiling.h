#ifndef DEQUANT_SWIGLU_QUANT_TILING_H
#define DEQUANT_SWIGLU_QUANT_TILING_H

struct DequantSwigluQuantTiling {
    int32_t T;
    int32_t twoH;
    int32_t H;
    int32_t swiglu_mode;
    int32_t activate_left;
    float clamp_limit;
    float glu_alpha;
    float glu_bias;
};

#endif
