#include "norm_rope_concat_kernel.h"

extern "C" __global__ __aicore__ void norm_rope_concat_custom_fp32(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR queryWeight, GM_ADDR queryBias,
    GM_ADDR keyWeight, GM_ADDR keyBias,
    GM_ADDR sin, GM_ADDR cos,
    GM_ADDR queryOut, GM_ADDR keyOut, GM_ADDR valueOut,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    NormRopeConcatKernel<float> kernel;
    kernel.Init(query, key, value, queryWeight, queryBias, keyWeight, keyBias,
                sin, cos, queryOut, keyOut, valueOut, tiling, &pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void norm_rope_concat_custom_fp16(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR queryWeight, GM_ADDR queryBias,
    GM_ADDR keyWeight, GM_ADDR keyBias,
    GM_ADDR sin, GM_ADDR cos,
    GM_ADDR queryOut, GM_ADDR keyOut, GM_ADDR valueOut,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    NormRopeConcatKernel<half> kernel;
    kernel.Init(query, key, value, queryWeight, queryBias, keyWeight, keyBias,
                sin, cos, queryOut, keyOut, valueOut, tiling, &pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void norm_rope_concat_custom_bf16(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR queryWeight, GM_ADDR queryBias,
    GM_ADDR keyWeight, GM_ADDR keyBias,
    GM_ADDR sin, GM_ADDR cos,
    GM_ADDR queryOut, GM_ADDR keyOut, GM_ADDR valueOut,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    NormRopeConcatKernel<bfloat16_t> kernel;
    kernel.Init(query, key, value, queryWeight, queryBias, keyWeight, keyBias,
                sin, cos, queryOut, keyOut, valueOut, tiling, &pipe);
    kernel.Process();
}

extern "C" void norm_rope_concat_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *queryWeight, uint8_t *queryBias,
    uint8_t *keyWeight, uint8_t *keyBias,
    uint8_t *sin, uint8_t *cos,
    uint8_t *queryOut, uint8_t *keyOut, uint8_t *valueOut,
    uint8_t *tiling)
{
    norm_rope_concat_custom_fp32<<<blockDim, nullptr, stream>>>(
        query, key, value, queryWeight, queryBias, keyWeight, keyBias,
        sin, cos, queryOut, keyOut, valueOut, tiling);
}

extern "C" void norm_rope_concat_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *queryWeight, uint8_t *queryBias,
    uint8_t *keyWeight, uint8_t *keyBias,
    uint8_t *sin, uint8_t *cos,
    uint8_t *queryOut, uint8_t *keyOut, uint8_t *valueOut,
    uint8_t *tiling)
{
    norm_rope_concat_custom_fp16<<<blockDim, nullptr, stream>>>(
        query, key, value, queryWeight, queryBias, keyWeight, keyBias,
        sin, cos, queryOut, keyOut, valueOut, tiling);
}

extern "C" void norm_rope_concat_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *queryWeight, uint8_t *queryBias,
    uint8_t *keyWeight, uint8_t *keyBias,
    uint8_t *sin, uint8_t *cos,
    uint8_t *queryOut, uint8_t *keyOut, uint8_t *valueOut,
    uint8_t *tiling)
{
    norm_rope_concat_custom_bf16<<<blockDim, nullptr, stream>>>(
        query, key, value, queryWeight, queryBias, keyWeight, keyBias,
        sin, cos, queryOut, keyOut, valueOut, tiling);
}
