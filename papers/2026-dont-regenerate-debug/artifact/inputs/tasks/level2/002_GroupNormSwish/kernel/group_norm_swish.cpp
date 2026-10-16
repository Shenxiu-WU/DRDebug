#include "group_norm_swish_kernel.h"

extern "C" __global__ __aicore__ void group_norm_swish_custom_fp32(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR output, GM_ADDR mean, GM_ADDR rstd, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    GroupNormSwishKernel<float> kernel;
    kernel.Init(x, weight, bias, output, mean, rstd, tiling, &pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void group_norm_swish_custom_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR output, GM_ADDR mean, GM_ADDR rstd, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    GroupNormSwishKernel<half> kernel;
    kernel.Init(x, weight, bias, output, mean, rstd, tiling, &pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void group_norm_swish_custom_bf16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR output, GM_ADDR mean, GM_ADDR rstd, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    GroupNormSwishKernel<bfloat16_t> kernel;
    kernel.Init(x, weight, bias, output, mean, rstd, tiling, &pipe);
    kernel.Process();
}

extern "C" void group_norm_swish_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *weight, uint8_t *bias,
    uint8_t *output, uint8_t *mean, uint8_t *rstd, uint8_t *tiling)
{
    group_norm_swish_custom_fp32<<<blockDim, nullptr, stream>>>(
        x, weight, bias, output, mean, rstd, tiling);
}

extern "C" void group_norm_swish_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *weight, uint8_t *bias,
    uint8_t *output, uint8_t *mean, uint8_t *rstd, uint8_t *tiling)
{
    group_norm_swish_custom_fp16<<<blockDim, nullptr, stream>>>(
        x, weight, bias, output, mean, rstd, tiling);
}

extern "C" void group_norm_swish_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *weight, uint8_t *bias,
    uint8_t *output, uint8_t *mean, uint8_t *rstd, uint8_t *tiling)
{
    group_norm_swish_custom_bf16<<<blockDim, nullptr, stream>>>(
        x, weight, bias, output, mean, rstd, tiling);
}
