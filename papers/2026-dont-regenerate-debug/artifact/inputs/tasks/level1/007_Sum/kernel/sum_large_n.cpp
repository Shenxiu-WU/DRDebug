#include "sum_large_n_kernel.h"

extern "C" __global__ __aicore__ void sum_large_n_custom_fp32(
    GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    SumLargeNKernel<float> kernel;
    kernel.Init(x, y, tiling, &pipe);
    kernel.Process();
}

extern "C" void sum_large_n_do_fp32(
    uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *tiling)
{
    sum_large_n_custom_fp32<<<blockDim, nullptr, stream>>>(x, y, tiling);
}

extern "C" __global__ __aicore__ void sum_large_n_custom_fp16(
    GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    SumLargeNKernel<half> kernel;
    kernel.Init(x, y, tiling, &pipe);
    kernel.Process();
}

extern "C" void sum_large_n_do_fp16(
    uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *tiling)
{
    sum_large_n_custom_fp16<<<blockDim, nullptr, stream>>>(x, y, tiling);
}

extern "C" __global__ __aicore__ void sum_large_n_custom_bf16(
    GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    SumLargeNKernel<bfloat16_t> kernel;
    kernel.Init(x, y, tiling, &pipe);
    kernel.Process();
}

extern "C" void sum_large_n_do_bf16(
    uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *tiling)
{
    sum_large_n_custom_bf16<<<blockDim, nullptr, stream>>>(x, y, tiling);
}
