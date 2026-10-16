#include "hyena_rfft_kernel.h"

extern "C" __global__ __aicore__ void hyena_rfft_custom_fp32(
    GM_ADDR x, GM_ADDR real, GM_ADDR imag, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::TPipe pipe;
    HyenaRfftKernel<float> kernel;
    kernel.Init(x, real, imag, tiling);
    kernel.Process();
}

extern "C" __global__ __aicore__ void hyena_rfft_custom_fp16(
    GM_ADDR x, GM_ADDR real, GM_ADDR imag, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::TPipe pipe;
    HyenaRfftKernel<half> kernel;
    kernel.Init(x, real, imag, tiling);
    kernel.Process();
}

extern "C" __global__ __aicore__ void hyena_rfft_custom_bf16(
    GM_ADDR x, GM_ADDR real, GM_ADDR imag, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::TPipe pipe;
    HyenaRfftKernel<bfloat16_t> kernel;
    kernel.Init(x, real, imag, tiling);
    kernel.Process();
}

extern "C" void hyena_rfft_do_fp32(
    uint32_t blockDim,
    void* stream,
    uint8_t* x,
    uint8_t* real,
    uint8_t* imag,
    uint8_t* tiling)
{
    hyena_rfft_custom_fp32<<<blockDim, nullptr, stream>>>(x, real, imag, tiling);
}

extern "C" void hyena_rfft_do_fp16(
    uint32_t blockDim,
    void* stream,
    uint8_t* x,
    uint8_t* real,
    uint8_t* imag,
    uint8_t* tiling)
{
    hyena_rfft_custom_fp16<<<blockDim, nullptr, stream>>>(x, real, imag, tiling);
}

extern "C" void hyena_rfft_do_bf16(
    uint32_t blockDim,
    void* stream,
    uint8_t* x,
    uint8_t* real,
    uint8_t* imag,
    uint8_t* tiling)
{
    hyena_rfft_custom_bf16<<<blockDim, nullptr, stream>>>(x, real, imag, tiling);
}
