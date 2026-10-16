#include "nllloss_kernel.h"

extern "C" __global__ __aicore__ void nllloss_reduce_fp16_custom(
    GM_ADDR input,
    GM_ADDR target,
    GM_ADDR weight,
    GM_ADDR out,
    GM_ADDR partialSum,
    GM_ADDR partialCount,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    NLLLossKernel<half> kernel;
    kernel.Init(input, target, weight, out, partialSum, partialCount, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void nllloss_reduce_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *input,
    uint8_t *target,
    uint8_t *weight,
    uint8_t *out,
    uint8_t *partialSum,
    uint8_t *partialCount,
    uint8_t *tiling)
{
    nllloss_reduce_fp16_custom<<<blockDim, nullptr, stream>>>(input, target, weight, out, partialSum, partialCount, tiling);
}
#endif
