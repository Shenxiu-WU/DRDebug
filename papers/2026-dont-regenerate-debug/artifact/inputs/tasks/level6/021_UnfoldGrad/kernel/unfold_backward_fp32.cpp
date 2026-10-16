#include "unfold_backward_kernel.h"

extern "C" __global__ __aicore__ void unfold_backward_fp32_custom(
    GM_ADDR grad_output,
    GM_ADDR grad_input,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    UnfoldBackwardKernel<float> kernel;
    kernel.Init(grad_output, grad_input, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void unfold_backward_fp32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *grad_output,
    uint8_t *grad_input,
    uint8_t *tiling)
{
    unfold_backward_fp32_custom<<<blockDim, nullptr, stream>>>(grad_output, grad_input, tiling);
}
#endif
