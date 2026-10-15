#include "add_kernel.h"

extern "C" __global__ __aicore__ void add_alpha_fp32(
    GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    AddAlphaKernel<float> kernel;
    kernel.Init(x, y, out, tiling, &pipe);
    kernel.Process();
}

extern "C" void add_alpha_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint8_t *out,
    uint8_t *tiling)
{
    add_alpha_fp32<<<blockDim, nullptr, stream>>>(x, y, out, tiling);
}
