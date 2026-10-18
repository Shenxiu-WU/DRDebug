#include "cumsum_kernel.h"

extern "C" __global__ __aicore__ void cumsum_custom_fp16(
    GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    CumsumKernel<half> kernel;
    kernel.Init(x, y, tiling, &pipe);
    kernel.Process();
}

extern "C" void cumsum_do_fp16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint8_t *tiling)
{
    cumsum_custom_fp16<<<blockDim, nullptr, stream>>>(x, y, tiling);
}
