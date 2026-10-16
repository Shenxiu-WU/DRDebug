#include "rms_norm_single_row_kernel.h"

extern "C" __global__ __aicore__ void rms_norm_single_row_custom_fp16(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    RmsNormSingleRowKernel<half> kernel;
    kernel.Init(x, gamma, y, invRms, tiling, &pipe);
    kernel.Process();
}

extern "C" void rms_norm_single_row_do_fp16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *y,
    uint8_t *invRms,
    uint8_t *tiling)
{
    rms_norm_single_row_custom_fp16<<<blockDim, nullptr, stream>>>(x, gamma, y, invRms, tiling);
}
