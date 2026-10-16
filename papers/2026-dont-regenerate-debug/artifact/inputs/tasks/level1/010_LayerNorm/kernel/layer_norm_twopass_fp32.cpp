#include "layer_norm_twopass_kernel.h"

extern "C" __global__ __aicore__ void layer_norm_twopass_custom_fp32(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    LayerNormTwopassKernel<float> kernel;
    kernel.Init(x, gamma, beta, y, tiling, &pipe);
    kernel.Process();
}

extern "C" void layer_norm_twopass_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *beta,
    uint8_t *y,
    uint8_t *tiling)
{
    layer_norm_twopass_custom_fp32<<<blockDim, nullptr, stream>>>(x, gamma, beta, y, tiling);
}
