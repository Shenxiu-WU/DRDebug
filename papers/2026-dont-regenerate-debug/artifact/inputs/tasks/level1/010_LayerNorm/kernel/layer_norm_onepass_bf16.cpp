#include "layer_norm_onepass_kernel.h"

extern "C" __global__ __aicore__ void layer_norm_onepass_custom_bf16(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    LayerNormOnepassKernel<bfloat16_t> kernel;
    kernel.Init(x, gamma, beta, y, tiling, &pipe);
    kernel.Process();
}

extern "C" void layer_norm_onepass_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *beta,
    uint8_t *y,
    uint8_t *tiling)
{
    layer_norm_onepass_custom_bf16<<<blockDim, nullptr, stream>>>(x, gamma, beta, y, tiling);
}
