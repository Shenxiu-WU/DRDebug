#include "masked_softmax_attn_dropout_bwd_kernel.h"

extern "C" __global__ __aicore__ void masked_softmax_attn_dropout_bwd_custom_fp32(
    GM_ADDR gradOutput, GM_ADDR pAttn, GM_ADDR mask, GM_ADDR dropoutMask, GM_ADDR out, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    MaskedSoftmaxAttnDropoutBwdKernel<float> kernel;
    kernel.Init(gradOutput, pAttn, mask, dropoutMask, out, tiling, &pipe);
    kernel.Process();
}

extern "C" void masked_softmax_attn_dropout_bwd_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *gradOutput,
    uint8_t *pAttn,
    uint8_t *mask,
    uint8_t *dropoutMask,
    uint8_t *out,
    uint8_t *tiling)
{
    masked_softmax_attn_dropout_bwd_custom_fp32<<<blockDim, nullptr, stream>>>(gradOutput, pAttn, mask, dropoutMask, out, tiling);
}
