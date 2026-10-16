#include "kernel_operator.h"
#include "tanh_gated_residual_add_backward_kernel.h"

extern "C" __global__ __aicore__ void tanh_gated_residual_add_backward_custom_bf16(
    GM_ADDR gradOutput, GM_ADDR gate, GM_ADDR hiddenStates, GM_ADDR mask,
    GM_ADDR gradResidual, GM_ADDR gradHiddenStates, GM_ADDR gradGate,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    TanhGatedResidualAddBackwardKernel<bfloat16_t> kernel;
    kernel.Init(gradOutput, gate, hiddenStates, mask,
                gradResidual, gradHiddenStates, gradGate,
                workspace, tiling, &pipe);
    kernel.Process();
}

extern "C" void tanh_gated_residual_add_backward_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *gradOutput,
    uint8_t *gate,
    uint8_t *hiddenStates,
    uint8_t *mask,
    uint8_t *gradResidual,
    uint8_t *gradHiddenStates,
    uint8_t *gradGate,
    uint8_t *workspace,
    uint8_t *tiling)
{
    tanh_gated_residual_add_backward_custom_bf16<<<blockDim, nullptr, stream>>>(
        gradOutput, gate, hiddenStates, mask,
        gradResidual, gradHiddenStates, gradGate,
        workspace, tiling);
}
