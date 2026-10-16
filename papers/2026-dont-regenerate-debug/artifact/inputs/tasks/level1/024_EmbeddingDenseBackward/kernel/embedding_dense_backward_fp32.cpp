#include "embedding_dense_backward_kernel.h"

extern "C" __global__ __aicore__ void embedding_dense_backward_custom_fp32(
    GM_ADDR gradOutput, GM_ADDR indices, GM_ADDR gradWeight, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    EmbeddingDenseBackwardKernel<float> kernel;
    kernel.Init(gradOutput, indices, gradWeight, tiling, &pipe);
    kernel.Process();
}

extern "C" void embedding_dense_backward_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *gradOutput,
    uint8_t *indices,
    uint8_t *gradWeight,
    uint8_t *tiling)
{
    embedding_dense_backward_custom_fp32<<<blockDim, nullptr, stream>>>(
        gradOutput, indices, gradWeight, tiling);
}
