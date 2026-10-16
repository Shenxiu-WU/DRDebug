#include "matmul_transb_kernel.h"

extern "C" __global__ __aicore__ void matmul_transb_custom_fp32(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    MatmulTransBKernel<float> kernel;
    kernel.Init(a, b, c, tiling, &pipe);
    kernel.Process();
}

extern "C" void matmul_transb_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *a,
    uint8_t *b,
    uint8_t *c,
    uint8_t *tiling)
{
    matmul_transb_custom_fp32<<<blockDim, nullptr, stream>>>(a, b, c, tiling);
}
