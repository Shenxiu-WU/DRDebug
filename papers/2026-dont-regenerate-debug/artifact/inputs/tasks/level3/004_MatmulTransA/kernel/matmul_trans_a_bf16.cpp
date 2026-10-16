/**
 * @file matmul_trans_a_bf16.cpp
 *
 * BFloat16 launcher for the MatmulTransA cube kernel.
 */
#include "matmul_trans_a.h"

extern "C" __global__ __aicore__ void matmul_trans_a_bf16(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    MatmulTransAKernel<bfloat16_t, bfloat16_t> kernel;
    kernel.Init(a, b, c, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void matmul_trans_a_do_bf16(uint32_t blockDim, void *stream,
                                       uint8_t *a, uint8_t *b, uint8_t *c,
                                       uint8_t *workspace, uint8_t *tiling)
{
    matmul_trans_a_bf16<<<blockDim, nullptr, stream>>>(a, b, c, workspace, tiling);
}
#endif
