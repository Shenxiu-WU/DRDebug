#include "reshape_matmul_quant.h"

extern "C" __global__ __aicore__ void reshape_matmul_quant_bf16(
    GM_ADDR x, GM_ADDR h, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    ReshapeMatmulQuantKernel<bfloat16_t, float, int8_t> kernel;
    kernel.Init(x, h, y, workspace, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void reshape_matmul_quant_do_bf16(uint32_t blockDim, void *stream,
                                              uint8_t *x, uint8_t *h, uint8_t *y,
                                              uint8_t *workspace, uint8_t *tiling)
{
    reshape_matmul_quant_bf16<<<blockDim, nullptr, stream>>>(x, h, y, workspace, tiling);
}
#endif
