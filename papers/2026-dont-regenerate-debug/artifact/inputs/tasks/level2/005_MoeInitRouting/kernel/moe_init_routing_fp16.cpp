#include "moe_init_routing_kernel.h"

extern "C" __global__ __aicore__ void moe_init_routing_fp16_custom(
    GM_ADDR x,
    GM_ADDR expandedRowIdx,
    GM_ADDR y,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    MoeInitRoutingKernel<half> kernel;
    kernel.Init(x, expandedRowIdx, y, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void moe_init_routing_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *expandedRowIdx,
    uint8_t *y,
    uint8_t *tiling)
{
    moe_init_routing_fp16_custom<<<blockDim, nullptr, stream>>>(x, expandedRowIdx, y, tiling);
}
#endif
