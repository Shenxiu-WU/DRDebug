#include "adamw_kernel.h"

extern "C" __global__ __aicore__ void adamw_apply_custom_fp32(
    GM_ADDR var,
    GM_ADDR m,
    GM_ADDR v,
    GM_ADDR grad,
    GM_ADDR varOut,
    GM_ADDR mOut,
    GM_ADDR vOut,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    AdamWKernel<float> kernel;
    kernel.Init(var, m, v, grad, varOut, mOut, vOut, tiling, &pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void adamw_apply_custom_fp16(
    GM_ADDR var,
    GM_ADDR m,
    GM_ADDR v,
    GM_ADDR grad,
    GM_ADDR varOut,
    GM_ADDR mOut,
    GM_ADDR vOut,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    AdamWKernel<half> kernel;
    kernel.Init(var, m, v, grad, varOut, mOut, vOut, tiling, &pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void adamw_apply_custom_bf16(
    GM_ADDR var,
    GM_ADDR m,
    GM_ADDR v,
    GM_ADDR grad,
    GM_ADDR varOut,
    GM_ADDR mOut,
    GM_ADDR vOut,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    AdamWKernel<bfloat16_t> kernel;
    kernel.Init(var, m, v, grad, varOut, mOut, vOut, tiling, &pipe);
    kernel.Process();
}

extern "C" void adamw_apply_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *var,
    uint8_t *m,
    uint8_t *v,
    uint8_t *grad,
    uint8_t *varOut,
    uint8_t *mOut,
    uint8_t *vOut,
    uint8_t *tiling)
{
    adamw_apply_custom_fp32<<<blockDim, nullptr, stream>>>(
        var, m, v, grad, varOut, mOut, vOut, tiling);
}

extern "C" void adamw_apply_do_fp16(
    uint32_t blockDim,
    void *stream,
    uint8_t *var,
    uint8_t *m,
    uint8_t *v,
    uint8_t *grad,
    uint8_t *varOut,
    uint8_t *mOut,
    uint8_t *vOut,
    uint8_t *tiling)
{
    adamw_apply_custom_fp16<<<blockDim, nullptr, stream>>>(
        var, m, v, grad, varOut, mOut, vOut, tiling);
}

extern "C" void adamw_apply_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *var,
    uint8_t *m,
    uint8_t *v,
    uint8_t *grad,
    uint8_t *varOut,
    uint8_t *mOut,
    uint8_t *vOut,
    uint8_t *tiling)
{
    adamw_apply_custom_bf16<<<blockDim, nullptr, stream>>>(
        var, m, v, grad, varOut, mOut, vOut, tiling);
}
