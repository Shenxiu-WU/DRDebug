#include "conv3d_kernel.h"

extern "C" __global__ __aicore__ void conv3d_custom_kernel(
    GM_ADDR input, GM_ADDR weight, GM_ADDR bias, GM_ADDR output,
    int N, int C, int D, int H, int W,
    int K, int D_out, int H_out, int W_out,
    int k_size, int stride, int padding, int dilation,
    int groups, int use_bias)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    Conv3dKernel<float> kernel;
    kernel.Init(input, weight, bias, output,
                N, C, D, H, W, K, D_out, H_out, W_out,
                k_size, stride, padding, dilation, groups, use_bias);
    kernel.Process();
}

extern "C" void conv3d_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *input,
    uint8_t *weight,
    uint8_t *bias,
    uint8_t *output,
    int N, int C, int D, int H, int W,
    int K, int D_out, int H_out, int W_out,
    int k_size, int stride, int padding, int dilation,
    int groups, int use_bias)
{
    conv3d_custom_kernel<<<blockDim, nullptr, stream>>>(
        input, weight, bias, output,
        N, C, D, H, W, K, D_out, H_out, W_out,
        k_size, stride, padding, dilation, groups, use_bias);
}
