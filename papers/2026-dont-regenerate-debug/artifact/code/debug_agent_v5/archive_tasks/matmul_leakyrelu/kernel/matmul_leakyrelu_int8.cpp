/**
 * @file matmul_leakyrelu_int8.cpp
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "matmul_leakyrelu.h"

extern "C" __global__ __aicore__ void matmul_leakyrelu_custom_int8(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    MatmulLeakyKernel<int8_t, int8_t, int32_t, float> matmulLeakyKernel;
    matmulLeakyKernel.Init(a, b, c, workspace, tiling, &pipe);
    matmulLeakyKernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void matmul_leakyrelu_do_int8(uint32_t blockDim, void *stream,
                                         uint8_t *a, uint8_t *b, uint8_t *c,
                                         uint8_t *workspace, uint8_t *tiling)
{
    matmul_leakyrelu_custom_int8<<<blockDim, nullptr, stream>>>(a, b, c, workspace, tiling);
}
#endif // #ifndef ASCENDC_CPU_DEBUG