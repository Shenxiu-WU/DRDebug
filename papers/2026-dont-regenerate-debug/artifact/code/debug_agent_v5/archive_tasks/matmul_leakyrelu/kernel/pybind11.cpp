/**
 * @file pybind11.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"

#include "matmul_leakyrelu_tiling.h"

static inline int64_t GetWorkspaceSize(const MatmulLeakyReluTiling *tiling, int accElementSize,
                                       uint32_t usedCoreNum)
{
    return tiling->baseM * tiling->baseN * WORKSPACE_DEPTH * accElementSize * usedCoreNum;
}

extern "C" void matmul_leakyrelu_do_fp16(uint32_t blockDim, void *stream,
                                         uint8_t *a, uint8_t *b, uint8_t *c,
                                         uint8_t *workspace, uint8_t *tiling);

extern "C" void matmul_leakyrelu_do_int8(uint32_t blockDim, void *stream,
                                         uint8_t *a, uint8_t *b, uint8_t *c,
                                         uint8_t *workspace, uint8_t *tiling);


namespace my_matmul_leakyrelu {

using LaunchFn = void(*)(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*);

at::Tensor run_matmul_leakyrelu(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.dim() == 2, "matmul_leakyrelu a must be two-dimensional.");
    TORCH_CHECK(b.dim() == 2, "matmul_leakyrelu b must be two-dimensional.");
    TORCH_CHECK(a.sizes()[1] == b.sizes()[0], "matmul_leakyrelu k must be same.");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "matmul_leakyrelu a and b must have same dtype.");

    int accElementSize;
    LaunchFn launchFn;

    if (a.scalar_type() == at::kHalf) {
        accElementSize = sizeof(float);
        launchFn = matmul_leakyrelu_do_fp16;
    } else if (a.scalar_type() == at::kChar) {
        accElementSize = sizeof(int32_t);
        launchFn = matmul_leakyrelu_do_int8;
    } else {
        TORCH_CHECK(false, "matmul_leakyrelu unsupported dtype, expected float16 or int8.");
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t usedCoreNum = 20;

    uint32_t m = a.sizes()[0];
    uint32_t n = b.sizes()[1];
    uint32_t k = a.sizes()[1];

    // output is always fp32
    at::Tensor c = at::empty({m, n}, at::device(at::kPrivateUse1).dtype(at::kFloat));

    // Fill lightweight tiling struct
    at::Tensor t = at::empty({sizeof(MatmulLeakyReluTiling)}, at::device(at::kCPU).dtype(at::kByte));
    auto *tiling_ptr = reinterpret_cast<MatmulLeakyReluTiling *>(t.data_ptr());
    tiling_ptr->M = m;
    tiling_ptr->N = n;
    tiling_ptr->K = k;
    tiling_ptr->baseM = DEFAULT_BASE_M;
    tiling_ptr->baseN = DEFAULT_BASE_N;
    tiling_ptr->baseK = DEFAULT_BASE_K;
    tiling_ptr->l1Prefetch = DEFAULT_L1_PREFETCH;
    auto tiling_npu = t.to(at::kPrivateUse1);

    auto workSpaceSize = GetWorkspaceSize(tiling_ptr, accElementSize, usedCoreNum);
    at::Tensor w = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));

    launchFn(usedCoreNum, acl_stream,
             static_cast<uint8_t*>(const_cast<void*>(a.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(b.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(c.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(w.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(tiling_npu.storage().data())));
    return c;
}
} // namespace my_matmul_leakyrelu

PYBIND11_MODULE(_matmul_leakyrelu_ext, m)
{
    m.doc() = "matmul_leakyrelu pybind11 interfaces"; // optional module docstring
    m.def("run_matmul_leakyrelu", &my_matmul_leakyrelu::run_matmul_leakyrelu, "");
}
