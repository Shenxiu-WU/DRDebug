#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"

#include "int8_matmul_scale_tiling.h"

static inline int64_t GetWorkspaceSize(const Int8MatmulScaleTiling *tiling, uint32_t usedCoreNum)
{
    return (int64_t)tiling->baseM * tiling->baseN * WORKSPACE_DEPTH * sizeof(int32_t) * usedCoreNum;
}

extern "C" void int8_matmul_scale_do(uint32_t blockDim, void *stream,
                                     uint8_t *a, uint8_t *b, uint8_t *scale,
                                     uint8_t *c, uint8_t *workspace, uint8_t *tiling);

namespace my_int8_matmul_scale {

at::Tensor run_int8_matmul_scale(const at::Tensor &a, const at::Tensor &b, const at::Tensor &scale)
{
    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D");
    TORCH_CHECK(scale.dim() == 1, "scale must be 1D");
    TORCH_CHECK(a.sizes()[1] == b.sizes()[0], "k dimension must match");
    TORCH_CHECK(a.scalar_type() == at::kChar, "a must be int8");
    TORCH_CHECK(b.scalar_type() == at::kChar, "b must be int8");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");
    TORCH_CHECK(scale.sizes()[0] == b.sizes()[1], "scale size must match N");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t usedCoreNum = 2;

    uint32_t m = a.sizes()[0];
    uint32_t n = b.sizes()[1];
    uint32_t k = a.sizes()[1];

    at::Tensor c = at::empty({m, n}, at::device(at::kPrivateUse1).dtype(at::kHalf));

    at::Tensor t = at::empty({(int64_t)sizeof(Int8MatmulScaleTiling)}, at::device(at::kCPU).dtype(at::kByte));
    auto *tiling_ptr = reinterpret_cast<Int8MatmulScaleTiling *>(t.data_ptr());
    tiling_ptr->M = m;
    tiling_ptr->N = n;
    tiling_ptr->K = k;
    tiling_ptr->baseM = DEFAULT_BASE_M;
    tiling_ptr->baseN = DEFAULT_BASE_N;
    tiling_ptr->baseK = DEFAULT_BASE_K;
    auto tiling_npu = t.to(at::kPrivateUse1);

    auto workSpaceSize = GetWorkspaceSize(tiling_ptr, usedCoreNum);
    at::Tensor w = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));

    int8_matmul_scale_do(usedCoreNum, acl_stream,
             static_cast<uint8_t*>(const_cast<void*>(a.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(b.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(scale.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(c.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(w.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(tiling_npu.storage().data())));
    return c;
}
} // namespace my_int8_matmul_scale

PYBIND11_MODULE(_quant_matmul_ext, m)
{
    m.doc() = "int8_matmul_scale pybind11 interface";
    m.def("run_int8_matmul_scale", &my_int8_matmul_scale::run_int8_matmul_scale, "");
}
