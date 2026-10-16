#include <algorithm>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "inplace_index_add_tiling.h"

extern "C" void inplace_index_add_fp32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *var,
    uint8_t *value,
    uint8_t *sortedIndices,
    uint8_t *alpha,
    uint8_t *out,
    uint8_t *tiling);

extern "C" void inplace_index_add_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *var,
    uint8_t *value,
    uint8_t *sortedIndices,
    uint8_t *alpha,
    uint8_t *out,
    uint8_t *tiling);

extern "C" void inplace_index_add_int32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *var,
    uint8_t *value,
    uint8_t *sortedIndices,
    uint8_t *alpha,
    uint8_t *out,
    uint8_t *tiling);

namespace inplace_index_add_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

inline LaunchFn ResolveLaunchFn(at::ScalarType dtype)
{
    if (dtype == at::kFloat) {
        return inplace_index_add_fp32_do;
    }
    if (dtype == at::kHalf) {
        return inplace_index_add_fp16_do;
    }
    if (dtype == at::kInt) {
        return inplace_index_add_int32_do;
    }
    TORCH_CHECK(false, "unsupported dtype, must be float16, float32 or int32");
    return nullptr;
}

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

at::Tensor run_inplace_index_add(
    const at::Tensor &var,
    const at::Tensor &value,
    const at::Tensor &sorted_indices,
    const at::Tensor &alpha)
{
    TORCH_CHECK(var.device().is_privateuseone(), "var must be on NPU");
    TORCH_CHECK(value.device().is_privateuseone(), "value must be on NPU");
    TORCH_CHECK(sorted_indices.device().is_privateuseone(), "sorted_indices must be on NPU");
    TORCH_CHECK(alpha.device().is_privateuseone(), "alpha must be on NPU");
    TORCH_CHECK(var.dim() == 2, "var must be 2D after permutation");
    TORCH_CHECK(value.dim() == 2, "value must be 2D after permutation");
    TORCH_CHECK(sorted_indices.dim() == 1, "sorted_indices must be 1D");
    TORCH_CHECK(var.is_contiguous(), "var must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(sorted_indices.is_contiguous(), "sorted_indices must be contiguous");
    TORCH_CHECK(sorted_indices.scalar_type() == at::kInt, "sorted_indices must be int32");
    TORCH_CHECK(alpha.numel() == 1, "alpha must be a scalar tensor");
    TORCH_CHECK(var.scalar_type() == value.scalar_type(), "var and value must have the same dtype");
    TORCH_CHECK(
        var.scalar_type() == at::kFloat || var.scalar_type() == at::kHalf || var.scalar_type() == at::kInt,
        "var must be float16, float32 or int32");

    const int32_t m = static_cast<int32_t>(var.size(0));
    const int32_t n = static_cast<int32_t>(var.size(1));
    const int32_t k = static_cast<int32_t>(value.size(0));
    TORCH_CHECK(value.size(1) == n, "value N must match var N");
    TORCH_CHECK(sorted_indices.size(0) == k, "sorted_indices K must match value K");

    const int32_t blockM = CeilDivI32(std::max(m, 1), std::min(DEFAULT_NUM_PHYSICAL_CORES, std::max(m, 1)));
    const int32_t mNum = CeilDivI32(std::max(m, 1), std::max(blockM, 1));
    const int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, std::max(mNum, 1));
    const int32_t tasksPerCore = CeilDivI32(std::max(mNum, 1), usedCoreNum);

    at::Tensor out = at::empty_like(var);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(InplaceIndexAddTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<InplaceIndexAddTiling *>(tilingCpu.data_ptr());
    tiling->M = m;
    tiling->N = n;
    tiling->K = k;
    tiling->blockM = blockM;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = ResolveLaunchFn(var.scalar_type());
    launch(
        static_cast<uint32_t>(usedCoreNum),
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(var.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(value.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(sorted_indices.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(alpha.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(out.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));
    return out;
}

} // namespace inplace_index_add_ext

PYBIND11_MODULE(_inplace_index_add_ext, m)
{
    m.doc() = "inplace_index_add AscendC extension";
    m.def("run_inplace_index_add", &inplace_index_add_ext::run_inplace_index_add, "");
}
