#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "gather_elements_v2_tiling.h"

extern "C" void gather_elements_v2_last_dim_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *index,
    uint8_t *rowMap,
    uint8_t *y,
    uint8_t *tiling);

extern "C" void gather_elements_v2_last_dim_fp32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *index,
    uint8_t *rowMap,
    uint8_t *y,
    uint8_t *tiling);

extern "C" void gather_elements_v2_transpose_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *index,
    uint8_t *rowMap,
    uint8_t *y,
    uint8_t *tiling);

extern "C" void gather_elements_v2_transpose_fp32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *index,
    uint8_t *rowMap,
    uint8_t *y,
    uint8_t *tiling);

extern "C" void gather_elements_v2_scalar_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *index,
    uint8_t *rowMap,
    uint8_t *y,
    uint8_t *tiling);

extern "C" void gather_elements_v2_scalar_fp32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *index,
    uint8_t *rowMap,
    uint8_t *y,
    uint8_t *tiling);

namespace current_task_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

inline LaunchFn ResolveLaunchFn(int32_t mode, at::ScalarType dtype)
{
    TORCH_CHECK(
        mode == GATHER_MODE_LAST_DIM || mode == GATHER_MODE_TRANSPOSE || mode == GATHER_MODE_SCALAR,
        "mode must be one of {0(last_dim), 1(transpose), 2(scalar)}");

    const bool isFp16 = (dtype == at::kHalf);
    const bool isFp32 = (dtype == at::kFloat);
    TORCH_CHECK(isFp16 || isFp32, "x must be float16 or float32");

    if (mode == GATHER_MODE_LAST_DIM) {
        return isFp16 ? gather_elements_v2_last_dim_fp16_do : gather_elements_v2_last_dim_fp32_do;
    }
    if (mode == GATHER_MODE_TRANSPOSE) {
        return isFp16 ? gather_elements_v2_transpose_fp16_do : gather_elements_v2_transpose_fp32_do;
    }
    return isFp16 ? gather_elements_v2_scalar_fp16_do : gather_elements_v2_scalar_fp32_do;
}

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

inline int32_t ChooseBlockM(int32_t m)
{
    int32_t blockM = 32;
    if (m < 32) {
        blockM = 16;
    }
    if (m < 16) {
        blockM = 8;
    }
    if (m < 8) {
        blockM = 4;
    }
    if (m < 4) {
        blockM = 2;
    }
    if (m < 2) {
        blockM = 1;
    }
    return blockM;
}

at::Tensor run_gather_elements_v2(
    const at::Tensor &x,
    const at::Tensor &index,
    const at::Tensor &row_map,
    int64_t ig,
    int64_t mode)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU");
    TORCH_CHECK(index.device().is_privateuseone(), "index must be on NPU");
    TORCH_CHECK(row_map.device().is_privateuseone(), "row_map must be on NPU");
    TORCH_CHECK(x.dim() == 2, "x must be [XRows, XStride]");
    TORCH_CHECK(index.dim() == 2, "index must be [M, YStride]");
    TORCH_CHECK(row_map.dim() == 1, "row_map must be [M]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(row_map.is_contiguous(), "row_map must be contiguous");
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(row_map.scalar_type() == at::kInt, "row_map must be int32");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat, "x must be float16 or float32");

    const int32_t mode32 = static_cast<int32_t>(mode);
    const int32_t m = static_cast<int32_t>(index.size(0));
    TORCH_CHECK(index.size(0) == row_map.size(0), "index and row_map row count must match");
    const int32_t xRows = static_cast<int32_t>(x.size(0));
    const int32_t xStride = static_cast<int32_t>(x.size(1));
    const int32_t yStride = static_cast<int32_t>(index.size(1));
    const int32_t ig32 = static_cast<int32_t>(ig);

    TORCH_CHECK(ig32 >= 0, "ig must be non-negative");
    TORCH_CHECK(ig32 <= yStride, "ig must not exceed index width");

    const int32_t blockM = ChooseBlockM(std::max(m, 1));
    const int32_t mNum = CeilDivI32(std::max(m, 1), blockM);
    const int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, std::max(mNum, 1));
    const int32_t tasksPerCore = CeilDivI32(mNum, usedCoreNum);
    const int32_t subBlockM = std::max(blockM / DEFAULT_VEC_NUM, 1);

    at::Tensor y = at::zeros({m, yStride}, x.options());

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(GatherElementsV2KernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<GatherElementsV2KernelTiling *>(tilingCpu.data_ptr());
    tiling->M = m;
    tiling->XRows = xRows;
    tiling->XG = xStride;
    tiling->IG = ig32;
    tiling->XStride = xStride;
    tiling->YStride = yStride;
    tiling->blockM = blockM;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->subBlockM = subBlockM;
    tiling->useRowMap = (mode32 == GATHER_MODE_LAST_DIM) ? 0 : 1;
    tiling->mode = mode32;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = ResolveLaunchFn(mode32, x.scalar_type());
    launch(
        static_cast<uint32_t>(usedCoreNum),
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(x.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(index.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(row_map.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(y.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));
    return y;
}

} // namespace current_task_ext

PYBIND11_MODULE(_current_task_ext, m)
{
    m.doc() = "current_task gather_elements_v2 AscendC extension";
    m.def("run_gather_elements_v2", &current_task_ext::run_gather_elements_v2, "");
}
