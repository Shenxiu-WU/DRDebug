#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "scatter_elements_v2_tiling.h"

extern "C" void scatter_elements_v2_last_dim_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *selfGM,
    uint8_t *indexGM,
    uint8_t *updatesGM,
    uint8_t *outGM,
    uint8_t *tiling);

extern "C" void scatter_elements_v2_last_dim_fp32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *selfGM,
    uint8_t *indexGM,
    uint8_t *updatesGM,
    uint8_t *outGM,
    uint8_t *tiling);

extern "C" void scatter_elements_v2_last_dim_int32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *selfGM,
    uint8_t *indexGM,
    uint8_t *updatesGM,
    uint8_t *outGM,
    uint8_t *tiling);

namespace current_task_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

inline LaunchFn ResolveLaunchFn(at::ScalarType dtype)
{
    if (dtype == at::kHalf) {
        return scatter_elements_v2_last_dim_fp16_do;
    }
    if (dtype == at::kFloat) {
        return scatter_elements_v2_last_dim_fp32_do;
    }
    if (dtype == at::kInt) {
        return scatter_elements_v2_last_dim_int32_do;
    }
    TORCH_CHECK(false, "Unsupported dtype for scatter_elements_v2 AscendC kernel");
    return nullptr;
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

at::Tensor run_scatter_elements_v2(
    const at::Tensor &self,
    const at::Tensor &index,
    const at::Tensor &updates,
    int64_t reduction_mode)
{
    TORCH_CHECK(self.device().is_privateuseone(), "self must be on NPU");
    TORCH_CHECK(index.device().is_privateuseone(), "index must be on NPU");
    TORCH_CHECK(updates.device().is_privateuseone(), "updates must be on NPU");
    TORCH_CHECK(self.dim() == 2, "self must be [XRows, XStride]");
    TORCH_CHECK(index.dim() == 2, "index must be [M, YStride]");
    TORCH_CHECK(updates.dim() == 2, "updates must be [M, YStride]");
    TORCH_CHECK(self.is_contiguous(), "self must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(updates.is_contiguous(), "updates must be contiguous");
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(updates.scalar_type() == self.scalar_type(), "updates dtype must match self dtype");
    TORCH_CHECK(
        self.scalar_type() == at::kHalf || self.scalar_type() == at::kFloat || self.scalar_type() == at::kInt,
        "self must be float16, float32 or int32"
    );

    const int32_t m = static_cast<int32_t>(index.size(0));
    const int32_t xRows = static_cast<int32_t>(self.size(0));
    const int32_t xStride = static_cast<int32_t>(self.size(1));
    const int32_t yStride = static_cast<int32_t>(index.size(1));
    const int32_t xg = static_cast<int32_t>(xStride);
    const int32_t ig = static_cast<int32_t>(yStride);

    const int32_t blockM = ChooseBlockM(std::max(m, 1));
    const int32_t mNum = CeilDivI32(std::max(m, 1), blockM);
    const int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, std::max(mNum, 1));
    const int32_t tasksPerCore = CeilDivI32(mNum, usedCoreNum);
    const int32_t subBlockM = std::max(blockM / DEFAULT_VEC_NUM, 1);

    // Initialize output as copy of self
    at::Tensor out = self.clone();

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(ScatterElementsV2KernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<ScatterElementsV2KernelTiling *>(tilingCpu.data_ptr());
    tiling->M = m;
    tiling->XRows = xRows;
    tiling->XG = xg;
    tiling->IG = ig;
    tiling->XStride = xStride;
    tiling->YStride = yStride;
    tiling->blockM = blockM;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->subBlockM = subBlockM;
    tiling->reductionMode = static_cast<int32_t>(reduction_mode);
    tiling->mode = SCATTER_MODE_LAST_DIM;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = ResolveLaunchFn(self.scalar_type());
    launch(
        static_cast<uint32_t>(usedCoreNum),
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(self.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(index.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(updates.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(out.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));
    return out;
}

} // namespace current_task_ext

PYBIND11_MODULE(_current_task_ext, m)
{
    m.doc() = "current_task scatter_elements_v2 AscendC extension";
    m.def("run_scatter_elements_v2", &current_task_ext::run_scatter_elements_v2, "");
}
