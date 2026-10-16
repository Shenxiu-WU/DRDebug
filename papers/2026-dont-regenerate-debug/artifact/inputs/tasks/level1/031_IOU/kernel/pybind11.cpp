#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "iou_tiling.h"

extern "C" void iou_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *bboxes_x1, uint8_t *bboxes_y1, uint8_t *bboxes_x2, uint8_t *bboxes_y2,
    uint8_t *gtboxes_x1, uint8_t *gtboxes_y1, uint8_t *gtboxes_x2, uint8_t *gtboxes_y2,
    uint8_t *out, uint8_t *tiling);

extern "C" void iou_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *bboxes_x1, uint8_t *bboxes_y1, uint8_t *bboxes_x2, uint8_t *bboxes_y2,
    uint8_t *gtboxes_x1, uint8_t *gtboxes_y1, uint8_t *gtboxes_x2, uint8_t *gtboxes_y2,
    uint8_t *out, uint8_t *tiling);

namespace iou_ext {

using LaunchFn = void (*)(uint32_t, void *,
    uint8_t *, uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, uint8_t *);

at::Tensor run_iou(
    const at::Tensor &bboxes_x1, const at::Tensor &bboxes_y1,
    const at::Tensor &bboxes_x2, const at::Tensor &bboxes_y2,
    const at::Tensor &gtboxes_x1, const at::Tensor &gtboxes_y1,
    const at::Tensor &gtboxes_x2, const at::Tensor &gtboxes_y2,
    int64_t mode)
{
    TORCH_CHECK(bboxes_x1.dim() == 1, "bboxes_x1 must be 1D");
    TORCH_CHECK(gtboxes_x1.dim() == 1, "gtboxes_x1 must be 1D");
    TORCH_CHECK(bboxes_x1.scalar_type() == at::kFloat || bboxes_x1.scalar_type() == at::kHalf,
                "dtype must be float32 or float16");
    TORCH_CHECK(gtboxes_x1.scalar_type() == bboxes_x1.scalar_type(), "dtype mismatch");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");

    int32_t N = static_cast<int32_t>(bboxes_x1.sizes()[0]);
    int32_t M = static_cast<int32_t>(gtboxes_x1.sizes()[0]);

    int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, M);
    int32_t tasksPerCore = (M + usedCoreNum - 1) / usedCoreNum;

    at::Tensor out = at::empty({M, N}, bboxes_x1.options());

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(IouKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<IouKernelTiling *>(tilingCpu.data_ptr());
    tiling->N = N;
    tiling->M = M;
    tiling->tileN = DEFAULT_TILE_N;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->mode = static_cast<int32_t>(mode);
    tiling->eps = 1e-10f;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (bboxes_x1.scalar_type() == at::kFloat) {
        launch = iou_do_fp32;
    } else if (bboxes_x1.scalar_type() == at::kHalf) {
        launch = iou_do_fp16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }
    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(bboxes_x1.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(bboxes_y1.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(bboxes_x2.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(bboxes_y2.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(gtboxes_x1.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(gtboxes_y1.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(gtboxes_x2.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(gtboxes_y2.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(out.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));
    return out;
}

} // namespace iou_ext

PYBIND11_MODULE(_iou_ext, m)
{
    m.doc() = "iou extension";
    m.def("run_iou", &iou_ext::run_iou, "");
}
