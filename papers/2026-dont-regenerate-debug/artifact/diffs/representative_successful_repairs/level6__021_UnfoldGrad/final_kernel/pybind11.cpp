#include <algorithm>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "unfold_backward_tiling.h"

extern "C" void unfold_backward_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *grad_output,
    uint8_t *grad_input,
    uint8_t *tiling);

extern "C" void unfold_backward_fp32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *grad_output,
    uint8_t *grad_input,
    uint8_t *tiling);

namespace current_task_ext {

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

at::Tensor run_unfold_backward_impl(
    const at::Tensor &grad_output,
    int64_t D,
    int64_t n_windows,
    int64_t size,
    int64_t step)
{
    TORCH_CHECK(grad_output.device().is_privateuseone(), "grad_output must be on NPU");
    TORCH_CHECK(grad_output.dim() == 3, "grad_output must be 3D [B, n_windows, size]");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.scalar_type() == at::kHalf || grad_output.scalar_type() == at::kFloat,
                "grad_output must be float16 or float32");

    const int32_t B = static_cast<int32_t>(grad_output.size(0));
    const int32_t D32 = static_cast<int32_t>(D);
    const int32_t nWindows = static_cast<int32_t>(n_windows);
    const int32_t size32 = static_cast<int32_t>(size);
    const int32_t step32 = static_cast<int32_t>(step);

    TORCH_CHECK(grad_output.size(1) >= nWindows, "grad_output dim 1 must be >= n_windows");
    TORCH_CHECK(grad_output.size(2) == size32, "grad_output dim 2 must be == size");

    at::Tensor grad_input = at::zeros({B, D32}, grad_output.options());

    const int32_t blockB = 16;
    const int32_t bNum = CeilDivI32(std::max(B, 1), blockB);
    const int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, std::max(bNum, 1));
    const int32_t tasksPerCore = CeilDivI32(bNum, usedCoreNum);
    const int32_t subBlockB = std::max(blockB / DEFAULT_VEC_NUM, 1);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(UnfoldBackwardTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<UnfoldBackwardTiling *>(tilingCpu.data_ptr());
    tiling->B = B;
    tiling->D = D32;
    tiling->nWindows = nWindows;
    tiling->size = size32;
    tiling->step = step32;
    tiling->blockB = blockB;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->subBlockB = subBlockB;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    void (*launch_fn)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *) = nullptr;
    if (grad_output.scalar_type() == at::kHalf) {
        launch_fn = unfold_backward_fp16_do;
    } else {
        launch_fn = unfold_backward_fp32_do;
    }

    launch_fn(
        static_cast<uint32_t>(usedCoreNum),
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(grad_output.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(grad_input.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return grad_input;
}

at::Tensor run_unfold_backward(
    const at::Tensor &grad_output,
    int64_t D,
    int64_t n_windows,
    int64_t size,
    int64_t step)
{
    if (grad_output.scalar_type() == at::kBFloat16) {
        auto grad_output_f32 = grad_output.to(at::kFloat);
        auto grad_input_f32 = run_unfold_backward_impl(grad_output_f32, D, n_windows, size, step);
        return grad_input_f32.to(at::kBFloat16);
    }
    return run_unfold_backward_impl(grad_output, D, n_windows, size, step);
}

} // namespace current_task_ext

PYBIND11_MODULE(_unfold_backward_ext, m)
{
    m.doc() = "unfold_backward AscendC extension";
    m.def("run_unfold_backward", &current_task_ext::run_unfold_backward, "");
}
