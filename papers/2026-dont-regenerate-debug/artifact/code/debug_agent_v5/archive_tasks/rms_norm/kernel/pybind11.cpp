#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "rms_norm_tiling.h"

extern "C" void rms_norm_merge_n_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *y,
    uint8_t *invRms,
    uint8_t *tiling);

extern "C" void rms_norm_merge_n_do_fp16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *y,
    uint8_t *invRms,
    uint8_t *tiling);

extern "C" void rms_norm_merge_n_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *y,
    uint8_t *invRms,
    uint8_t *tiling);

extern "C" void rms_norm_single_row_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *y,
    uint8_t *invRms,
    uint8_t *tiling);

extern "C" void rms_norm_single_row_do_fp16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *y,
    uint8_t *invRms,
    uint8_t *tiling);

extern "C" void rms_norm_single_row_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *y,
    uint8_t *invRms,
    uint8_t *tiling);

extern "C" void rms_norm_splitd_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *y,
    uint8_t *invRms,
    uint8_t *tiling);

extern "C" void rms_norm_splitd_do_fp16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *y,
    uint8_t *invRms,
    uint8_t *tiling);

extern "C" void rms_norm_splitd_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *y,
    uint8_t *invRms,
    uint8_t *tiling);

namespace rms_norm_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

pybind11::tuple run_rms_norm(const at::Tensor &x, const at::Tensor &gamma, double eps)
{
    TORCH_CHECK(x.dim() == 2, "x must be [M, N]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [N]");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
        "x must be float16, float32, or bfloat16");
    TORCH_CHECK(gamma.scalar_type() == x.scalar_type(), "gamma must have the same dtype as x");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(x.sizes()[1] == gamma.sizes()[0], "gamma shape mismatch");

    const auto m = static_cast<int32_t>(x.sizes()[0]);
    const auto n = static_cast<int32_t>(x.sizes()[1]);

    const int32_t mNum = (m + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    const int32_t tasksPerCore = (mNum + usedCoreNum - 1) / usedCoreNum;

    at::Tensor y = at::empty_like(x);
    at::Tensor invRms = at::empty({m}, x.options());

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(RmsNormKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<RmsNormKernelTiling *>(tilingCpu.data_ptr());
    tiling->M = m;
    tiling->N = n;
    tiling->blockM = DEFAULT_BLOCK_M;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->rowFactor = DEFAULT_ROW_FACTOR;
    tiling->eps = static_cast<float>(eps);
    tiling->invN = 1.0f / static_cast<float>(n);
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (x.scalar_type() == at::kFloat) {
        if (n <= 1024) {
            launch = rms_norm_merge_n_do_fp32;
        } else if (n > 8192) {
            launch = rms_norm_splitd_do_fp32;
        } else {
            launch = rms_norm_single_row_do_fp32;
        }
    } else if (x.scalar_type() == at::kHalf) {
        if (n <= 1024) {
            launch = rms_norm_merge_n_do_fp16;
        } else if (n > 8192) {
            launch = rms_norm_splitd_do_fp16;
        } else {
            launch = rms_norm_single_row_do_fp16;
        }
    } else if (x.scalar_type() == at::kBFloat16) {
        if (n <= 1024) {
            launch = rms_norm_merge_n_do_bf16;
        } else if (n > 8192) {
            launch = rms_norm_splitd_do_bf16;
        } else {
            launch = rms_norm_single_row_do_bf16;
        }
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }
    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(x.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(gamma.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(y.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(invRms.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));
    return pybind11::make_tuple(y, invRms);
}

}  // namespace rms_norm_ext

PYBIND11_MODULE(_rms_norm_ext, m)
{
    m.doc() = "rms_norm extension";
    m.def("run_rms_norm", &rms_norm_ext::run_rms_norm, "");
}
