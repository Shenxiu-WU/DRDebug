#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "matmul_transb_tiling.h"

extern "C" void matmul_transb_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *a,
    uint8_t *b,
    uint8_t *c,
    uint8_t *tiling);

extern "C" void matmul_transb_do_fp16(
    uint32_t blockDim,
    void *stream,
    uint8_t *a,
    uint8_t *b,
    uint8_t *c,
    uint8_t *tiling);

extern "C" void matmul_transb_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *a,
    uint8_t *b,
    uint8_t *c,
    uint8_t *tiling);

namespace matmul_transb_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

at::Tensor run_matmul_transb(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.dim() == 2, "A must be 2-D [M, K]");
    TORCH_CHECK(b.dim() == 2, "B must be 2-D [N, K]");
    TORCH_CHECK(a.sizes()[1] == b.sizes()[1], "K dimension must match between A and B");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "A and B must have the same dtype");
    TORCH_CHECK(
        a.scalar_type() == at::kFloat || a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
        "A must be float32, float16, or bfloat16");
    TORCH_CHECK(a.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "B must be contiguous");

    const auto m = static_cast<int32_t>(a.sizes()[0]);
    const auto k = static_cast<int32_t>(a.sizes()[1]);
    const auto n = static_cast<int32_t>(b.sizes()[0]);

    const int32_t mNum = (m + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    int32_t usedCoreNum = std::min<int32_t>(static_cast<int32_t>(DEFAULT_NUM_PHYSICAL_CORES), mNum);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    const int32_t tasksPerCore = (mNum + usedCoreNum - 1) / usedCoreNum;

    at::Tensor c = at::empty({m, n}, a.options());

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(MatmulTransBKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<MatmulTransBKernelTiling *>(tilingCpu.data_ptr());
    tiling->M = m;
    tiling->N = n;
    tiling->K = k;
    tiling->blockM = DEFAULT_BLOCK_M;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (a.scalar_type() == at::kFloat) {
        launch = matmul_transb_do_fp32;
    } else if (a.scalar_type() == at::kHalf) {
        launch = matmul_transb_do_fp16;
    } else if (a.scalar_type() == at::kBFloat16) {
        launch = matmul_transb_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        static_cast<uint32_t>(usedCoreNum),
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(a.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(b.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(c.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));
    return c;
}

}  // namespace matmul_transb_ext

PYBIND11_MODULE(_matmul_transb_ext, m)
{
    m.doc() = "matmul_transb extension";
    m.def("run_matmul_transb", &matmul_transb_ext::run_matmul_transb, "");
}
