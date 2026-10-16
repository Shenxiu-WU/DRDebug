#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"

#include "batch_matmul_tiling.h"

extern "C" void batch_matmul_do_fp16(uint32_t blockDim, void *stream,
                                     uint8_t *a, uint8_t *b, uint8_t *c,
                                     uint8_t *tiling);

namespace my_batch_matmul {

at::Tensor run_batch_matmul(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.dim() == 3, "batch_matmul a must be 3D.");
    TORCH_CHECK(b.dim() == 3, "batch_matmul b must be 3D.");
    TORCH_CHECK(a.sizes()[0] == b.sizes()[0], "batch_matmul batch must match.");
    TORCH_CHECK(a.sizes()[2] == b.sizes()[1], "batch_matmul k must match.");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "batch_matmul a and b must have same dtype.");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "batch_matmul only supports float16 in kernel.");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t usedCoreNum = 20;

    int B = a.sizes()[0];
    int M = a.sizes()[1];
    int K = a.sizes()[2];
    int N = b.sizes()[2];

    at::Tensor c = at::empty({B, M, N}, at::device(at::kPrivateUse1).dtype(at::kFloat));

    at::Tensor t = at::empty({sizeof(BatchMatmulTiling)}, at::device(at::kCPU).dtype(at::kByte));
    auto *tiling_ptr = reinterpret_cast<BatchMatmulTiling *>(t.data_ptr());
    tiling_ptr->B = B;
    tiling_ptr->M = M;
    tiling_ptr->N = N;
    tiling_ptr->K = K;
    tiling_ptr->baseM = DEFAULT_BASE_M;
    tiling_ptr->baseN = DEFAULT_BASE_N;
    tiling_ptr->baseK = DEFAULT_BASE_K;
    auto tiling_npu = t.to(at::kPrivateUse1);

    batch_matmul_do_fp16(usedCoreNum, acl_stream,
                         static_cast<uint8_t*>(const_cast<void*>(a.storage().data())),
                         static_cast<uint8_t*>(const_cast<void*>(b.storage().data())),
                         static_cast<uint8_t*>(const_cast<void*>(c.storage().data())),
                         static_cast<uint8_t*>(const_cast<void*>(tiling_npu.storage().data())));
    return c;
}

} // namespace my_batch_matmul

PYBIND11_MODULE(_batch_matmul_ext, m)
{
    m.doc() = "batch_matmul pybind11 interfaces";
    m.def("run_batch_matmul", &my_batch_matmul::run_batch_matmul, "");
}
