#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"

#include "reshape_matmul_quant_tiling.h"

extern "C" void reshape_matmul_quant_do_bf16(uint32_t blockDim, void *stream,
                                              uint8_t *x, uint8_t *h, uint8_t *y,
                                              uint8_t *workspace, uint8_t *tiling);

namespace my_reshape_matmul_quant {

at::Tensor run_reshape_matmul_quant(const at::Tensor &x, const at::Tensor &h)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(h.dim() == 2, "h must be 2D");
    TORCH_CHECK(h.sizes()[0] == h.sizes()[1], "h must be square");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(h.scalar_type() == at::kBFloat16, "h must be bfloat16");

    int32_t M = x.sizes()[0];
    int32_t N = x.sizes()[1];
    int32_t H_K = h.sizes()[0];

    TORCH_CHECK(N % H_K == 0, "N must be divisible by H_K");
    TORCH_CHECK(M % DEFAULT_BASE_M == 0, "M must be divisible by baseM");
    TORCH_CHECK(N % DEFAULT_BASE_N == 0, "N must be divisible by baseN");
    TORCH_CHECK(H_K % DEFAULT_BASE_K == 0, "H_K must be divisible by baseK");

    int32_t nTiles = N / DEFAULT_BASE_N;
    int32_t nTilesPerH = H_K / DEFAULT_BASE_N;
    int32_t mNum = M / DEFAULT_BASE_M;  // one core per m-block

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t usedCoreNum = mNum;

    // Output: int8 (M, N)
    at::Tensor y = at::empty({M, N}, at::device(at::kPrivateUse1).dtype(at::kChar));

    // Workspace: float32 (M, N)
    int64_t wsSize = (int64_t)M * N * sizeof(float);
    at::Tensor w = at::empty({wsSize}, at::device(at::kPrivateUse1).dtype(at::kByte));

    // Tiling
    at::Tensor t = at::empty({(int64_t)sizeof(ReshapeMatmulQuantTiling)},
                              at::device(at::kCPU).dtype(at::kByte));
    auto *tp = reinterpret_cast<ReshapeMatmulQuantTiling *>(t.data_ptr());
    tp->M = M;
    tp->N = N;
    tp->H_K = H_K;
    tp->baseM = DEFAULT_BASE_M;
    tp->baseN = DEFAULT_BASE_N;
    tp->baseK = DEFAULT_BASE_K;
    tp->K_L1 = DEFAULT_K_L1;
    tp->nTiles = nTiles;
    tp->nTilesPerH = nTilesPerH;
    auto tiling_npu = t.to(at::kPrivateUse1);

    reshape_matmul_quant_do_bf16(
        usedCoreNum, acl_stream,
        static_cast<uint8_t*>(const_cast<void*>(x.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(h.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(y.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(w.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(tiling_npu.storage().data())));

    return y;
}

} // namespace my_reshape_matmul_quant

PYBIND11_MODULE(_reshape_matmul_rowwise_quant_int8_ext, m)
{
    m.doc() = "reshape_matmul_rowwise_quant_int8 pybind11 interfaces";
    m.def("run_reshape_matmul_quant", &my_reshape_matmul_quant::run_reshape_matmul_quant, "");
}
