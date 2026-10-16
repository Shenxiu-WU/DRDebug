/**
 * @file pybind11.cpp
 *
 * Host entry point for the MatmulTransA AscendC operator.
 *
 * Logical op:    C = A.T @ B,  with A:(m,n), B:(m,k), C:(n,k)
 *
 * Layout strategy:
 *   - The Python wrapper is responsible for transposing A and zero-padding
 *     A.T and B so that all of (n, m, k) become multiples of 16. This pybind
 *     therefore receives `A_T` (shape padded_n x padded_m) and `B` (shape
 *     padded_m x padded_k), both already padded.
 *   - Internally we relabel (M = padded_n, K = padded_m, N = padded_k) and
 *     dispatch a vanilla GEMM cube launcher per dtype.
 *   - Output of this pybind call is the padded `C` of shape (padded_n, padded_k);
 *     the Python wrapper slices it back to (n, k) when needed.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"

#include "matmul_trans_a_tiling.h"

extern "C" void matmul_trans_a_do_fp16(uint32_t blockDim, void *stream,
                                       uint8_t *a, uint8_t *b, uint8_t *c,
                                       uint8_t *workspace, uint8_t *tiling);

extern "C" void matmul_trans_a_do_bf16(uint32_t blockDim, void *stream,
                                       uint8_t *a, uint8_t *b, uint8_t *c,
                                       uint8_t *workspace, uint8_t *tiling);

extern "C" void matmul_trans_a_do_fp32(uint32_t blockDim, void *stream,
                                       uint8_t *a, uint8_t *b, uint8_t *c,
                                       uint8_t *workspace, uint8_t *tiling);

namespace my_matmul_trans_a {

using LaunchFn = void(*)(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*);

at::Tensor run_matmul_trans_a(const at::Tensor &a_t, const at::Tensor &b)
{
    TORCH_CHECK(a_t.dim() == 2, "matmul_trans_a: A_T must be 2D");
    TORCH_CHECK(b.dim() == 2,   "matmul_trans_a: B must be 2D");
    TORCH_CHECK(a_t.scalar_type() == b.scalar_type(),
                "matmul_trans_a: A_T and B must share dtype");
    TORCH_CHECK(a_t.sizes()[1] == b.sizes()[0],
                "matmul_trans_a: A_T.shape[1] (padded_m) must equal B.shape[0]");

    int32_t M = static_cast<int32_t>(a_t.sizes()[0]); // padded_n
    int32_t K = static_cast<int32_t>(a_t.sizes()[1]); // padded_m
    int32_t N = static_cast<int32_t>(b.sizes()[1]);   // padded_k

    TORCH_CHECK(M % DEFAULT_BASE_M == 0, "matmul_trans_a: padded_n must be a multiple of baseM");
    TORCH_CHECK(N % DEFAULT_BASE_N == 0, "matmul_trans_a: padded_k must be a multiple of baseN");
    TORCH_CHECK(K % DEFAULT_BASE_K == 0, "matmul_trans_a: padded_m must be a multiple of baseK");

    LaunchFn launchFn = nullptr;
    at::ScalarType outDtype = a_t.scalar_type();

    if (a_t.scalar_type() == at::kHalf) {
        launchFn = matmul_trans_a_do_fp16;
    } else if (a_t.scalar_type() == at::kBFloat16) {
        launchFn = matmul_trans_a_do_bf16;
    } else if (a_t.scalar_type() == at::kFloat) {
        launchFn = matmul_trans_a_do_fp32;
    } else {
        TORCH_CHECK(false, "matmul_trans_a: unsupported dtype, expected float16/bfloat16/float32");
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    // 20 AIC cores match the design's persistent schedule. The BlockScheduler
    // distributes (M/baseM)*(N/baseN) tiles round-robin so it is safe to run
    // even when there are more cores than tiles (extra cores get zero work).
    uint32_t usedCoreNum = 20;

    at::Tensor c = at::empty({M, N}, at::device(at::kPrivateUse1).dtype(outDtype));

    // Tiling struct (CPU side, then pushed to device)
    at::Tensor t = at::empty({(int64_t)sizeof(MatmulTransATiling)},
                              at::device(at::kCPU).dtype(at::kByte));
    auto *tp = reinterpret_cast<MatmulTransATiling *>(t.data_ptr());
    tp->M          = M;
    tp->N          = N;
    tp->K          = K;
    tp->baseM      = DEFAULT_BASE_M;
    tp->baseN      = DEFAULT_BASE_N;
    tp->baseK      = DEFAULT_BASE_K;
    tp->l1Prefetch = DEFAULT_L1_PREFETCH;
    auto tiling_npu = t.to(at::kPrivateUse1);

    // Workspace is unused by this pure-cube kernel but the launcher
    // signature still expects an address; allocate a tiny stub.
    at::Tensor w = at::empty({64}, at::device(at::kPrivateUse1).dtype(at::kByte));

    launchFn(usedCoreNum, acl_stream,
             static_cast<uint8_t*>(const_cast<void*>(a_t.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(b.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(c.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(w.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(tiling_npu.storage().data())));
    return c;
}

} // namespace my_matmul_trans_a

PYBIND11_MODULE(_matmul_trans_a_ext, m)
{
    m.doc() = "matmul_trans_a pybind11 interfaces";
    m.def("run_matmul_trans_a", &my_matmul_trans_a::run_matmul_trans_a, "");
}
