#include <algorithm>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "hyena_rfft_tiling.h"

extern "C" void hyena_rfft_do_fp32(
    uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* real, uint8_t* imag, uint8_t* tiling);

extern "C" void hyena_rfft_do_fp16(
    uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* real, uint8_t* imag, uint8_t* tiling);

extern "C" void hyena_rfft_do_bf16(
    uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* real, uint8_t* imag, uint8_t* tiling);

namespace hyena_rfft_ext {

using LaunchFn = void (*)(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*);

pybind11::tuple run_hyena_rfft(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() == 3, "x must be [B, C, L]");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
        "x must be float16, float32, or bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int32_t B = static_cast<int32_t>(x.sizes()[0]);
    int32_t C = static_cast<int32_t>(x.sizes()[1]);
    int32_t L = static_cast<int32_t>(x.sizes()[2]);
    int32_t N = 2 * L;
    int32_t totalSeq = B * C;

    auto options = x.options().dtype(at::kFloat);
    at::Tensor real = at::empty({B, C, L + 1}, options);
    at::Tensor imag = at::empty({B, C, L + 1}, options);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(HyenaRfftTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto* tiling = reinterpret_cast<HyenaRfftTiling*>(tilingCpu.data_ptr());
    tiling->B = B;
    tiling->C = C;
    tiling->L = L;
    tiling->N = N;
    tiling->totalSeq = totalSeq;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (x.scalar_type() == at::kFloat) {
        launch = hyena_rfft_do_fp32;
    } else if (x.scalar_type() == at::kHalf) {
        launch = hyena_rfft_do_fp16;
    } else if (x.scalar_type() == at::kBFloat16) {
        launch = hyena_rfft_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        totalSeq,
        aclStream,
        static_cast<uint8_t*>(const_cast<void*>(x.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(real.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(imag.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(tilingNpu.storage().data())));

    return pybind11::make_tuple(real, imag);
}

}  // namespace hyena_rfft_ext

PYBIND11_MODULE(_hyena_rfft_ext, m)
{
    m.doc() = "hyena_rfft extension";
    m.def("run_hyena_rfft", &hyena_rfft_ext::run_hyena_rfft, "");
}
