#include <algorithm>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "norm_rope_concat_tiling.h"

extern "C" void norm_rope_concat_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *queryWeight, uint8_t *queryBias,
    uint8_t *keyWeight, uint8_t *keyBias,
    uint8_t *sin, uint8_t *cos,
    uint8_t *queryOut, uint8_t *keyOut, uint8_t *valueOut,
    uint8_t *tiling);

extern "C" void norm_rope_concat_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *queryWeight, uint8_t *queryBias,
    uint8_t *keyWeight, uint8_t *keyBias,
    uint8_t *sin, uint8_t *cos,
    uint8_t *queryOut, uint8_t *keyOut, uint8_t *valueOut,
    uint8_t *tiling);

extern "C" void norm_rope_concat_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *queryWeight, uint8_t *queryBias,
    uint8_t *keyWeight, uint8_t *keyBias,
    uint8_t *sin, uint8_t *cos,
    uint8_t *queryOut, uint8_t *keyOut, uint8_t *valueOut,
    uint8_t *tiling);

namespace norm_rope_concat_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                          uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

pybind11::tuple run_norm_rope_concat(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &queryWeight,
    const at::Tensor &queryBias,
    const at::Tensor &keyWeight,
    const at::Tensor &keyBias,
    const at::Tensor &sin,
    const at::Tensor &cos,
    int64_t normType,
    int64_t ropeType,
    double eps)
{
    TORCH_CHECK(query.dim() == 4, "query must be 4D [B,S,N,D]");
    TORCH_CHECK(key.dim() == 4, "key must be 4D [B,S,N,D]");
    TORCH_CHECK(value.dim() == 4, "value must be 4D [B,S,N,D]");
    TORCH_CHECK(query.sizes() == key.sizes() && key.sizes() == value.sizes(),
                "query/key/value must have same shape");
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(),
                "query/key/value must have same dtype");
    TORCH_CHECK(query.scalar_type() == at::kFloat || query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "query must be float16, float32, or bfloat16");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
                "inputs must be contiguous");

    const auto B = query.sizes()[0];
    const auto S = query.sizes()[1];
    const auto N = query.sizes()[2];
    const auto D = query.sizes()[3];
    const int32_t M = static_cast<int32_t>(B * S * N);
    const int32_t mNum = (M + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, mNum);

    auto options = query.options();
    at::Tensor queryOut = at::empty({B, N, S, D}, options);
    at::Tensor keyOut = at::empty({B, N, S, D}, options);
    at::Tensor valueOut = at::empty({B, N, S, D}, options);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(NormRopeConcatKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<NormRopeConcatKernelTiling *>(tilingCpu.data_ptr());
    tiling->B = static_cast<int32_t>(B);
    tiling->S = static_cast<int32_t>(S);
    tiling->N = static_cast<int32_t>(N);
    tiling->D = static_cast<int32_t>(D);
    tiling->blockM = DEFAULT_BLOCK_M;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = (mNum + usedCoreNum - 1) / usedCoreNum;
    tiling->normType = static_cast<int32_t>(normType);
    tiling->ropeType = static_cast<int32_t>(ropeType);
    tiling->eps = static_cast<float>(eps);
    tiling->invD = 1.0f / static_cast<float>(D);
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (query.scalar_type() == at::kFloat) {
        launch = norm_rope_concat_do_fp32;
    } else if (query.scalar_type() == at::kHalf) {
        launch = norm_rope_concat_do_fp16;
    } else if (query.scalar_type() == at::kBFloat16) {
        launch = norm_rope_concat_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(query.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(key.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(value.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(queryWeight.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(queryBias.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(keyWeight.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(keyBias.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(sin.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(cos.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(queryOut.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(keyOut.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(valueOut.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return pybind11::make_tuple(queryOut, keyOut, valueOut);
}

} // namespace norm_rope_concat_ext

PYBIND11_MODULE(_norm_rope_concat_ext, m)
{
    m.doc() = "norm_rope_concat extension";
    m.def("run_norm_rope_concat", &norm_rope_concat_ext::run_norm_rope_concat, "");
}
