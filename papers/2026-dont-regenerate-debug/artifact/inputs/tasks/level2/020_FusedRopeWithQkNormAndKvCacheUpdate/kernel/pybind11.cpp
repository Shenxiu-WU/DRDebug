#include <algorithm>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "fused_rope_qk_norm_kv_cache_tiling.h"

extern "C" void fused_rope_qk_norm_kv_cache_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *positionIds, uint8_t *cachePosition,
    uint8_t *qNormWeight, uint8_t *kNormWeight, uint8_t *invFreq,
    uint8_t *queryRotated, uint8_t *keyRotated,
    uint8_t *keyCacheOut, uint8_t *valueCacheOut,
    uint8_t *tiling);

extern "C" void fused_rope_qk_norm_kv_cache_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *positionIds, uint8_t *cachePosition,
    uint8_t *qNormWeight, uint8_t *kNormWeight, uint8_t *invFreq,
    uint8_t *queryRotated, uint8_t *keyRotated,
    uint8_t *keyCacheOut, uint8_t *valueCacheOut,
    uint8_t *tiling);

extern "C" void fused_rope_qk_norm_kv_cache_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *positionIds, uint8_t *cachePosition,
    uint8_t *qNormWeight, uint8_t *kNormWeight, uint8_t *invFreq,
    uint8_t *queryRotated, uint8_t *keyRotated,
    uint8_t *keyCacheOut, uint8_t *valueCacheOut,
    uint8_t *tiling);

namespace fused_rope_qk_norm_kv_cache_ext {

using LaunchFn = void (*)(uint32_t, void *,
    uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

pybind11::tuple run_fused_rope_qk_norm_kv_cache(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &positionIds, const at::Tensor &cachePosition,
    const at::Tensor &qNormWeight, const at::Tensor &kNormWeight, const at::Tensor &invFreq,
    const at::Tensor &keyCache, const at::Tensor &valueCache,
    double eps)
{
    TORCH_CHECK(query.dim() == 4, "query must be 4D");
    TORCH_CHECK(key.dim() == 4, "key must be 4D");
    TORCH_CHECK(value.dim() == 4, "value must be 4D");
    TORCH_CHECK(positionIds.dim() == 2, "position_ids must be 2D");
    TORCH_CHECK(cachePosition.dim() == 1, "cache_position must be 1D");
    TORCH_CHECK(qNormWeight.dim() == 1, "q_norm_weight must be 1D");
    TORCH_CHECK(kNormWeight.dim() == 1, "k_norm_weight must be 1D");
    TORCH_CHECK(invFreq.dim() == 1, "inv_freq must be 1D");
    TORCH_CHECK(keyCache.dim() == 4, "key_cache must be 4D");
    TORCH_CHECK(valueCache.dim() == 4, "value_cache must be 4D");

    const int32_t B = static_cast<int32_t>(query.sizes()[0]);
    const int32_t Hq = static_cast<int32_t>(query.sizes()[1]);
    const int32_t S = static_cast<int32_t>(query.sizes()[2]);
    const int32_t D = static_cast<int32_t>(query.sizes()[3]);
    const int32_t Hkv = static_cast<int32_t>(key.sizes()[1]);
    const int32_t maxS = static_cast<int32_t>(keyCache.sizes()[2]);

    TORCH_CHECK(key.sizes()[0] == B && key.sizes()[2] == S && key.sizes()[3] == D,
                "key shape mismatch");
    TORCH_CHECK(value.sizes()[0] == B && value.sizes()[1] == Hkv && value.sizes()[2] == S && value.sizes()[3] == D,
                "value shape mismatch");
    TORCH_CHECK(positionIds.sizes()[0] == B && positionIds.sizes()[1] == S,
                "position_ids shape mismatch");
    TORCH_CHECK(cachePosition.sizes()[0] == S, "cache_position shape mismatch");
    TORCH_CHECK(qNormWeight.sizes()[0] == D, "q_norm_weight shape mismatch");
    TORCH_CHECK(kNormWeight.sizes()[0] == D, "k_norm_weight shape mismatch");
    TORCH_CHECK(invFreq.sizes()[0] == D / 2, "inv_freq shape mismatch");
    TORCH_CHECK(keyCache.sizes()[0] == B && keyCache.sizes()[1] == Hkv && keyCache.sizes()[3] == D,
                "key_cache shape mismatch");
    TORCH_CHECK(valueCache.sizes()[0] == B && valueCache.sizes()[1] == Hkv && valueCache.sizes()[3] == D,
                "value_cache shape mismatch");

    at::Tensor queryRotated = at::empty_like(query);
    at::Tensor keyRotated = at::empty_like(key);
    at::Tensor keyCacheOut = keyCache.clone();
    at::Tensor valueCacheOut = valueCache.clone();

    const int32_t Mq = B * Hq * S;
    const int32_t Mk = B * Hkv * S;
    const int32_t totalRows = Mq + 2 * Mk;
    const int32_t mNum = (totalRows + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    const int32_t tasksPerCore = (mNum + usedCoreNum - 1) / usedCoreNum;

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(FusedRopeQkNormKvCacheTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<FusedRopeQkNormKvCacheTiling *>(tilingCpu.data_ptr());
    tiling->B = B;
    tiling->Hq = Hq;
    tiling->Hkv = Hkv;
    tiling->S = S;
    tiling->D = D;
    tiling->maxS = maxS;
    tiling->Mq = Mq;
    tiling->Mk = Mk;
    tiling->totalRows = totalRows;
    tiling->blockM = DEFAULT_BLOCK_M;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->eps = static_cast<float>(eps);
    tiling->invD = 1.0f / static_cast<float>(D);
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (query.scalar_type() == at::kFloat) {
        launch = fused_rope_qk_norm_kv_cache_do_fp32;
    } else if (query.scalar_type() == at::kHalf) {
        launch = fused_rope_qk_norm_kv_cache_do_fp16;
    } else if (query.scalar_type() == at::kBFloat16) {
        launch = fused_rope_qk_norm_kv_cache_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(query.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(key.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(value.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(positionIds.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(cachePosition.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(qNormWeight.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(kNormWeight.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(invFreq.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(queryRotated.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(keyRotated.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(keyCacheOut.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(valueCacheOut.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return pybind11::make_tuple(queryRotated, keyRotated, keyCacheOut, valueCacheOut);
}

}  // namespace fused_rope_qk_norm_kv_cache_ext

PYBIND11_MODULE(_fused_rope_qk_norm_kv_cache_ext, m)
{
    m.doc() = "fused_rope_qk_norm_kv_cache extension";
    m.def("run_fused_rope_qk_norm_kv_cache", &fused_rope_qk_norm_kv_cache_ext::run_fused_rope_qk_norm_kv_cache, "");
}
