#include "fused_rope_qk_norm_kv_cache_kernel.h"

extern "C" __global__ __aicore__ void fused_rope_qk_norm_kv_cache_custom_fp32(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR positionIds, GM_ADDR cachePosition,
    GM_ADDR qNormWeight, GM_ADDR kNormWeight, GM_ADDR invFreq,
    GM_ADDR queryRotated, GM_ADDR keyRotated,
    GM_ADDR keyCacheOut, GM_ADDR valueCacheOut,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    FusedRopeQkNormKvCacheKernel<float> kernel;
    kernel.Init(query, key, value, positionIds, cachePosition,
                qNormWeight, kNormWeight, invFreq,
                queryRotated, keyRotated, keyCacheOut, valueCacheOut,
                tiling, &pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void fused_rope_qk_norm_kv_cache_custom_fp16(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR positionIds, GM_ADDR cachePosition,
    GM_ADDR qNormWeight, GM_ADDR kNormWeight, GM_ADDR invFreq,
    GM_ADDR queryRotated, GM_ADDR keyRotated,
    GM_ADDR keyCacheOut, GM_ADDR valueCacheOut,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    FusedRopeQkNormKvCacheKernel<half> kernel;
    kernel.Init(query, key, value, positionIds, cachePosition,
                qNormWeight, kNormWeight, invFreq,
                queryRotated, keyRotated, keyCacheOut, valueCacheOut,
                tiling, &pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void fused_rope_qk_norm_kv_cache_custom_bf16(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR positionIds, GM_ADDR cachePosition,
    GM_ADDR qNormWeight, GM_ADDR kNormWeight, GM_ADDR invFreq,
    GM_ADDR queryRotated, GM_ADDR keyRotated,
    GM_ADDR keyCacheOut, GM_ADDR valueCacheOut,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    FusedRopeQkNormKvCacheKernel<bfloat16_t> kernel;
    kernel.Init(query, key, value, positionIds, cachePosition,
                qNormWeight, kNormWeight, invFreq,
                queryRotated, keyRotated, keyCacheOut, valueCacheOut,
                tiling, &pipe);
    kernel.Process();
}

extern "C" void fused_rope_qk_norm_kv_cache_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *positionIds, uint8_t *cachePosition,
    uint8_t *qNormWeight, uint8_t *kNormWeight, uint8_t *invFreq,
    uint8_t *queryRotated, uint8_t *keyRotated,
    uint8_t *keyCacheOut, uint8_t *valueCacheOut,
    uint8_t *tiling)
{
    fused_rope_qk_norm_kv_cache_custom_fp32<<<blockDim, nullptr, stream>>>(
        query, key, value, positionIds, cachePosition,
        qNormWeight, kNormWeight, invFreq,
        queryRotated, keyRotated, keyCacheOut, valueCacheOut, tiling);
}

extern "C" void fused_rope_qk_norm_kv_cache_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *positionIds, uint8_t *cachePosition,
    uint8_t *qNormWeight, uint8_t *kNormWeight, uint8_t *invFreq,
    uint8_t *queryRotated, uint8_t *keyRotated,
    uint8_t *keyCacheOut, uint8_t *valueCacheOut,
    uint8_t *tiling)
{
    fused_rope_qk_norm_kv_cache_custom_fp16<<<blockDim, nullptr, stream>>>(
        query, key, value, positionIds, cachePosition,
        qNormWeight, kNormWeight, invFreq,
        queryRotated, keyRotated, keyCacheOut, valueCacheOut, tiling);
}

extern "C" void fused_rope_qk_norm_kv_cache_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *positionIds, uint8_t *cachePosition,
    uint8_t *qNormWeight, uint8_t *kNormWeight, uint8_t *invFreq,
    uint8_t *queryRotated, uint8_t *keyRotated,
    uint8_t *keyCacheOut, uint8_t *valueCacheOut,
    uint8_t *tiling)
{
    fused_rope_qk_norm_kv_cache_custom_bf16<<<blockDim, nullptr, stream>>>(
        query, key, value, positionIds, cachePosition,
        qNormWeight, kNormWeight, invFreq,
        queryRotated, keyRotated, keyCacheOut, valueCacheOut, tiling);
}
