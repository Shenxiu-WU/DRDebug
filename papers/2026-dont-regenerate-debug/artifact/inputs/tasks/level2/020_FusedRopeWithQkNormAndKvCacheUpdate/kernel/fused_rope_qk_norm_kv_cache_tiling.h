#ifndef FUSED_ROPE_QK_NORM_KV_CACHE_TILING_H
#define FUSED_ROPE_QK_NORM_KV_CACHE_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_M = 64;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct FusedRopeQkNormKvCacheTiling {
    int32_t B;
    int32_t Hq;
    int32_t Hkv;
    int32_t S;
    int32_t D;
    int32_t maxS;
    int32_t Mq;
    int32_t Mk;
    int32_t totalRows;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    float eps;
    float invD;
};

#endif
