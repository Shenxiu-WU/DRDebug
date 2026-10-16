#ifndef NORM_ROPE_CONCAT_TILING_H
#define NORM_ROPE_CONCAT_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_M = 32;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct NormRopeConcatKernelTiling {
    int32_t B;
    int32_t S;
    int32_t N;
    int32_t D;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t normType;
    int32_t ropeType;
    float eps;
    float invD;
};

#endif
