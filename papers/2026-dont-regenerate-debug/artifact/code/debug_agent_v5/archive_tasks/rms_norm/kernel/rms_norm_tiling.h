#ifndef RMS_NORM_TILING_H
#define RMS_NORM_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_M = 64;
constexpr uint32_t DEFAULT_ROW_FACTOR = 8;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct RmsNormKernelTiling {
    int32_t M;
    int32_t N;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t rowFactor;
    float eps;
    float invN;
};

#endif
