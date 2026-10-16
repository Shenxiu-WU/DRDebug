#pragma once

#include <cstdint>

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr int32_t DEFAULT_VEC_NUM = 2;

constexpr int32_t SCATTER_MODE_LAST_DIM = 0;

struct ScatterElementsV2KernelTiling {
    int32_t M;
    int32_t XRows;
    int32_t XG;
    int32_t IG;
    int32_t XStride;
    int32_t YStride;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t subBlockM;
    int32_t reductionMode;
    int32_t mode;
};
