#pragma once

#include <cstdint>

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr int32_t DEFAULT_VEC_NUM = 2;

constexpr int32_t GATHER_MODE_LAST_DIM = 0;
constexpr int32_t GATHER_MODE_TRANSPOSE = 1;
constexpr int32_t GATHER_MODE_SCALAR = 2;

struct GatherElementsV2KernelTiling {
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
    int32_t useRowMap;
    int32_t mode;
};
