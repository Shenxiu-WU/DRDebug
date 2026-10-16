#ifndef IOU_TILING_H
#define IOU_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_TILE_N = 128;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct IouKernelTiling {
    int32_t N;
    int32_t M;
    int32_t tileN;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t mode;
    float eps;
};

#endif
