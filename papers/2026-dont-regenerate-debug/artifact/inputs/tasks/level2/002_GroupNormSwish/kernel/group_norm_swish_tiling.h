#ifndef GROUP_NORM_SWISH_TILING_H
#define GROUP_NORM_SWISH_TILING_H

#include <cstdint>

constexpr uint32_t kTileSpatial = 64;
constexpr uint32_t kNumPhysicalCores = 20;

struct GroupNormSwishTiling {
    int32_t N;
    int32_t C;
    int32_t spatialSize;
    int32_t numGroups;
    int32_t groupSize;
    int32_t M;
    int32_t K;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    float eps;
    float invK;
    float swishScale;
};

#endif
