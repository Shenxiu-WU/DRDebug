#ifndef CUMSUM_TILING_H
#define CUMSUM_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct CumsumTiling {
    int32_t S;              // scan dimension size
    int32_t P;              // product of remaining dimensions
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t scanMode;
};

#endif
