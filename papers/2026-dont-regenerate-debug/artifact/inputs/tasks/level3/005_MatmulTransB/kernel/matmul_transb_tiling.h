#ifndef MATMUL_TRANSB_TILING_H
#define MATMUL_TRANSB_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr uint32_t DEFAULT_BLOCK_M = 1;

struct MatmulTransBKernelTiling {
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
};

#endif
