#ifndef TANH_GATED_RESIDUAL_ADD_BACKWARD_TILING_H
#define TANH_GATED_RESIDUAL_ADD_BACKWARD_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_SIZE = 2048;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct TanhGatedResidualAddBackwardTiling {
    int32_t numElements;
    int32_t blockSize;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t numBlocks;
};

#endif
