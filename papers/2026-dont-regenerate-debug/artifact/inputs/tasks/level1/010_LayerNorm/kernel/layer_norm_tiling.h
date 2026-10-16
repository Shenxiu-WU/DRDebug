#ifndef LAYER_NORM_TILING_H
#define LAYER_NORM_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_M = 64;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr uint32_t DEFAULT_VEC_NUM = 2;
constexpr uint32_t DEFAULT_BLOCK_N_CHUNK = 2048;

struct LayerNormKernelTiling {
    int32_t M;
    int32_t N;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t subBlockM;
    float eps;
    float invN;
    int32_t hasBias;
    int32_t nChunks;  // for twopass: ceildiv(N, blockNChunk)
    int32_t blockNChunk;
};

#endif
