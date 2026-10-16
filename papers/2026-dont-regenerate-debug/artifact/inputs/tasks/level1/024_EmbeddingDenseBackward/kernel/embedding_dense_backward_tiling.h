#ifndef EMBEDDING_DENSE_BACKWARD_TILING_H
#define EMBEDDING_DENSE_BACKWARD_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_WEIGHTS = 64;
constexpr uint32_t DEFAULT_BLOCK_D = 256;
constexpr uint32_t DEFAULT_BLOCK_I = 64;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct EmbeddingDenseBackwardTiling {
    int32_t totalIndices;
    int32_t D;
    int32_t numWeights;
    int32_t paddingIdx;
    int32_t scaleGradByFreq;  // 0 or 1
    int32_t blockWeights;
    int32_t blockD;
    int32_t blockI;
    int32_t totalBlocks;
    int32_t dNum;
    int32_t tasksPerCore;
};

#endif
