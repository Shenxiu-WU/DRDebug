#ifndef TIME_DECAY_EXP_STABILIZATION_TILING_H
#define TIME_DECAY_EXP_STABILIZATION_TILING_H

#include <cstdint>

constexpr uint32_t TILE_SIZE = 2048;
constexpr uint32_t HIDDEN_SIZE = 2048;
constexpr uint32_t NUM_TILES = HIDDEN_SIZE / TILE_SIZE;

struct TimeDecayExpStabilizationTiling {
    int32_t batch;
    int32_t seqLen;
    int32_t hiddenSize;
    int32_t tileSize;
    int32_t numTiles;
    int32_t blockNum;
};

#endif
