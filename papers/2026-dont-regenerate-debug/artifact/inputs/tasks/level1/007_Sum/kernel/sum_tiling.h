#ifndef SUM_TILING_H
#define SUM_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_M = 32;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr uint32_t DEFAULT_TILE_N = 2048;

struct SumKernelTiling {
    int32_t M;
    int32_t N;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t tileN;          // tile size for large_n path
    int32_t nTiles;         // number of tiles along N
    int32_t lastTileValid;  // elements in the last (possibly partial) tile
};

#endif
