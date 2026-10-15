#ifndef ADD_TILING_H
#define ADD_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_TILE_SIZE = 2048;
constexpr uint32_t DEFAULT_TILE_SIZE_SMALL = 4096;  // for fp16/bf16
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct AddKernelTiling {
    int32_t N;              // total number of elements
    int32_t tileSize;       // elements per tile
    int32_t usedCoreNum;    // actual cores used
    int32_t tasksPerCore;   // tiles assigned per core
    int32_t totalTiles;     // total number of tiles
    float alpha;            // scalar multiplier for y
};

#endif
