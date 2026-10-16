#ifndef CONV1D_TILING_H
#define CONV1D_TILING_H

#include <cstdint>

constexpr int32_t CONV1D_DEFAULT_NUM_PHYSICAL_CORES = 20;

struct Conv1DKernelTiling {
    int32_t N;
    int32_t cIn;
    int32_t cOut;
    int32_t lIn;
    int32_t lOut;

    int32_t kSize;
    int32_t stride;
    int32_t padding;
    int32_t dilation;
    int32_t groups;

    int32_t cInPerGroup;
    int32_t cOutPerGroup;

    int32_t hasBias;        // 1 if bias is present, 0 otherwise
    int32_t totalBlocks;    // N * lOut
    int32_t usedCoreNum;
    int32_t tasksPerCore;

    // Reserved 4-byte slots so the struct stays a multiple of 32 bytes —
    // simplifies CopyTiling() (which copies in int32_t units).
    int32_t reserved0;
    int32_t reserved1;
    int32_t reserved2;
    int32_t reserved3;
};

#endif
