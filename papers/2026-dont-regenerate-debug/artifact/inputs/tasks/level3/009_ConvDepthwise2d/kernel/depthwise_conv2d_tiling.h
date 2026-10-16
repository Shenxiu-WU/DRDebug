#ifndef DEPTHWISE_CONV2D_TILING_H
#define DEPTHWISE_CONV2D_TILING_H

#include <cstdint>

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr int32_t DEFAULT_VEC_NUM = 2;

struct DepthwiseConv2DKernelTiling {
    int32_t N;
    int32_t C;
    int32_t H;
    int32_t W;

    int32_t OH;
    int32_t OW;

    int32_t kH;
    int32_t kW;

    int32_t sH;
    int32_t sW;

    int32_t pH;
    int32_t pW;

    int32_t hasBias;

    int32_t inSpatial;     // H * W
    int32_t outSpatial;    // OH * OW
    int32_t weightCount;   // kH * kW * C
    int32_t mOut;          // N * OH * OW

    int32_t blockM;
    int32_t subBlockM;
    int32_t mNum;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t launchBlocks;

    int32_t reserved0;
    int32_t reserved1;
    int32_t reserved2;
};

#endif
