#pragma once

#include <cstdint>

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr int32_t DEFAULT_VEC_NUM = 2;

struct MoeInitRoutingKernelTiling {
    int32_t M;             // total output rows = N * K
    int32_t N;             // input rows
    int32_t H;             // feature dim
    int32_t XStride;       // padded inner stride for X (in elements)
    int32_t YStride;       // padded inner stride for Y (in elements)
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t subBlockM;
};
