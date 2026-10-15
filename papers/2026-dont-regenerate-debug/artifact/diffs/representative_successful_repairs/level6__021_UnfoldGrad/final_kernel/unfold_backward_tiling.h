#pragma once

#include <cstdint>

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr int32_t DEFAULT_VEC_NUM = 2;

struct UnfoldBackwardTiling {
    int32_t B;
    int32_t D;
    int32_t nWindows;
    int32_t size;
    int32_t step;
    int32_t blockB;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t subBlockB;
};
