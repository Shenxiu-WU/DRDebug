#pragma once
#include <cstdint>

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr int32_t DEFAULT_VEC_NUM = 2;

struct InplaceIndexAddTiling {
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
};
