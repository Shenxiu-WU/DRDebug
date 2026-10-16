#pragma once

#include <cstdint>

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr int32_t DEFAULT_VEC_NUM = 2;

constexpr int32_t REDUCTION_NONE = 0;
constexpr int32_t REDUCTION_SUM = 1;
constexpr int32_t REDUCTION_MEAN = 2;

struct NLLLossTiling {
    int32_t numPos;
    int32_t C;
    int32_t reduction;
    int32_t hasWeight;
    int32_t ignoreIndex;
    int32_t blockN;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t subBlockN;
    int32_t slotsPerCore;
};
