#ifndef ADAMW_TILING_H
#define ADAMW_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_N = 8192;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr uint32_t DEFAULT_VEC_NUM = 2;
constexpr uint32_t DEFAULT_SUB_BLOCK_N = DEFAULT_BLOCK_N / DEFAULT_VEC_NUM;

struct AdamWKernelTiling {
    int32_t nTotal;             // total number of elements (after flattening)
    int32_t blockN;             // block size, default DEFAULT_BLOCK_N
    int32_t subBlockN;           // per-AIV slice = blockN / vec_num
    int32_t usedCoreNum;        // physical cores actually used (= AIC blocks)
    int32_t tasksPerCore;       // ceildiv(nNum, usedCoreNum)
    int32_t nNum;               // ceildiv(nTotal, blockN)
    int32_t useNesterov;        // 0 or 1
    int32_t pad0;               // alignment to 8 bytes for the float section
    float beta1;
    float beta2;
    float oneMinusBeta1;        // 1 - beta1
    float oneMinusBeta2;        // 1 - beta2
    float invBias1;             // 1 / (1 - beta1_power)
    float invBias2;             // 1 / (1 - beta2_power)
    float lr;
    float negLr;                // -lr
    float epsilon;
    float coeffGradForNesterov; // (1 - beta1) / (1 - beta1_power)
};

#endif  // ADAMW_TILING_H
