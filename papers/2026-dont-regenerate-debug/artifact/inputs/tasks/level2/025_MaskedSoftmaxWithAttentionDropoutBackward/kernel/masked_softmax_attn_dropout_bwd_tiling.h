#ifndef MASKED_SOFTMAX_ATTN_DROPOUT_BWD_TILING_H
#define MASKED_SOFTMAX_ATTN_DROPOUT_BWD_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct MaskedSoftmaxAttnDropoutBwdTiling {
    int32_t M;
    int32_t T;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    float p_dropout;
    float scale;
};

#endif
