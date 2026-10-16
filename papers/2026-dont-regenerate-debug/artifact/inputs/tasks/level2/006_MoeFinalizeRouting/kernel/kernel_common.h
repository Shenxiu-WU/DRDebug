#ifndef MOE_FINALIZE_ROUTING_KERNEL_COMMON_H
#define MOE_FINALIZE_ROUTING_KERNEL_COMMON_H

#include <cstddef>
#include <cstdint>

#include "kernel_operator.h"

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

__aicore__ inline uint32_t Align32(uint32_t bytes)
{
    return ((bytes + 31U) / 32U) * 32U;
}

template <typename T>
__aicore__ inline void CopyTiling(T *tiling, GM_ADDR tilingGM)
{
    int32_t *dst = reinterpret_cast<int32_t *>(tiling);
    auto *src = reinterpret_cast<__gm__ int32_t *>(tilingGM);
    for (size_t i = 0; i < sizeof(T) / sizeof(int32_t); ++i) {
        dst[i] = src[i];
    }
}

#endif
