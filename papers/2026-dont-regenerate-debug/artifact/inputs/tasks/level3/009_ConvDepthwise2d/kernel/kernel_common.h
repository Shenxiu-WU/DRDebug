#ifndef DEPTHWISE_CONV2D_KERNEL_COMMON_H
#define DEPTHWISE_CONV2D_KERNEL_COMMON_H

#include <cstddef>
#include <cstdint>

#include "kernel_operator.h"

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
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

template <typename T>
__aicore__ inline void LoadGmToUb(
    AscendC::LocalTensor<T> &dst,
    AscendC::GlobalTensor<T> src,
    uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void StoreUbToGm(
    AscendC::GlobalTensor<T> dst,
    AscendC::LocalTensor<T> &src,
    uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
}

#endif
