#ifndef RMS_NORM_VECTOR_TILE_H
#define RMS_NORM_VECTOR_TILE_H

#include "kernel_operator.h"

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
