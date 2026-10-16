#ifndef KERNEL_COMMON_H
#define KERNEL_COMMON_H

__aicore__ inline uint32_t CeilDiv32(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template<typename T>
__aicore__ inline void CopyTiling(T *tiling, GM_ADDR tilingGM)
{
    int32_t *ptr = reinterpret_cast<int32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ int32_t *>(tilingGM);
    for (size_t i = 0; i < sizeof(T) / sizeof(int32_t); ++i, ++ptr) {
        *ptr = *(tiling32 + i);
    }
}

#endif // KERNEL_COMMON_H
