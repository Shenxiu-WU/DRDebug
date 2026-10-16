#ifndef BATCH_MATMUL_TILING_H
#define BATCH_MATMUL_TILING_H

#include <cstdint>

constexpr int32_t DEFAULT_BASE_M = 16;
constexpr int32_t DEFAULT_BASE_N = 16;
constexpr int32_t DEFAULT_BASE_K = 16;

#pragma pack(push, 8)
struct BatchMatmulTiling {
    int32_t B;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
};
#pragma pack(pop)

#endif // BATCH_MATMUL_TILING_H
