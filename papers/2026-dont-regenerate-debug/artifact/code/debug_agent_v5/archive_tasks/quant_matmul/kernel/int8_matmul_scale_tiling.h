#ifndef INT8_MATMUL_SCALE_TILING_H
#define INT8_MATMUL_SCALE_TILING_H

#include <cstdint>

constexpr int32_t DEFAULT_BASE_M = 128;
constexpr int32_t DEFAULT_BASE_N = 128;
constexpr int32_t DEFAULT_BASE_K = 128;
constexpr int32_t WORKSPACE_DEPTH = 4;

#pragma pack(push, 8)
struct Int8MatmulScaleTiling {
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
};
#pragma pack(pop)

#endif // INT8_MATMUL_SCALE_TILING_H
