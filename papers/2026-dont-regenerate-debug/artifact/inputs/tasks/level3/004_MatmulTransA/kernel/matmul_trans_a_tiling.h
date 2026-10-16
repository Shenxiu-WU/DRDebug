/**
 * @file matmul_trans_a_tiling.h
 *
 * Tiling parameters for the MatmulTransA cube kernel.
 *
 * The host pads `(m, n, k)` to multiples of `DEFAULT_BASE_*` before
 * launching, so the kernel can assume aligned shapes throughout.
 */
#ifndef MATMUL_TRANS_A_TILING_H
#define MATMUL_TRANS_A_TILING_H

#include <cstdint>

constexpr int32_t DEFAULT_BASE_M = 16;
constexpr int32_t DEFAULT_BASE_N = 16;
constexpr int32_t DEFAULT_BASE_K = 16;
constexpr int32_t DEFAULT_L1_PREFETCH = 1;

#pragma pack(push, 8)
struct MatmulTransATiling {
    int32_t M;          // = padded n (rows of output)
    int32_t N;          // = padded k (cols of output)
    int32_t K;          // = padded m (reduction dim)
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    int32_t l1Prefetch;
};
#pragma pack(pop)

#endif // MATMUL_TRANS_A_TILING_H
