/**
 * @file matmul_leakyrelu_tiling.h
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef MATMUL_LEAKYRELU_TILING_H
#define MATMUL_LEAKYRELU_TILING_H

#include <cstdint>

constexpr int32_t DEFAULT_BASE_M = 128;
constexpr int32_t DEFAULT_BASE_N = 128;
constexpr int32_t DEFAULT_BASE_K = 128;
constexpr int32_t DEFAULT_L1_PREFETCH = 4;
constexpr int32_t WORKSPACE_DEPTH = 4;

#pragma pack(push, 8)
struct MatmulLeakyReluTiling {
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    int32_t l1Prefetch;
};
#pragma pack(pop)

#endif // MATMUL_LEAKYRELU_TILING_H
