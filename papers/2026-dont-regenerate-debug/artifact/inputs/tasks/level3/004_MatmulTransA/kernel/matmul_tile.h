/**
 * @file matmul_tile.h
 *
 * Tile-level data movement helpers for the MatmulTransA cube kernel.
 *
 * The helpers are deliberately templatised on the element type so a
 * single MatmulKernel template can serve all three accepted input
 * dtypes (`half`, `bfloat16_t`, `float`). The L1->L0B path uses the
 * 3D LoadData primitive which works uniformly across element widths
 * (2B for half/bf16, 4B for float).
 */
#ifndef MATMUL_TILE_H
#define MATMUL_TILE_H

#include "kernel_operator.h"

// GM(ND) -> L1(Nz)  : load an (m, n) ND tile into an (m, n) Nz L1 buffer.
// Pre-condition: m % 16 == 0 && n % C0 == 0.
template<typename T>
__aicore__ inline void LoadNdGmToNzL1(const AscendC::LocalTensor<T> &dst,
                                      const AscendC::GlobalTensor<T> &src,
                                      uint32_t m, uint32_t n, uint32_t ld)
{
    AscendC::Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = m;
    params.dValue = n;
    params.srcNdMatrixStride = 0;
    params.srcDValue = ld;
    params.dstNzC0Stride = m;
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    AscendC::DataCopy(dst, src, params);
}

// L1(Nz) -> L0A(Zz)
// Generic across element widths because LoadData3DParamsV2 expresses
// extents in *elements*, not bytes.
template<typename T>
__aicore__ inline void LoadNzL1ToZzL0A(const AscendC::LocalTensor<T> &dst,
                                       const AscendC::LocalTensor<T> &src,
                                       uint32_t m, uint32_t k, uint32_t colC0Stride)
{
    AscendC::LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = k;
    params.kExtension = k;
    params.mExtension = m;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    AscendC::LoadData(dst, src, params);
}

// L1(Nz) -> L0B(Zn) — generic version that works for half / bfloat16 / float.
template<typename T>
__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<T> &dst,
                                       const AscendC::LocalTensor<T> &src,
                                       uint32_t k, uint32_t n, uint32_t colC0Stride)
{
    AscendC::LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = n;
    params.kExtension = n;
    params.mExtension = k;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    AscendC::LoadData(dst, src, params);
}

// L0C(Nz) -> GM(ND): write an (m, n) Nz fragment back to an ND row of a
// larger GM matrix whose row stride is `dstStride` (in elements).
// `quantPre` carries any optional CO1 -> dst dtype cast (NoQuant for
// float, F322F16 for half, F322BF16 for bfloat16_t).
//
// Note: `QuantMode_t` is a *global* enum declared by the Bisheng cce
// intrinsics header, not a member of namespace AscendC.
template<typename T>
__aicore__ inline void FixpipeNzL0cToNdGm(const AscendC::GlobalTensor<T> &dst,
                                          const AscendC::LocalTensor<float> &src,
                                          uint32_t m, uint32_t n, uint32_t dstStride,
                                          QuantMode_t quantPre)
{
    AscendC::FixpipeParamsV220 params;
    params.nSize = n;
    params.mSize = m;
    params.srcStride = m;
    params.dstStride = dstStride;
    params.ndNum = 1;
    params.srcNdStride = 0;
    params.dstNdStride = 0;
    params.quantPre = quantPre;
    AscendC::Fixpipe(dst, src, params);
}

#endif // MATMUL_TILE_H
