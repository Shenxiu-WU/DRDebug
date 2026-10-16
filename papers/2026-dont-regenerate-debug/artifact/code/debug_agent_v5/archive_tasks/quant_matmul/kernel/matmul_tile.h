#ifndef MATMUL_TILE_H
#define MATMUL_TILE_H

#include "kernel_operator.h"

// GM(ND) -> L1(Nz)
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

// L1(Nz) -> L0B(Zn) for int8
__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<int8_t> &dst,
                                       const AscendC::LocalTensor<int8_t> &src,
                                       uint32_t k, uint32_t n, uint32_t colC0Stride)
{
    static constexpr uint32_t C0_SIZE = 32;
    static constexpr uint32_t FRAC_K  = C0_SIZE;
    uint32_t dstOffset = FRAC_K * n;
    uint32_t srcOffset = FRAC_K * C0_SIZE;
    AscendC::LoadData2dTransposeParams loadDataParams;
    loadDataParams.repeatTimes = n / C0_SIZE;
    loadDataParams.srcStride = colC0Stride / FRAC_K;
    loadDataParams.dstGap = 1;
    loadDataParams.dstFracGap = 0;
    for (int i = 0; i < (int)(k / FRAC_K); ++i) {
        AscendC::LoadDataWithTranspose(dst[i * dstOffset], src[i * srcOffset], loadDataParams);
    }
}

// L0C(Nz) -> GM(ND)
template<typename T>
__aicore__ inline void FixpipeNzL0cToNdGm(const AscendC::GlobalTensor<T> &dst,
                                      const AscendC::LocalTensor<T> &src,
                                      uint32_t m, uint32_t n)
{
    AscendC::FixpipeParamsV220 params;
    params.nSize = n;
    params.mSize = m;
    params.srcStride = m;
    params.dstStride = n;
    params.ndNum = 1;
    params.srcNdStride = 0;
    params.dstNdStride = 0;
    AscendC::Fixpipe(dst, src, params);
}

#endif // MATMUL_TILE_H
