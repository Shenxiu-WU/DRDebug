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

class BlockScheduler {
public:
    __aicore__ inline void Init(int M, int N, int baseM, int baseN, int numBlocks, int blockIdx)
    {
        mBlocks_ = M / baseM;
        nBlocks_ = N / baseN;
        int totalBlocks = mBlocks_ * nBlocks_;
        int blocksPerCore = totalBlocks / numBlocks;
        int remainder = totalBlocks % numBlocks;
        startBlock_ = blockIdx * blocksPerCore + (blockIdx < remainder ? blockIdx : remainder);
        endBlock_ = startBlock_ + blocksPerCore + (blockIdx < remainder ? 1 : 0);
        current_ = startBlock_;
    }

    __aicore__ inline bool HasNext()
    {
        return current_ < endBlock_;
    }

    __aicore__ inline void Next(int &mIdx, int &nIdx)
    {
        mIdx = current_ / nBlocks_;
        nIdx = current_ % nBlocks_;
        current_++;
    }

private:
    int mBlocks_;
    int nBlocks_;
    int startBlock_;
    int endBlock_;
    int current_;
};

#endif // KERNEL_COMMON_H
