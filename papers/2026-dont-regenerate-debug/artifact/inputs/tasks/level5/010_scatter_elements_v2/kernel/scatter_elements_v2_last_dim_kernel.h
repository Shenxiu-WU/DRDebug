#pragma once

#include "scatter_elements_v2_common_kernel.h"

template <typename DataType>
class ScatterElementsV2LastDimKernel : public ScatterElementsV2CommonKernel<DataType> {
public:
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const ScatterElementsV2KernelTiling &tiling = this->Tiling();
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            for (int localIdx = 0; localIdx < tiling.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling.tasksPerCore + localIdx;
                if (bx >= this->BlockCount()) {
                    continue;
                }

                const int rowBase = bx * tiling.blockM + subBlockIdx * tiling.subBlockM;
                for (int row = 0; row < this->SubBlockRows(); ++row) {
                    const int rowIdx = rowBase + row;
                    if (rowIdx < tiling.M) {
                        this->ProcessScatterRow(rowIdx);
                    }
                }
            }
        }
    }
};
