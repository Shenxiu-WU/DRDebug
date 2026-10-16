#pragma once

#include "gather_elements_v2_common_kernel.h"

template <typename DataType>
class GatherElementsV2ScalarKernel : public GatherElementsV2CommonKernel<DataType> {
public:
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const GatherElementsV2KernelTiling &tiling = this->Tiling();
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
                        const int32_t xRowIdx = this->ResolveXRowFromMap(rowIdx);
                        this->ProcessGatherRow(rowIdx, xRowIdx);
                    }
                }
            }
        }
    }
};
