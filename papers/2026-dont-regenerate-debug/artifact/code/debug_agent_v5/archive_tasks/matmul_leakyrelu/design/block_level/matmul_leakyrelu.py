"""Block-level TileLang design for matmul_leakyrelu.

This file keeps only block scheduling, pipeline skeleton, and cross-scope sync.
Fine-grained compute details are intentionally marked as TODO comments and should
be implemented in tile-level design.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True
}


@tilelang.jit(out_idx=[2], workspace_idx=3, pass_configs=pass_configs)
def matmul_leakyrelu(M,
                     N,
                     K,
                     dtype="float16",
                     accum_dtype="float",
                     negative_slope=0.01):
    block_M, block_N, block_K, K_L1 = 128, 256, 64, 256
    num_physical_cores = 20
    assert M % block_M == 0
    assert N % block_N == 0
    m_num = M // block_M
    n_num = N // block_N
    total_blocks = m_num * n_num
    assert total_blocks > 0
    usedCoreNum = min(num_physical_cores, total_blocks)
    tasksPerCore = (total_blocks + usedCoreNum - 1) // usedCoreNum

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
        workspace: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid

            # Block-level persistent-kernel pipeline:
            #   1) Fixed physical cores iterate over a contiguous task range.
            #   2) For each task, Cube scope produces one tile and Vector scope consumes it.
            for localIdx in T.serial(tasksPerCore):
                task_id = coreIdx * tasksPerCore + localIdx
                bx = task_id // n_num
                by = task_id % n_num

                with T.Scope("C"):
                    if task_id < total_blocks:
                        # TODO(tile-level):
                        # - implement block matmul (A @ B) for tile (bx, by)
                        # - hand off matmul tile to vector stage (via workspace/cross-scope sync)
                        # - ensure Cube-side pipeline/synchronization is correct
                        T.set_cross_flag("FIX", 0)

                with T.Scope("V"):
                    if task_id < total_blocks:
                        T.wait_cross_flag(0)

                        # TODO(tile-level):
                        # - wait for Cube-side matmul tile readiness
                        # - implement vector epilogue: leaky_relu(A @ B, negative_slope)
                        # - write final tile result to C
                        _ = negative_slope
                        _ = workspace
                        _ = A
                        _ = B
                        _ = C
                        _ = bx
                        _ = by
                        _ = vid
                        pass

    return main
