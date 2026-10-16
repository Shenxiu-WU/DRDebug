import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], workspace_idx=3, pass_configs=pass_configs)
def matmul_leakyrelu(
    M,
    N,
    K,
    dtype="float16",
    accum_dtype="float",
    negative_slope=0.01,
):
    baseM, baseN, baseK, l1Prefetch = 128, 128, 128, 4
    num_physical_cores = 20
    assert M % baseM == 0
    assert N % baseN == 0
    assert K % (l1Prefetch * baseK) == 0
    m_num = M // baseM
    n_num = N // baseN
    total_blocks = m_num * n_num
    assert total_blocks > 0
    usedCoreNum = min(num_physical_cores, total_blocks)
    tasksPerCore = (total_blocks + usedCoreNum - 1) // usedCoreNum
    vec_num = 2
    use_float_accum = accum_dtype == "float"
    negative_slope_const = T.float32(negative_slope)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), "float32"),
        # Persistent-kernel workspace: one tile buffer per physical core.
        workspace: T.Tensor((usedCoreNum, baseM, baseN), accum_dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid

            A_L1 = T.alloc_L1((baseM, l1Prefetch * baseK), dtype)
            B_L1 = T.alloc_L1((l1Prefetch * baseK, baseN), dtype)
            T.annotate_layout({
                A_L1: make_zn_layout(A_L1),
                B_L1: make_zn_layout(B_L1),
            })
            A_L0 = T.alloc_L0A((baseM, baseK), dtype)
            B_L0 = T.alloc_L0B((baseK, baseN), dtype)
            C_L0 = T.alloc_L0C((baseM, baseN), accum_dtype)

            c_accum_ub = T.alloc_ub((baseM // vec_num, baseN), accum_dtype)
            c_out_ub = T.alloc_ub((baseM // vec_num, baseN), "float32")

            # Fixed physical cores; each core walks a contiguous task range.
            for localIdx in T.serial(tasksPerCore):
                task_id = coreIdx * tasksPerCore + localIdx
                mIdx = task_id // n_num
                nIdx = task_id % n_num

                with T.Scope("C"):
                    if task_id < total_blocks:
                        loop_k = K // (l1Prefetch * baseK)
                        for k in T.serial(loop_k):
                            outer = k * l1Prefetch
                            T.copy(A[mIdx * baseM, outer * baseK], A_L1)
                            T.copy(B[outer * baseK, nIdx * baseN], B_L1)
                            for kk in T.serial(l1Prefetch):
                                T.copy(A_L1[0, kk * baseK], A_L0)
                                T.copy(B_L1[kk * baseK, 0], B_L0)
                                T.mma(
                                    A_L0,
                                    B_L0,
                                    C_L0,
                                    init=T.And(k == 0, kk == 0),
                                )

                        T.copy(C_L0, workspace[coreIdx, 0, 0])
                        T.set_cross_flag("FIX", 0)

                with T.Scope("V"):
                    if task_id < total_blocks:
                        T.wait_cross_flag(0)
                        T.copy(
                            workspace[coreIdx, vid * baseM // vec_num, 0],
                            c_accum_ub,
                        )
                        if use_float_accum:
                            T.copy(c_accum_ub, c_out_ub)
                        else:
                            T.tile.cast(
                                c_out_ub,
                                c_accum_ub,
                                mode="CAST_NONE",
                                count=(baseM // vec_num) * baseN,
                            )
                        T.tile.leaky_relu(c_out_ub, c_out_ub, negative_slope_const)
                        T.copy(c_out_ub, C[mIdx * baseM + vid * baseM // vec_num, nIdx * baseN])

    return main
