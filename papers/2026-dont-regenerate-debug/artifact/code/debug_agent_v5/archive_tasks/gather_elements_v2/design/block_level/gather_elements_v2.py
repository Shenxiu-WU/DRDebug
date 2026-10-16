import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def gather_elements_v2(M, XG, IG, dtype="float32"):
    block_M = 32
    if M < 32:
        block_M = 16
    if M < 16:
        block_M = 8
    if M < 8:
        block_M = 4
    if M < 4:
        block_M = 2
    if M < 2:
        block_M = 1

    m_num = T.ceildiv(M, block_M)
    num_physical_cores = 20
    used_core_num = min(num_physical_cores, m_num)
    tasks_per_core = T.ceildiv(m_num, used_core_num)
    vec_num = 2
    sub_block_M = max(block_M // vec_num, 1)

    @T.prim_func
    def single_row(
        X: T.Tensor((M * XG,), dtype),
        Index: T.Tensor((M * IG,), "int32"),
        Y: T.Tensor((M * IG,), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        row_base = bx * block_M + vid * sub_block_M
                        for row in T.serial(sub_block_M):
                            row_idx = row_base + row
                            if row_idx < M:
                                x_base = row_idx * XG
                                y_base = row_idx * IG
                                # TODO(tile-level):
                                # - use linearized 1D addresses for X/Index/Y
                                # - normalize negative indices in-kernel
                                # - read X[x_base + gather_idx] and write Y[y_base + col]
                                _ = X
                                _ = Index
                                _ = Y
                                _ = x_base
                                _ = y_base

    return single_row
