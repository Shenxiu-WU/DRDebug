import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def gather_elements_v2(
    M,
    XRows,
    XG,
    IG,
    XStride,
    YStride,
    mode="last_dim",
    dtype="float32",
):
    block_m = 32
    if M < 32:
        block_m = 16
    if M < 16:
        block_m = 8
    if M < 8:
        block_m = 4
    if M < 4:
        block_m = 2
    if M < 2:
        block_m = 1

    m_num = T.ceildiv(M, block_m)
    num_physical_cores = 20
    used_core_num = min(num_physical_cores, m_num)
    tasks_per_core = T.ceildiv(m_num, used_core_num)
    vec_num = 2
    sub_block_m = max(block_m // vec_num, 1)

    @T.prim_func
    def last_dim_kernel(
        X: T.Tensor((M, XStride), dtype),
        Index: T.Tensor((M, YStride), "int32"),
        Y: T.Tensor((M, YStride), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((1, XStride), dtype)
            index_ub = T.alloc_ub((1, YStride), "int32")
            out_ub = T.alloc_ub((1, YStride), dtype)

            with T.Scope("V"):
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        row_base = bx * block_m + vid * sub_block_m
                        for row in T.serial(sub_block_m):
                            row_idx = row_base + row
                            if row_idx < M:
                                T.copy(X[row_idx, :], x_ub)
                                T.copy(Index[row_idx, :], index_ub)
                                for col in T.serial(IG):
                                    gather_idx = T.cast(index_ub[0, col], "int32")
                                    out_ub[0, col] = x_ub[0, gather_idx]
                                T.copy(out_ub, Y[row_idx, :])

    @T.prim_func
    def indexed_kernel(
        X: T.Tensor((XRows, XStride), dtype),
        Index: T.Tensor((M, YStride), "int32"),
        RowMap: T.Tensor((M,), "int32"),
        Y: T.Tensor((M, YStride), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((1, XStride), dtype)
            index_ub = T.alloc_ub((1, YStride), "int32")
            out_ub = T.alloc_ub((1, YStride), dtype)

            with T.Scope("V"):
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        row_base = bx * block_m + vid * sub_block_m
                        for row in T.serial(sub_block_m):
                            idx_row = row_base + row
                            if idx_row < M:
                                x_row = T.cast(RowMap[idx_row], "int32")
                                T.copy(X[x_row, :], x_ub)
                                T.copy(Index[idx_row, :], index_ub)
                                for col in T.serial(IG):
                                    gather_idx = T.cast(index_ub[0, col], "int32")
                                    out_ub[0, col] = x_ub[0, gather_idx]
                                T.copy(out_ub, Y[idx_row, :])

    @T.prim_func
    def transpose_kernel(
        X: T.Tensor((XRows, XStride), dtype),
        Index: T.Tensor((M, YStride), "int32"),
        RowMap: T.Tensor((M,), "int32"),
        Y: T.Tensor((M, YStride), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((1, XStride), dtype)
            index_ub = T.alloc_ub((1, YStride), "int32")
            out_ub = T.alloc_ub((1, YStride), dtype)

            with T.Scope("V"):
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        row_base = bx * block_m + vid * sub_block_m
                        for row in T.serial(sub_block_m):
                            idx_row = row_base + row
                            if idx_row < M:
                                x_row = T.cast(RowMap[idx_row], "int32")
                                T.copy(X[x_row, :], x_ub)
                                T.copy(Index[idx_row, :], index_ub)
                                for col in T.serial(IG):
                                    gather_idx = T.cast(index_ub[0, col], "int32")
                                    out_ub[0, col] = x_ub[0, gather_idx]
                                T.copy(out_ub, Y[idx_row, :])

    if mode == "last_dim":
        return last_dim_kernel
    if mode == "transpose":
        return transpose_kernel
    return indexed_kernel
