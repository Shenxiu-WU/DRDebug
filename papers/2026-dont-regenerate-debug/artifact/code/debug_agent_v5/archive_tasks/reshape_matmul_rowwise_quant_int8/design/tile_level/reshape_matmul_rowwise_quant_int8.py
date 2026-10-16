import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], workspace_idx=3, pass_configs=pass_configs)
def reshape_matmul_rowwise_quant_int8(
    M,
    N,
    H_K,
    dtype="bfloat16",
    accum_dtype="float",
):
    block_M, block_N, block_K, K_L1 = 128, 256, 64, 256
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2
    sub_block_M = block_M // vec_num
    n_tiles_per_h = H_K // block_N

    inv_127 = T.float32(1.0 / 127.0)

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        H: T.Tensor((H_K, H_K), dtype),
        Y: T.Tensor((M, N), "int8"),
        workspace: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            T.annotate_layout({
                A_L1: make_zn_layout(A_L1),
                B_L1: make_zn_layout(B_L1),
            })
            A_L0 = T.alloc_L0A((block_M, block_K), dtype)
            B_L0 = T.alloc_L0B((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            out_fp32_ub = T.alloc_ub((sub_block_M, block_N), "float32")
            out_abs_ub = T.alloc_ub((sub_block_M, block_N), "float32")
            row_absmax_ub = T.alloc_ub((sub_block_M,), "float32")
            row_absmax_tile_ub = T.alloc_ub((sub_block_M,), "float32")
            row_scale_ub = T.alloc_ub((sub_block_M,), "float32")
            out_fp16_ub = T.alloc_ub((sub_block_M, block_N), "float16")
            out_int8_ub = T.alloc_ub((sub_block_M, block_N), "int8")
            reduce_tmp_ub = T.alloc_ub((2 * sub_block_M * block_N,), "uint8")

            with T.Scope("C"):
                loop_h = T.ceildiv(H_K, K_L1)
                loop_kk = T.ceildiv(K_L1, block_K)
                for by in T.serial(n_num):
                    group_id = by // n_tiles_per_h
                    col_in_group = by % n_tiles_per_h
                    for h_block in T.serial(loop_h):
                        T.copy(
                            X[
                                bx * block_M,
                                group_id * H_K + h_block * K_L1,
                            ],
                            A_L1,
                        )
                        T.copy(
                            H[
                                h_block * K_L1,
                                col_in_group * block_N,
                            ],
                            B_L1,
                        )
                        for kk in T.serial(loop_kk):
                            T.copy(A_L1[0, kk * block_K], A_L0)
                            T.copy(B_L1[kk * block_K, 0], B_L0)
                            T.mma(
                                A_L0,
                                B_L0,
                                C_L0,
                                init=T.And(h_block == 0, kk == 0),
                            )
                    T.copy(C_L0, workspace[bx * block_M, by * block_N])
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)

                T.tile.fill(row_absmax_ub, 0.0)
                for by in T.serial(n_num):
                    T.copy(
                        workspace[bx * block_M + vid * sub_block_M, by * block_N],
                        out_fp32_ub,
                    )
                    T.tile.abs(out_abs_ub, out_fp32_ub)
                    T.reduce_max(out_abs_ub, row_absmax_tile_ub, reduce_tmp_ub, dim=-1)
                    T.tile.max(row_absmax_ub, row_absmax_ub, row_absmax_tile_ub)

                T.tile.mul(row_scale_ub, row_absmax_ub, inv_127)

                for by in T.serial(n_num):
                    T.copy(
                        workspace[bx * block_M + vid * sub_block_M, by * block_N],
                        out_fp32_ub,
                    )
                    for i in T.serial(sub_block_M):
                        T.tile.div(out_fp32_ub[i, :], out_fp32_ub[i, :], row_scale_ub[i])
                    T.tile.cast(
                        out_fp16_ub,
                        out_fp32_ub,
                        mode="CAST_NONE",
                        count=sub_block_M * block_N,
                    )
                    T.tile.cast(
                        out_int8_ub,
                        out_fp16_ub,
                        mode="CAST_NONE",
                        count=sub_block_M * block_N,
                    )
                    T.copy(
                        out_int8_ub,
                        Y[bx * block_M + vid * sub_block_M, by * block_N],
                    )

    return main
