import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True
}


@tilelang.jit(out_idx=[3], workspace_idx=4, pass_configs=pass_configs)
def int8_matmul_scale(
    M,
    N,
    K,
    dtype="int8",
    accum_dtype="int32",
):
    block_M, block_N, block_K, K_L1 = 128, 256, 64, 256
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        Scale: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), "float16"),
        workspace: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            T.annotate_layout({
                A_L1: make_zn_layout(A_L1),
                B_L1: make_zn_layout(B_L1),
            })
            A_L0 = T.alloc_L0A((block_M, block_K), dtype)
            B_L0 = T.alloc_L0B((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            acc_i32_ub = T.alloc_ub((sub_block_M, block_N), accum_dtype)
            acc_fp32_ub = T.alloc_ub((sub_block_M, block_N), "float32")
            out_ub = T.alloc_ub((sub_block_M, block_N), "float16")
            scale_row_ub = T.alloc_ub((1, block_N), "float32")
            row_fp32_ub = T.alloc_ub((1, block_N), "float32")

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                loop_kk = T.ceildiv(K_L1, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    for kk in T.serial(loop_kk):
                        T.copy(A_L1[0, kk * block_K], A_L0)
                        T.copy(B_L1[kk * block_K, 0], B_L0)
                        T.mma(
                            A_L0,
                            B_L0,
                            C_L0,
                            init=T.And(k == 0, kk == 0),
                        )

                T.copy(C_L0, workspace[bx * block_M, by * block_N])
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)
                T.copy(
                    workspace[bx * block_M + vid * sub_block_M, by * block_N],
                    acc_i32_ub,
                )
                T.copy(Scale[by * block_N], scale_row_ub)
                T.tile.cast(
                    acc_fp32_ub,
                    acc_i32_ub,
                    mode="CAST_NONE",
                    count=sub_block_M * block_N,
                )
                for i in T.serial(sub_block_M):
                    T.copy(acc_fp32_ub[i, 0], row_fp32_ub)
                    T.tile.mul(row_fp32_ub, row_fp32_ub, scale_row_ub)
                    T.copy(row_fp32_ub, acc_fp32_ub[i, 0])
                T.tile.cast(
                    out_ub,
                    acc_fp32_ub,
                    mode="CAST_NONE",
                    count=sub_block_M * block_N,
                )
                T.copy(
                    out_ub,
                    C[bx * block_M + vid * sub_block_M, by * block_N],
                )

    return main
