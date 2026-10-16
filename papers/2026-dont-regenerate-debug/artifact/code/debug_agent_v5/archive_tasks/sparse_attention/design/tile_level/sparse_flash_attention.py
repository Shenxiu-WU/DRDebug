import tilelang
from tilelang import DataType, language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7, 8, 9], pass_configs=pass_configs)
def sparse_flash_attention_fwd(
    batch,
    q_seq_len,
    kv_seq_len,
    heads,
    kv_heads,
    dim,
    sparse_size,
):
    group_size = heads // kv_heads
    block_group = max(16, ((group_size + 15) // 16) * 16)
    block_sparse = max(16, ((sparse_size + 15) // 16) * 16)

    dtype = "float16"
    accum_dtype = "float"
    sm_scale = (1.0 / dim) ** 0.5

    total_bkvh = batch * kv_heads
    block_num = total_bkvh * q_seq_len
    subgroup = block_group // 2

    q_shape = [total_bkvh, q_seq_len, block_group, dim]
    kv_shape = [total_bkvh, kv_seq_len, dim]
    sparse_index_shape = [total_bkvh, q_seq_len, sparse_size]
    output_shape = [total_bkvh, q_seq_len, block_group, dim]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape, dtype),  # type: ignore
        V: T.Tensor(kv_shape, dtype),  # type: ignore
        SparseIndex: T.Tensor(sparse_index_shape, "int32"),  # type: ignore
        Output: T.Tensor(output_shape, dtype),  # type: ignore
        workspace_scores: T.Tensor([block_num, block_group, block_sparse], accum_dtype),
        workspace_probs: T.Tensor([block_num, block_group, block_sparse], dtype),
        workspace_out: T.Tensor([block_num, block_group, dim], accum_dtype),
        workspace_k: T.Tensor([block_num, block_sparse, dim], dtype),
        workspace_v: T.Tensor([block_num, block_sparse, dim], dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            global_bkvh = cid // q_seq_len
            q_idx = cid % q_seq_len
            subgroup_begin = vid * subgroup

            q_l1 = T.alloc_L1([block_group, dim], dtype)
            k_l1 = T.alloc_L1([block_sparse, dim], dtype)
            v_l1 = T.alloc_L1([block_sparse, dim], dtype)
            prob_l1 = T.alloc_L1([block_group, block_sparse], dtype)

            scores_l0c = T.alloc_L0C([block_group, block_sparse], accum_dtype)
            out_l0c = T.alloc_L0C([block_group, dim], accum_dtype)

            selected_k_ub = T.alloc_ub([block_sparse, dim], dtype)
            selected_v_ub = T.alloc_ub([block_sparse, dim], dtype)
            scores_ub = T.alloc_ub([subgroup, block_sparse], accum_dtype)
            probs_ub = T.alloc_ub([subgroup, block_sparse], dtype)
            out_ub = T.alloc_ub([subgroup, dim], accum_dtype)
            out_half = T.alloc_ub([subgroup, dim], dtype)
            row_max = T.alloc_ub([subgroup], accum_dtype)
            row_sum = T.alloc_ub([subgroup], accum_dtype)
            reduce_tmp = T.alloc_ub(
                [3 * DataType(accum_dtype).bits // 8 * subgroup * block_sparse],
                "uint8",
            )

            with T.Scope("C"):
                T.wait_cross_flag(0)
                T.copy(Q[global_bkvh, q_idx, :, :], q_l1)
                T.copy(workspace_k[cid, :, :], k_l1)
                T.gemm_v0(q_l1, k_l1, scores_l0c, transpose_B=True, init=True)
                T.copy(scores_l0c, workspace_scores[cid, :, :])
                T.set_cross_flag("FIX", 1)

                T.wait_cross_flag(2)
                T.copy(workspace_probs[cid, :, :], prob_l1)
                T.copy(workspace_v[cid, :, :], v_l1)
                T.gemm_v0(prob_l1, v_l1, out_l0c, init=True)
                T.copy(out_l0c, workspace_out[cid, :, :])
                T.set_cross_flag("FIX", 3)

            with T.Scope("V"):
                T.tile.fill(selected_k_ub, 0.0)
                for sparse_i in range(sparse_size):
                    token_idx = SparseIndex[global_bkvh, q_idx, sparse_i]
                    T.copy(K[global_bkvh, token_idx, :], selected_k_ub[sparse_i, :])
                T.copy(selected_k_ub, workspace_k[cid, :, :])
                T.set_cross_flag("MTE3", 0)

                T.wait_cross_flag(1)
                T.copy(
                    workspace_scores[
                        cid,
                        subgroup_begin:subgroup_begin + subgroup,
                        :,
                    ],
                    scores_ub,
                )
                T.tile.mul(scores_ub, scores_ub, sm_scale)
                for row_i in range(subgroup):
                    for sparse_i in range(sparse_size, block_sparse):
                        scores_ub[row_i, sparse_i] = -2**30
                T.reduce_max(scores_ub, row_max, reduce_tmp, dim=-1)
                for row_i in range(subgroup):
                    T.tile.sub(scores_ub[row_i, :], scores_ub[row_i, :], row_max[row_i])
                T.tile.exp(scores_ub, scores_ub)
                T.reduce_sum(scores_ub, row_sum, reduce_tmp, dim=-1)
                for row_i in range(subgroup):
                    T.tile.div(scores_ub[row_i, :], scores_ub[row_i, :], row_sum[row_i])
                T.copy(scores_ub, probs_ub)
                T.copy(
                    probs_ub,
                    workspace_probs[
                        cid,
                        subgroup_begin:subgroup_begin + subgroup,
                        :,
                    ],
                )
                T.tile.fill(selected_v_ub, 0.0)
                for sparse_i in range(sparse_size):
                    token_idx = SparseIndex[global_bkvh, q_idx, sparse_i]
                    T.copy(V[global_bkvh, token_idx, :], selected_v_ub[sparse_i, :])
                T.copy(selected_v_ub, workspace_v[cid, :, :])
                T.set_cross_flag("MTE3", 2)

                T.wait_cross_flag(3)
                T.copy(
                    workspace_out[
                        cid,
                        subgroup_begin:subgroup_begin + subgroup,
                        :,
                    ],
                    out_ub,
                )
                T.copy(out_ub, out_half)
                T.copy(
                    out_half,
                    Output[
                        global_bkvh,
                        q_idx,
                        subgroup_begin:subgroup_begin + subgroup,
                        :,
                    ],
                )

    return main
