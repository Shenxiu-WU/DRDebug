"""Block-level TileLang design for sparse_flash_attention.

This block-level decomposition keeps the irregular sparse gather inside the
kernel, but places it on the Vector side. Each block first uses sparse indices
to gather K/V rows into dense UB tiles, then fuses:

1. Q @ K_selected^T
2. scaled softmax across the sparse token axis
3. P @ V_selected

The file intentionally leaves tile math as TODO(tile-level) while fixing the
block ownership, workspace contract, and C/V synchronization skeleton.
"""

import tilelang
import tilelang.language as T


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

    total_bkvh = batch * kv_heads
    block_num = total_bkvh * q_seq_len

    q_shape = [total_bkvh, q_seq_len, block_group, dim]
    kv_shape = [total_bkvh, kv_seq_len, dim]
    sparse_index_shape = [total_bkvh, q_seq_len, sparse_size]
    output_shape = [total_bkvh, q_seq_len, block_group, dim]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        SparseIndex: T.Tensor(sparse_index_shape, "int32"),
        Output: T.Tensor(output_shape, dtype),
        workspace_scores: T.Tensor([block_num, block_group, block_sparse], accum_dtype),
        workspace_probs: T.Tensor([block_num, block_group, block_sparse], dtype),
        workspace_out: T.Tensor([block_num, block_group, dim], accum_dtype),
        workspace_k: T.Tensor([block_num, block_sparse, dim], dtype),
        workspace_v: T.Tensor([block_num, block_sparse, dim], dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            global_bkvh = cid // q_seq_len
            q_idx = cid % q_seq_len
            subgroup = block_group // 2
            subgroup_begin = vid * subgroup

            _ = (Q, K, V, SparseIndex, Output)
            _ = (workspace_scores, workspace_probs, workspace_out, workspace_k, workspace_v)
            _ = (global_bkvh, q_idx, subgroup_begin)

            # Block ownership:
            #   global_bkvh = batch_id * kv_heads + kv_head_id
            #   q_idx       = query sequence position
            #
            # One kernel block owns Output[global_bkvh, q_idx, :, :], i.e. all
            # GQA-group rows for a single query position under one kv head.
            #
            # Inputs are pre-packed by model_new_tilelang.py as:
            #   Q           : [B * N2, S1, block_group, D]
            #   K / V       : [B * N2, S2, D]
            #   SparseIndex : [B * N2, S1, sparse_size]
            #
            # `block_group` / `block_sparse` may include zero-padding so the
            # tile-level kernel can use stable matrix tile shapes. The first
            # `sparse_size` rows of the K/V staging tiles are filled by sparse
            # row-wise `T.copy` on the V side into UB and then written to
            # workspace_k / workspace_v for the C side to consume.

            with T.Scope("C"):
                # TODO(tile-level):
                # - load the dense padded Q tile for (global_bkvh, q_idx)
                # - wait until workspace_k[cid, :, :] is ready from the V side
                # - load the gathered dense K tile from workspace_k into L1
                # - compute scores = Q @ K_selected^T
                # - store scores into workspace_scores[cid, :, :]
                # - signal the Vector softmax/gather-V stage
                T.wait_cross_flag(0)
                T.set_cross_flag("FIX", 1)

                # TODO(tile-level):
                # - wait until workspace_probs[cid, :, :] and workspace_v are ready
                # - load the dense V tile from workspace_v into L1
                # - compute workspace_out[cid, :, :] = P @ V_selected
                # - signal the Vector writeback stage
                T.wait_cross_flag(2)
                T.set_cross_flag("FIX", 3)

            with T.Scope("V"):
                # Vector side works on half of the padded group rows per `vid`.
                # It owns:
                #   subgroup_begin : subgroup_begin + subgroup

                # TODO(tile-level):
                # - use SparseIndex[global_bkvh, q_idx, :] to gather K rows
                #   from GM into a dense UB tile via row-wise T.copy
                # - zero-fill padded rows
                # - write the gathered tile to workspace_k[cid, :, :]
                # - signal the Cube QK stage
                T.set_cross_flag("MTE3", 0)

                # TODO(tile-level):
                # - wait for workspace_scores[cid, :, :] to become ready
                # - apply scale and mask padded sparse columns
                # - compute softmax along the sparse axis
                # - write probabilities into workspace_probs
                # - gather V rows into a dense UB tile and write workspace_v
                # - signal the Cube PV stage
                T.wait_cross_flag(1)
                T.set_cross_flag("MTE3", 2)

                # TODO(tile-level):
                # - wait for workspace_out[cid, :, :] to become ready
                # - cast and write the owned subgroup rows into Output
                T.wait_cross_flag(3)

    return main
