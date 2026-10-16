import tilelang
from tilelang import DataType, language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7], pass_configs=pass_configs)
def flash_attention_gqa_fwd(
    batch,
    q_seq_len,
    kv_seq_len,
    heads,
    kv_heads,
    dim,
):
    block_q_packed = 64 # block_gs1
    block_kv_seq = 64
    block_bkvh = 2 # block_BN2
    prelaunch = 2
    ring_slots = prelaunch + 1

    dtype = "float16"
    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    group_size = heads // kv_heads
    packed_q_seq_len = group_size * q_seq_len

    q_shape = [batch * kv_heads, packed_q_seq_len, dim]
    kv_shape = [batch * kv_heads, kv_seq_len, dim]
    output_shape = [batch * kv_heads, packed_q_seq_len, dim]

    total_bkvh = batch * kv_heads
    q_blocks_per_head = packed_q_seq_len // block_q_packed
    q_blocks = block_bkvh * q_blocks_per_head
    bkvh_blocks = T.ceildiv(total_bkvh, block_bkvh)
    block_num = bkvh_blocks
    kv_loops = T.ceildiv(kv_seq_len, block_kv_seq)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape, dtype),  # type: ignore
        V: T.Tensor(kv_shape, dtype),  # type: ignore
        Output: T.Tensor(output_shape, dtype),  # type: ignore
        workspace_1: T.Tensor([block_num, ring_slots, block_q_packed, block_kv_seq], accum_dtype),
        workspace_2: T.Tensor([block_num, ring_slots, block_q_packed, block_kv_seq], dtype),
        workspace_3: T.Tensor([block_num, ring_slots, block_q_packed, dim], accum_dtype),
        workspace_meta: T.Tensor([block_num, ring_slots, block_q_packed, 2], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bkvh_block_id = cid

            q_l1 = T.alloc_L1([block_q_packed, dim], dtype)
            k_l1 = T.alloc_L1([block_kv_seq, dim], dtype)
            v_l1 = T.alloc_L1([block_kv_seq, dim], dtype)

            acc_s_l1 = T.alloc_L1([block_q_packed, block_kv_seq], dtype)

            acc_s_l0c = T.alloc_L0C([block_q_packed, block_kv_seq], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_q_packed, dim], accum_dtype)

            acc_o = T.alloc_ub([block_q_packed // 2, dim], accum_dtype)
            sumexp = T.alloc_ub([block_q_packed // 2], accum_dtype)
            m_i = T.alloc_ub([block_q_packed // 2], accum_dtype)

            acc_s_ub = T.alloc_ub([block_q_packed // 2, block_kv_seq], accum_dtype)
            m_i_prev = T.alloc_ub([block_q_packed // 2], accum_dtype)
            acc_s_ub_ = T.alloc_ub([block_q_packed // 2, block_kv_seq], accum_dtype)
            tmp_ub = T.alloc_ub(
                [3 * DataType(accum_dtype).bits // 8 * block_q_packed // 2 * block_kv_seq],
                "uint8",
            )
            sumexp_i_ub = T.alloc_ub([block_q_packed // 2], accum_dtype)
            acc_s_half = T.alloc_ub([block_q_packed // 2, block_kv_seq], dtype)
            acc_o_ub = T.alloc_ub([block_q_packed // 2, dim], accum_dtype)
            acc_o_half = T.alloc_ub([block_q_packed // 2, dim], dtype)
            alpha_ub = T.alloc_ub([block_q_packed // 2], accum_dtype)
            sumexp_meta_ub = T.alloc_ub([block_q_packed // 2], accum_dtype)

            for bx in T.serial(q_blocks):
                bkvh_i = bx // q_blocks_per_head
                q_block_idx = bx % q_blocks_per_head
                global_bkvh = bkvh_block_id * block_bkvh + bkvh_i
                if global_bkvh < total_bkvh:
                    with T.Scope("C"):
                        T.copy(
                            Q[
                                global_bkvh,
                                q_block_idx * block_q_packed:(q_block_idx + 1) * block_q_packed,
                                :,
                            ],
                            q_l1,
                        )
                    with T.Scope("V"):
                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, -2**30)

                    for t in T.serial(kv_loops + prelaunch):
                        if t < kv_loops:
                            slot_prod = t % ring_slots
                            with T.Scope("C"):
                                T.copy(K[global_bkvh, t * block_kv_seq:(t + 1) * block_kv_seq, :], k_l1)
                                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                                T.copy(acc_s_l0c, workspace_1[cid, slot_prod, :, :])
                                T.set_cross_flag("FIX", 0)

                            with T.Scope("V"):
                                T.wait_cross_flag(0)
                                T.tile.fill(acc_s_ub, 0.0)
                                T.copy(m_i, m_i_prev)
                                T.copy(
                                    workspace_1[
                                        cid,
                                        slot_prod,
                                        vid * block_q_packed // 2:vid * block_q_packed // 2 + block_q_packed // 2,
                                        :,
                                    ],
                                    acc_s_ub_,
                                )
                                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                                T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.tile.exp(m_i_prev, m_i_prev)
                                for h_i in range(block_q_packed // 2):
                                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                                T.tile.exp(acc_s_ub, acc_s_ub)
                                T.reduce_sum(acc_s_ub, sumexp_i_ub, tmp_ub, dim=-1)

                                T.copy(acc_s_ub, acc_s_half)
                                T.copy(
                                    acc_s_half,
                                    workspace_2[
                                        cid,
                                        slot_prod,
                                        vid * block_q_packed // 2:vid * block_q_packed // 2 + block_q_packed // 2,
                                        :,
                                    ],
                                )
                                for h_i in range(block_q_packed // 2):
                                    workspace_meta[
                                        cid,
                                        slot_prod,
                                        vid * block_q_packed // 2 + h_i,
                                        0,
                                    ] = m_i_prev[h_i]
                                    workspace_meta[
                                        cid,
                                        slot_prod,
                                        vid * block_q_packed // 2 + h_i,
                                        1,
                                    ] = sumexp_i_ub[h_i]
                                T.set_cross_flag("MTE3", 1)

                        if t >= prelaunch:
                            now_k = t - prelaunch
                            slot_cons = now_k % ring_slots
                            with T.Scope("C"):
                                T.wait_cross_flag(1)
                                T.copy(workspace_2[cid, slot_cons, :, :], acc_s_l1)
                                T.copy(
                                    V[global_bkvh, now_k * block_kv_seq:(now_k + 1) * block_kv_seq, :],
                                    v_l1,
                                )
                                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                                T.copy(acc_o_l0c, workspace_3[cid, slot_cons, :, :])
                                T.set_cross_flag("FIX", 2)

                            with T.Scope("V"):
                                T.wait_cross_flag(2)
                                for h_i in range(block_q_packed // 2):
                                    alpha_ub[h_i] = workspace_meta[
                                        cid,
                                        slot_cons,
                                        vid * block_q_packed // 2 + h_i,
                                        0,
                                    ]
                                    sumexp_meta_ub[h_i] = workspace_meta[
                                        cid,
                                        slot_cons,
                                        vid * block_q_packed // 2 + h_i,
                                        1,
                                    ]
                                T.copy(
                                    workspace_3[
                                        cid,
                                        slot_cons,
                                        vid * block_q_packed // 2:vid * block_q_packed // 2 + block_q_packed // 2,
                                        :,
                                    ],
                                    acc_o_ub,
                                )
                                for h_i in range(block_q_packed // 2):
                                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], alpha_ub[h_i])
                                T.tile.add(acc_o, acc_o, acc_o_ub)
                                T.tile.mul(sumexp, sumexp, alpha_ub)
                                T.tile.add(sumexp, sumexp, sumexp_meta_ub)

                    with T.Scope("V"):
                        for h_i in range(block_q_packed // 2):
                            T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

                        T.copy(acc_o, acc_o_half)
                        T.copy(
                            acc_o_half,
                            Output[
                                global_bkvh,
                                q_block_idx * block_q_packed + vid * block_q_packed // 2:q_block_idx * block_q_packed + vid * block_q_packed // 2 + block_q_packed // 2,
                                :,
                            ],
                        )

    return main
