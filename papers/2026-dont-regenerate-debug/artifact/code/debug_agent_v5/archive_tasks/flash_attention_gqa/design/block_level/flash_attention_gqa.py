"""Block-level TileLang design for flash_attention_gqa.

This is the block-level decomposition reverse engineered from the existing
tile-level implementation. It keeps the same launch topology, packed-GQA data
layout, ring-buffer protocol, and stage boundaries, while intentionally leaving
the fine-grained math to the tile-level implementation.
"""

import tilelang
import tilelang.language as T


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
    block_q_packed = 64
    block_kv_seq = 64
    block_bkvh = 2
    prelaunch = 2
    ring_slots = prelaunch + 1

    dtype = "float16"
    accum_dtype = "float"

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
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(output_shape, dtype),
        workspace_s: T.Tensor([block_num, ring_slots, block_q_packed, block_kv_seq], accum_dtype),
        workspace_p: T.Tensor([block_num, ring_slots, block_q_packed, block_kv_seq], dtype),
        workspace_o: T.Tensor([block_num, ring_slots, block_q_packed, dim], accum_dtype),
        workspace_meta: T.Tensor([block_num, ring_slots, block_q_packed, 2], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bkvh_block_id = cid

            _ = (Q, K, V, Output, workspace_s, workspace_p, workspace_o, workspace_meta)
            _ = (bkvh_block_id, vid)

            # Packed GQA mapping:
            #   global_bkvh = batch_id * kv_heads + kv_head_id
            #   Q  shape    = [B * N2, group_size * S_q, D]
            #   K/V shape   = [B * N2, S_kv, D]
            #   Output shape= [B * N2, group_size * S_q, D]
            #
            # One kernel block owns up to `block_bkvh` packed (batch, kv_head)
            # rows. Inside that block, it serially iterates over all q-blocks
            # belonging to those packed rows.
            #
            # For a local loop index `bx`:
            #   bkvh_i      = bx // q_blocks_per_head
            #   q_block_idx = bx % q_blocks_per_head
            #   global_bkvh = bkvh_block_id * block_bkvh + bkvh_i
            #
            # So the q tile owned by this iteration is:
            #   Q[global_bkvh,
            #     q_block_idx * block_q_packed:(q_block_idx + 1) * block_q_packed, :]
            #
            # Ring timeline with prelaunch=2:
            #   t=0: C1(0) / V1(0)
            #   t=1: C1(1) / V1(1)
            #   t=2: C1(2) + C2(0) / V1(2) + V2(0)
            #   t=3: C1(3) + C2(1) / V1(3) + V2(1)
            #   ...
            #   tail: C2(...) / V2(...)

            for bx in T.serial(q_blocks):
                bkvh_i = bx // q_blocks_per_head
                q_block_idx = bx % q_blocks_per_head
                global_bkvh = bkvh_block_id * block_bkvh + bkvh_i

                if global_bkvh < total_bkvh:
                    with T.Scope("C"):
                        # TODO(tile-level, C0):
                        # - preload the packed Q tile for (global_bkvh, q_block_idx)
                        # - keep it resident across the whole C1/C2 pipeline
                        _ = q_block_idx

                    with T.Scope("V"):
                        # TODO(tile-level, V0):
                        # - initialize the running output accumulator, running max,
                        #   and running sumexp for this vid-owned half tile
                        _ = q_block_idx

                    # Vector side works on half of the q rows per `vid`.
                    # The half-tile row range is:
                    #   vid * block_q_packed // 2 :
                    #   vid * block_q_packed // 2 + block_q_packed // 2
                    for t in T.serial(kv_loops + prelaunch):
                        if t < kv_loops:
                            slot_prod = t % ring_slots

                            with T.Scope("C"):
                                # TODO(tile-level, C1):
                                # - load K tile for kv step t
                                # - compute S = Q @ K^T
                                # - store S into workspace_s[cid, slot_prod, ...]
                                # - signal that C1(t) is ready for V1(t)
                                _ = slot_prod
                                T.set_cross_flag("FIX", 0)

                            with T.Scope("V"):
                                # TODO(tile-level, V1):
                                # - wait until C1(t) has produced workspace_s[cid, slot_prod, ...]
                                # - load this vid-owned half tile
                                # - apply scale + online softmax update
                                # - store probabilities into workspace_p
                                # - store alpha / local sumexp into workspace_meta
                                # - signal that V1(t) is ready for C2(t)
                                _ = slot_prod
                                T.wait_cross_flag(0)
                                T.set_cross_flag("MTE3", 1)

                        if t >= prelaunch:
                            now_k = t - prelaunch
                            slot_cons = now_k % ring_slots

                            with T.Scope("C"):
                                # TODO(tile-level, C2):
                                # - wait until V1(now_k) has produced workspace_p[cid, slot_cons, ...]
                                # - load V tile for kv step now_k
                                # - compute O_tmp = P @ V
                                # - store O_tmp into workspace_o[cid, slot_cons, ...]
                                # - signal that C2(now_k) is ready for V2(now_k)
                                _ = slot_cons
                                T.wait_cross_flag(1)
                                T.set_cross_flag("FIX", 2)

                            with T.Scope("V"):
                                # TODO(tile-level, V2):
                                # - wait until C2(now_k) has produced workspace_o[cid, slot_cons, ...]
                                # - read alpha / local sumexp from workspace_meta
                                # - rescale running accumulator and merge O_tmp
                                # - update the running output/sumexp state for this half tile
                                _ = (now_k, slot_cons)
                                T.wait_cross_flag(2)

                    with T.Scope("V"):
                        # TODO(tile-level, V2-epilogue):
                        # - after the final kv tile, normalize and write Output
                        #   at the packed-q location of (global_bkvh, q_block_idx)
                        _ = q_block_idx

    return main
