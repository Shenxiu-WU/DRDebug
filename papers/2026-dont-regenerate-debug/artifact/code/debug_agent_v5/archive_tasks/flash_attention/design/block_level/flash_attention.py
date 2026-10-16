"""Block-level TileLang design for flash_attention.

This file captures task decomposition, stage boundaries, ring-buffer ownership,
and cross-scope synchronization. Fine-grained math is intentionally left as
TODOs and should be filled in by the tile-level design.
"""

import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7], pass_configs=pass_configs)
def flash_attention_fwd(
    batch,
    seq_len,
    heads,
    dim,
):
    block_M, block_N = 64, 64
    prelaunch = 2
    ring_slots = prelaunch + 1

    dtype = "float16"
    accum_dtype = "float"

    shape = [batch, heads, seq_len, dim]
    block_num = seq_len // block_M * heads * batch
    kv_loops = T.ceildiv(seq_len, block_N)

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
        workspace_s: T.Tensor([block_num, ring_slots, block_M, block_N], accum_dtype),
        workspace_p: T.Tensor([block_num, ring_slots, block_M, block_N], dtype),
        workspace_o: T.Tensor([block_num, ring_slots, block_M, dim], accum_dtype),
        workspace_meta: T.Tensor([block_num, ring_slots, block_M, 2], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch

            _ = (Q, K, V, Output, workspace_s, workspace_p, workspace_o, workspace_meta)
            _ = (bx, by, bz, vid)

            # Block-level task partition:
            #   Each kernel instance handles one (batch, head, q-block) tile:
            #   Q[bz, by, bx * block_M:(bx + 1) * block_M, :]
            #
            # Ring pipeline timeline for prelaunch=2:
            #   t=0: C1(0) / V1(0)
            #   t=1: C1(1) / V1(1)
            #   t=2: C1(2) + C2(0) / V1(2) + V2(0)
            #   t=3: C1(3) + C2(1) / V1(3) + V2(1)
            #   ...
            #   tail: C2(...) / V2(...)

            with T.Scope("C"):
                # TODO(tile-level, C0):
                # - preload the current q-block into on-chip Cube-side memory
                # - keep it resident across the whole C1/C2 pipeline
                pass

            with T.Scope("V"):
                # TODO(tile-level, V0):
                # - initialize the running output accumulator, running max, and
                #   running sumexp for this vid-owned half tile before V1/V2
                pass

            for t in T.serial(kv_loops + prelaunch):
                if t < kv_loops:
                    slot_prod = t % ring_slots

                    with T.Scope("C"):
                        # TODO(tile-level, C1):
                        # - load K tile for kv-step t
                        # - compute S = Q @ K^T for current q-block
                        # - store S into workspace_s[cid, slot_prod, ...]
                        # - signal that C1(t) is ready for V1(t)
                        _ = slot_prod
                        T.set_cross_flag("FIX", 0)

                    with T.Scope("V"):
                        # TODO(tile-level, V1):
                        # - wait until C1(t) has produced workspace_s[cid, slot_prod, ...]
                        # - apply scaling and online softmax update on current half-tile
                        # - store P into workspace_p[cid, slot_prod, ...]
                        # - store merge metadata into workspace_meta[cid, slot_prod, ...]
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
                        # - load V tile for now_k
                        # - compute O_tmp = P @ V
                        # - store O_tmp into workspace_o[cid, slot_cons, ...]
                        # - signal that C2(now_k) is ready for V2(now_k)
                        _ = slot_cons
                        T.wait_cross_flag(1)
                        T.set_cross_flag("FIX", 2)

                    with T.Scope("V"):
                        # TODO(tile-level, V2):
                        # - wait until C2(now_k) has produced workspace_o[cid, slot_cons, ...]
                        # - load alpha / local sumexp produced by V1(now_k)
                        # - rescale previous accumulator and merge O_tmp
                        # - update the running output/sumexp state for this half tile
                        _ = slot_cons
                        T.wait_cross_flag(2)

            with T.Scope("V"):
                # TODO(tile-level, V2-epilogue):
                # - normalize the running accumulator by the final sumexp
                # - cast and write the final output half tile to Output
                pass

    return main
