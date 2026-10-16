"""Block-level TileLang design for reshape-view matmul + row-wise int8 quant.

Computation target (equivalent to `current_task/model.py`):
1) X: (M, N), H: (H_K, H_K), where N % H_K == 0.
2) View X as (M * (N / H_K), H_K), run matmul with H, reshape back to (M, N).
3) Per-row dynamic int8 quant on the reshaped output:
   scale_i = max(abs(row_i)) / 127, clamp_min(1e-12)
   y_i = round(row_i / scale_i), clipped to int8 range.

This block-level file keeps only scheduling/pipeline/cross-scope skeleton.
Tile-level compute details are filled in by
`design/tile_level/reshape_matmul_rowwise_quant_int8.py`.
"""

import tilelang
import tilelang.language as T


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
    n_tiles_per_h = H_K // block_N

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        H: T.Tensor((H_K, H_K), dtype),
        Y: T.Tensor((M, N), "int8"),
        workspace: T.Tensor((M, N), accum_dtype),
    ):
        # One kernel task handles one output row tile (block_M rows).
        # It computes all N-tiles for this row tile, then quantizes row-wise.
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid

            with T.Scope("C"):
                # TODO(tile-level):
                # - loop by over all output column tiles
                # - map each by to (group_id, col_in_group) for reshape-view matmul
                # - run block matmul and write tile result to workspace
                # - signal Vector scope when all N-tiles are ready
                _ = n_num
                _ = n_tiles_per_h
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)
                # TODO(tile-level):
                # - pass 1: compute per-row absmax across all N tiles
                # - derive row scale = max(abs(row))/127 with clamp_min(1e-12)
                # - pass 2: divide by row scale, round, cast/saturate to int8
                # - write final quantized result to Y
                pass

    return main
