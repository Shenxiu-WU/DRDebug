"""Block-level TileLang design for int8 matmul + per-column scale.

This layer only captures block scheduling, Cube/Vector collaboration, and the
cross-scope handoff. Tile-level compute details are intentionally left as TODOs
and are filled in by `design/tile_level/int8_matmul_scale.py`.
"""

import tilelang
import tilelang.language as T

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

            with T.Scope("C"):
                # TODO(tile-level):
                # - stage int8 A/B tiles through L1/L0
                # - run block matmul for tile (bx, by) with int32 accumulation
                # - store the accumulator tile into workspace
                # - signal Vector scope when the tile is ready
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)

                # TODO(tile-level):
                # - load the accumulator tile from workspace
                # - broadcast the per-column scale tile
                # - convert int32 -> float32, apply scale, cast to float16
                # - store the final tile into C
                pass

    return main
