"""Block-level TileLang design for avg_pool3_d.

This file mirrors tile-level mode selection and scheduling, but keeps only
coarse-grained block partition and Vector-side pipeline skeleton.
Fine-grained compute details are intentionally left as TODO comments.
"""

import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def avg_pool3_d(
    N_batch,
    C,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    kD,
    kH,
    kW,
    sD,
    sH,
    sW,
    pD,
    pH,
    pW,
    count_include_pad,
    divisor_override,
    block_C=0,
    dtype="float32",
    split_mode=0,
    split_w_tile_kw=0,
    multi_w_window_w_num=1,
):
    in_spatial = D * H * W
    out_spatial = OD * OH * OW
    M_out = N_batch * out_spatial

    block_M = 0
    for candidate in (64, 32, 16, 8, 4, 2):
        if candidate <= M_out and M_out % candidate == 0:
            block_M = candidate
            break
    if block_M == 0:
        raise ValueError(
            f"Unsupported output spatial size: M_out={M_out} is not divisible by any block_M in [2,4,8,16,32,64]"
        )

    m_num = M_out // block_M
    assert m_num > 0

    num_physical_cores = 20
    usedCoreNumM = min(num_physical_cores, m_num)
    tasksPerCoreM = (m_num + usedCoreNumM - 1) // usedCoreNumM

    vec_num = 2
    sub_block_M = block_M // vec_num

    pool_size = kD * kH * kW
    HW = H * W

    split_block_C = block_C if block_C > 0 else 1
    c_num = C // split_block_C
    total_blocks_mc = m_num * c_num
    usedCoreNumMC = min(num_physical_cores, total_blocks_mc)
    tasksPerCoreMC = (total_blocks_mc + usedCoreNumMC - 1) // usedCoreNumMC

    split_w_step = split_w_tile_kw if split_w_tile_kw > 0 else kW
    multi_w_window = multi_w_window_w_num if multi_w_window_w_num > 0 else 1

    @T.prim_func
    def generic_3d(
        X: T.Tensor((N_batch * in_spatial, C), dtype),
        Y: T.Tensor((M_out, C), dtype),
    ):
        with T.Kernel(usedCoreNumM, is_npu=True) as (cid, vid):
            coreIdx = cid

            # Block-level persistent-kernel pipeline:
            #   1) Fixed physical cores iterate over contiguous output-row blocks.
            #   2) Each AIV sub-block handles one half block_M rows.
            #   3) generic_3d traverses full (kD, kH, kW) window per output row.
            for localIdx in T.serial(tasksPerCoreM):
                bx = coreIdx * tasksPerCoreM + localIdx

                with T.Scope("V"):
                    if bx < m_num:
                        row_base = bx * block_M + vid * sub_block_M
                        for m in T.serial(sub_block_M):
                            out_row = row_base + m

                            n_idx = out_row // out_spatial
                            out_rem = out_row - n_idx * out_spatial
                            od = out_rem // (OH * OW)
                            od_rem = out_rem - od * (OH * OW)
                            oh = od_rem // OW
                            ow = od_rem - oh * OW

                            # TODO(tile-level):
                            # - allocate/load UB tiles for one output row on channel span C
                            # - traverse (kd, kh, kw), check boundaries, and accumulate X
                            # - apply avg divisor (divisor_override / pool_size / valid_pool)
                            # - store result to Y[out_row, :]
                            _ = X
                            _ = Y
                            _ = n_idx
                            _ = od
                            _ = oh
                            _ = ow

    @T.prim_func
    def reduce_d(
        X: T.Tensor((N_batch * in_spatial, C), dtype),
        Y: T.Tensor((M_out, C), dtype),
    ):
        with T.Kernel(usedCoreNumM, is_npu=True) as (cid, vid):
            coreIdx = cid

            # Block-level persistent-kernel pipeline:
            #   1) Task mapping is same as generic_3d.
            #   2) Special shape path where only depth dimension is reduced.
            for localIdx in T.serial(tasksPerCoreM):
                bx = coreIdx * tasksPerCoreM + localIdx

                with T.Scope("V"):
                    if bx < m_num:
                        row_base = bx * block_M + vid * sub_block_M
                        for m in T.serial(sub_block_M):
                            out_row = row_base + m

                            n_idx = out_row // out_spatial
                            out_rem = out_row - n_idx * out_spatial
                            od = out_rem // (OH * OW)
                            od_rem = out_rem - od * (OH * OW)
                            oh = od_rem // OW
                            ow = od_rem - oh * OW

                            # TODO(tile-level):
                            # - load one (oh, ow) location and reduce only along kD
                            # - apply avg divisor policy and store Y[out_row, :]
                            _ = X
                            _ = Y
                            _ = n_idx
                            _ = od
                            _ = oh
                            _ = ow

    @T.prim_func
    def split_c(
        X: T.Tensor((N_batch * in_spatial, C), dtype),
        Y: T.Tensor((M_out, C), dtype),
    ):
        with T.Kernel(usedCoreNumMC, is_npu=True) as (cid, vid):
            coreIdx = cid

            # Block-level persistent-kernel pipeline:
            #   1) Logical task is (output-row block, channel block).
            #   2) Different cores process disjoint channel tiles of one output row.
            for localIdx in T.serial(tasksPerCoreMC):
                task_id = coreIdx * tasksPerCoreMC + localIdx
                bc = task_id // m_num
                bx = task_id % m_num

                with T.Scope("V"):
                    if task_id < total_blocks_mc:
                        row_base = bx * block_M + vid * sub_block_M
                        c_base = bc * split_block_C

                        for m in T.serial(sub_block_M):
                            out_row = row_base + m

                            n_idx = out_row // out_spatial
                            out_rem = out_row - n_idx * out_spatial
                            od = out_rem // (OH * OW)
                            od_rem = out_rem - od * (OH * OW)
                            oh = od_rem // OW
                            ow = od_rem - oh * OW

                            # TODO(tile-level):
                            # - process pooling window for channel tile [c_base:c_base+split_block_C)
                            # - apply avg divisor policy and store Y[out_row, c_base:...]
                            _ = X
                            _ = Y
                            _ = c_base
                            _ = n_idx
                            _ = od
                            _ = oh
                            _ = ow

    @T.prim_func
    def split_w(
        X: T.Tensor((N_batch * in_spatial, C), dtype),
        Y: T.Tensor((M_out, C), dtype),
    ):
        with T.Kernel(usedCoreNumM, is_npu=True) as (cid, vid):
            coreIdx = cid

            # Block-level persistent-kernel pipeline:
            #   1) Task mapping is same as generic_3d.
            #   2) Width reduction loop is split into chunks of split_w_step.
            for localIdx in T.serial(tasksPerCoreM):
                bx = coreIdx * tasksPerCoreM + localIdx

                with T.Scope("V"):
                    if bx < m_num:
                        row_base = bx * block_M + vid * sub_block_M
                        for m in T.serial(sub_block_M):
                            out_row = row_base + m

                            n_idx = out_row // out_spatial
                            out_rem = out_row - n_idx * out_spatial
                            od = out_rem // (OH * OW)
                            od_rem = out_rem - od * (OH * OW)
                            oh = od_rem // OW
                            ow = od_rem - oh * OW

                            # TODO(tile-level):
                            # - split kw traversal into kw_base/kw_local by split_w_step
                            # - accumulate pooled value for full channel span C
                            # - apply avg divisor policy and store Y[out_row, :]
                            _ = X
                            _ = Y
                            _ = split_w_step
                            _ = n_idx
                            _ = od
                            _ = oh
                            _ = ow

    @T.prim_func
    def multi_w(
        X: T.Tensor((N_batch * in_spatial, C), dtype),
        Y: T.Tensor((M_out, C), dtype),
    ):
        with T.Kernel(usedCoreNumM, is_npu=True) as (cid, vid):
            coreIdx = cid

            # Block-level persistent-kernel pipeline:
            #   1) Task mapping is same as generic_3d.
            #   2) For aligned ow, one task computes multi_w_window consecutive outputs.
            #   3) Prefix branch handles unaligned head of each row segment.
            for localIdx in T.serial(tasksPerCoreM):
                bx = coreIdx * tasksPerCoreM + localIdx

                with T.Scope("V"):
                    if bx < m_num:
                        row_base = bx * block_M + vid * sub_block_M
                        for m in T.serial(sub_block_M):
                            out_row = row_base + m

                            n_idx = out_row // out_spatial
                            out_rem = out_row - n_idx * out_spatial
                            od = out_rem // (OH * OW)
                            od_rem = out_rem - od * (OH * OW)
                            oh = od_rem // OW
                            ow = od_rem - oh * OW

                            if multi_w_window > 1:
                                if ow % multi_w_window == 0:
                                    for local_w in T.serial(multi_w_window):
                                        if local_w < sub_block_M - m:
                                            if local_w < OW - ow:
                                                cur_ow = ow + local_w
                                                # TODO(tile-level):
                                                # - compute one output at (od, oh, cur_ow)
                                                # - apply divisor_override/count_include_pad policy
                                                # - store Y[out_row + local_w, :]
                                                _ = X
                                                _ = Y
                                                _ = n_idx
                                                _ = od
                                                _ = oh
                                                _ = cur_ow
                                elif m == 0:
                                    prefix_len = multi_w_window - (ow % multi_w_window)
                                    for local_w in T.serial(multi_w_window):
                                        if local_w < prefix_len:
                                            if local_w < sub_block_M - m:
                                                if local_w < OW - ow:
                                                    cur_ow = ow + local_w
                                                    # TODO(tile-level):
                                                    # - handle unaligned prefix outputs before steady-state multi_w
                                                    # - store Y[out_row + local_w, :]
                                                    _ = X
                                                    _ = Y
                                                    _ = n_idx
                                                    _ = od
                                                    _ = oh
                                                    _ = cur_ow

    split_c_ready = block_C > 0 and C % block_C == 0

    # reduce_d fast path: only depth dimension participates in pooling.
    if kH == 1 and kW == 1 and sH == 1 and sW == 1 and pH == 0 and pW == 0:
        return reduce_d
    if split_mode == 1:
        if split_c_ready:
            return split_c
        raise ValueError(f"split_c requires divisible channel tiles, got C={C}, block_C={block_C}")
    if split_mode == 2:
        return split_w
    if split_mode == 3:
        if multi_w_window > 1:
            return multi_w
        return generic_3d
    if split_c_ready:
        return split_c
    _ = count_include_pad
    _ = divisor_override
    _ = pool_size
    return generic_3d
