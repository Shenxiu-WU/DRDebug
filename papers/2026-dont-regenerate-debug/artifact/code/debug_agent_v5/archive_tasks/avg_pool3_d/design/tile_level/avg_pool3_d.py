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
    vec_num = 2
    sub_block_M = block_M // vec_num

    pool_size = kD * kH * kW
    HW = H * W

    split_block_C = block_C if block_C > 0 else 1
    c_num = C // split_block_C
    split_w_step = split_w_tile_kw if split_w_tile_kw > 0 else kW
    multi_w_window = multi_w_window_w_num if multi_w_window_w_num > 0 else 1

    @T.prim_func
    def generic_3d(
        X: T.Tensor((N_batch * in_spatial, C), dtype),
        Y: T.Tensor((M_out, C), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_M + vid * sub_block_M

            acc_ub = T.alloc_ub((1, C), dtype)
            inp_ub = T.alloc_ub((1, C), dtype)
            out_ub = T.alloc_ub((1, C), dtype)

            with T.Scope("V"):
                for m in T.serial(sub_block_M):
                    out_row = row_base + m

                    n_idx = out_row // out_spatial
                    out_rem = out_row - n_idx * out_spatial
                    od = out_rem // (OH * OW)
                    od_rem = out_rem - od * (OH * OW)
                    oh = od_rem // OW
                    ow = od_rem - oh * OW

                    T.tile.fill(acc_ub, T.float32(0.0))

                    for kd in T.serial(kD):
                        id_val = od * sD - pD + kd
                        if id_val >= 0:
                            if id_val < D:
                                for kh in T.serial(kH):
                                    ih_val = oh * sH - pH + kh
                                    if ih_val >= 0:
                                        if ih_val < H:
                                            for kw in T.serial(kW):
                                                iw_val = ow * sW - pW + kw
                                                if iw_val >= 0:
                                                    if iw_val < W:
                                                        in_row = n_idx * in_spatial + id_val * HW + ih_val * W + iw_val
                                                        T.copy(X[in_row, 0], inp_ub)
                                                        T.tile.add(acc_ub, acc_ub, inp_ub)

                    if divisor_override > 0:
                        T.tile.mul(out_ub, acc_ub, T.float32(1.0 / divisor_override))
                    else:
                        T.tile.mul(out_ub, acc_ub, T.float32(1.0 / pool_size))

                    T.copy(out_ub, Y[out_row, 0])

    @T.prim_func
    def reduce_d(
        X: T.Tensor((N_batch * in_spatial, C), dtype),
        Y: T.Tensor((M_out, C), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_M + vid * sub_block_M

            acc_ub = T.alloc_ub((1, C), dtype)
            inp_ub = T.alloc_ub((1, C), dtype)
            out_ub = T.alloc_ub((1, C), dtype)

            with T.Scope("V"):
                for m in T.serial(sub_block_M):
                    out_row = row_base + m

                    n_idx = out_row // out_spatial
                    out_rem = out_row - n_idx * out_spatial
                    od = out_rem // (OH * OW)
                    od_rem = out_rem - od * (OH * OW)
                    oh = od_rem // OW
                    ow = od_rem - oh * OW

                    T.tile.fill(acc_ub, T.float32(0.0))

                    for kd in T.serial(kD):
                        id_val = od * sD - pD + kd
                        if id_val >= 0:
                            if id_val < D:
                                in_row = n_idx * in_spatial + id_val * HW + oh * W + ow
                                T.copy(X[in_row, 0], inp_ub)
                                T.tile.add(acc_ub, acc_ub, inp_ub)

                    if divisor_override > 0:
                        T.tile.mul(out_ub, acc_ub, T.float32(1.0 / divisor_override))
                    else:
                        T.tile.mul(out_ub, acc_ub, T.float32(1.0 / pool_size))

                    T.copy(out_ub, Y[out_row, 0])

    @T.prim_func
    def split_c(
        X: T.Tensor((N_batch * in_spatial, C), dtype),
        Y: T.Tensor((M_out, C), dtype),
    ):
        with T.Kernel(m_num * c_num, is_npu=True) as (cid, vid):
            bx = cid % m_num
            bc = cid // m_num
            row_base = bx * block_M + vid * sub_block_M
            c_base = bc * split_block_C

            acc_ub = T.alloc_ub((1, split_block_C), dtype)
            inp_ub = T.alloc_ub((1, split_block_C), dtype)
            out_ub = T.alloc_ub((1, split_block_C), dtype)

            with T.Scope("V"):
                for m in T.serial(sub_block_M):
                    out_row = row_base + m

                    n_idx = out_row // out_spatial
                    out_rem = out_row - n_idx * out_spatial
                    od = out_rem // (OH * OW)
                    od_rem = out_rem - od * (OH * OW)
                    oh = od_rem // OW
                    ow = od_rem - oh * OW

                    T.tile.fill(acc_ub, T.float32(0.0))

                    for kd in T.serial(kD):
                        id_val = od * sD - pD + kd
                        if id_val >= 0:
                            if id_val < D:
                                for kh in T.serial(kH):
                                    ih_val = oh * sH - pH + kh
                                    if ih_val >= 0:
                                        if ih_val < H:
                                            for kw in T.serial(kW):
                                                iw_val = ow * sW - pW + kw
                                                if iw_val >= 0:
                                                    if iw_val < W:
                                                        in_row = n_idx * in_spatial + id_val * HW + ih_val * W + iw_val
                                                        T.copy(X[in_row, c_base], inp_ub)
                                                        T.tile.add(acc_ub, acc_ub, inp_ub)

                    if divisor_override > 0:
                        T.tile.mul(out_ub, acc_ub, T.float32(1.0 / divisor_override))
                    else:
                        T.tile.mul(out_ub, acc_ub, T.float32(1.0 / pool_size))

                    T.copy(out_ub, Y[out_row, c_base])

    @T.prim_func
    def split_w(
        X: T.Tensor((N_batch * in_spatial, C), dtype),
        Y: T.Tensor((M_out, C), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_M + vid * sub_block_M

            acc_ub = T.alloc_ub((1, C), dtype)
            inp_ub = T.alloc_ub((1, C), dtype)
            out_ub = T.alloc_ub((1, C), dtype)

            with T.Scope("V"):
                for m in T.serial(sub_block_M):
                    out_row = row_base + m

                    n_idx = out_row // out_spatial
                    out_rem = out_row - n_idx * out_spatial
                    od = out_rem // (OH * OW)
                    od_rem = out_rem - od * (OH * OW)
                    oh = od_rem // OW
                    ow = od_rem - oh * OW

                    T.tile.fill(acc_ub, T.float32(0.0))

                    for kd in T.serial(kD):
                        id_val = od * sD - pD + kd
                        if id_val >= 0:
                            if id_val < D:
                                for kh in T.serial(kH):
                                    ih_val = oh * sH - pH + kh
                                    if ih_val >= 0:
                                        if ih_val < H:
                                            for kw_base in T.serial((kW + split_w_step - 1) // split_w_step):
                                                for kw_local in T.serial(split_w_step):
                                                    kw = kw_base * split_w_step + kw_local
                                                    if kw < kW:
                                                        iw_val = ow * sW - pW + kw
                                                        if iw_val >= 0:
                                                            if iw_val < W:
                                                                in_row = n_idx * in_spatial + id_val * HW + ih_val * W + iw_val
                                                                T.copy(X[in_row, 0], inp_ub)
                                                                T.tile.add(acc_ub, acc_ub, inp_ub)

                    if divisor_override > 0:
                        T.tile.mul(out_ub, acc_ub, T.float32(1.0 / divisor_override))
                    else:
                        T.tile.mul(out_ub, acc_ub, T.float32(1.0 / pool_size))

                    T.copy(out_ub, Y[out_row, 0])

    @T.prim_func
    def multi_w(
        X: T.Tensor((N_batch * in_spatial, C), dtype),
        Y: T.Tensor((M_out, C), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_M + vid * sub_block_M

            acc_ub = T.alloc_ub((1, C), dtype)
            inp_ub = T.alloc_ub((1, C), dtype)
            out_ub = T.alloc_ub((1, C), dtype)

            with T.Scope("V"):
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
                                        T.tile.fill(acc_ub, T.float32(0.0))

                                        for kd in T.serial(kD):
                                            id_val = od * sD - pD + kd
                                            if id_val >= 0:
                                                if id_val < D:
                                                    for kh in T.serial(kH):
                                                        ih_val = oh * sH - pH + kh
                                                        if ih_val >= 0:
                                                            if ih_val < H:
                                                                for kw in T.serial(kW):
                                                                    iw_val = cur_ow * sW - pW + kw
                                                                    if iw_val >= 0:
                                                                        if iw_val < W:
                                                                            in_row = n_idx * in_spatial + id_val * HW + ih_val * W + iw_val
                                                                            T.copy(X[in_row, 0], inp_ub)
                                                                            T.tile.add(acc_ub, acc_ub, inp_ub)

                                        if divisor_override > 0:
                                            T.tile.mul(out_ub, acc_ub, T.float32(1.0 / divisor_override))
                                        elif count_include_pad > 0:
                                            T.tile.mul(out_ub, acc_ub, T.float32(1.0 / pool_size))
                                        else:
                                            d_start = od * sD - pD
                                            h_start = oh * sH - pH
                                            w_start = cur_ow * sW - pW

                                            d_begin = d_start
                                            h_begin = h_start
                                            w_begin = w_start
                                            if d_begin < 0:
                                                d_begin = 0
                                            if h_begin < 0:
                                                h_begin = 0
                                            if w_begin < 0:
                                                w_begin = 0

                                            d_end = d_start + kD
                                            h_end = h_start + kH
                                            w_end = w_start + kW

                                            if d_end > D:
                                                d_end = D
                                            if h_end > H:
                                                h_end = H
                                            if w_end > W:
                                                w_end = W

                                            valid_d = d_end - d_begin
                                            valid_h = h_end - h_begin
                                            valid_w = w_end - w_begin
                                            if valid_d < 0:
                                                valid_d = 0
                                            if valid_h < 0:
                                                valid_h = 0
                                            if valid_w < 0:
                                                valid_w = 0

                                            valid_pool = valid_d * valid_h * valid_w
                                            if valid_pool <= 0:
                                                T.tile.fill(out_ub, T.float32(0.0))
                                            else:
                                                T.tile.mul(out_ub, acc_ub, T.float32(1.0 / valid_pool))

                                        T.copy(out_ub, Y[out_row + local_w, 0])
                        elif m == 0:
                            prefix_len = multi_w_window - (ow % multi_w_window)
                            for local_w in T.serial(multi_w_window):
                                if local_w < prefix_len:
                                    if local_w < sub_block_M - m:
                                        if local_w < OW - ow:
                                            cur_ow = ow + local_w
                                            T.tile.fill(acc_ub, T.float32(0.0))

                                            for kd in T.serial(kD):
                                                id_val = od * sD - pD + kd
                                                if id_val >= 0:
                                                    if id_val < D:
                                                        for kh in T.serial(kH):
                                                            ih_val = oh * sH - pH + kh
                                                            if ih_val >= 0:
                                                                if ih_val < H:
                                                                    for kw in T.serial(kW):
                                                                        iw_val = cur_ow * sW - pW + kw
                                                                        if iw_val >= 0:
                                                                            if iw_val < W:
                                                                                in_row = n_idx * in_spatial + id_val * HW + ih_val * W + iw_val
                                                                                T.copy(X[in_row, 0], inp_ub)
                                                                                T.tile.add(acc_ub, acc_ub, inp_ub)

                                            if divisor_override > 0:
                                                T.tile.mul(out_ub, acc_ub, T.float32(1.0 / divisor_override))
                                            elif count_include_pad > 0:
                                                T.tile.mul(out_ub, acc_ub, T.float32(1.0 / pool_size))
                                            else:
                                                d_start = od * sD - pD
                                                h_start = oh * sH - pH
                                                w_start = cur_ow * sW - pW

                                                d_begin = d_start
                                                h_begin = h_start
                                                w_begin = w_start
                                                if d_begin < 0:
                                                    d_begin = 0
                                                if h_begin < 0:
                                                    h_begin = 0
                                                if w_begin < 0:
                                                    w_begin = 0

                                                d_end = d_start + kD
                                                h_end = h_start + kH
                                                w_end = w_start + kW

                                                if d_end > D:
                                                    d_end = D
                                                if h_end > H:
                                                    h_end = H
                                                if w_end > W:
                                                    w_end = W

                                                valid_d = d_end - d_begin
                                                valid_h = h_end - h_begin
                                                valid_w = w_end - w_begin
                                                if valid_d < 0:
                                                    valid_d = 0
                                                if valid_h < 0:
                                                    valid_h = 0
                                                if valid_w < 0:
                                                    valid_w = 0

                                                valid_pool = valid_d * valid_h * valid_w
                                                if valid_pool <= 0:
                                                    T.tile.fill(out_ub, T.float32(0.0))
                                                else:
                                                    T.tile.mul(out_ub, acc_ub, T.float32(1.0 / valid_pool))

                                            T.copy(out_ub, Y[out_row + local_w, 0])

    split_c_ready = block_C > 0 and C % block_C == 0

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
    return generic_3d
