# Flash Attention 计算分解示例

## 算子信息
- op_name: flash_attention
- category: attention
- 计算模式: **多阶段 C/V 融合 (Online Softmax + Tiled MatMul)**
- 输入 shape: Q/K/V [batch, heads, seq_len, dim], dtype: float16
- 输出 shape: [batch, heads, seq_len, dim], dtype: float16
- 内部累加精度: float32

## 参考实现来源
- reference: `scaled_dot_product_attention(Q, K, V)` 即 `softmax(QK^T / sqrt(d)) × V`
- 参考 TileLang 设计: `archive_tasks/flash_attention/design/tile_level/flash_attention.py`
- 参考实现: 无 AscendC kernel/（仅 TileLang 设计）

## 计算链分解

### Step 0: 输入
- Q: shape [batch, heads, seq_len, dim], dtype: float16
- K: shape [batch, heads, seq_len, dim], dtype: float16
- V: shape [batch, heads, seq_len, dim], dtype: float16
- sm_scale = 1 / sqrt(dim)
- block_M = 64, block_N = 64 (分块大小)
- kv_loops = ceil(seq_len / block_N) (KV 维度迭代次数)

### Step 1: QK^T 矩阵乘 (Cube 侧)
- 操作: `S = Q_block × K_block^T` (矩阵乘, shape [block_M, block_N])
- 输入: Q[block_M, dim] × K[block_N, dim]^T
- 输出 shape: [block_M, block_N], dtype: float32 (Cube 输出到 L0C)
- **精度风险点**:
  - Cube 运算 (gemm_v0), 硬件精度由 accumulation dtype 决定
  - transpose_B=True, K 需要转置后相乘
  - 结果通过 Fixpipe 从 L0C 写出到 workspace, 需要 cross-core flag 同步

### Step 2: Scale + Online ReduceMax (Vector 侧)
- 操作: `S = S * sm_scale`, `m_new = max(m_old, rowmax(S))`
- 输入: Step 1 的 S (从 workspace 读取) + 上一轮的 m_old
- 输出: 更新后的 m_i [block_M/2], 更新后的 S [block_M/2, block_N]
- **精度风险点**:
  - **cross-core 同步**: Vector 必须等 Cube 写完 workspace 后才能读取 (wait_cross_flag)
  - sm_scale 乘法必须在 softmax 计算之前
  - **online max 更新**: `m_new = max(m_old, rowmax(S_current))`, 必须保留全局 max, 否则 exp 溢出
  - m_i 初始化为 -2^30 (足够小的负数), 确保第一个 tile 的 max 被正确捕获

### Step 3: Online Softmax 归一化 (Vector 侧)
- 操作:
  1. `alpha = exp(m_old - m_new)` (旧累加值的修正因子)
  2. `P = exp(S - m_new)` (当前 tile 的 softmax 分子)
  3. `sumexp = sumexp * alpha + sum(P)` (更新分母)
- 输入: Step 2 的 S 和 m_i + 上一轮的 sumexp
- 输出: P [block_M/2, block_N] (softmax 概率), 更新后的 sumexp [block_M/2]
- **精度风险点**:
  - **alpha 修正遗漏**: 若不乘 `exp(m_old - m_new)`, 旧 sumexp 基于旧 max, 与新 exp 值不一致
  - exp 计算: S - m_new ≤ 0, 所以 exp 值 ∈ (0, 1], 不会溢出
  - 若 m_new 计算错误 (如 padding 抬高), exp(S - m_new) 中部分值 > 1 → 溢出风险
  - P 需要 cast 为 float16 后传给 Cube 做 PV 矩阵乘

### Step 4: PV 矩阵乘 + Output 修正 (Cube + Vector)
- 操作:
  1. Cube: `O_new = P × V_block` (矩阵乘, shape [block_M, dim])
  2. Vector: `acc_o = acc_o * alpha + O_new` (用 alpha 修正旧累加值后加上新结果)
- 输入: Step 3 的 P + V[block_N, dim] + 上一轮的 acc_o
- 输出: 更新后的 acc_o [block_M/2, dim]
- **精度风险点**:
  - **acc_o 修正**: 必须先乘 alpha (= exp(m_old - m_new)) 再加 O_new
  - 若修正遗漏, acc_o 中旧 tile 的贡献基于旧 max, 与新 tile 不可加 → 输出系统性偏大
  - Cube 侧的 V 加载需要与 Vector 侧的 softmax 计算流水对齐 (prelaunch)

### Step 5: 最终归一化
- 操作: `output = acc_o / sumexp` (逐行除以归一化常数)
- 输入: 所有 KV tile 遍历完成后的 acc_o [block_M, dim] + sumexp [block_M]
- 输出 shape: [block_M, dim], dtype: float16 (cast 后写回 GM)
- **精度风险点**:
  - sumexp 是所有 tile 的 exp sum 的修正累加值, 必须与最终 acc_o 对应同一个 max
  - 若中间某个 tile 的 alpha 修正遗漏, sumexp 与 acc_o 不匹配 → 输出整体偏差
  - float32 → float16 cast 引入精度截断

## 误差传播链

```
Step 2 m_i (row max) 错误 (padding 抬高 / tile 遗漏)
  → Step 3 exp(S - m_new) 溢出或 sumexp 偏大
    → Step 4 alpha 修正因子错误, acc_o 累加不一致
      → Step 5 output / sumexp 不匹配
        → 输出系统性偏大或 NaN
典型 pattern: all_wrong 或 partial_wrong (按 KV tile 边界分布)

Step 3 alpha 修正遗漏 (sumexp 或 acc_o 未乘 alpha)
  → 旧 tile 贡献未修正
    → 前几个 tile 权重过大, 后几个 tile 权重过小
      → 输出偏向早期 KV 内容
典型 pattern: dimension_concentration (序列位置相关)

Step 1→2 cross-core 同步缺失
  → Vector 读到 Cube 未写完的脏数据
    → S 值随机错误 → softmax 完全错误
典型 pattern: all_wrong (随机)
```

## Tiling 策略要点

- `block_M = 64, block_N = 64` (Q 和 KV 的分块大小)
- block_num = seq_len/block_M × heads × batch (总任务数, 按 Cube core 分配)
- 每个 block 处理 Q 的 block_M 行, 遍历所有 KV tile
- prelaunch = 2 (流水深度), ring_slots = 3 (workspace 环形缓冲)
- vec_num = 2 (每 block 1 AIC + 2 AIV)
- workspace: 4 个 GM workspace (S_float32, S_float16, O_float32, meta) 用于 C/V 跨核通信
- C/V 同步通过 `set_cross_flag("FIX", idx)` / `wait_cross_flag(idx)` 实现

## 与 AscendC Kernel 的对照要点

1. **cross-core 同步**: Cube 写 workspace 后是否 SetCrossCoreFlag? Vector 是否 WaitCrossCoreFlag?
2. **sm_scale**: 是否在 softmax 前乘了 1/sqrt(dim)?
3. **m_i 初始化**: 是否初始化为足够小的负数 (-2^30 或 -INFINITY)?
4. **alpha 修正**: sumexp 和 acc_o 是否都乘了 exp(m_old - m_new)?
5. **最终归一化**: 所有 KV tile 遍历完成后 acc_o 是否除以 sumexp?
6. **workspace 环形缓冲**: ring_slots 是否 = prelaunch + 1? 读写 slot index 是否正确?
7. **P 的 dtype cast**: softmax 概率 P 是否 cast 为 float16 后再传给 Cube 做 PV 矩阵乘?

## AscendC 通用实现约束

> DataCopyPad / TBuf/TQue / work_buf 初始化 / SyncAll / Fixpipe 同步等通用约束
> 参见 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_cv.md`
> 和 `dsl2Ascendc.md`（第四章常见陷阱速查表）。
> C/V 融合算子的跨核同步约束参见 `dsl2Ascendc_cross_core_sync.md`。
