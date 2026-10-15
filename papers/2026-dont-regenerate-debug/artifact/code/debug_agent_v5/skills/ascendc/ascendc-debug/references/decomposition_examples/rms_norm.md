# RMSNorm 计算分解示例

## 算子信息
- op_name: rms_norm
- category: normalization
- 计算模式: **单行归约 (单 pass)**
- 归约维度: dim=-1 (最后一维, features 维)
- 输入 shape: x [M, N], gamma [N], dtype: float32 (支持 float16/bfloat16, 内部 float32 累加)
- 输出 shape: y [M, N], inv_rms [M], dtype: 同输入

## 参考实现来源
- reference: `y = x * gamma / sqrt(mean(x^2) + eps)`
- 参考 TileLang 设计: `archive_tasks/rms_norm/design/tile_level/rms_norm.py`
- 参考 AscendC kernel: `archive_tasks/rms_norm/kernel/`

## 计算链分解

### Step 0: 输入
- x: shape [M, N], dtype: float32
- gamma: shape [N], dtype: float32 (per-element 缩放权重)
- eps: 1e-5
- 数值范围: 取决于输入, 通常 torch.rand → [0, 1)

### Step 1: Square
- 操作: `x_sq = x * x` (逐元素)
- 输入: Step 0 的 x, shape [M, N]
- 输出 shape: [M, N]
- 数值范围预期: [0, 1) (输入 [0,1) 的平方)
- **精度风险点**:
  - 逐元素乘法, 精度安全
  - Padding 区域的平方值: 若 padding 为 0, x_sq padding 也为 0 → 对后续 ReduceSum 安全

### Step 2: ReduceSum → Mean Square
- 操作: `sum_sq = sum(x_sq, dim=-1)`, `ms = sum_sq / N`
- 输入: Step 1 的 x_sq, shape [M, N]
- 输出 shape: [M, 1] (每行一个标量)
- 数值范围预期: ms ≈ 1/3 (uniform [0,1) 的 E[x²] = 1/3)
- **精度风险点**:
  - ReduceSum count 必须 64 倍数对齐, N 不满足时 padding 区域必须为 0
  - **分母必须是 N (原始维度长度), 非对齐后的 count**
  - TileLang 中使用 `inv_n_const = 1.0 / N` 乘法替代除法
  - 多 tile 场景 (N > block_N) 需要跨 tile 累加 partial sum

### Step 3: Add eps → Rsqrt → inv_rms
- 操作: `inv_rms = rsqrt(ms + eps) = 1 / sqrt(ms + eps)`
- 输入: Step 2 的 ms + eps=1e-5
- 输出 shape: [M, 1] (每行一个标量)
- 数值范围预期: ≈ 1.73 (1/sqrt(1/3 + 1e-5))
- **精度风险点**:
  - **eps 必须在 rsqrt 之前加** (即 rsqrt(ms + eps)), 而非 rsqrt(ms) + eps
  - TileLang 使用 `T.tile.rsqrt` 一步完成, AscendC 中可能需要 `Sqrt` + `Reciprocal` 或等效 API
  - 若 ms 被 Padding 污染偏大, inv_rms 偏小, 输出整体偏小

### Step 4: Broadcast inv_rms → Mul x
- 操作: `normalized = x * inv_rms` (广播乘法, inv_rms 从 [M,1] 广播到 [M,N])
- 输入: Step 0 的 x + Step 3 的 inv_rms
- 输出 shape: [M, N]
- 数值范围预期: 取决于 inv_rms, 通常 x × 1.73 ≈ [0, 1.73)
- **精度风险点**:
  - 广播: AscendC 中需要 Duplicate 或 Broadcast 将标量/列向量复制到行宽度
  - TileLang 使用 `T.tile.broadcast(rstd_broad_ub, inv_rms_ub, tmp)` 完成

### Step 5: Mul gamma → output
- 操作: `y = normalized * gamma` (逐元素乘法)
- 输入: Step 4 的 normalized + gamma [N]
- 输出 shape: [M, N]
- 数值范围预期: 与 normalized 相同 (gamma 初始值通常为全 1)
- **精度风险点**:
  - gamma 是 per-element 的 (shape [N]), 需要正确的 GM 偏移
  - gamma 在所有行之间共享, GM 偏移不含 row_idx
  - gamma 可在 Init() 中一次性加载到 UB 缓存, 所有行复用

## 误差传播链

```
Step 2 Padding 污染 sum_sq (padding 非 0 参与求和)
  → ms 偏大
    → Step 3 inv_rms 偏小
      → Step 4 normalized 整体偏小
        → Step 5 输出整体偏小
典型 pattern: uniform_offset (系统性偏小)

Step 2 分母错误 (用 count 代替 N)
  → ms 偏小 (分母偏大)
    → inv_rms 偏大
      → 输出偏大
典型 pattern: uniform_offset (系统性偏大)
```

## Tiling 策略要点

- `n_cores = 20`, 按行块分配: `block_M = 64`, `tasks_per_core = ceil(m_num / n_cores)`
- `block_N = 1024`, `n_num = ceil(N / block_N)` — merge_n 要求 N ≤ 1024 (单 tile)
- vec_num = 2 (每 block 内 AIC/AIV 分工), sub_block_M = block_M / 2 = 32
- row_factor = 8 (每次处理 8 行)
- gamma 在 Init 阶段一次性加载, 所有行共享
- 3 个 prim_func 按 N 大小路由: merge_n (N≤1024), single_row (1024<N≤8192), splitd (N>8192)

## 与 AscendC Kernel 的对照要点

1. ReduceSum 的 count 是否 64 倍数对齐? 分母是否 = N (不是 count)?
2. ReduceSum 前 work_buf 是否 `Duplicate(0.0f, count)` 初始化?
3. eps 是否在 `rsqrt(ms + eps)` 中的 sqrt 之前加?
4. gamma 的 GM 偏移是否为 `col_start` (per-element, 不含 row_idx)?
5. gamma 是否一次性加载复用, 而非每行重复加载?
6. inv_rms 广播到 [M, N] 时是否使用了正确的 Broadcast/Duplicate?
7. 多 prim_func (merge_n / single_row / splitd) 是否对应多个 kernel 入口?

## AscendC 通用实现约束

> DataCopyPad / TBuf/TQue / work_buf 初始化 / SyncAll 等通用约束
> 参见 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md`
> 和 `dsl2Ascendc.md`（第四章常见陷阱速查表）。
