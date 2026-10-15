# Softmax 计算分解示例

## 算子信息
- op_name: softmax
- category: activation (含归约)
- 计算模式: **单行归约**
- 归约维度: dim=-1 (最后一维, 即 features 维)
- 输入 shape: [32, 768], dtype: float32
- 输出 shape: [32, 768], dtype: float32

## 参考实现来源
- reference: `F.softmax(x, dim=self.dim)`
- 参考 TileLang 设计: `archive_tasks/flash_attention/model_new_tilelang.py`（含 softmax 子算法）

## 计算链分解

### Step 0: 输入
- tensor: x
- shape: [32, 768]
- dtype: float32
- 数值范围: 取决于 `get_inputs()` 中的 `torch.rand`, 通常 [0, 1)

### Step 1: ReduceMax (数值稳定性)
- 操作: `row_max = max(x, dim=-1, keepdim=True)`
- 输入: Step 0 的输出, shape [32, 768]
- 输出 shape: [32, 1] (每行一个标量)
- 数值范围预期: [0, 1) (因为输入是 torch.rand)
- **精度风险点**:
  - Padding 值为 0 时参与 Max: 若输入全负则 max 被错误抬高到 0
  - ReduceMax 的 count 参数必须为 64 的倍数, 不对齐时硬件行为未定义
  - Buffer 未用 Duplicate(-INF) 初始化时, padding 区域的默认值可能干扰结果

### Step 2: Sub (去中心化)
- 操作: `x_shifted = x - row_max` (广播减法)
- 输入: Step 0 的 x + Step 1 的 row_max
- 输出 shape: [32, 768]
- 数值范围预期: (-∞, 0] (最大值变为 0, 其余为负)
- **精度风险点**:
  - 如果 Step 1 的 max 不正确 (如被 padding 抬高), x_shifted 中将出现正值
  - 正值会导致 Step 3 的 Exp 溢出
  - 此步本身不引入精度误差, 但会传播 Step 1 的误差

### Step 3: Exp
- 操作: `exp_val = exp(x_shifted)`
- 输入: Step 2 的 x_shifted, shape [32, 768]
- 输出 shape: [32, 768]
- 数值范围预期: (0, 1] (因为 x_shifted ≤ 0, exp(0)=1, exp(负数)<1)
- **精度风险点**:
  - 若 Step 2 输入有正值 (Step 1 错误导致), exp 可能溢出 (float32 下 exp(88) ≈ 1.65e38)
  - **Padding 区域**: exp(0) = 1 (若 padding 为 0), 这些虚假的 1 会污染后续 ReduceSum
  - 若 padding 初始化为 -INF (ReduceMax 正确做法), 则 Sub 后 padding 仍为 -INF, exp(-INF)=0, 不污染 — 这是理想情况

### Step 4: ReduceSum
- 操作: `exp_sum = sum(exp_val, dim=-1, keepdim=True)`
- 输入: Step 3 的 exp_val, shape [32, 768]
- 输出 shape: [32, 1] (每行一个标量)
- 数值范围预期: (0, 768] (每行 exp 值之和, 最大 768 × 1 = 768)
- **精度风险点**:
  - Padding 区域 exp(0)=1 会增大 sum, 导致最终 softmax 结果偏小
  - ReduceSum 的 count 参数同样要求 64 倍数对齐
  - Padding 区域对 Sum 的语义: 正确值应为 0 (即 padding 元素不贡献到 sum)

### Step 5: Div (归一化)
- 操作: `output = exp_val / exp_sum` (广播除法)
- 输入: Step 3 的 exp_val + Step 4 的 exp_sum
- 输出 shape: [32, 768]
- 数值范围预期: [0, 1], 每行和为 1
- **精度风险点**:
  - 若 exp_sum 接近 0 (不应该在正常情况发生, 但 NaN 输入可能导致)
  - 若 exp_sum 被 padding 污染而偏大, 所有输出值都会偏小 → uniform_offset 模式

## 误差传播链

```
Step 1 (Max) 错误
  → Step 2 (Sub) 所有元素偏移不正确
    → Step 3 (Exp) 部分值溢出或偏大
      → Step 4 (Sum) 值偏大
        → Step 5 (Div) 所有值偏小
```

典型表现: pattern = uniform_offset 或 all_wrong

## Tiling 策略要点

- `n_cores = 32`, 按行分配 (`rows_per_core = rows // n_cores`)
- `tile_length = cols = 768` (整行放入 UB, 无需多 tile)
- 归约维 (dim=-1) 在单核内完整处理, **不跨核切分**
- 4 个 UB buffer: row_ub, exp_ub, shared_ub, out_ub

## 与 AscendC Kernel 的对照要点

1. Compute() 中 ReduceMax 的 count 参数是否 = tileLength (而非动态 actualLength)
2. ReduceMax 前 buffer 是否用 Duplicate(-INF) 初始化
3. Exp 前 Sub 的标量来源是否正确 (从 sharedLocal.GetValue(0) 提取)
4. ReduceSum 前 padding 区域是否已被正确初始化为 0
5. ReduceSum 的 count 参数是否与 ReduceMax 一致
6. Div 的除数来源是否正确 (sharedLocal.GetValue(0) 在 ReduceSum 之后)
7. Host TilingFunc 中 tileLength 是否 = cols (保持归约维完整)

## AscendC 通用实现约束

> DataCopyPad / TBuf/TQue / work_buf 初始化 / SyncAll 等通用约束
> 参见 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md`
> 和 `dsl2Ascendc.md`（第四章常见陷阱速查表）。
