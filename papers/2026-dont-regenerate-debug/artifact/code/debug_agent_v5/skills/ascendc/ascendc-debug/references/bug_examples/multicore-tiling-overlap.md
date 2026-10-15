# 案例：多核 tiling 区间重叠

## 现象

- 小 shape（单核处理）通过
- 大 shape（多核处理）失败
- 实验 A（blockDim=1）临时改单核后通过

## 误差分析输出

```
shape=(65536,)  dtype=torch.float32  block_dim=40
错误元素: 1638/65536 (2.50%)
首错线性下标: 1638  多维坐标: [1638]
错误间隔: [1] → 从首错起连续错误到 3276
```

## 根因

op_host 的 tiling 计算中，formerLength 和 tailLength 的计算有 off-by-one，导致相邻核的 GM 区间有重叠，后写入的核覆盖了先写入核的正确结果。

```cpp
// ❌ 错误：formerLength 向上取整导致溢出
int64_t formerLength = (totalLength + coreNum - 1) / coreNum;
int64_t tailLength   = totalLength - formerLength * (coreNum - 1);
// 当 totalLength=65536, coreNum=40 → formerLength=1639, 1639*40=65560 > 65536

// ✅ 正确
int64_t formerLength = totalLength / coreNum;
int64_t remainder    = totalLength % coreNum;
// 前 remainder 个核处理 formerLength+1，其余处理 formerLength
```

## 定位关键

1. 实验 A：blockDim=1 过 → 确认核间问题
2. 首错下标 1638 ≈ totalLength/coreNum → 位于核边界处（tiling 反推验证）
3. Sub-step 2.3 核间边界检查：formerNum × formerLength + tailNum × tailLength ≠ totalLength → 区间不正确
