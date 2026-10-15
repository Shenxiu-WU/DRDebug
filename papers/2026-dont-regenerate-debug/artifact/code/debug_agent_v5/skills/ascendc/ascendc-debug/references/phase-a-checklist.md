# Phase A/C 模板：[REFERENCE_IMPL_SPEC] 与 [KERNEL_STEP_TRACE]

> **读取时机**：进入 Sub-step 2.2 填写 `[REFERENCE_IMPL_SPEC]` 之前（同时获取 Sub-step 2.3 Phase C 所需的 `[KERNEL_STEP_TRACE]` 模板），必须 Read 本文件。

---

## [REFERENCE_IMPL_SPEC] 格式模板

从参考实现与文档中提取并填写如下规范：

```
[REFERENCE_IMPL_SPEC]
  参考实现来源: <lowering example 文件路径>

  TQue/TBuf 分配规范 (来自参考实现):
    - inQueue (TQue<VECIN>): <用途, DataCopy GM→UB 的目标 buffer>
    - outQueue (TQue<VECOUT>): <用途, DataCopy UB→GM 的源 buffer, 必须经 EnQue/DeQue>
    - TBuf (VECCALC): <用途, 中间计算 buffer, 不可直接作为 DataCopy src/dst>
    - TBuf→GM 正确路径: TBuf计算结果 → outQueue.AllocTensor → 写入outLocal → EnQue → DeQue → DataCopy

  关键 API 规范 (来自参考实现):
    - ReduceMax: <调用签名; work_buf 是否需要 Duplicate(-3.402823466e+38f, count) 初始化>
    - ReduceSum: <count 对齐要求 (64的倍数); work_buf 是否需要 Duplicate(0.0f, count) 初始化>
    - SyncAll: <是否需要, 插入位置 (跨核写GM后/读GM前)>

  非对齐处理规范 (来自 dsl2Ascendc_compute_vector.md):
    - 触发条件: count × sizeof(dtype) 不是 32 的倍数
    - GM→UB: DataCopyPad(dst, src, {1, count*sizeof(T), 0, 0}, {false, 0, 0, 0})
    - UB→GM: DataCopyPad(dst, src, {1, count*sizeof(T), 0, 0})
    - 本算子 tileLength 的对齐状况: <tileLength × sizeof(dtype) = ? 字节, 是否32字节对齐>

  error_correction 禁用模式 (来自 dsl2Ascendc.md §常见陷阱速查表):
    - 禁止 float↔uint 强制类型转换 (改用 float n = (float)int_val)
    - 禁止标量上下文调用向量 Log API (改用 AscendC::Log(tmp,tmp,1); tmp.GetValue(0))
    - 禁止向量上下文调用标量 AscendC::Sqrt (改用 sqrt(val))
    - 本算子代码中是否出现上述模式: <逐一检查>
```

---

## [KERNEL_STEP_TRACE] 格式模板

```
[KERNEL_STEP_TRACE]
  Kernel 计算步骤 (从 Compute() 函数提取):
    K-Step 1: <AscendC API 名称>
      - 代码位置: <kernel 文件名>.cpp 第 <line> 行
      - 参数: count=<value>, src=<buffer>, dst=<buffer>
      - 对应计算链: Step <N> (<operation_name>)
      - 匹配状态: ✅ 匹配 / ⚠️ 参数偏差: <描述> / ❌ 缺失或多余
      - 实测中间值 (来自 L5_PROBE): <值，如 "P2 后 out[0]=0.5312" / N/A（此阶段未被探针覆盖）>

    K-Step 2: ...
    ...

  Host tiling 参数:
    - TilingData 结构体字段: <从 _tiling.h 中列出所有字段名和类型>
    - tileLength = <值> (来源: pybind11.cpp TilingFunc 第 <line> 行)
    - 其他 TilingData 字段: <列出 field=value>
    - 归约维度完整性: tileLength <>=<> 归约轴长度 <length> → 完整 / 被切分

  跨核通信验证: (仅跨核归约模式)
    - workspace buffer: GM 中是否分配, 大小是否 = n_cores
    - 各核写入: DataCopy 后是否有同步
    - Core 0 读取: 是否在所有核完成后才读取
    - 全局归约: ReduceSum 的 count 是否正确 (= n_cores, 而非 tile_size)
    - 最终除法: 分母是否 = total_elems

  算子类型专项检查 (根据 L8 op_type 选择对应项):

    [Pooling 类] DataCopy 维度一致性:
      - 输入内存布局: <NCDHW / NCHW / NHWC, 来自 L6>
      - tileC 含义: <沿 C 维度的 tile 大小>
      - DataCopy count=tileC 读取的是: <C 维度 tileC 个通道 还是 W 维度 tileC 个元素?>
      - C 维度在内存中的 stride: <C_stride = D*H*W (NCDHW) / H*W (NCHW)>
      - ⚠️ 检查: tileC 个连续地址是否真的对应 tileC 个通道? 若 C_stride > 1, 连续地址实为沿 W/空间维读取
      - input base offset 公式: <写出 b/c0/d/h/w 各维度的 offset 计算, 标出 c0 的系数是否为 C_stride>
      - output base offset 公式: <写出 b/c0/od/oh/ow 各维度的 offset 计算, 标出 c0 的系数>
      - ⚠️ 检查: outBase 中 c0 的系数是否为 outD*outH*outW (正确) 而非 1 (错误)

    [Reduction / Normalization 类] 工作 Buffer 初始化:
      - ReduceMax work buffer: <调用前是否 Duplicate(work, -INF, count) 初始化?>
      - ReduceSum work buffer: <调用前是否 Duplicate(work, 0, count) 初始化?>
      - ⚠️ 检查: work buffer 是否从上一步骤残留了非零数据 (如 ReduceMax work buffer 含有上一步 maxVal 残留)
      - 负无穷写法: <代码中使用 -3.402823466e+38f / (float)(-INFINITY) / -65504.0f (float16 错误!)>

    [MatMul / 分块累加类] 累加器初始化:
      - 累加器 (acc buffer) 初始化位置: <在外层循环前 Duplicate(0) / 未初始化>
      - ⚠️ 检查: 多个 tile 间累加器是否在每个输出位置开始时被正确重置

    [TQue / TBuf 数据流] 同步验证 (所有算子类型必填):
      - inQueue 流程: AllocTensor → DataCopy(GM→UB) → EnQue → DeQue → (计算) → FreeTensor ✅/❌
      - outQueue 流程: AllocTensor → (计算写入) → EnQue → DeQue → DataCopy(UB→GM) → FreeTensor ✅/❌
      - TBuf 用途: <VECCALC 中间计算, 不参与 DMA 传输>
      - ⚠️ 严重: TBuf.Get() 直接作为 DataCopy dst 写 GM = 绕过 outQueue 同步 = 数据未写出 = 输出全零
      - ⚠️ 检查: CopyOut 函数中 maxLocal/accLocal 等 TBuf 变量是否直接用于 DataCopy(outputGm[], ...)

  步骤对齐结论:
    - 全部匹配: 是 / 否
    - 缺失步骤: <列出, 或 "无">
    - 参数偏差: <列出, 或 "无">
    - 新增/多余步骤: <列出, 或 "无">

  L7 代码位置映射 (手动):
    - worst element index=<index> → 对应 kernel 中的 <函数/代码块>
    - 该元素位于 main block / tail block?
    - 对应的 K-Step: <编号>
    - 对应的 L5_PROBE 探针阶段: <P1/P2/P3，或"探针未覆盖">
```
