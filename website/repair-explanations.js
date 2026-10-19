/* Explanations for the constructed CPU examples, not reports of NPU runs. */
(() => {
  "use strict";
  const explanations = {
    row_stride: {
      what: [
        "行起点决定读取哪一组数据。误把行数当作每行长度，仍会读到合法数字，但内容已错。",
        "A row's offset determines where its data starts in memory. Confusing the row count with the row length still loads valid numbers, but they belong to the wrong input."
      ],
      where: [
        "第 2 行 stride=rows 从索引 2 读到 [2,2,1]；改用 cols，应从索引 3 读取 [2,1,0]。",
        "On line 2, stride=rows starts the second row at index 2 and loads [2, 2, 1]. stride=cols starts at index 3 and loads the intended [2, 1, 0]."
      ],
      impact: [
        "A、B 错误地各占约 42.2%；修复后恢复约 66.5% 和 24.5%，完整输出匹配参考。",
        "A and B incorrectly receive about 42.2% each. Correcting the offset restores about 66.5% and 24.5%, and the full output matches the reference."
      ]
    },
    mask_broadcast: {
      what: [
        "掩码规定哪些位置可以获得权重。每行的有效位置可能不同，不能共用第一行的规则。",
        "The mask defines which positions may receive weight. Different batch rows may have different valid positions, so one row's mask cannot stand in for the whole batch."
      ],
      where: [
        "第 4 行 valid[0] 将 A、B 掩码套到第二行，而该行实际只允许 A、C；改用完整 valid 逐行匹配。",
        "On line 4, valid[0] broadcasts row 1's A/B mask into row 2, which actually allows A/C. Using the full valid mask preserves each row's own rules."
      ],
      impact: [
        "无效 B 获得约 26.9%，C 却被清零；修复后 B 为 0%，C 恢复约 11.9%。",
        "Invalid B receives about 26.9% while C is suppressed. After repair, B receives 0% and C about 11.9%, restoring weight to the correct positions."
      ]
    },
    padding_mask: {
      what: [
        "补齐空位不是真实内容，权重应为 0。但将分数设为 0，仍会产生 exp(0)=1，参与归一化。",
        "Padding is not real content and should receive zero weight. Replacing its score with zero does not exclude it: exp(0)=1 still enters normalization."
      ],
      where: [
        "第 4 行 where(valid,x,0) 未排除空位；改填 −INF，让其指数为 0，不再计入分母。",
        "On line 4, where(valid, x, 0) leaves padding with a participating score. Replacing zero with −INF makes its exponential zero and excludes it from the denominator."
      ],
      impact: [
        "真实分数为 −2、−3、−4 时，空位抢走约 83.1%；修复后空位为 0%，真实位置恢复正确权重。",
        "With real scores −2, −3 and −4, padding steals about 83.1% of the weight. After repair it receives 0%, and the real positions receive about 66.5%, 24.5% and 9.0%."
      ]
    },
    reduction_axis: {
      what: [
        "每行应独立归一化。沿错的维度求和，会混合独立输入，让同一行的各位置除以不同的跨行总和。",
        "Softmax normalizes each row independently. Summing the wrong axis mixes separate inputs and divides each position by a different cross-row total instead of one row's denominator."
      ],
      where: [
        "第 8 行 axis=0 混合两行同列的 e；改为 axis=−1，汇总本行全部指数作为共同分母。",
        "On line 8, axis=0 combines e from the same column across the batch. axis=−1 sums the final dimension, providing one denominator for all positions in that row."
      ],
      impact: [
        "第二行权重之和约为 150%；修复后约为 100%，各位置也匹配逐行计算的独立参考。",
        "The second row's weights total about 150%, exposing incorrect normalization. After repair they total about 100%, with every position matching the independent row-wise reference."
      ]
    },
    tile_local_sum: {
      what: [
        "长行可以分块，但分母必须覆盖整行。每块若分别归一化，都会分配完整的 100% 权重。",
        "Long rows may be processed in tiles, but the denominator must cover the whole row. Normalizing each tile separately allocates a full 100% of weight per tile."
      ],
      where: [
        "第 8 行 sum_per_tile 只算两项的局部分母；改为汇总整行 e，保留原有全行最大值。",
        "On line 8, sum_per_tile(e, tile=2) forms a separate denominator for each pair. This case keeps the correct global maximum and replaces only the local denominators with a whole-row sum of e."
      ],
      impact: [
        "两个分块各为 100%，整行达到 200%；合并分母后整行约为 100%，块间比例也恢复正确。",
        "Both tiles independently total 100%, so the row totals 200%. Merging the denominator restores about 100% overall and the correct relative weights between the two tiles."
      ]
    },
    sum_rounding: {
      what: [
        "4,096 个指数均为 1。但逐次 FP16 舍入时，2,048+1 会舍回 2,048，分母停止增长。",
        "An insufficient accumulator precision can lose small increments. All 4,096 exponentials equal 1, but sequential FP16 rounding makes 2,048 + 1 round back to 2,048."
      ],
      where: [
        "第 9 行 accumulator=FP16 丢失增量；改为 FP32 累加，输出仍为 FP16。",
        "On line 9, accumulator=FP16 rounds each addition at low precision. Using FP32 improves the reduction's precision while leaving the final output in FP16."
      ],
      impact: [
        "分母少一半使每项权重翻倍，整行总和为 200%；修复后分母达到 4,096，总和为 100%。",
        "The wrong denominator is 2,048 instead of 4,096, doubling every weight and producing a 200% total. The repaired denominator reaches 4,096 and the total returns to 100%."
      ]
    },
    exp_overflow: {
      what: [
        "有限分数的指数也会溢出。exp(90) 超出 FP32 范围，Inf/Inf 使输出出现 NaN。",
        "Finite scores can still produce unrepresentable exponentials. exp(90) exceeds FP32 range; an infinite numerator and denominator then yield NaN through Inf/Inf."
      ],
      where: [
        "第 6 行 z=s 漏减最大值；改为 s−m，将 [90,89,88] 平移至 [0,−1,−2]，数学权重不变。",
        "On line 6, z=s omits the maximum subtraction. z=s−m shifts [90, 89, 88] to [0, −1, −2] without changing the mathematical weights."
      ],
      impact: [
        "修复前含 NaN；修复后输出均有限，约为 66.5%、24.5%、9.0%，与参考一致。",
        "Before repair, NaN outputs cannot serve as valid weights. After shifting, the exponentials are representable and the outputs match the reference at about 66.5%, 24.5% and 9.0%."
      ]
    },
    exp_underflow: {
      what: [
        "极小的指数也可能无法表示。−104、−105、−106 的 FP32 指数均舍入为 0，分母也成为 0。",
        "Very small exponentials can also fall outside the representable range. Direct FP32 exponentials of −104, −105 and −106 all round to zero, making the denominator zero."
      ],
      where: [
        "第 6 行 z=s 未作平移；减去最大值 −104 后变为 [0,−1,−2]，仍保留原来的相对差距。",
        "On line 6, z=s retains the large negative scores. Subtracting the row maximum, −104, produces [0, −1, −2] while preserving their relative differences."
      ],
      impact: [
        "修复前 0/0 使输出全为 NaN；修复后分母约为 1.503，输出有限且与参考一致。",
        "Every output is NaN from 0/0 before repair. Afterwards the denominator is about 1.503 and the finite outputs match the reference at about 66.5%, 24.5% and 9.0%."
      ]
    },
    masked_max: {
      what: [
        "掩码也影响稳定化使用的最大值。若仍包含空位，虽然已减最大值，仍可能发生下溢。",
        "Masking affects both normalization and the maximum used for numerical stability. If padding still contributes to that maximum, subtracting it may fail to prevent underflow."
      ],
      where: [
        "第 5 行 max(x) 从 [−104,−105,−106,0] 取到空位的 0；应改用屏蔽后的 max(s)，得到 −104。",
        "On line 5, max(x) takes padding's zero from raw [−104, −105, −106, 0]. max(s) instead uses the masked maximum −104, giving z=[0, −1, −2, −INF]."
      ],
      impact: [
        "错误最大值使有效指数全为 0，输出含 NaN；修复后真实权重正确，空位的指数与权重均为 0。",
        "The wrong maximum makes all valid exponentials zero, leaving NaN outputs. After repair, real positions regain the correct weights and padding receives zero exponential and zero weight."
      ]
    }
  };
  globalThis.DRDebugExplanations = explanations;
  if (typeof module !== "undefined" && module.exports) module.exports = explanations;
})();
