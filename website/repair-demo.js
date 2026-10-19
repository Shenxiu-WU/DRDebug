/* Constructed CPU models of isolated masked-Softmax defects, not NPU runs. */
(() => {
  "use strict";
  const fixedCode = [
    "x = fp32(load_rows(scores, stride=cols))",
    "s = where(valid, x, -INF)",
    "z = s - max(s, axis=-1)",
    "e = exp_fp32(z)",
    "d = sum(e, axis=-1, accumulator=FP32)",
    "p = fp16(e / d)"
  ];
  // The same six mathematical steps, expanded into ten readable display lines.
  const displayCode = [
    {text:"x = load_rows(scores,", family:"index"},
    {text:"              stride=cols)", family:"index"},
    {text:"x = fp32(x)", family:"numeric"},
    {text:"s = where(valid, x, -INF)", family:"mask"},
    {text:"m = max(s, axis=-1)", family:"mask"},
    {text:"z = s - m", family:"numeric"},
    {text:"e = exp_fp32(z)", family:"numeric"},
    {text:"d = sum(e, axis=-1,", family:"reduce"},
    {text:"        accumulator=FP32)", family:"numeric"},
    {text:"p = fp16(e / d)", family:"output"}
  ];
  const codePatches = {
    row_stride:{line:2, before:"              stride=rows)"},
    mask_broadcast:{line:4, before:"s = where(valid[0], x, -INF)"},
    padding_mask:{line:4, before:"s = where(valid, x, 0)"},
    reduction_axis:{line:8, before:"d = sum(e, axis=0,"},
    tile_local_sum:{line:8, before:"d = sum_per_tile(e, tile=2,"},
    sum_rounding:{line:9, before:"        accumulator=FP16)"},
    exp_overflow:{line:6, before:"z = s"},
    exp_underflow:{line:6, before:"z = s"},
    masked_max:{line:5, before:"m = max(x, axis=-1)"}
  };
  function codeView(id) {
    const patch = codePatches[id];
    if (!patch) throw new Error("Unknown code illustration");
    return {line:patch.line, before:patch.before, after:displayCode[patch.line - 1].text,
      family:displayCode[patch.line - 1].family, lines:displayCode.map(item => item.text)};
  }
  function fp16(value) {
    if (value === 0 || !Number.isFinite(value)) return value;
    const sign = Math.sign(value), magnitude = Math.abs(value);
    if (magnitude >= 65520) return sign * Infinity;
    const step = Math.pow(2, Math.max(-24, Math.floor(Math.log2(magnitude)) - 10));
    const scaled = magnitude / step, lower = Math.floor(scaled), fraction = scaled - lower;
    const rounded = fraction < .5 ? lower : fraction > .5 ? lower + 1 : lower + (lower % 2);
    return sign * rounded * step;
  }
  const f32 = Math.fround;
  const fixtures = {
    row_stride: {rows:2, cols:3, row:1, scores:[0,1,2,2,1,0], masks:[[true,true,true],[true,true,true]], line:1},
    mask_broadcast: {rows:2, cols:4, row:1, scores:[2,1,0,-1,2,1,0,-1], masks:[[true,true,false,false],[true,false,true,false]], line:2},
    padding_mask: {rows:1, cols:4, row:0, scores:[-2,-3,-4,0], masks:[[true,true,true,false]], line:2},
    reduction_axis: {rows:2, cols:3, row:1, scores:[0,1,2,2,1,0], masks:[[true,true,true],[true,true,true]], line:5},
    tile_local_sum: {rows:1, cols:4, row:0, scores:[2,1,0,-1], masks:[[true,true,true,true]], line:5},
    sum_rounding: {rows:1, cols:4096, row:0, scores:Array(4096).fill(0), masks:[Array(4096).fill(true)], line:5},
    exp_overflow: {rows:1, cols:3, row:0, scores:[90,89,88], masks:[[true,true,true]], line:3},
    exp_underflow: {rows:1, cols:3, row:0, scores:[-104,-105,-106], masks:[[true,true,true]], line:3},
    masked_max: {rows:1, cols:4, row:0, scores:[-104,-105,-106,0], masks:[[true,true,true,false]], line:3}
  };
  function sum(values, round) {
    return values.reduce((total, value) => round(total + value), 0);
  }
  function run(input, defect) {
    const stride = defect === "row_stride" ? input.rows : input.cols;
    const loaded = Array.from({length:input.rows}, (_, row) =>
      input.scores.slice(row * stride, row * stride + input.cols).map(f32));
    const masked = loaded.map((row, r) => row.map((value, c) =>
      (defect === "mask_broadcast" ? input.masks[0] : input.masks[r])[c]
        ? value : defect === "padding_mask" ? 0 : -Infinity));
    const shifted = masked.map((row, r) => {
      const maximum = Math.max(...(defect === "masked_max" ? loaded[r] : row));
      return row.map(value => defect === "exp_overflow" || defect === "exp_underflow"
        ? value : f32(value - maximum));
    });
    const exp = shifted.map(row => row.map(value => f32(Math.exp(value))));
    const r = input.row;
    let denominator;
    if (defect === "reduction_axis") {
      denominator = exp[r].map((_, c) => sum(exp.map(row => row[c]), f32));
    } else if (defect === "tile_local_sum") {
      denominator = exp[r].map((_, c) => sum(exp[r].slice(Math.floor(c / 2) * 2, Math.floor(c / 2) * 2 + 2), f32));
    } else {
      denominator = sum(exp[r], defect === "sum_rounding" ? fp16 : f32);
    }
    const probabilities = exp[r].map((value, c) =>
      fp16(f32(value / (Array.isArray(denominator) ? denominator[c] : denominator))));
    return {sourceOffset:r * stride, loaded:loaded[r], masked:masked[r], shifted:shifted[r], exp:exp[r],
      denominator, probabilities, mass:probabilities.reduce((total, value) => total + value, 0)};
  }
  function evaluateCase(id) {
    const input = fixtures[id];
    if (!input) throw new Error("Unknown repair illustration");
    // This valid-only reference never uses candidate masking or its FP32 sum.
    const actual = input.scores.slice(input.row * input.cols, (input.row + 1) * input.cols);
    const valid = input.masks[input.row], maximum = Math.max(...actual.filter((_, c) => valid[c]));
    const numerators = actual.map((value, c) => valid[c] ? Math.exp(value - maximum) : 0);
    const denominator = numerators.reduce((total, value) => total + value, 0);
    const reference = numerators.map(value => fp16(value / denominator));
    const before = run(input, id), after = run(input);
    for (const result of [before, after]) result.matches = result.probabilities.every((value, c) =>
      Number.isFinite(value) && value === reference[c]);
    return {id, input, reference, before, after};
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {fp16, evaluateCase, fixedCode, codeView};
  if (typeof document === "undefined") return;
  const demo = document.querySelector("[data-repair-demo]");
  if (!demo) return;
  const copies = {
    row_stride: {
      title:["第二行数据，读错了起点。","The second row starts at the wrong offset."],
      input:["FP16 · 2 行 × 3 列；第二行应为 [2, 1, 0]","FP16 · 2 × 3 matrix; the second row is [2, 1, 0]"],
      metric:["第二行读取起点","Second-row offset"],
      diagnosis:["行起点误乘了行数，读到 [2, 2, 1]。改乘每行列数，才能读取正确的 [2, 1, 0]。","Multiplying by the row count loads [2, 2, 1]. Using the column count loads the intended [2, 1, 0]."],
    },
    mask_broadcast: {
      title:["第二行，误用了第一行的掩码。","The second row uses the first row's mask."],
      input:["FP16 · 两行分数均为 [2, 1, 0, −1]，有效位置各不相同","FP16 · Both rows score [2, 1, 0, −1], with different valid positions"],
      metric:["本应排除的 B 位置","Weight on invalid position B"],
      diagnosis:["第一行允许 A、B，第二行只允许 A、C。广播第一行掩码，会放进 B 并漏掉 C；逐行匹配才能修正输出。","Row 1 allows A/B; row 2 allows A/C. Reusing row 1's mask admits B and suppresses C. Each row needs its own mask."],
    },
    padding_mask: {
      title:["空位拿走了 83% 的注意力。","Padding steals 83% of the attention."],
      input:["FP16 · A −2、B −3、C −4，加一个补齐空位","FP16 · Scores A −2, B −3, C −4, plus padding"],
      metric:["补齐空位的权重","Weight on padding"],
      diagnosis:["无效位置填 0，仍会产生非零指数并参与分母。改填 −∞，指数为 0，空位不再分走权重。","A zero score still has a nonzero exponential and enters the denominator. With −∞ its exponential is zero, excluding padding."],
    },
    reduction_axis: {
      title:["跨行求和，混淆了两组权重。","A column sum mixes two independent rows."],
      input:["FP16 · 两行分数：[0, 1, 2] 与 [2, 1, 0]","FP16 · Two rows: [0, 1, 2] and [2, 1, 0]"],
      metric:["第二行权重之和","Total weight in row 2"],
      diagnosis:["分母沿 batch 方向求和，混入了另一行的指数。改为最后一维归约，每行才独立归一化。","Summing across the batch includes another row's exponentials. Reducing the last axis normalizes each row independently."],
    },
    tile_local_sum: {
      title:["每块都归一化，全行却多了一倍。","Each tile sums to one. The row sums to two."],
      input:["FP16 · [2, 1, 0, −1] 分两块处理；每块 2 项仅为示意","FP16 · [2, 1, 0, −1] in two illustrative tiles of 2"],
      metric:["整行权重之和","Total row weight"],
      diagnosis:["长序列可能分块处理；若每块各用自己的分母，整行就重复分配权重。这里保留全行最大值，只修复分母的跨块合并。","Long rows may be tiled. Separate denominators allocate a full unit of weight per tile. This case keeps the global maximum and fixes only the missing denominator merge."],
    },
    sum_rounding: {
      title:["本该 100% 的权重，算出了 200%。","Weights should total 100%. They total 200%."],
      input:["FP16 · 4,096 个有效位置，分数均为 0，指数均为 1","FP16 · 4,096 valid zero scores; every exponential is 1"],
      metric:["整行权重之和","Total row weight"],
      diagnosis:["逐次 FP16 舍入时，2,048 + 1 又变回 2,048，分母就此停住。用 FP32 累加得到 4,096，权重恢复正确。","With sequential FP16 rounding, 2,048 + 1 rounds back to 2,048. An FP32 accumulator reaches 4,096 and restores the correct weights."],
    },
    exp_overflow: {
      title:["分数合法，指数却溢出了。","Finite scores produce overflowing exponentials."],
      input:["FP16 · 分数 [90, 89, 88]，内部指数按 FP32 计算","FP16 · Scores [90, 89, 88]; exponentials use FP32"],
      metric:["输出是否为有限数","Finite outputs"],
      diagnosis:["exp(90) 超出 FP32 范围，随后 Inf / Inf 产生 NaN。先减去行最大值，指数输入变为 [0, −1, −2]，仍表示同一组权重。","exp(90) exceeds FP32 range; Inf / Inf then yields NaN. Subtracting the row maximum gives [0, −1, −2] without changing the mathematical weights."],
    },
    exp_underflow: {
      title:["指数全变成 0，分母也成了 0。","All exponentials vanish. The denominator is zero."],
      input:["FP16 · 分数 [−104, −105, −106]，内部指数按 FP32 计算","FP16 · Scores [−104, −105, −106]; exponentials use FP32"],
      metric:["输出是否为有限数","Finite outputs"],
      diagnosis:["直接计算的 FP32 指数全部下溢为 0，0 / 0 产生 NaN。减去行最大值后，指数输入同样变为 [0, −1, −2]。","Direct FP32 exponentials all underflow to zero, so 0 / 0 yields NaN. Subtracting the row maximum again gives [0, −1, −2]."],
    },
    masked_max: {
      title:["空位已经屏蔽，为什么仍然出现 NaN？","Padding is masked. Why does NaN remain?"],
      input:["FP16 · 有效分数 [−104, −105, −106]，补齐位置的原始值为 0","FP16 · Valid scores [−104, −105, −106]; the raw padding value is 0"],
      metric:["输出是否为有限数","Finite outputs"],
      diagnosis:["虽然已将空位屏蔽为 −∞，最大值却误从原始 x 中取得 0，导致有效项的指数全部下溢。应从屏蔽后的 s 求最大值 −104。","Padding is masked to −∞, but the maximum is still taken from raw x, where padding is 0. Valid exponentials underflow. Taking the maximum from masked s gives −104."],
    }
  };
  let selected = null, rangeInput = "exp_overflow";
  const cache = new Map();
  const get = id => {
    if (!cache.has(id)) cache.set(id, evaluateCase(id));
    return cache.get(id);
  };
  const reducedMotion = typeof matchMedia === "function" ? matchMedia("(prefers-reduced-motion: reduce)") : null;
  const motions = new Map();
  const motionNodes = [demo.querySelector(".repair-folio"), document.getElementById("repair-range-inputs"),
    ...demo.querySelectorAll("[data-repair-code-line]"), document.getElementById("repair-detail")];
  function stopMotion() {
    for (const [node, motion] of motions) {
      motions.delete(node);
      motion.animation.cancel();
      motion.cleanup();
    }
  }
  function measure(node) {
    const style = getComputedStyle(node), height = node.getBoundingClientRect().height;
    return {height, visible:style.display !== "none" && height > 0, display:style.display,
      opacity:style.opacity, paddingTop:style.paddingTop, paddingBottom:style.paddingBottom,
      borderTopWidth:style.borderTopWidth, borderBottomWidth:style.borderBottomWidth};
  }
  function move(node, before, after) {
    if (!before.visible && !after.visible) return;
    const detail = node.id === "repair-detail";
    if (before.visible === after.visible && Math.abs(before.height - after.height) < .5 && !detail) return;
    const properties = ["overflow", "display", "min-height"], saved = properties.map(key => node.style.getPropertyValue(key));
    const wasInert = node.inert;
    const cleanup = () => {
      properties.forEach((key, i) => saved[i] ? node.style.setProperty(key, saved[i]) : node.style.removeProperty(key));
      node.inert = wasInert;
    };
    node.style.overflow = "hidden";
    node.style.minHeight = "0";
    // The final hidden state is already set by render. Keep only the outgoing
    // visual box briefly; inert prevents interaction while it collapses.
    if (!after.visible) { node.style.display = before.display; node.inert = true; }
    const frame = state => ({height:state.visible ? state.height + "px" : "0px",
      opacity:state.visible ? state.opacity : 0,
      paddingTop:state.visible ? state.paddingTop : "0px", paddingBottom:state.visible ? state.paddingBottom : "0px",
      borderTopWidth:state.visible ? state.borderTopWidth : "0px", borderBottomWidth:state.visible ? state.borderBottomWidth : "0px"});
    const from = frame(before), to = frame(after);
    if (detail && before.visible && after.visible) from.opacity = Math.min(Number(before.opacity), .75);
    try {
      const animation = node.animate([from, to], {duration:240, easing:"cubic-bezier(.22,.68,.25,1)"});
      const motion = {animation, cleanup};
      motions.set(node, motion);
      animation.onfinish = () => {
        // A canceled animation must never clean up a newer transition.
        if (motions.get(node) !== motion) return;
        motions.delete(node); cleanup();
      };
    } catch (_) { cleanup(); }
  }
  function update(animate = false) {
    const enabled = animate && !(reducedMotion && reducedMotion.matches) &&
      typeof getComputedStyle === "function" && motionNodes.every(node => typeof node.animate === "function");
    // Read the current animated sizes before canceling, so a rapid reversal
    // starts from what is on screen rather than from an obsolete endpoint.
    const before = enabled ? motionNodes.map(measure) : [];
    stopMotion();
    render();
    if (!enabled) return;
    const after = motionNodes.map(measure);
    motionNodes.forEach((node, i) => move(node, before[i], after[i]));
  }
  if (reducedMotion && typeof reducedMotion.addEventListener === "function") reducedMotion.addEventListener("change", () => {
    if (reducedMotion.matches) stopMotion();
  });
  function render() {
    const en = document.documentElement.lang === "en", lang = en ? 1 : 0;
    const id = selected === "exp_range" ? rangeInput : selected;
    const text = (key, value) => { document.getElementById(key).textContent = value; };
    const percent = value => Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "NaN";
    const patch = id ? codeView(id) : null;
    for (const row of demo.querySelectorAll("[data-repair-code-line]")) {
      const line = Number(row.dataset.repairCodeLine), active = !!patch && line === patch.line;
      row.classList.toggle("is-active", active);
      if (active) row.setAttribute("aria-current", "step"); else row.removeAttribute("aria-current");
      row.querySelector(".repair-line-before .repair-line-text").textContent = active ? patch.before : displayCode[line - 1].text;
      row.querySelector(".repair-line-before .repair-line-sign").textContent = active ? "−" : "";
      row.querySelector(".repair-line-after .repair-line-text").textContent = displayCode[line - 1].text;
      row.querySelector(".repair-line-after").hidden = !active;
    }
    for (const button of demo.querySelectorAll("[data-repair-scenario]")) {
      const active = button.dataset.repairScenario === selected;
      button.setAttribute("aria-pressed", String(active));
      button.setAttribute("aria-expanded", String(active));
    }
    document.getElementById("repair-range-inputs").hidden = selected !== "exp_range";
    for (const button of demo.querySelectorAll("[data-repair-range]")) button.setAttribute("aria-pressed", String(selected === "exp_range" && button.dataset.repairRange === rangeInput));
    const detail = document.getElementById("repair-detail");
    detail.hidden = !id;
    detail.classList.toggle("has-chart", !!id && id !== "sum_rounding");
    document.getElementById("repair-detail-link").hidden = !id;
    if (!id) {
      text("repair-code-status", en ? "Correct implementation" : "正确实现");
      text("repair-case-summary", en ? "The correct implementation is shown. Select a problem to inspect its patch; select it again to restore the code." : "默认是正确实现。点选问题展开补丁，再点同一问题即可复原。");
      document.getElementById("repair-weight-chart").hidden = true;
      return;
    }
    const {input, before, after} = get(id), copy = copies[id];
    const explanation = globalThis.DRDebugExplanations && globalThis.DRDebugExplanations[id];
    const what = explanation ? explanation.what[lang] : copy.diagnosis[lang];
    const where = explanation ? explanation.where[lang] : (en ? "Inspect the highlighted code line." : "见当前高亮的代码行。");
    const impact = explanation ? explanation.impact[lang] : copy.diagnosis[lang];
    text("repair-detail-title", copy.title[lang]); text("repair-input", copy.input[lang]);
    text("repair-problem", what); text("repair-location", where); text("repair-impact", impact);
    text("repair-case-summary", what + " " + impact);
    text("repair-metric-label", copy.metric[lang]);
    const failedNumbers = id === "exp_overflow" || id === "exp_underflow" || id === "masked_max";
    const topic = [...demo.querySelectorAll("[data-repair-scenario]")].find(button => button.dataset.repairScenario === selected);
    text("repair-code-status", (en ? "Line " : "第 ") + patch.line + (en ? " · " : " 行 · ") + topic.textContent);
    let values, notes;
    if (id === "row_stride") {
      values = [String(before.sourceOffset), String(after.sourceOffset)];
      notes = [before, after].map(result => "[" + result.loaded.join(", ") + "]");
    } else if (id === "padding_mask" || id === "mask_broadcast") {
      const c = id === "padding_mask" ? 3 : 1;
      values = [percent(before.probabilities[c]), percent(after.probabilities[c])];
      notes = en ? ["Invalid position gets weight", "Invalid position excluded"] : ["无效位置获得了权重", "无效位置已排除"];
    } else if (failedNumbers) {
      values = en ? ["No", "Yes"] : ["含 NaN", "全部有限"];
      notes = [before, after].map(result => (en ? "denominator = " : "分母 = ") + (Number.isFinite(result.denominator)
        ? result.denominator.toLocaleString("en-US", {maximumFractionDigits:4}) : "∞"));
    } else {
      values = [percent(before.mass), percent(after.mass)];
      notes = id === "sum_rounding" ? (en ? ["denominator = 2,048", "denominator = 4,096"] : ["归约分母 = 2,048", "归约分母 = 4,096"])
        : (en ? ["Wrong normalization domain", "Matches the whole-row reference"] : ["归一化范围错误", "与全行参考一致"]);
    }
    for (let i = 0; i < 2; i++) {
      text("repair-evidence-" + (i === 0 ? "before" : "after"), values[i]);
      text("repair-note-" + (i === 0 ? "before" : "after"), notes[i]);
    }
    const chart = document.getElementById("repair-weight-chart");
    chart.hidden = id === "sum_rounding";
    text("repair-chart-title", failedNumbers
      ? (en ? "Repaired weights; before contains NaN" : "修复后权重；修复前含 NaN")
      : (en ? "Output weights" : "输出权重"));
    for (const row of demo.querySelectorAll("[data-position]")) {
      const c = Number(row.dataset.position);
      row.hidden = c >= Math.min(input.cols, 4);
      if (row.hidden) continue;
      const invalid = !input.masks[input.row][c];
      const padding = id === "padding_mask" || id === "masked_max";
      row.classList.toggle("is-padding", invalid);
      row.querySelector(".repair-position").textContent = invalid
        ? (padding ? (en ? "PAD" : "空位") : "ABCD"[c] + "×") : "ABCD"[c];
      const label = invalid ? (padding ? (en ? "Padding" : "空位")
        : "ABCD"[c] + (en ? " (masked)" : "（应屏蔽）")) : "ABCD"[c];
      for (const [version, result] of [["before", before], ["after", after]]) {
        const weight = result.probabilities[c], finite = Number.isFinite(weight);
        row.querySelector(".is-" + version + " .repair-weight-track>span").style.width = finite ? (weight * 100) + "%" : "0%";
        row.querySelector(".is-" + version + " strong").textContent = percent(weight);
      }
      row.setAttribute("aria-label", label + (en ? ": before " : "：修复前 ") + percent(before.probabilities[c]) +
        (en ? ", after " : "，修复后 ") + percent(after.probabilities[c]));
    }
    text("repair-simulation-note", id === "sum_rounding"
      ? (en ? "Sequential FP16 rounding on CPU; not a model of a particular NPU reduction tree." : "CPU 逐次 FP16 舍入模拟，不代表特定 NPU 的归约树。")
      : (en ? "Constructed input; each case isolates one defect. Full outputs are checked by the CPU script." : "构造输入，每组只引入一种缺陷；CPU 脚本比对完整输出。"));
    text("repair-before-verdict", before.matches ? (en ? "Pass" : "通过") : (en ? "Fail" : "不通过"));
    text("repair-after-verdict", after.matches ? (en ? "Pass" : "通过") : (en ? "Fail" : "不通过"));
  }
  for (const button of demo.querySelectorAll("[data-repair-scenario]")) button.addEventListener("click", () => {
    selected = selected === button.dataset.repairScenario ? null : button.dataset.repairScenario; update(true);
  });
  for (const button of demo.querySelectorAll("[data-repair-range]")) button.addEventListener("click", () => {
    rangeInput = button.dataset.repairRange; update(true);
  });
  for (const button of demo.querySelectorAll("[data-repair-reset]")) button.addEventListener("click", () => {
    const opener = [...demo.querySelectorAll("[data-repair-scenario]")].find(node => node.dataset.repairScenario === selected);
    selected = null; update(true);
    if (opener && typeof opener.focus === "function") opener.focus();
  });
  document.addEventListener("site:language", () => update());
  render();
})();
