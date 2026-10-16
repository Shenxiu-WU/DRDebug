(function () {
  "use strict";

  const script = document.currentScript;
  const dataUrl = new URL(
    "./data/leaderboard-2026-10-15.json",
    script && script.src ? script.src : document.baseURI
  );
  const state = {
    data: null,
    comparison: "level4",
    operator: "MlaProlog",
    lang: document.documentElement.lang.toLowerCase().startsWith("en") ? "en" : "zh",
    failed: false
  };

  // Public task definitions: https://cannbench.com/api/benchmarks/official-tasks
  // Official-Tasks 1.1.2, checked on 2026-10-16. The challenges below are
  // engineering summaries of the public computation graphs, not diagnoses of
  // hidden cases or descriptions of private repair mechanisms.
  const operatorProfiles = {
    GQA: {
      zh: {
        purpose: "多个 Query 头共享较少的 KV 头，对已分头的 Q/K/V 计算缩放点积注意力。",
        challenge: "需要协调 Query 与 KV 头的分组映射、两次矩阵乘、Softmax 归约及可选因果掩码。"
      },
      en: {
        purpose: "Computes scaled dot-product attention with multiple query heads sharing fewer key/value heads.",
        challenge: "Combines grouped head mapping, two matrix multiplications, softmax reductions and optional causal masking."
      }
    },
    GroupedMatmulSwigluQuant: {
      zh: {
        purpose: "按专家分组执行矩阵乘，随后完成反量化、SwiGLU 激活和逐 Token 再量化。",
        challenge: "分组边界、矩阵累加、双侧缩放、激活、逐行归约和量化舍入形成连续的数据流。"
      },
      en: {
        purpose: "Performs grouped expert matmul followed by dequantization, SwiGLU activation and per-token requantization.",
        challenge: "Couples group boundaries, matrix accumulation, scaling, activation, row reductions and quantization rounding."
      }
    },
    GRU: {
      zh: {
        purpose: "通过重置门和更新门，沿序列逐步更新隐藏状态。",
        challenge: "矩阵投影与 Sigmoid/Tanh 门控相互衔接，同时存在跨时间步的状态依赖。"
      },
      en: {
        purpose: "Updates a recurrent hidden state across a sequence using reset and update gates.",
        challenge: "Connects matrix projections and sigmoid/tanh gating with dependencies between successive time steps."
      }
    },
    LSTM: {
      zh: {
        purpose: "通过输入门、遗忘门和输出门维护细胞状态与隐藏状态，处理序列依赖。",
        challenge: "四路门控与候选计算、多种非线性及两类状态的跨步传递构成复杂计算链。"
      },
      en: {
        purpose: "Maintains cell and hidden states through input, forget and output gates to process sequential dependencies.",
        challenge: "Combines four gate/candidate branches, nonlinear activations and coupled cell/hidden-state transitions across time."
      }
    },
    MHA: {
      zh: {
        purpose: "对已分头的 Q/K/V 执行多头缩放点积注意力，支持可选因果掩码。",
        challenge: "两次矩阵乘之间需要完成缩放、掩码及 Softmax 归约，多个阶段共同决定输出。"
      },
      en: {
        purpose: "Computes multi-head scaled dot-product attention from already split Q/K/V tensors, with optional causal masking.",
        challenge: "Links two matrix multiplications through scaling, masking and softmax reductions."
      }
    },
    MLA: {
      zh: {
        purpose: "组合 Q/K 的内容与位置编码分量，执行 MLA 注意力核心及 Value 聚合。",
        challenge: "内容与位置分量、共享 KV 头、不同的 QK 与 Value 维度及 Softmax 归约需要协同处理。"
      },
      en: {
        purpose: "Combines content and positional Q/K components to compute the MLA attention core and value aggregation.",
        challenge: "Coordinates split components, shared KV heads, different QK/value dimensions and softmax reductions."
      }
    },
    MlaProlog: {
      zh: {
        purpose: "完成 MLA 的 Query/Key 投影、归一化与旋转位置编码前处理。",
        challenge: "串联四次矩阵乘、两次 RMSNorm 和两次 RoPE，衔接矩阵运算、归约与位置编码。"
      },
      en: {
        purpose: "Prepares MLA queries and keys through projections, normalization and rotary positional encoding.",
        challenge: "Connects four matrix multiplications, two RMSNorm operations and two RoPE operations in one computation pipeline."
      }
    },
    SparseFlashAttention: {
      zh: {
        purpose: "仅对输入索引指定的 KV 位置子集计算注意力，支持共享 KV 头及可选因果掩码。",
        challenge: "不规则索引访问、按头分组、掩码和 Softmax 归约需要共同遵守稀疏注意力语义。"
      },
      en: {
        purpose: "Computes attention over KV positions supplied by sparse indices, with shared KV heads and optional causal masking.",
        challenge: "Combines irregular indexed access, grouped heads, masking and softmax reductions under sparse attention semantics."
      }
    }
  };

  const copy = {
    zh: {
      standard: "标准测试",
      hidden: "隐藏测试",
      passed: "通过",
      remaining: "剩余未通过",
      allPassed: "全部通过",
      oneRemaining: "剩余 1 例",
      challenge: "实现难点",
      error: "互动数据暂时无法读取，已保留页面中的静态快照。可通过公开榜单与数据链接核验。",
      comparison: (view, total, max) => `${view === "level4" ? "Level 4" : "全部 53 个算子"} · 标准 + 隐藏测试 · 每个方案完整覆盖 ${total} 例。横条表示剩余未通过用例，越少越好；坐标从 0 到 ${max}。展示精选的完整覆盖方案。`,
      operatorLabel: (name, passed, total) => `${name}：${passed}/${total} 例通过，查看标准与隐藏测试明细`,
      comparisonLabel: (name, failed, total) => `${name}：${total} 例中剩余 ${failed} 例未通过`
    },
    en: {
      standard: "Standard tests",
      hidden: "Hidden tests",
      passed: "passed",
      remaining: "remaining failures",
      allPassed: "All passed",
      oneRemaining: "1 remaining",
      challenge: "Why it is challenging",
      error: "Interactive data could not be loaded. The static snapshot has been preserved. Use the public leaderboard and data links to verify the results.",
      comparison: (view, total, max) => `${view === "level4" ? "Level 4" : "All 53 operators"} · standard + hidden tests · ${total} cases fully covered per solution. Bars show remaining failures: fewer is better, on an axis from 0 to ${max}. Selected solutions with complete coverage are shown.`,
      operatorLabel: (name, passed, total) => `${name}: ${passed}/${total} cases passed. Show standard and hidden test details`,
      comparisonLabel: (name, failed, total) => `${name}: ${failed} remaining failures out of ${total} cases`
    }
  };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function formatNumber(value, fractionDigits) {
    return new Intl.NumberFormat(state.lang === "zh" ? "zh-CN" : "en-US", {
      maximumFractionDigits: fractionDigits || 0
    }).format(value);
  }

  function formatRate(value) {
    return new Intl.NumberFormat(state.lang === "zh" ? "zh-CN" : "en-US", {
      style: "percent",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  }

  function countPair(value) {
    return `${formatNumber(value.passed)} / ${formatNumber(value.total)}`;
  }

  function assert(condition) {
    if (!condition) throw new Error("Invalid public leaderboard snapshot");
  }

  function validatePair(value) {
    assert(value && Number.isInteger(value.passed) && Number.isInteger(value.total));
    assert(value.total > 0 && value.passed >= 0 && value.passed <= value.total);
  }

  function validateMetric(value) {
    validatePair(value);
    assert(value.failed === value.total - value.passed);
    assert(Number.isFinite(value.pass_rate));
    assert(Math.abs(value.pass_rate - value.passed / value.total) < 1e-9);
  }

  function validateData(data) {
    assert(data && data.schema_version === 1 && data.solution && data.scope);
    assert(typeof data.solution.name === "string" && data.solution.name.length > 0);
    assert(/^\d{4}-\d{2}-\d{2}$/.test(data.snapshot_date));
    validateMetric(data.solution.overall);
    validateMetric(data.solution.level4);
    validatePair(data.solution.standard);
    validatePair(data.solution.hidden);
    assert(data.solution.standard.passed + data.solution.hidden.passed === data.solution.overall.passed);
    assert(data.solution.standard.total + data.solution.hidden.total === data.solution.overall.total);
    assert(data.solution.overall.total === data.scope.complete_cases);
    assert(data.solution.level4.total === data.scope.level4_complete_cases);

    for (const view of ["overall", "level4"]) {
      const rows = data[`${view}_comparison`];
      assert(Array.isArray(rows) && rows.length > 1);
      for (const row of rows) {
        validateMetric(row);
        assert(typeof row.name === "string" && row.name.length > 0);
        assert(row.total === data.solution[view].total);
      }
      const own = rows.find((row) => row.name === data.solution.name);
      assert(own && own.passed === data.solution[view].passed);
    }

    const operators = data.level4_operators;
    assert(Array.isArray(operators) && operators.length === 8);
    assert(new Set(operators.map((operator) => operator.name)).size === 8);
    for (const operator of operators) {
      validateMetric(operator);
      validatePair(operator.standard);
      validatePair(operator.hidden);
      assert(typeof operator.name === "string" && operator.name.length > 0);
      assert(operator.total === 100);
      assert(operator.standard.total + operator.hidden.total === operator.total);
      assert(operator.standard.passed + operator.hidden.passed === operator.passed);
    }
    assert(operators.reduce((sum, operator) => sum + operator.passed, 0) === data.solution.level4.passed);
    assert(operators.reduce((sum, operator) => sum + operator.total, 0) === data.solution.level4.total);
    return data;
  }

  function renderComparison() {
    const data = state.data;
    const text = copy[state.lang];
    const rows = data[`${state.comparison}_comparison`];
    const max = Math.max(...rows.map((row) => row.failed), 1);
    const bars = document.getElementById("comparison-bars");
    if (bars) {
      const fragment = document.createDocumentFragment();
      for (const row of rows) {
        const item = element("div", "comparison-row");
        if (row.name === data.solution.name) item.classList.add("is-ours");
        const label = element("span", "comparison-name", row.name);
        const track = element("span", "comparison-track");
        const fill = element("span", "comparison-fill");
        fill.style.width = `${row.failed / max * 100}%`;
        fill.setAttribute("aria-hidden", "true");
        track.setAttribute("aria-hidden", "true");
        track.append(fill);
        const value = element("span", "comparison-value", formatNumber(row.failed));
        item.setAttribute("role", "group");
        item.setAttribute("aria-label", text.comparisonLabel(row.name, row.failed, row.total));
        item.append(label, track, value);
        fragment.append(item);
      }
      bars.replaceChildren(fragment);
    }

    const scale = document.getElementById("comparison-scale");
    if (scale) {
      const ticks = [0, max / 2, max].map((value) => element("span", "comparison-tick", formatNumber(value, 1)));
      scale.replaceChildren(...ticks);
      scale.setAttribute("aria-hidden", "true");
    }
    const caption = document.getElementById("comparison-caption");
    if (caption) caption.textContent = text.comparison(state.comparison, formatNumber(rows[0].total), formatNumber(max));
    for (const button of document.querySelectorAll("button[data-comparison]")) {
      const selected = button.dataset.comparison === state.comparison;
      button.setAttribute("aria-pressed", String(selected));
      button.classList.toggle("is-active", selected);
    }
  }

  function renderProfile(name, lang) {
    if (!Object.prototype.hasOwnProperty.call(operatorProfiles, name)) return null;
    const profile = operatorProfiles[name][lang];
    if (!profile) return null;
    const purpose = element("p", "operator-detail-purpose", profile.purpose);
    const challenge = element("div", "operator-detail-challenge");
    challenge.append(element("h5", "", copy[lang].challenge), element("p", "", profile.challenge));
    return { purpose, challenge };
  }

  function renderOperatorDetail() {
    const detail = document.getElementById("operator-detail");
    if (!detail) return;
    const operator = state.data.level4_operators.find((item) => item.name === state.operator);
    if (!operator) return;
    const text = copy[state.lang];
    const heading = element("h4", "operator-detail-name", operator.name);
    const total = element("p", "operator-detail-count", countPair(operator));
    const profile = renderProfile(operator.name, state.lang);
    const metrics = element("dl", "operator-detail-metrics");
    for (const [key, label] of [["standard", text.standard], ["hidden", text.hidden]]) {
      const group = element("div", "operator-detail-metric");
      group.append(element("dt", "", label), element("dd", "", countPair(operator[key])));
      metrics.append(group);
    }
    const children = [heading, total];
    if (profile) children.push(profile.purpose);
    children.push(metrics);
    if (profile) children.push(profile.challenge);
    detail.replaceChildren(...children);
    detail.setAttribute("aria-live", "polite");
    detail.setAttribute("aria-atomic", "true");
  }

  function renderOperators() {
    const grid = document.getElementById("operator-grid");
    if (grid) {
      const fragment = document.createDocumentFragment();
      for (const operator of state.data.level4_operators) {
        const selected = operator.name === state.operator;
        const button = element("button", "operator-tile");
        button.type = "button";
        button.dataset.operator = operator.name;
        button.setAttribute("aria-pressed", String(selected));
        button.setAttribute("aria-controls", "operator-detail");
        button.setAttribute("aria-label", copy[state.lang].operatorLabel(operator.name, operator.passed, operator.total));
        button.classList.toggle("is-active", selected);
        button.classList.toggle("is-partial", operator.failed > 0);
        const status = operator.failed === 0 ? copy[state.lang].allPassed : operator.failed === 1 ? copy[state.lang].oneRemaining : `${operator.failed} ${copy[state.lang].remaining}`;
        button.append(
          element("span", "operator-name", operator.name),
          element("span", "operator-count", countPair(operator)),
          element("span", "operator-state", status)
        );
        fragment.append(button);
      }
      grid.replaceChildren(fragment);
    }
    renderOperatorDetail();
  }

  function renderStats() {
    const solution = state.data.solution;
    const values = {
      overallPassed: formatNumber(solution.overall.passed),
      overallTotal: formatNumber(solution.overall.total),
      overallRate: formatRate(solution.overall.pass_rate),
      level4Passed: formatNumber(solution.level4.passed),
      level4Total: formatNumber(solution.level4.total),
      level4Rate: formatRate(solution.level4.pass_rate)
    };
    for (const node of document.querySelectorAll("[data-stat]")) {
      if (Object.prototype.hasOwnProperty.call(values, node.dataset.stat)) node.textContent = values[node.dataset.stat];
    }
    const date = document.getElementById("snapshot-date");
    if (date) {
      const parts = state.data.snapshot_date.split("-").map(Number);
      date.textContent = new Intl.DateTimeFormat(state.lang === "zh" ? "zh-CN" : "en-GB", {
        year: "numeric", month: state.lang === "zh" ? "numeric" : "long", day: "numeric", timeZone: "UTC"
      }).format(new Date(Date.UTC(parts[0], parts[1] - 1, parts[2])));
      if (date.tagName === "TIME") date.setAttribute("datetime", state.data.snapshot_date);
    }
  }

  function renderError() {
    const status = document.getElementById("data-status");
    if (!status) return;
    status.hidden = false;
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    status.textContent = copy[state.lang].error;
  }

  function render() {
    if (!state.data) {
      if (state.failed) renderError();
      return;
    }
    renderStats();
    renderComparison();
    renderOperators();
    const status = document.getElementById("data-status");
    if (status) {
      status.textContent = "";
      status.hidden = true;
    }
  }

  document.addEventListener("site:language", function (event) {
    if (!event.detail || !["zh", "en"].includes(event.detail.lang)) return;
    state.lang = event.detail.lang;
    render();
  });

  document.addEventListener("click", function (event) {
    if (!state.data || !(event.target instanceof Element)) return;
    const comparison = event.target.closest("button[data-comparison]");
    if (comparison && ["level4", "overall"].includes(comparison.dataset.comparison)) {
      state.comparison = comparison.dataset.comparison;
      renderComparison();
      return;
    }
    const operator = event.target.closest("#operator-grid button[data-operator]");
    if (operator && state.data.level4_operators.some((item) => item.name === operator.dataset.operator)) {
      state.operator = operator.dataset.operator;
      // Keep the focused button in place when its detail panel changes.
      for (const tile of document.querySelectorAll("#operator-grid button[data-operator]")) {
        const selected = tile.dataset.operator === state.operator;
        tile.setAttribute("aria-pressed", String(selected));
        tile.classList.toggle("is-active", selected);
      }
      renderOperatorDetail();
    }
  });

  fetch(dataUrl, { credentials: "omit" })
    .then(function (response) {
      if (!response.ok) throw new Error("Public snapshot unavailable");
      return response.json();
    })
    .then(function (data) {
      state.data = validateData(data);
      if (!state.data.level4_operators.some((item) => item.name === state.operator)) state.operator = state.data.level4_operators[0].name;
      render();
    })
    .catch(function () {
      state.failed = true;
      renderError();
    });
})();
