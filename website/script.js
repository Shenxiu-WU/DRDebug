(() => {
  "use strict";
  const translations = [...document.querySelectorAll("[data-en]")].map(node => ({node, zh: node.textContent, en: node.dataset.en}));
  const attributes = ["aria", "alt"].flatMap(type => [...document.querySelectorAll(`[data-en-${type}]`)].map(node => ({node, attr: type === "aria" ? "aria-label" : "alt", zh: node.getAttribute(type === "aria" ? "aria-label" : "alt"), en: node.getAttribute(`data-en-${type}`)})));
  let language = "zh";
  let stage = 0;
  const stages = {
    zh: [
      ["01 / 先让失败可复现", "从一个真实的失败开始。", "保留已有候选，先建立可复现的失败。引擎记录构建、运行与精度证据，再交由工作智能体提出修改。", "已有内核 + 失败测例", "可复现的基线"],
      ["02 / 用证据缩小问题", "定位偏差，而非猜测答案。", "按需启用取证与插桩，结合编译、运行和精度反馈定位问题。结构化领域知识帮助工作智能体判断错误类型与可行的修复方向。", "错误反馈 + 按需取证", "有依据的修复假设"],
      ["03 / 保留已有进展", "围绕问题，做定向修改。", "工作智能体在同一任务中保留上下文，对已有候选提出补丁。修改围绕当前失败与已收集的证据展开，并受任务预算和约束控制。", "修复假设 + 领域知识", "待验证的内核补丁"],
      ["04 / 让验证决定接纳", "过关的标准，由引擎掌握。", "引擎检查内核完整性，并执行完整测试集验证。正确性与性能分别记录；验证未通过的修改不能仅凭大语言模型的自述被视为成功。", "候选补丁 + 完整测试集", "独立验证的接纳决定"],
      ["05 / 累积可复用的经验", "保留状态，继续下一轮。", "记录证据、修复轨迹与验证结果，通过检查点和回滚保留当前最优状态。经过核验的经验可进入结构化知识，供后续任务复用。", "验证记录 + 修复轨迹", "检查点与可复用经验"]
    ],
    en: [
      ["01 / REPRODUCE THE FAILURE", "Start with a real failing case.", "Keep the existing candidate and establish a reproducible failure. The engine records build, execution and precision evidence before asking the worker for a patch.", "Candidate + failing case", "A reproducible baseline"],
      ["02 / NARROW THE PROBLEM", "Locate the discrepancy.", "Use on-demand forensics and instrumentation alongside build, runtime and precision feedback. Structured domain knowledge helps the worker identify failure types and plausible repair directions.", "Failure feedback + forensics", "An evidence-based hypothesis"],
      ["03 / KEEP THE PROGRESS", "Make a targeted edit.", "The worker retains context within one repair task and proposes a patch to the existing candidate. Edits follow the observed failure and collected evidence, within task constraints and budgets.", "Hypothesis + domain knowledge", "A kernel patch to validate"],
      ["04 / VALIDATION DECIDES", "The engine owns acceptance.", "The engine checks kernel integrity and validates the complete test suite. Correctness and performance are recorded separately; the large language model’s own report cannot turn an unvalidated patch into a success.", "Patch + complete test suite", "An independently checked decision"],
      ["05 / RETAIN WHAT WAS LEARNED", "Keep state for the next round.", "Record evidence, repair traces and validation results. Checkpoints and rollback preserve the best-known state; verified experience can become structured knowledge for later tasks.", "Validation record + repair trace", "Checkpoints + reusable knowledge"]
    ]
  };
  function renderStage() {
    const [eyebrow, heading, description, input, output] = stages[language][stage];
    const container = document.getElementById("stage-detail");
    const intro = document.createElement("div");
    for (const [tag, content, className] of [["p",eyebrow,"eyebrow"],["h3",heading,""],["p",description,""]]) {
      const node = document.createElement(tag); node.textContent = content; node.className = className; intro.append(node);
    }
    const evidence = document.createElement("div"); evidence.className = "stage-evidence";
    for (const [tag, content] of [["span",language === "zh" ? "输入证据" : "EVIDENCE IN"],["strong",input],["span",language === "zh" ? "引擎产物" : "ENGINE OUTPUT"],["strong",output]]) {
      const node = document.createElement(tag); node.textContent = content; evidence.append(node);
    }
    container.replaceChildren(intro,evidence);
    for (const button of document.querySelectorAll("[data-stage]")) button.setAttribute("aria-pressed",String(Number(button.dataset.stage) === stage));
  }
  function setLanguage(lang) {
    language = lang === "en" ? "en" : "zh";
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.title = language === "zh" ? "DRDebug · 接近成功，继续修好" : "DRDebug · Don’t regenerate. Debug.";
    for (const item of translations) item.node.textContent = item[language];
    for (const item of attributes) item.node.setAttribute(item.attr,item[language]);
    for (const button of document.querySelectorAll("[data-language]")) button.setAttribute("aria-pressed",String(button.dataset.language === language));
    renderStage();
    const skillGuide = document.querySelector(".skill-guide");
    if (skillGuide) skillGuide.href = `https://github.com/Shenxiu-WU/DRDebug/blob/main/docs/skill-guide${language === "zh" ? ".zh-CN" : ""}.md`;
    const skillExample = document.querySelector(".skill-example");
    if (skillExample) skillExample.href = `https://github.com/Shenxiu-WU/DRDebug/blob/main/examples/cumsum/README${language === "zh" ? ".zh-CN" : ""}.md`;
    const copyStatus = document.getElementById("skill-copy-status");
    if (copyStatus) copyStatus.textContent = "";
    document.dispatchEvent(new CustomEvent("site:language",{detail:{lang:language}}));
    try { localStorage.setItem("debug-agent-language",language); } catch (_) { /* Storage is optional. */ }
  }
  for (const button of document.querySelectorAll("[data-language]")) button.addEventListener("click",() => setLanguage(button.dataset.language));
  for (const button of document.querySelectorAll("[data-stage]")) button.addEventListener("click",() => {stage = Number(button.dataset.stage); renderStage();});
  const dialog = document.getElementById("figure-dialog");
  const dialogImage = document.getElementById("dialog-image");
  for (const link of document.querySelectorAll("[data-figure]")) link.addEventListener("click",event => {
    if (typeof dialog.showModal !== "function") return;
    event.preventDefault();
    dialogImage.src = link.href; dialogImage.alt = link.querySelector("img").alt;
    document.getElementById("figure-original").href = link.href;
    dialog.showModal();
  });
  document.getElementById("close-figure").addEventListener("click",() => dialog.close());
  dialog.addEventListener("click",event => { if(event.target === dialog) {const r = dialog.getBoundingClientRect(); if(event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();} });
  const installTargets = {
    codex: {args: "--codex", destination: "~/.agents/skills/drdebug"},
    claude: {args: "--claude", destination: "~/.claude/skills/drdebug"},
    other: {args: "--dest /path/to/your-agent/skills", destination: "/path/to/your-agent/skills/drdebug"}
  };
  for (const button of document.querySelectorAll("[data-install-host]")) button.addEventListener("click", () => {
    const host = button.dataset.installHost;
    const target = installTargets[host];
    document.getElementById("skill-install-command").textContent = `git clone https://github.com/Shenxiu-WU/DRDebug.git\ncd DRDebug\n./install.sh ${target.args}`;
    document.getElementById("skill-install-destination").textContent = target.destination;
    document.getElementById("skill-generic-note").hidden = host !== "other";
    for (const target of document.querySelectorAll("[data-install-host]")) target.setAttribute("aria-pressed", String(target.dataset.installHost === host));
    document.getElementById("skill-copy-status").textContent = "";
  });
  for (const button of document.querySelectorAll("[data-copy-target]")) button.addEventListener("click", async () => {
    const command = document.getElementById(button.dataset.copyTarget);
    const status = document.getElementById("skill-copy-status");
    try {
      await navigator.clipboard.writeText(command.textContent);
      status.textContent = language === "zh" ? "安装命令已复制。" : "Installation commands copied.";
    } catch (_) {
      status.textContent = language === "zh" ? "可选中上方命令，手动复制。" : "Select the commands above to copy them manually.";
    }
  });
  let initial = "zh";
  try { if(localStorage.getItem("debug-agent-language") === "en") initial = "en"; } catch (_) { /* Chinese remains the default. */ }
  setLanguage(initial);
})();
