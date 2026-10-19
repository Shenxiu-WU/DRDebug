# 项目展示网站

网站以中文为默认语言，可在页面内切换 English。HTML、CSS、JavaScript 和精选公开资产都在本仓库的 `website/` 中，直接在此维护。公开仓库以独立历史保存论文冻结 artifact，GitHub Pages 只发布 `website/`。

## 本地预览

在本公开仓库（本地 checkout 为 `public-showcase/`）的根目录运行：

```bash
python3 -m http.server 8000 --directory website
```

在浏览器打开 <http://localhost:8000>，检查中文与英文、手机与桌面布局、图表、图片及链接。静态资产使用相对路径，使本地预览和 GitHub Pages 的项目子路径都可访问。网站无需 npm 安装或构建。

## 内容维护

- 所有排行榜或评测 snapshot 必须标注采集日期、硬件、指标口径及分母；更新数字时同步检查图表、正文和中英文文案。
- 区分 CANNBench 官方结果与论文中早期生成流程（NPUKernelBench 任务）的实验结果，不将不同基准的数字混为同一排名或结论。正式成果链接到原始公开来源。
- 论文中的 PDF 图资产用于说明研究动机、机制和实验，不代表论文全文。论文标题及 arXiv 编号链接到[公开摘要页](https://arxiv.org/abs/2608.02712)，另提供[全文 PDF](https://arxiv.org/pdf/2608.02712) 入口；展示题名和作者机构标号以 arXiv 公开 PDF v2（2026-10-15）的 Accelerator Kernels 为准。
- 仅把用于公开展示的精选资产放入 `website/`，运行回执、原始会话和机器本地配置继续由原有目录维护。

## GitHub Pages 发布

首次由仓库管理员在 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。仓库可见性保持现状；此工作流不会修改它。GitHub Free 的 Pages 需要公开仓库，私有仓库需支持 Pages 的付费方案；即使源码仓库私有，普通 Pages 站点仍公开。

发布工作流为 [deploy-pages.yml](../.github/workflows/deploy-pages.yml)：

- `main` 上的 `website/**` 或工作流自身变更触发发布。
- 也可在 **Actions → Deploy project website → Run workflow** 手动运行。
- 工作流只上传 `website/`，并通过 `github-pages` environment 发布；成功后可在部署记录查看实际网址。

默认项目网址为 <https://Shenxiu-WU.github.io/DRDebug/>。若配置了自定义域名，以 Pages 设置和部署输出为准。

说明参考：[GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)、[发布源和公开性](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。

## 图表与交互

- `dashboard.js` 读取同目录下带日期的公开 JSON；正文保留静态数据作为网络失败时的回退。比较图可切换全量与 Level4，横条统一表示剩余未通过用例，坐标从 0 开始。
- Level4 明确说明为 Official-Tasks 的最高难度等级，包含 8 个复杂算子任务、800 个测例（标准 160 + 隐藏 640）。两列对比直接标明 Debug Agent 与 ForgeKernel 的剩余未通过用例数，避免将 1 和 12 表达为分数；两者覆盖范围一致。
- 首屏用 FP16 顺序求和的构造样例解释“失败复现 → 中间态取证 → 定向补丁 → 验证”流程；`examples/fp16-reduction-demo.py` 可在 CPU 上复现明确的舍入轨迹及 6 组构造回归。它不是 Ascend 内核、真实 Agent 日志或 CANNBench 任务，不能据此声明 NPU 正式验证通过。下方另行展示带日期的真实 CANNBench 成绩。
- 首屏数据来源条完整标注 official-tasks、版本系列 1.1 / 任务包 1.1.2、A3 / NPU 910C、标准集与隐藏集、算子和测例覆盖、聚合名及快照日期。算子明细只展示公开的标准/隐藏集聚合计数。
- `script.js` 提供语言切换、修复机制分步说明与论文图放大。首次访问默认中文，之后记住访客选择；无需外部字体或脚本。
- `assets/research/` 两份 SVG 为用户提供的论文图直接转换的矢量版本，未重画科研内容。架构图对应 `Latest.pdf`（单页图，不是论文全文），机制图对应 `Figure2_AAAI27_borderless.pdf`。论文实验叙事来自原稿，图和文字不表示论文已被接收或已正式发表。

更新日期快照时，须同时修改 `dashboard.js` 的数据文件名与 HTML 中的静态数字、图注、来源链接；不要把旧日期改成今天而不重新采集数据。所有排行榜陈述均是正确性指标，不代表综合得分第一。

## 颜色主题

页头的太阳、月亮和显示器按钮分别选择亮色、深色与跟随系统。首次访问保持深色；之后用独立的 `debug-agent-theme` 保存选择，语言偏好不受影响。`theme.js` 在样式绘制前恢复主题，并在跟随系统时响应系统外观变化；禁用浏览器存储时仍可切换。

章节使用全宽底色区分：成果为冷灰，方法为浅紫，研究为暖纸色，资源为灰蓝；深色版有对应的低亮度分区。图表、控件、选中项和弹窗采用统一颜色变量。原论文矢量图保留白底与原始配色。

## 公开发布维护

公开仓库为 [DRDebug](https://github.com/Shenxiu-WU/DRDebug)，本地独立 checkout 为 `public-showcase/`。其中的 `website/` 是唯一维护的网页源码目录。网页源码入口提供论文冻结代码与数据，CANNBench 快照与论文实验分别呈现。

日常直接编辑本目录，在公开仓库根目录启动本地预览，并检查中英文、主题、响应式布局与链接。收到发布指示后，在本公开仓库提交并推送 `main`，由现有 Pages 工作流部署。网站更新与论文 artifact 分开组织；旧的私仓网站副本、导出工具和导出哈希清单已退役。

## 研究内容与任务说明

- 首屏以“算子精度修复智能体 / ASCEND NPU”概括定位；结果正文明确排名对象为测试用例通过数。
- 8个 Level4 算子的功能依据 [Official-Tasks 公开任务定义](https://cannbench.com/api/benchmarks/official-tasks)，实现难点为对其公开计算图的工程概括。详情只显示通过计数、标准/隐藏明细及各任务的功能和难点，不推测隐藏输入。
- 研究区采用当前公开论文 v2 题名、9位作者机构上标及共同第一/通讯作者标记。Token图的口径是总all tokens除以成功算子数：Regenerate为4,372,531,063/11=397.50M，Debug为511,901,478/18=28.44M，降幅92.8456%。包含失败运行和缓存token；M表示百万token。数值来自冻结artifact的token_accounting_summary.json。
- 论文任务和测例单独放在公开 checkout 的 [dataset/](../papers/2026-dont-regenerate-debug/dataset/)，不改写已经打标签的 artifact。此处提供 27 个参考实现、compact/full 公开输入规格及哈希溯源，区别于近成功候选和 NPU 运行环境；网页通过“论文任务与测例”链接到该目录。
- 评测演进分别说明固定公开任务与社区标准/隐藏集的用途，不声称所有27个任务直接来自生产客户，也不把两个平台的数字合并比较。


## 可安装 Skill 展示

`#skills` 保留原有灰绿分区、能力卡、安装命令切换与复制按钮，展示当前原生宿主协议。Codex、Claude Code、Kimi Code 和其他兼容 Agent 分列为四个安装选项；安装目标只选择 Skill 目录，不选择引擎内部模型。Kimi Code 安装到 `~/.config/agents/skills/drdebug`，再用 `kimi --skills-dir "$HOME/.config/agents/skills"` 显式加载 skills 父目录；该参数替代本次会话的用户／项目自动发现，可重复指定以加入其他技能目录。其他 Agent 保留通用 `--dest` 入口，使用宿主实际识别的 skills 父目录。

宿主使用自己的模型、工具和对话；CLI 只运行确定性步骤，不启动嵌套 worker 或调用 provider。`next` 返回请求 ID、隔离候选目录与完整指令，宿主编辑候选并在其外部写真实结果 JSON；`submit --request-id ID --result RESULT_JSON` 进行验证和推进。循环到终态后独立 `verify --casewise --json`。宿主的 completed 不表示内核通过。

新 schema-4 工作区默认最多 5 次尝试、86,400 秒总墙钟，包含待处理宿主时间与确定性命令；不伪造 turns。旧 run 入口退役，历史工作区只保留原合同下的 status／受支持独立 verify，不热迁移。

统一安装器在 `../install.sh`，当前双语指南在 `../docs/skill-guide.zh-CN.md` 和 `../docs/skill-guide.md`。安装仅需 Python 3.10+ 与标准库；NPU 执行依赖另行说明，doctor 不要求 Claude 或其他宿主 CLI。模型密钥、权限和 SSH 由宿主管理。网页不执行安装器或 CLI。

Cumsum 教程与全新安装验证记录分别链接，并随语言切换。原生宿主记录包含真实请求、提交、源码身份与独立 51 例验收；使用公开参考引导，宿主配置与验证边界详见结果矩阵。历史验证回执单独保留。安装、环境预检、历史重放和新的真实修复分别表述，论文归档保持不变。
