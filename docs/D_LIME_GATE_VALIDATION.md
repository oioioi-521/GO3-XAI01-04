# D 的 LIME 原始预算门禁独立校验入口

更新日期：2026-10-16。此工具支持旧 schema 1 的只读接收与新 schema 2 门禁验收。
新 DCU 正式表格包已接收并完成统计，但不等于本工具已拿到全部服务器预算门禁 NPY。
以下软件测试材料都是显式合成 fixture，不能当作六组真实门禁结果。

schema 2 由 [修复后的通用门禁](D_DCU_INTEGRATION_REVIEW_20261016.md)生成，增加源图片字节、
实际预处理张量、代码/机器/device 身份。校验器会从本机受控源图片重算预处理张量 SHA，
核对 plan/record 绑定和动态 summary 计数。schema 1 仍允许缺省来源字段，但明确不证明其来源。
两个格式均不自动批准预算或证明原始生成过程；新门禁不得复用旧目录。

2026-10-16 最终软件复验：本校验器 **29 项通过**；集成全仓 **227 passed、2 条既有 warning**，
`pip check` 无冲突。新格式测试核对输入/张量绑定及拒绝变更，历史格式 25 项保留。

历史软件验证（2026-10-14）：新增 **25 项通过**；当时 D 全仓 **133 passed、2 条既有 warning，135.70 s**，
`pip check` 无冲突。日志在 ignored `results/e_lime_review_20261013/d_branch_tests_20261014.log`。
这与 E 分支原有的 94 项测试、D 原有的 108 项测试分别记录，不混成实验完成数量。

## 已准备好的工具

[`analysis/validate_lime_budget_gate.py`](../analysis/validate_lime_budget_gate.py) 不导入 E 的
adapter/runner，不执行交接包中的 Python、PowerShell 或 shell 文件。它读取单个门禁目录的
plan、40 份 records、原始 NPY 与 summary，以本项目受控 metadata 为依据独立校验。

```powershell
.\.venv\Scripts\python.exe -m analysis.validate_lime_budget_gate `
  results\received_e_gate\resnet50_voc `
  --report results\analysis\lime_gate_resnet50_voc_validation.json
```

这是收到真实文件后的命令示例，不表示该目录现在存在。工具只接受已安全解压的目录；
ZIP 本身仍须先校验路径、CRC、大小和整包 SHA，不执行里面的脚本。
报告路径必须位于被核对的门禁目录之外，不改写原始输入；不指定 `--report` 时只打印结果。

## 输入与验收边界

每个单元分别提供以下目录，不把新服务器和旧 4080 的不同设置混成一个目录：

```text
unit_gate/
├── plan.json
├── records/              # 40 个按 image_id SHA 前缀命名的 JSON
├── maps_float/           # 每图预算 × seeds 的原始 float32 NPY
├── summary.json
├── timings.csv           # 原 CSV 保留，工具不将它作为统计真值
├── comparisons.csv
├── variance.csv
├── failures.jsonl        # 如有失败/重试
└── 运行日志与环境/代码/机器说明
```

工具实际核对：

- plan 的 canonical 配置身份、完整 debug 40 张冻结 metadata 字节 SHA/ID 顺序、
  ImageNet softmax / VOC sigmoid、预算/seed 集合、次数和环境版本字段。
- 每图 target、config hash，以及预算 × seeds 的唯一完整组合；两预算 × 三 seeds 应有
  240 runs、120 预算比较、80 条 budget variance、240 个 seed pairs。
- 每份 map 的受限相对路径、字节 SHA、`float32 224×224`、有限性和 `[0,1]` 范围。
- 从原始 map 重算逐图预算 Spearman/RMSE、跨 seeds 的像素样本方差（ddof=1）及所有
  seed-pair 比较，并核对 JSON records 和 summary；常量图明确计数，不静默丢弃。
- 时间字段非负有限，以及按完整 timed runs 重算的均值/总和；允许负的有限 weighted R²。
- 如记录提供 `source_image_sha256`，与 D 本机受控 metadata 路径对应的图片字节独立核对。

不自动验收：

- **正式预算是否够用**：没有自动相似度阈值，不把低相似度图掩盖为均值通过。
- generation-time 输入/权重/环境来源、服务器实际运行与代码版本、暂停/恢复/重试完整历史。
  schema 1 缺少输入字节绑定，schema 2 已要求生成/续跑检查；后补 SHA 仍不能证明原生成时来源。
- 原 CSV 与 JSON 是否一致、实际 weighted R² 是否来自正确回归、代码是否执行了正确模型。
  工具以 NPY/records 为统计依据；CSV、完整日志与来源仍需另审。
- 总墙钟时间：`current_invocation_wall_seconds` 仅是当前调用，不是经历恢复的全部耗时。
- 460 张正式 eval、Issue #8 RGB-noise 稳定性、全组排名/ANOVA/Pareto。

因此成功报告也明确包含 `formal_budget_approved=false` 和
`generation_time_provenance_verified=false`。只证明所交 arrays 与统计记录内部一致，
不能据此直接把六份配置的 `budget_frozen` 改为 true。

## E 修复与交接顺序

1. 按 [D 固定提交审查](D_LIME_CODE_REVIEW_20261013.md) 修复完整记录读取验证、
   target/统计复算以及输入图片字节身份绑定；提供新的固定提交和配置。
2. 六组原始门禁分别交付，保留完整计划、maps、records、日志和代码/机器信息。
   三个中断单元需完成 40 图，不用单图 smoke 或 pilot 代替。
3. D 执行该工具并审查低相似度样本、计时异常和来源，再共同确定预算。
4. 审阅通过后才冻结正式配置并启动隔离的 460 张 eval。

可给 E 的简短说明（未代发）：

> D 已完成 `64d2ffd` 的工程复测和代码审查。现有测试通过，但门禁续跑会接受缺记录、
> 错 target/统计，且输入图片替换不改变计划身份。请先按审查文档修复，发六组原始门禁
> 的 plan/records/maps/summary/日志与固定代码 SHA；完成三组中断门禁后再一起审核预算，
> 暂不要直接冻结并跑正式 eval。D 已准备好独立 arrays/统计校验器。
