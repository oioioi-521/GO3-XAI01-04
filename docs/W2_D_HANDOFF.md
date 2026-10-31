# 成员 D 交接：KernelSHAP 与统计分析

更新时间：2026-10-16。

## 2026-10-16 后续收尾

已在 D 分支本地集成 PR #17 固定提交 `7eb0a316b097073fe43da1f95e2ec9c8897c356a`，
解决交付入口冲突并保留 A 的运行证据和 D 的统计交付。LIME 通用门禁 P1/P2 已增加
记录/统计重算、输入字节/张量与执行身份绑定、写出前检查及独占输出锁；旧目录只读，
不自动迁移或回填来源。参考 LIME、向量化归因算法及已有正式测量值未修改。
详见 [集成修复记录](D_DCU_INTEGRATION_REVIEW_20261016.md)和
[E/A 可直接整合清单](D_TO_E_HANDOFF_20261016.md)。没有启动新 GPU 实验、改动 main 或发送微信群消息。

## 当前交付：六方法 DCU 表格统计已完成

已收到 `dcu_team_handoff_20261014.zip`（SHA256
`14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc`）。
36 单元、66,240 行逐图指标、144 行汇总、82,800 条 trace、预测/正确组及 90 份执行配置
已通过本仓只读独立检查；没有执行归档脚本，没有启动或恢复 GPU 模型。

已完成模型命中组配对主分析、共同命中组/全样本敏感性分析、共同命中组跨模型比较、
Friedman 秩置换、Wilcoxon/符号检验、配对效应量、分层 bootstrap CI、Holm 校正、
方法内探索性相关及质量/记录时间 Pareto。分析以原图为单位，不把 5 次噪声重复视为独立样本。
报告、四图、全部派生 JSON 和主分析质量配对表见
[最新统计报告](DCU_STATISTICAL_ANALYSIS_20261015.md)与[统一入口](D_DELIVERY_INDEX.md)。

该批次与此前本机 GPU 的 KernelSHAP 六单元是不同运行，不能混合耗时或复用旧数值作新结论。
IG 平均 AUC 六组中五组最低，RISE 平均噪声稳定性六组最高；方法选择需看取舍，
不宣布全指标冠军。计时来自两轮共享 DCU，只作记录比较，不宣称受控加速比。
没有独立重算未存盘噪声图，没有证明 LIME 预算收敛，也不宣称普通三因素 ANOVA 完成。
新的表格输入已解除旧“A/C/E 材料不齐”的统计阻塞；旧门禁审查仍限于当时的提交。
E 若需要完整热图 Demo，应获取外部大归档，不影响此处 D 统计交付。

以下按原时间保留实现、门禁和旧运行的历史，不应将历史“待收齐/预算未批”误作当前 DCU 表格状态。

## 已完成并可独立验收

- 开发分支 `feat/kernelshap-analysis` 已接入 `origin/integrate/a-voc-eval` 的当前代码；
  这是 D 的分支工作，**没有合并 `main`**。
- KernelSHAP 适配器已注册到统一 runner：单图接口、规则网格特征、归一化黑色基线、
  固定随机种子、二维 `[0,1]` 输出。三通道 SHAP 值按有符号算术均值聚合，保持
  MoRF 对正/负贡献的排序语义，不把强负贡献用绝对值变成“高重要性”。
- 1 个离线 smoke 配置和 6 个 40 张候选门禁配置（3 模型 × ImageNet/VOC）。成本试跑后，
  D 选择 `feature_grid_size=7`、`n_samples=2048` 作为下一阶段候选。2026-10-01 六单元
  40 张门禁已全部完成并通过结果、数组与续跑校验；它们仍不是 460 张正式结果。
- 已对齐 A 的 `faithfulness_morf_auc_raw`（21 个删除比例点）、归因预热、PNG 与
  float32 NPY 保存，以及 VOC20 `sigmoid` 多标签输出契约。
- 结果 schema 校验、聚合、三因素 ANOVA 与效应量、假设诊断、Friedman 对照、Pareto、
  Pearson/Spearman 分析骨架。
- 自动化测试既覆盖假后端接口与续跑，也覆盖真实 Captum 在合成图/小模型上的调用、
  runner → CSV → PNG/NPY 链路。**合成结果不能作为课程实验结果。**
- D 已完成 PR #13（头提交 `6d5645f`）的 Occlusion 门禁与统计口径静态复核，结论为
  有条件通过；正式比较前需解决跨 GPU 效率混杂、机器可读门禁摘要和六目录合并流程。
  详见 `docs/D_OCCLUSION_GATE_STAT_REVIEW.md`。
- 2026-09-27 已验收 B 的六单元正式 eval 轻量交接包：2,760 条预测、5,520 条逐图
  指标均通过身份、样本集合和汇总重算门禁，并补充正确/错误分组及三模型共同正确样本表。
  faithfulness 主分析冻结为正确组，跨模型配对使用共同正确样本。2026-09-30 又完成
  2,760 张 float32 NPY 和 12 个 stability 单图/40 张门禁的接收验收；门禁不等于 460 张
  正式稳定性结果。详见 `docs/D_OCCLUSION_EVAL_ACCEPTANCE.md` 和
  `docs/STABILITY_D_STATISTICAL_REVIEW.md`。
- 2026-10-09 又独立验收 B 的六单元 460 张 Occlusion 正式稳定性交接包：
  每单元 460 图、2,300 条有效重复记录，逐图/单位汇总、配对种子、配置/协议哈希、
  2,760 张原有 float32 参考图绑定及基础结果未变均已核对；六组立即续跑均为
  `processed=0, skipped=460`。测量值、包哈希和来源限制见
  `docs/D_OCCLUSION_STABILITY_EVAL_ACCEPTANCE.md`。
- 2026-10-09 接收并独立校验 C 的六单元 460 张 RISE 正式基础指标轻量包；2026-10-12 再验通过：
  六组各 460 张、逐图指标与正确/错误组汇总可重算，六组立即续跑均为
  `processed=0, skipped=460`。该包未附原始归因图与权重，因此只接受轻量 CSV 的
  受限统计输入，不接受原图来源完整验收或 RISE 稳定性完成。详见
  `docs/D_RISE_EVAL_LIGHT_ACCEPTANCE.md`。
- 2026-10-12 完成 E 的 LIME 中期包受限接收检查：五张 float32 示例图和 NPZ 一致，
  LIME 图哈希、VOC debug 8173 真值目标、本地 checkpoint/输入图核对通过；原面板不改动复制到
  `analysis/lime_midterm_same_image.png`。六组 40 张门禁仅收到摘要，不能代替原始记录或代码验收；
  六组耗时估算与均值推算未对齐，正式预算未冻结、460 张 eval 未启动。
  限制见 `docs/D_LIME_MIDTERM_RECEIPT.md`，统一可交付入口见 `docs/D_DELIVERY_INDEX.md`。

## 历史依赖与阻塞（后续变更见上方当前交付）

中期汇报已另行整理 [`MIDTERM_D_COMPARISON.md`](MIDTERM_D_COMPARISON.md)：四方法
ImageNet 全样本 raw MoRF AUC 描述图，以及 A 同一 GPU 的 IG/Grad-CAM 成对双目标
Pareto 图和各一句限定范围的结论。该图制作时 C 的 RISE 仅使用已提交的三单元正式
摘要；2026-10-12 已按收到并再次验收的六单元轻量逐图 CSV 更新摘要并重画。
尚未验收 RISE 原始归因图，不替代正确组主分析、稳定性或最终排名。

1. PR #10 已于 2026-09-30 合并到集成分支（merge commit `00d73b4`），并已合入 D
   分支；PR #14 仍为 draft。正式提 PR 前仍需确认目标分支与 review 顺序。
2. 本工作区的项目 `.venv` 已安装 Captum 0.9.0 与 scikit-learn；真实图片、三个 VOC20
   checkpoint、Occlusion 轻量结果和 float32 包均已通过哈希/格式验证并放入 Git 忽略目录。
   这些二进制和结果仍不进入 Git。
3. ImageNet 三个 torchvision 官方 DEFAULT 权重和三个 VOC20 checkpoint 均已在当前机器
   成功加载；二进制仍位于 Git 忽略目录，不进入提交。
4. 六份当前 YAML 是 40 张候选门禁口径，不是正式 eval 配置。六单元已经在目标 GPU
   完成，合计 240 张图、480 条逐图指标、240 张 PNG 和 240 张 float32 NPY；完整性、
   有限性、非恒定性、范围和 resume 均通过。结果与边界见
   `docs/KERNELSHAP_GATE_RESULTS.md`。
5. D 已完成 PR #10 修复提交 `9891eec` 的统计复核，并验收 PR #14 的 Occlusion 单图/
   40 张候选门禁；KernelSHAP 六个稳定性单图门禁也已通过，30/30 repeats valid，旧参考图
   均经独立重算绑定。Issue #8 于 2026-10-01 记录协议冻结并链接 C/D 最终复核；D 已完成
   KernelSHAP 六单元 40 张稳定性门禁：240 张图片、1,200/1,200 repeats valid；每单元均有
   40 行 `stability_spearman`、40 行 `stability_valid_rate`、200 条 trace，seed、逐图聚合、
   单元均值/样本标准差与 `n=40` 已独立重算核对，立即续跑均为
   `processed=0, skipped=40`。完整测试 89 passed、2 条 `torch.load` FutureWarning，
   `pip check` 正常。详见 `docs/KERNELSHAP_GATE_RESULTS.md`。六份 KernelSHAP eval 配置
   已按冻结候选参数生成并通过 460 张数据清单预检；ImageNet 权重缓存、VOC checkpoint
   均已确认。正式批次已按单元生成独立 eval 基础指标、float32 图与 provenance，并完成
   460 张稳定性补跑；结果存于 `results/kernelshap_eval/`，六单元验收证据见下项。
   六个 460 张 Occlusion stability eval 已由 D 接收验收；正式 ANOVA、三维 Pareto、
   相关性和 RQ1–RQ3 最终结论仍需真实全组结果。
   A/B/D 的门禁不能被当成全项目最终结论。
6. KernelSHAP 六个 460 张正式 eval 单元已由 `analysis.validate_kernelshap_formal` 全部
   独立验收：每单元各有 460 张基础/稳定性逐图指标、2,300/2,300 条有效重复 trace、
   完整 float32 参考图来源和立即续跑 `processed=0, skipped=460`；六单元合计 2,760 张图、
   13,800 条有效 trace。批次退出码为 0，完整测试 95 passed、2 条既有 FutureWarning，
   `pip check` 无冲突。测量值及配置哈希见 `docs/KERNELSHAP_GATE_RESULTS.md`，方法内描述图
   原位于 `results/analysis/kernelshap_formal_summary.png`，可交付图现位于
   `analysis/kernelshap_formal_summary.png`。这完成的是 KernelSHAP 六单元，
   全组 36 单元排名、ANOVA、相关性和最终 Pareto 仍依赖其他方法完整且可配对的正式输入。
7. 2026-10-12 重验 KernelSHAP 六单元和 RISE 六单元基础轻量包通过；LIME 中期
   示例受限接收、图表交付入口与 KernelSHAP 计时离群审计已完成。完整回归测试
   108 passed、2 条既有 FutureWarning，`pip check` 正常；未启动新 GPU 实验或改动原始结果。
8. 2026-10-13 对 E 的固定提交 `64d2ffde8a1c206d6b5bac9cc8b2c83622e702cb` 完成
   LIME 代码审查与隔离工程复测：专项 26 passed，全仓 94 passed、2 条既有 warning，
   依赖检查通过；两个 CPU 合成 smoke 的 map SHA 一致且立即续跑 0/1。
   D 附加探针为 1 passed、3 strict xfailed，复现了门禁续跑记录验证和输入图片字节身份缺口。
   这三项 xfail 是缺陷，不是验收通过；预算不批准冻结，仍需 E 修复及原始门禁回执。
   审查、复现及 E 待交清单见 `docs/D_LIME_CODE_REVIEW_20261013.md`。
   没有替 E 跑真实 GPU 门禁、启动 460 张 eval 或重新运行 D 的正式实验。
   D 分支收尾仍为 108 passed、2 条既有 warning，pip check 无冲突；pytest 默认
   搜集已排除 ignored results/outputs，防止审查副本与第三方依赖测试污染本项目回归。
9. 2026-10-14：D 为下一次 E 原始包验收准备 `analysis.validate_lime_budget_gate`，
   对完整 40 图逐个验证 runs/target/map SHA，并从 float32 NPY 独立重算预算比较、
   跨 seed 样本方差和 summary。成功报告仍不批准正式预算或证明生成时来源。
   输入、命令、缺项与交接顺序见 `docs/D_LIME_GATE_VALIDATION.md`。
   E 远端仍为 `64d2ffd`；没有新原始门禁包可独立验收，未启动真实 GPU 工作。
   新增校验器 25 项测试通过；D 全仓 133 passed、2 条既有 warning（135.70 s），
   `pip check` 无冲突，`git diff --check` 通过。数字表示软件测试，不是 36 单元结果。

## 已完成的本地验证与后续命令

```bash
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
# 以下仅为可选工程 smoke，不需要重跑已验收的正式实验：
.\.venv\Scripts\python.exe experiments\run_unit.py --config configs\kernelshap_smoke_imagenet.yaml
```

smoke 配置使用随机模型、1 张图片和 8 次采样，只证明管线能运行，不可写入报告的性能结论。
六份候选配置使用 `n_samples=2048`、`feature_grid_size=7`、`max_images=40`。选择依据是
10 张 ResNet50/ImageNet 试跑相对 4096 样本参考的平均 Spearman 0.853、最低 0.782，
以及约 3.07 秒/图的归因成本。稳定性六单元 40 张门禁和六单元 460 张正式 eval 均已完成；
门禁数值与正式结果继续分开报告。正式结果使用 eval split 的 KernelSHAP 基础指标表、
float32 参考归因图及 provenance，并与 debug 输出隔离。
已知本地冻结数据 CSV 的
Windows CRLF 换行会导致 SHA-256 校验失败；工作区已机械恢复为仓库规定的 LF，
没有改动标签内容。
