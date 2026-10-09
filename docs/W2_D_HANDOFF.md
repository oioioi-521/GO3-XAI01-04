# 成员 D 交接：KernelSHAP 接入与统计分析骨架

更新时间：2026-10-09。

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

## 当前依赖与阻塞

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
   均已确认。正式批次按单元先生成独立 eval 基础指标/float32 图与 provenance，再执行
   460 张稳定性补跑；结果存于 `results/kernelshap_eval/`，完成前不当作最终统计结论。
   六个 460 张 Occlusion stability eval
   尚未开始，正式 ANOVA、三维 Pareto、相关性和 RQ1–RQ3 最终结论仍需真实全组结果。
   A/B/D 的门禁不能被当成全项目最终结论。
6. KernelSHAP 的 ImageNet/ResNet50 与 ImageNet/DenseNet121 两个 460 张正式 eval 单元
   已由 `analysis.validate_kernelshap_formal` 独立验收：各有 460 张基础/稳定性逐图指标、
   2,300/2,300 条有效重复 trace、完整 float32 参考图来源和立即续跑
   `processed=0, skipped=460`。测量值及配置哈希见 `docs/KERNELSHAP_GATE_RESULTS.md`。
   其余四单元正在补跑；整批完成前仍不生成六单元或全组最终排名。

## 已完成的本地验证与后续命令

```bash
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
# 以下命令要有冻结的 ImageNet debug 图片，当前工作区尚不具备：
.\.venv\Scripts\python.exe experiments\run_unit.py --config configs\kernelshap_smoke_imagenet.yaml
```

smoke 配置使用随机模型、1 张图片和 8 次采样，只证明管线能运行，不可写入报告的性能结论。
六份候选配置使用 `n_samples=2048`、`feature_grid_size=7`、`max_images=40`。选择依据是
10 张 ResNet50/ImageNet 试跑相对 4096 样本参考的平均 Spearman 0.853、最低 0.782，
以及约 3.07 秒/图的归因成本。稳定性六单元 40 张门禁已完成，但在完成各单元 460 张
eval 稳定性补跑前，不得把门禁数值写成正式实验排名。正式补跑还依赖 eval split 的
KernelSHAP 基础指标表、float32 参考归因图及 provenance；这些必须与 debug 输出隔离。
已知本地冻结数据 CSV 的
Windows CRLF 换行会导致 SHA-256 校验失败；工作区已机械恢复为仓库规定的 LF，
没有改动标签内容。
