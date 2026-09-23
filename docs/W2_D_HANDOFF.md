# 成员 D 交接：KernelSHAP 接入与统计分析骨架

更新时间：2026-09-23。

## 已完成并可独立验收

- 开发分支 `feat/kernelshap-analysis` 已接入 `origin/integrate/a-voc-eval` 的当前代码；
  这是 D 的分支工作，**没有合并 `main`**。
- KernelSHAP 适配器已注册到统一 runner：单图接口、规则网格特征、归一化黑色基线、
  固定随机种子、二维 `[0,1]` 输出。三通道 SHAP 值按有符号算术均值聚合，保持
  MoRF 对正/负贡献的排序语义，不把强负贡献用绝对值变成“高重要性”。
- 1 个离线 smoke 配置和 6 个 40 张候选门禁配置（3 模型 × ImageNet/VOC）。成本试跑后，
  D 选择 `feature_grid_size=7`、`n_samples=2048` 作为下一阶段候选；它们仍不是正式参数。
- 已对齐 A 的 `faithfulness_morf_auc_raw`（21 个删除比例点）、归因预热、PNG 与
  float32 NPY 保存，以及 VOC20 `sigmoid` 多标签输出契约。
- 结果 schema 校验、聚合、三因素 ANOVA 与效应量、假设诊断、Friedman 对照、Pareto、
  Pearson/Spearman 分析骨架。
- 自动化测试既覆盖假后端接口与续跑，也覆盖真实 Captum 在合成图/小模型上的调用、
  runner → CSV → PNG/NPY 链路。**合成结果不能作为课程实验结果。**

## 当前依赖与阻塞

1. C/B/A 的接口代码已在集成分支，但相关 PR 尚未合并到 `main`。D 分支目前基于该
   集成状态；正式提 PR 前需确认目标分支与 review 顺序。
2. 本工作区的项目 `.venv` 已安装 Captum 0.9.0 与 scikit-learn；根依赖文件已补上
   KernelSHAP 实际需要的 `scikit-learn`。尚未安装/复制真实图片、ImageNet 权重缓存与
   B 的 VOC20 checkpoint。
3. B 已有 VOC20 checkpoint 的交接清单和哈希，但权重二进制不进 Git；必须通过获批准
   的共享渠道取得并校验，不能用 ImageNet 1000 类输出替代。
4. 六份当前 YAML 是 40 张候选门禁口径，不是正式 eval 配置。首图和 10 图成本曲线已经完成；
   需在目标 GPU 上完成三模型、两数据集共六单元的 40 张运行，检查失败率、有限性、
   成本与归因图质量，再决定是否冻结为正式参数。
5. D 已完成 PR #10（提交 `ab60d54`）统计聚合和退化计分复核并给出通过结论；协议仍需
   C 完成 runner/schema 复核后才能标记 frozen。正式 ANOVA、三维 Pareto、相关性和 RQ1–RQ3 最终结论
   仍需真实全组结果和统一统计方案。A 的阶段结果不能被当成全项目最终结论。

## 已完成的本地验证与后续命令

```bash
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
# 以下命令要有冻结的 ImageNet debug 图片，当前工作区尚不具备：
.\.venv\Scripts\python.exe experiments\run_unit.py --config configs\kernelshap_smoke_imagenet.yaml
```

smoke 配置使用随机模型、1 张图片和 8 次采样，只证明管线能运行，不可写入报告的性能结论。
六份候选配置当前使用 `n_samples=2048`、`feature_grid_size=7`、`max_images=40`。选择依据是
10 张 ResNet50/ImageNet 试跑相对 4096 样本参考的平均 Spearman 0.853、最低 0.782，
以及约 3.07 秒/图的归因成本。40 张六单元门禁完成前，不得把该候选写成正式参数。
已知本地冻结数据 CSV 的
Windows CRLF 换行会导致 SHA-256 校验失败；工作区已机械恢复为仓库规定的 LF，
没有改动标签内容。
