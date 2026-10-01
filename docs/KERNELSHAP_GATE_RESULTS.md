# KernelSHAP 六单元 40 张候选门禁

完成日期：2026-10-01。

## 结论

`feature_grid_size=7`、`n_samples=2048`、`perturbations_per_eval=16` 的候选参数已在
ImageNet/VOC × ResNet50/DenseNet121/VGG16 六个 debug 单元完成 40 张门禁。六单元均为
`processed=40, skipped=0`；立即续跑均为 `processed=0, skipped=40`。这证明候选参数在
当前机器上可执行、结果可保存且能安全续跑，但 **40 张门禁不是 460 张正式 eval，不能
据此作最终方法排名或冻结正式结论**。

运行环境：Windows、Python 3.13.9、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、
Captum 0.9.0、NVIDIA GeForce RTX 4060 Laptop GPU。ImageNet 使用 torchvision 官方
DEFAULT 权重；VOC 使用项目交接的三个 VOC20 checkpoint。

## 门禁结果

`faithfulness_morf_auc_raw` 越低越好。表中的正确数表示冻结 target 与模型最高分输出一致
的图片数；“正确组 raw MoRF”只在这些图片上计算，避免错误预测的低目标分数人为降低 AUC。
计时只可在本次同机、同协议的六单元内部解释。

| 数据集 | 模型 | 正确数 | 归因耗时 ms（均值 ± std） | raw MoRF 全部 | raw MoRF 正确组 | 配置哈希 |
|---|---|---:|---:|---:|---:|---|
| ImageNet | ResNet50 | 36/40 | 5222.094 ± 90.885 | 0.113755 | 0.126157 | `e98172812911` |
| ImageNet | DenseNet121 | 34/40 | 5013.756 ± 94.110 | 0.041120 | 0.048140 | `b3a5a9ddba36` |
| ImageNet | VGG16 | 31/40 | 10282.054 ± 142.226 | 0.032253 | 0.041043 | `f2511aa1ba80` |
| VOC | ResNet50 | 32/40 | 6895.317 ± 1722.165 | 0.466700 | 0.490698 | `7e1664df54dc` |
| VOC | DenseNet121 | 31/40 | 8819.774 ± 2637.963 | 0.334590 | 0.344281 | `38e5804f7f4c` |
| VOC | VGG16 | 30/40 | 10257.963 ± 154.694 | 0.158647 | 0.175627 | `16d520edc279` |

不同模型的预测正确集合不完全相同，因此不能直接用表中正确组均值进行严格配对比较；正式
跨模型统计应先构造共同正确样本集合。当前数值只作门禁诊断。

## 完整性校验

- `per_image.csv`：480 行，六单元 × 40 图 × 2 指标；每个 metric/单元恰好 40 行。
- `units.csv`：12 行，六单元 × 2 指标；每行 `n=40`。
- 归因产物：240 张 PNG 和 240 张 float32 NPY；每单元各 40 张。
- 全部 NPY 均为 `float32`、`224×224`、有限、非恒定且位于 `[0,1]`。
- 六份立即续跑日志均为 `processed=0, skipped=40`，没有重复追加逐图结果。
- `analysis.validate_results` 已通过；结果文件位于 Git 忽略的 `results/`，不提交大文件。

## 稳定性单图门禁

PR #10 的稳定性 runner 合入 D 分支后，六单元又按 `rgb-gaussian-spearman-v1` 完成了
单图 × 5 repeats 门禁。旧参考图先用当前模型、基础配置和 checkpoint 独立重算；六张图
均在容差内一致，sidecar 均写为 `verified_recomputation`。30 条 trace 全部为 `valid`，
seed 均符合 `dataset/image_id/repeat` 派生规则；立即续跑均为
`processed=0, skipped=1`。

| 数据集 | 模型 | 单图 stability_spearman | valid repeats | 配置哈希 |
|---|---|---:|---:|---|
| ImageNet | ResNet50 | 0.997204 | 5/5 | `25083ef5de85` |
| ImageNet | DenseNet121 | 0.988959 | 5/5 | `91ef149f422c` |
| ImageNet | VGG16 | 0.994510 | 5/5 | `bba2e5602363` |
| VOC | ResNet50 | 0.994388 | 5/5 | `af6eef9a2658` |
| VOC | DenseNet121 | 0.998367 | 5/5 | `f052b6e4a5be` |
| VOC | VGG16 | 0.998959 | 5/5 | `961b8c86ba02` |

这些值只证明参考图绑定、扰动、归因、trace、聚合和 resume 链路在 KernelSHAP 上可执行；
单图数值不能代替 40 张稳定性门禁，更不能代替 460 张正式稳定性结果。

## 下一阶段边界

基础 40 张门禁和稳定性单图门禁通过后，下一步仍需完成六单元 40 张稳定性门禁；参数
正式冻结后才能生成 460 张 eval 配置并运行六个主体单元。正式统计应沿用正确预测主分析、
全样本敏感性分析、共同正确样本配对和同硬件效率比较的边界。
