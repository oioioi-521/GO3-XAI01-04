# W2 成员 C 历史交接说明：RISE + pipeline

> 本文记录 2026-09-16 的初始实现。正式配置、4000-mask 门禁结果和当前运行要求以
> [`RISE_FORMAL_HANDOFF.md`](RISE_FORMAL_HANDOFF.md) 与
> [`RISE_SAMPLING_RESULTS.md`](RISE_SAMPLING_RESULTS.md) 为准。稳定性 runner/schema
> 最终复核见 [`STABILITY_C_RUNNER_REVIEW.md`](STABILITY_C_RUNNER_REVIEW.md)。

## 已完成

- RISE：批量随机掩码、随机平移与双线性上采样、固定种子、归一化黑色基线。
- `run_unit.py`：YAML 驱动、单元运行、断点续跑、配置快照、逐图与单元 CSV、运行日志。
- W2 指标：RISE wall-clock 耗时，以及 MoRF 删除曲线 AUC（越低越好）。
- 6 个 RISE 单元配置：3 模型 × ImageNet/VOC；另有一个离线 smoke 配置。
- 数据版本冻结：`data/DATA_VERSION.json` 记录规模与四个关键元数据文件的 SHA-256。
- 结果合并：`experiments/merge_results.py` 去重合并逐图结果并重建单元汇总。

## 演示命令

```bash
# 1. 无需下载模型权重的管线冒烟测试（随机权重，不可作为实验结论）
python experiments/run_unit.py --config configs/rise_smoke_imagenet.yaml

# 2. 正式 ImageNet 单元（当前配置不再是小规模 debug）
python experiments/run_unit.py --config configs/rise_resnet50_imagenet.yaml

# 3. 再次执行同一命令会跳过已完成图片；显式重跑整个单元
python experiments/run_unit.py --config configs/rise_resnet50_imagenet.yaml --force

# 4. 测试
pytest -q

# 5. 校验共享数据是否与冻结版本一致
python preprocessing/verify_data_version.py
```

结果写入 `results/per_image.csv`、`results/units.csv`、`results/run_log.jsonl`，灰度归因图写入
`results/maps/`，运行时配置快照写入 `results/configs/`。`results/` 按组内约定不提交 Git，请通过共享盘交接。

## 与 B 对接的阻塞项

ImageNet 模型输出 1000 个 ImageNet 类别，不能把 VOC 的 0–19 标签直接当作其输出类别。三个 VOC 配置因此要求 B 提供以下 20 类分类 checkpoint：

- `models/checkpoints/vgg16_voc20.pt`
- `models/checkpoints/resnet50_voc20.pt`
- `models/checkpoints/densenet121_voc20.pt`

checkpoint 应为完整 `state_dict`（也可包装在 `{"state_dict": ...}` 中），类别顺序必须与
`preprocessing/extract_voc_labels.py` 中的 `VOC_CLASSES` 一致。三份权重已经 Issue #4
验收；本机于 2026-10-09 从 `muti-data` 接收并通过大小、SHA-256、严格加载、20 类
有限前向及 RISE 首图检查，详见 [`C_VOC20_RECEIPT.md`](C_VOC20_RECEIPT.md)。
三个 VOC 正式单元已安排顺序跑批。

## W4 正式运行前调整（已于 2026-09-23 完成）

6 份 YAML 已统一为 `eval`、`max_images: null`、4000 masks、21 点 raw MoRF、一次不计时 warmup，并保存 float32 NPY。正式跑批前仍须由 B 复核 VOC checkpoint、目标类别和首批热力图。
