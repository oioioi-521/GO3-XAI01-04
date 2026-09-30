# RISE 正式跑批交接

状态：代码与 6 个正式配置已对齐统一契约，采样预算门禁已通过并冻结为 4000 masks；ResNet50/ImageNet 正式单元已于 2026-09-30 完成并通过立即续跑验收，其余 5 个单元待跑。三份 VOC-20 checkpoint 已在 Issue #4 完成团队验收，但当前机器尚未取得被 Git 忽略的二进制文件；开始 VOC 跑批前须通过组内批准渠道接收并按 manifest 校验。门禁数据与决策见 [`RISE_SAMPLING_RESULTS.md`](RISE_SAMPLING_RESULTS.md)，正式结果见 [`RISE_FORMAL_RESULTS.md`](RISE_FORMAL_RESULTS.md)。

## 已锁定的正式契约

- 3 模型 × ImageNet/VOC 共 6 个 `eval` 单元，每单元 460 张；
- 真值 target；ImageNet 使用 softmax，VOC 多标签使用 sigmoid；
- `faithfulness_morf_auc_raw` 使用 21 个删除比例点；
- attribution 计时前执行 1 次不计时 warmup；
- 同时保存 PNG 与 float32 NPY，供冻结后的稳定性协议独立补跑；
- 正式预算已依据 1 张成本 smoke 与冻结 40 张门禁确定为 4000 masks。

## 采样预算门禁（已完成；以下命令用于复核或更换 GPU）

在最终跑批 GPU 上先执行 1 张 smoke：

```powershell
.venv\Scripts\python.exe analysis/rise_sampling_gate.py `
  --config configs/rise_resnet50_imagenet.yaml `
  --budgets 1000 2000 4000 `
  --max-images 1 `
  --output analysis/pilots/rise_budget_smoke.csv
```

确认显存与耗时后，在同一 GPU 上对冻结的 40 张 debug 集执行：

```powershell
.venv\Scripts\python.exe analysis/rise_sampling_gate.py `
  --config configs/rise_resnet50_imagenet.yaml `
  --budgets 1000 2000 4000 `
  --max-images 40 `
  --output analysis/pilots/rise_budget_40.csv
```

CSV 同时记录每图耗时、相对最高预算图的 Pearson 相关与 MAE。评审时据 40 张分布选择预算，并将 6 份 YAML 的 `num_masks` 一次性冻结为同一值。

## 本机审计（2026-09-23）

- ImageNet/VOC 原图均为 500 张，冻结 metadata 均为 eval=460、debug=40；
- RTX 4060 Laptop GPU 8GB 可见，但最初环境为 CPU 版 PyTorch；
- ImageNet ResNet50 权重已缓存；
- Issue #4 已确认三份 VOC-20 checkpoint 的训练、严格加载和正式使用均通过验收；本机的 `models/checkpoints/` 目录仍缺失，必须从组内批准渠道接收，并按 `docs/VOC20_CHECKPOINT_MANIFEST.json` 校验字节数与 SHA-256 后再跑 VOC；
- Issue #8 的稳定性实现当前仍在 `feat/stability-pilot`，其文档标记为候选协议。冻结并合入后使用独立补跑入口，不重复 MoRF，也不计入 `efficiency_time_ms`。

正式单元示例：

```powershell
.venv\Scripts\python.exe experiments/run_unit.py --config configs/rise_resnet50_imagenet.yaml
```

每个单元验收首轮日志 `processed=460, skipped=0, warmup_runs=1`；立即续跑应为 `processed=0, skipped=460`。结果摘要、预算门禁 CSV、配置快照、运行日志和失败重跑记录均需入库；大体积 maps 可按团队约定外部保存。
