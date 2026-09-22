# 正式稳定性协议与补跑入口

状态：**候选实现，等待 Issue #8 的 C/D 复核后冻结**。B 已于 2026-09-22
确认本口径可用于 Occlusion。候选依据和 10/40 张结果见
[`STABILITY_PILOT_RESULTS.md`](STABILITY_PILOT_RESULTS.md)。

## 协议 v1

唯一协议文件为 [`configs/stability_protocol_v1.yaml`](../configs/stability_protocol_v1.yaml)：

- 反归一化到 RGB `[0,1]` 后加入零均值高斯噪声，`sigma=0.005`，裁剪到
  `[0,1]` 后重新归一化；
- 每图 5 次，target 固定为 metadata 真值；
- seed 由 `dataset/image_id/repeat` 的 SHA-256 派生，不包含模型和方法，因而相同
  图片与重复在所有模型、方法间复用同一扰动；
- 主指标为全像素 Spearman，方向为 `max`；top-10% Jaccard 仅作诊断；
- 常量图、非有限图或未定义相关系数保留明确状态；该重复的正式 `score=0`，不伪造
  相关系数；
- 每图 `stability_spearman` 是 5 个正式 score（含退化 0 分）的均值，
  `stability_valid_rate` 是有效重复占比；
- 不保存扰动归因图，不重复 MoRF，扰动归因耗时不计入 `efficiency_time_ms`。

## 为什么使用独立补跑入口

正式忠实性和效率已经完成。若直接给原 YAML 增加稳定性字段，配置哈希变化会使现有
resume 逻辑删除旧单元结果并重跑 MoRF。`experiments/run_stability.py` 因此采用补跑设计：

1. 读取原正式 YAML，以完全相同的模型、checkpoint、方法参数和真值 target 重新构建
   归因器；
2. 从 `results/maps_float/<method>_<model>_<dataset>/<image_id>.npy` 读取原始
   float32 `A(x)`，缺失、dtype/shape 不符或非有限时立即失败；
3. 只计算 5 个 `A(x+delta)`，向统一 `per_image.csv` 追加两个稳定性 metric；
4. 只替换 `units.csv` 中本单元的两个稳定性汇总，已有忠实性/效率行及其配置哈希保持
   不变；
5. 使用独立状态文件和 trace，部分写入会在续跑时清理，不产生重复结果。

## 输出 schema

统一长表保持既有列不变：

```text
results/per_image.csv
image_id,dataset,model,method,metric,value,time_ms

results/units.csv
method,model,dataset,metric,mean,std,n,config_hash
```

新增的 per-image metric：

- `stability_spearman`：5 次 score 的均值；
- `stability_valid_rate`：状态为 `valid` 的重复比例。

逐重复审计表默认为 `results/stability_trace.csv`，包含：

```text
image_id,dataset,model,method,repeat,seed,protocol_version,config_hash,target,
sigma,status,spearman,score,top10_jaccard,original_pred,perturbed_pred,
prediction_preserved,original_target_score,perturbed_target_score,
target_score_abs_delta,perturbation_mae_pixel,attribution_time_ms
```

退化重复的 `spearman` / `top10_jaccard` 留空、`score=0`，防止下游误把 0 当作
实际相关系数。trace 同时保存协议版本和补跑配置哈希。配置快照包含完整原 YAML、原配置
哈希、协议文件和有效 `max_images`。

## 运行与门禁

协议冻结前只允许单图验证：

```bash
python experiments/run_stability.py \
  --config configs/ig_resnet50_imagenet.yaml \
  --protocol configs/stability_protocol_v1.yaml \
  --max-images 1
```

确认输出、trace、续跑和已有 metric 保留无误后，正式运行去掉 `--max-images`：

```bash
python experiments/run_stability.py \
  --config configs/ig_resnet50_imagenet.yaml \
  --protocol configs/stability_protocol_v1.yaml
```

`--force` 只清理并重跑当前单元的稳定性 metric/trace，不会删除原忠实性和效率结果。
每个单元验收 `processed=460`、`skipped=0`、`warmup_runs=1`，随后再运行一次确认
`processed=0`、`skipped=460`。12 个 A 单元全部完成后才进入三维 Pareto 和 RQ2/RQ3。
