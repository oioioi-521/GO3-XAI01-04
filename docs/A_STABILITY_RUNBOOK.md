# A 的 12 单元稳定性补跑

状态（2026-10-01）：协议已通过 C/D 复核并合入集成分支，冻结记录已同步。
本文件和跑批入口已准备；尚未启动 12 个 460 张正式稳定性单元。

## 输入与执行

使用原 GPU 实验副本的完整 `data/`、`results/`、ImageNet 权重缓存和三个 VOC20
checkpoint。保持 12 份基础 YAML 原样，原结果须是每单元 460 张的忠实性/效率结果。
在项目根目录使用 CUDA 版项目 Python：

```bash
python scripts/run_a_stability.py --preflight
python scripts/run_a_stability.py
```

预检逐单元核对冻结 eval ID、图片是否存在、VOC checkpoint 哈希、基础指标覆盖、
基础汇总配置身份，以及 460 张有限的 float32 NPY。运行入口要求可用 CUDA，
不在 CPU 上意外启动完整跑批。

只执行一个单元时：

```bash
python scripts/run_a_stability.py --unit ig_resnet50_imagenet
```

跑批串行执行。每个单元先把首图和对应基础指标复制到
`results/a_stability_gates/<method>_<model>_<dataset>/`，在副本上完成单图、立即续跑和
trace 验收；来源重算的 sidecar 只写在门禁副本。随后使用未修改的原 YAML 补跑
正式结果，并立即再次执行，确认 `processed=0, skipped=460`。
门禁和正式结果使用独立长表、trace 和状态，重复执行入口不会用首图状态替换正式状态。

## 验收与恢复

- 首次正式完整运行应为每单元 `processed=460, skipped=0`；中断恢复保留已完成图，
  如实记录 processed/skipped，不把续跑宣称为重新处理 460 张。
- 每个单元恰有 920 行新稳定性指标、2,300 条 trace；ID、target、重复号、配对 seed、
  score/退化状态、逐图均值、valid_rate、单元均值/样本标准差与配置身份均需匹配。
- 跑批开始时备份基础 CSV；每次门禁和正式运行后比较所有原忠实性/效率行，确保
  数值和基础 config_hash 保留；原参考 NPY 的 SHA-256 也须保持不变。
- 验收回执逐单元写入 `results/a_stability_batches/<UTC时间>/validation_report.json`；
  首次运行的基础 CSV 备份保存在同目录。失败时入口停止，修复后重复同一命令续跑。
- 保留 Grad-CAM 的合法常量图，退化重复按冻结协议计 0，另报有效率。

旧图首次绑定时会增加一次原图归因重算，因此总时间包含来源核验；该时间不进入
既有 `efficiency_time_ms`。12 个单元全部验收后再提交正式稳定性结果摘要和交接包。
全组结果齐全后才生成最终三维 Pareto 和 RQ1–RQ3 结论。

协议和验收依据见 [`STABILITY_PROTOCOL.md`](STABILITY_PROTOCOL.md)。
