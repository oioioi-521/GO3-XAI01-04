# A 的 12 单元稳定性补跑

状态（2026-10-01）：协议已通过 C/D 复核并合入集成分支，冻结记录已同步。
12 个 460 张正式稳定性单元已在 desktop / RTX 2080 Ti 上完成并通过验收。
整批于 2026-10-01 22:30（Asia/Shanghai）结束，退出码 `0`，立即续跑均为
`processed=0, skipped=460`。正式汇总见 [A_STABILITY_RESULTS.md](A_STABILITY_RESULTS.md)。

## 本次执行记录

- 代码提交：`a96749ac153c80f01e7cd32c1384f17a331eca5b`，GPG 签名有效；
- 独立远端目录：`/home/hycx233/Courses/machine-learning/GO3-XAI01-04-a-stability-eval`；
- 环境：Python 3.13.9、PyTorch 2.8.0、Captum 0.9.0、SciPy 1.17.1、CUDA；
- 全部 12 个单元的输入预检已通过，5,520 张原参考 NPY 就位；
- 后台日志：`results/a_stability_batch_20261001.log`；
- 进程文件：`results/a_stability_batch_20261001.pid`；完成/失败退出码写入
  `results/a_stability_batch_20261001.exit`，`0` 为整批验收成功；
- 每个单元完成后写入 `results/a_stability_batches/<UTC时间>/validation_report.json`。

本次回执位于 `results/a_stability_batches/20261001T132233566810Z/validation_report.json`。
完成后复核了全部 trace、基础 CSV、预测表和 5,520 张原参考 NPY，均通过；本地已取回
`results/a_stability_eval_20261001/A_STABILITY_EVAL_LIGHT_20261001.zip`，ZIP 和包内
逐文件 SHA-256 与实验机一致。该交接包包含表格、trace、日志、快照、状态、单图门禁、
原图 provenance 与验收回执，原参考 NPY 沿用已有正式产物。

使用 `nohup` 运行，SSH 断连不会中断批次。原实验副本保留，当前结果是其独立副本，
不会改写原副本的 CSV 或 NPY。若退出码非零，先检查日志末尾错误，再重用同一命令续跑。

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
