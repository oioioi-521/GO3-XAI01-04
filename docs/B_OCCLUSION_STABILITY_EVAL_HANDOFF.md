# B：六组 Occlusion 正式稳定性 eval 交接（2026-10-01）

## 验收结论与边界

ResNet50、DenseNet121、VGG16 × ImageNet、VOC 的六组 **460 张 eval 稳定性补跑均完成并通过逐组验收**。每组有 460 个唯一 eval image_id、2,300 个互异 `(image_id, repeat)` trace、920 行新增稳定性逐图指标；六组合计 2,760 个单元内图片、13,800 条 trace、5,520 行新增指标。13,800 次均为 `valid`，没有失败或退化计分；六组立即 resume 均为 `processed=0, skipped=460`，没有重复追加。

这是 B 的正式 Occlusion 稳定性结果，不等于 36 个全组单元齐全，也不代表跨方法最终排名。本交接不代替 C/D 对全组汇总、Pareto 或报告的最终验收。Issue #8 正文已经冻结协议 v1；PR #10 已由 `00d73b4` 合入集成分支。C 的 `09e850e` runner/schema 复核和 D 的 `174b050` 统计复核及 B 单图/40 张门禁接收均已完成，不能再列为待确认事项。A 的 PR #16 进展不计作 B 的六组结果。

## 协议、代码与输入身份

- 冻结协议 Git blob：`configs/stability_protocol_v1.yaml`，SHA-256 `324315f1ecf62a807ec91e77b2cba1ea53593e9a7ad98a2a90cce09a9e32dcdf`。Windows `core.autocrlf` 将工作树文件变成 CRLF，工作树字节哈希不同；**冻结的是 Git blob 与解析后参数**，未改协议。
- A 稳定性算法/来源验证来自 `9891eecc4ec869c0bc14a054a2cd85a0beca4266`。B 基础 eval 原运行提交为 `6d5645fecfcfdc0af1665435186a81f775faf8e2`。B 本轮的备份、正式运行、独立验收及打包脚本另见本分支提交；没有把 A 实现归为 B 的新算法。
- 扰动在反归一化 RGB `[0,1]` 域加零均值高斯噪声，`sigma=0.005`，裁剪后重新归一化；每图 5 次，seed 由 `dataset/image_id/repeat` 派生，跨模型/方法配对。target 固定 metadata 真值。主相似度为全像素 Spearman，top-10% Jaccard 只作诊断；未定义/常量/非有限重复保留状态且主分数记 0。每图主值为五次 score 的平均，valid_rate 为有效重复数/5；单元统计用 460 个逐图值的均值和**样本标准差**。
- 原图与扰动图均使用 Occlusion 32×32 窗口、16×16 步长、`perturbations_per_eval=16`。ImageNet softmax、VOC sigmoid；官方 ImageNet 权重与三份 VOC checkpoint 的字节数、完整 SHA-256 均与已有 manifest 匹配。冻结数据版本校验通过，eval target、预测上下文与 460 ID 逐项匹配，未混入 debug。
- 运行直接引用原六份 `configs/occlusion_<unit>.yaml`；没有因搬迁而改变基础配置哈希。稳定性专属 protocol、trace、state、快照、日志及验收报告位于被 Git 忽略的 `results/occlusion_stability_eval/<unit>/`。原 `results/occlusion_eval/<unit>/per_image.csv` 与 `units.csv` 按 PR #10 schema 只追加稳定性两指标；预测表、基础指标数值/hash、run_state、日志、460 张 float32 NPY 与权重均未改变。
- 六个基础结果在运行前备份至 `results/occlusion_stability_eval/backup/`，共 5,568 文件、558,559,434 字节。`backup_manifest.json` SHA-256 `17f7eeb96d1ddc5f697c9903c96a7dd84621fe6bf3bd2e2476803ac54d7926fe`；逐组验收将备份与现存源文件逐项比较。旧图无 sidecar，runner 先按当前模型/配置重算 `A(x)`，必须通过 `np.allclose(rtol=1e-4, atol=1e-5)` 才写 `verified_recomputation` sidecar。此绑定只证明**本次复核**的一致性，不追认旧图初次生成时的权重。

## 六组实测

下表 `Spearman` 为 460 个逐图值的均值及样本标准差；valid_rate 六组均为 1.000。墙钟秒数是各组所有本轮进程（包括立即 resume，以及 VGG16/ImageNet 的失败尝试）耗时之和，不是原 `efficiency_time_ms`，也不是只取最后一次进程。扰动归因毫秒数仅来自 2,300 条 trace 的计时，来源图重算不算入其中。显存是本进程 CUDA allocated 峰值，不是整机占用。

| 单元 | 稳定性 config_hash | Spearman 均值 ± 样本 SD | 扰动归因均值 ms | 累计进程墙钟 s | CUDA 峰值 MiB | 失败尝试 |
|---|---|---:|---:|---:|---:|---:|
| ResNet50 / ImageNet | `21eb94fdd740` | 0.916753 ± 0.070779 | 126.928 | 399.19 | 300.98 | 0 |
| ResNet50 / VOC | `b5523d68e7e9` | 0.915317 ± 0.056971 | 126.923 | 396.97 | 293.24 | 0 |
| DenseNet121 / ImageNet | `08f31fbeb979` | 0.987817 ± 0.012083 | 155.241 | 500.76 | 224.78 | 0 |
| DenseNet121 / VOC | `3ecdd05d0880` | 0.987885 ± 0.012658 | 157.225 | 499.04 | 220.88 | 0 |
| VGG16 / ImageNet | `e9d3f9ab5823` | 0.990120 ± 0.012890 | 260.973 | 811.10 | 1354.83 | 1 |
| VGG16 / VOC | `79262a50113f` | 0.992078 ± 0.008619 | 260.718 | 784.20 | 1339.07 | 0 |

六组累计进程墙钟约 3,391.24 秒，包含加载、460 张旧图来源校验、正式扰动归因、立即续跑和一次失败的模型加载；不能与单纯归因时间相加或比较。各组验收报告分别在 `results/occlusion_stability_eval/<unit>/validation_report.json`，机器可读摘要见 `docs/B_OCCLUSION_STABILITY_EVAL_MANIFEST.json`。

VGG16/ImageNet 曾在**模型加载前**因 torchvision 读取项目缓存（该处没有 VGG）而发起下载，下载流哈希不符而停止，耗时 23.43 秒。该失败没有生成 trace/sidecar，也未改基础 CSV；原先预检并验过 SHA-256 的官方权重实际存于用户 torch 缓存。B 只调整本轮包装器使用该已核验缓存，并核实失败前文件不变，随后**仅重试一次**成功。保留 `console_first.log`、`console_retry1.log` 与 `process_runs.jsonl`，没有删除失败证据、替换权重或改变系统安全设置。

## 验收方法与交付

`scripts/validate_occlusion_stability_eval.py` 逐组校验：460 eval 身份/target、五个配对 seed、trace/status/score、从 trace 独立重算逐图主值与 valid_rate、单元均值/样本 SD/n、配置哈希与 checkpoint、来源 sidecar、基础逐图值与汇总哈希、预测表、参考 NPY、其它原始文件和立即续跑记录。每组 `per_image.csv` 最终为 1,840 行（920 基础 + 920 稳定性），`units.csv` 为 4 行；验收均通过。稳定性 runner 不调用 MoRF，也未改基础效率计时。

环境为项目 `.venv-gpu`，Python 3.12.14、torch 2.11.0+cu128、torchvision 0.26.0+cu128、Captum 0.9.0、SciPy 1.18.1、Matplotlib 3.11.2、RTX 5080 Laptop GPU；完整 pytest 与 `pip check` 的最终结果随 PR 交接记录。轻量结果包在 `results/occlusion_stability_eval/delivery/`，包括合并后的 CSV、预测表、trace、基础/冻结/单元配置、环境快照、来源 sidecar、验收报告、日志、备份清单及逐文件 SHA manifest；不含 NPY、checkpoint 或备份二进制。包大小与 SHA 见同目录 `receipt.json`。接收方先验 ZIP SHA-256，再核包内 `file_manifest.json` 的路径、字节数及 SHA-256，按六个单元读取。原 float32 NPY 仍在本机基础结果目录，已经另行交付过基础 FLOAT32 包；如需此轮独立二进制传输，应使用组内批准渠道。**本轮轻量包尚未传输**，本机路径不是下载链接。

目前还需组员实际接收并核验本轮轻量包，并在全组 36 单元齐全后完成统一统计和报告；这些不是 B 六组正式稳定性 eval 未完成的借口。本交接不合并 PR、不关闭 Issue、不发表跨 softmax/sigmoid 或跨方法正式排名。
