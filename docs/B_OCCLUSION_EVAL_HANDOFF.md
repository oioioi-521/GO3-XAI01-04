# B：六组 Occlusion 基础 eval 交接（待组内验收）

本批运行使用代码提交 **`6d5645fecfcfdc0af1665435186a81f775faf8e2`**，不是本文档提交。
六组各处理冻结 `eval` 的 460 张唯一图片，合计 2,760 条预测、2,760 张
`224×224` 有限 float32 原图归因 NPY、5,520 行逐图指标；六组各自
`processed=460, skipped=0, failed=0, constant_maps=0`，与 40 张 debug 无交集。
每组的 `faithfulness_morf_auc_raw` 都有限且位于 `[0,1]`。**基础 eval 已完成，
但尚未获得组内结果验收；稳定性未运行，未生成正式排名。** 旧五图 CPU pilot
及 40 张 GPU 门禁都未计入本批 460 张。

## 运行配置与可复核结果

六份源配置位于 `configs/occlusion_<model>_<dataset>.yaml`，运行时快照位于各自
`results/occlusion_eval/<model>_<dataset>/configs/`。共同参数为 32×32 遮挡窗、
16×16 步长、`perturbations_per_eval=16`；21 点（0、0.05、…、1）的
`faithfulness_morf_auc_raw` 删除网格；一次不计时 warm-up、seed 42、
`max_images: null`、`resume: true`、保存 float32 原图归因图。ImageNet 为 1000 类
softmax，VOC 为 20 类 sigmoid。输入按 `preprocessing/dataset.py` 缩放为
`224×224`、RGB `[0,1]` 并使用 ImageNet mean/std 归一化。所用 VOC 三份
checkpoint 与 `docs/VOC20_CHECKPOINT_MANIFEST.json` 的大小及 SHA-256 相符；
ImageNet 为 torchvision 官方 DEFAULT 权重（具体枚举及本机缓存 SHA-256 见清单）。

下表均由现有 CSV 和 `validation_report.json` 重新读取复核；均值和标准差按
460 张逐图 raw AUC 计算。`wall_s` 是 `console.log` 起止时间跨度，包含模型加载、
warm-up、MoRF 和 I/O，**不是** `efficiency_time_ms` 的逐图归因计时。

| 组合 / 结果相对目录 | 配置哈希 | raw AUC 均值 ± 样本标准差 | 归因 ms/图 | wall_s | 失败 / 常量 |
|---|---|---:|---:|---:|---:|
| `resnet50_imagenet` / `results/occlusion_eval/resnet50_imagenet/` | `ba1ecb269647` | 0.193132 ± 0.149224 | 126.458 | 109.090 | 0 / 0 |
| `resnet50_voc` / `results/occlusion_eval/resnet50_voc/` | `1e267b87c4ef` | 0.610883 ± 0.169287 | 126.279 | 107.214 | 0 / 0 |
| `densenet121_imagenet` / `results/occlusion_eval/densenet121_imagenet/` | `32e3c7d7a0fa` | 0.081563 ± 0.127249 | 169.560 | 200.644 | 0 / 0 |
| `densenet121_voc` / `results/occlusion_eval/densenet121_voc/` | `650c5c6bc824` | 0.453345 ± 0.220642 | 169.002 | 195.720 | 0 / 0 |
| `vgg16_imagenet` / `results/occlusion_eval/vgg16_imagenet/` | `c675f036cd41` | 0.039684 ± 0.055036 | 258.315 | 156.486 | 0 / 0 |
| `vgg16_voc` / `results/occlusion_eval/vgg16_voc/` | `c97adcbd896e` | 0.256823 ± 0.251152 | 259.073 | 154.421 | 0 / 0 |

每个结果目录包含 `per_image.csv`（920 行）、`units.csv`（两项指标）、
`predictions.csv`（460 行）、`run_log.jsonl`、`console.log`、配置快照、
`run_state/`、`maps_float/`（460 张 NPY）及事后校验形成的
`validation_report.json`。PNG 预览位于 `maps/`，但不属于 float32 接收包。
六个输出目录和配置哈希互相隔离，没有覆盖 IG/Grad-CAM 或旧 pilot 结果。

## 证据与接收

受版本控制的 `docs/B_OCCLUSION_EVAL_MANIFEST.json` 记录六组概要、数据版本、
类别顺序、预处理、模型权重身份和详细清单哈希。被 Git 忽略的
`results/occlusion_handoff/B_OCCLUSION_EVAL_FILES.json` 逐文件列出源配置、
运行快照、预测表、逐图指标、汇总、状态、日志、校验报告及全部 2,760 张
float32 NPY 的仓库相对路径、字节数和 SHA-256。

交接使用两个本地压缩包：`B_OCCLUSION_EVAL_LIGHT.zip`（CSV、快照、日志、
校验报告、交接文档和两个 manifest）与 `B_OCCLUSION_EVAL_FLOAT32.zip`
（六组原图 NPY）。包大小与 SHA-256 另见同目录的 `package_receipt.json`；
包没有上传至 Git、Release 或公开网盘。本机 `results/occlusion_handoff/` 路径
不是组员可下载链接。**当前尚未确认经批准且可用的二进制传输渠道，结果包已
准备但尚未传输。** 接收方应按顺序：

1. 经组内批准的渠道取得两个包和独立传递的 `package_receipt.json`，先核对
   两个包的 SHA-256 与字节大小。
2. 解包到独立仓库工作副本，保留 ZIP 内的相对路径；不得覆盖已有同名结果。
3. 核对详细清单本身与简要 manifest 中的 SHA-256，再逐项检查所需文件的
   相对路径、大小及 SHA-256；六组各核对 460 张 NPY 和三份 CSV。
4. 核查配置哈希与快照，按六个单元独立读取和汇总。float32 原图 NPY 应保留，
   供稳定性补跑复用；不能从 PNG 替代还原。

运行时已有配置哈希、运行状态和预测表的 checkpoint SHA-256；**逐文件清单及
缓存权重的再次哈希是在交接时补算**，不是运行时记录。基础 runner 未在每张
NPY 写入时将其与权重 SHA-256 做密码学绑定；当前路径、配置快照、预测表与
补算哈希相符，并不能独立证明每张图当时使用的权重，接收方应保留此证据缺口。

## 尚需确认

- 请 C/A 与组内确认 32×32 窗口、16×16 步长、每批 16 个扰动、21 点 raw MoRF
  及正式结果目录，并按统一 schema 确认单元汇总和正确/错误分组；ImageNet
  softmax 与 VOC sigmoid 不可直接跨数据集比较，更不能据此发布方法排名。
- 请接收方确认可批准的传输渠道与两个结果包的验收回执。checkpoint 二进制
  不在本次结果包中；VOC 权重另按 checkpoint 清单交付。
- [Issue #8](https://github.com/oioioi-521/GO3-XAI01-04/issues/8) 中 B 已同意
  sigma=0.005、5 次扰动的候选稳定性协议，但 C/D 的 runner/schema 与统计验收
  尚未见完成记录。[PR #10](https://github.com/oioioi-521/GO3-XAI01-04/pull/10)
  仍开放；B 的具体 review 指出参考 NPY 来源校验、debug/eval trace/state 隔离、
  基础 runner 重汇总改变稳定性哈希的风险。待这些问题经复核、修复并合入后，
  Occlusion 稳定性仍须按 **单图 → 40 张 debug → 460 张 eval** 另行门控执行。

本交接不修改稳定性代码，不启动稳定性补跑，也不替组员作最终验收。
