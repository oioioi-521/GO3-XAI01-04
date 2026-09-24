# B：Occlusion 效率与 raw MoRF 门禁（2026-09-24）

基线为 `origin/integrate/a-voc-eval` 的 `2f120e84d1bd2a7bca66bc23731b6cc4c0a2a3c0`。
这是冻结 `debug` 图像上的工程门禁，**不是 460 张正式 eval，也不是稳定性排名**。
旧 `occlusion_resnet50_imagenet_debug{,_5}.yaml` 是 CPU/legacy ratio/5 点网格 pilot，
本次没有复用其结果目录或计入下表。

## 配置、口径和复现

- 六个配置在 `configs/occlusion_gate_<model>_<dataset>.yaml`。共同候选参数：
  `32×32` 遮挡窗、`16×16` 步长、`perturbations_per_eval=16`。输入为现有
  `224×224`、ImageNet mean/std 归一化图像；baseline 是归一化后的 RGB 黑色。
- Occlusion 对真值 `target` 的 logit 归因，通道有符号均值后逐图 min-max；
  非恒定图保持 MoRF 的降序排名，但输出的 0 不是原始零贡献。候选参数还未获
  Issue #8 最终冻结，不应直接解释为最优参数。
- 指标只含 `efficiency_time_ms` 和主指标 `faithfulness_morf_auc_raw`；后者
  使用 0.00 到 1.00、步长 0.05 的 21 点删除网格。legacy ratio
  `faithfulness_morf_auc` 未写入门禁 CSV。ImageNet 是 1000 类 softmax，
  VOC 是 20 类 sigmoid，并严格校验 VOC checkpoint 清单 SHA-256。
- `warmup_runs=1` 不计入归因耗时，`seed=42`，CUDA 串行运行；一次只占用一个
  GPU 计算进程。门禁器开始前检查是否有其他 GPU compute app，禁止 CPU 回退、
  未缓存的 ImageNet 权重自动下载、eval split 和混合输出目录。
- 单图用 `--images 1` 生成独立的、被 Git 忽略的有效 YAML，40 张用源配置；
  对应 `results/occlusion_gates/single/` 与 `results/occlusion_gates/debug40/`。
  每组保留逐图 CSV、预测表、配置快照、PNG、float32 NPY、日志和
  `gate_report.json`/`gate_runs.jsonl`。所有检查会核对原结果 schema、
  配置哈希、图像集合、`(224,224)` 有限 `[0,1]` float32 图、常量图、
  raw 值域以及预测类别/置信度。

运行例子（项目根目录、已有 `.venv-gpu`）：

```powershell
.\.venv-gpu\Scripts\python.exe experiments\run_occlusion_gate.py --config configs\occlusion_gate_resnet50_imagenet.yaml --images 1
.\.venv-gpu\Scripts\python.exe experiments\run_occlusion_gate.py --config configs\occlusion_gate_resnet50_imagenet.yaml --images 40
```

冻结数据两个 raw 目录各有 500 张。三个 CSV 在当前 Windows 工作树为 CRLF，
直接字节 SHA 与 `DATA_VERSION.json` 中 Git/LF SHA 不同；规范化换行后四份
元数据哈希均匹配。`preprocessing/verify_data_version.py` 现按 `.gitattributes`
声明的 LF 形式验证文本，未修改任何元数据或图片。三份 VOC checkpoint 的
字节数与完整 SHA-256 均匹配 `docs/VOC20_CHECKPOINT_MANIFEST.json`，
ImageNet 三份 torchvision 官方 DEFAULT 权重均从既有本地缓存读取。

## 实测结果

均为 RTX 5080 Laptop GPU、torch 2.11.0+cu128。`wall_s` 是本次完整进程的
墙钟时间（含加载、warm-up、MoRF、I/O），`attr_ms` 是 warm-up 后逐图
归因耗时的均值；`peak_alloc_MiB` 为本进程 PyTorch 峰值已分配显存，
不是整机显存总占用。各组均为 `0` 失败、`0` 常量图，结果文件未进 Git。

| 门禁 | 模型 | 数据 | 数量 | wall_s | attr_ms/图 | peak_alloc_MiB | raw AUC 均值 | 配置哈希 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 单图 | ResNet50 | ImageNet | 1 | 1.555 | 125.309 | 300.4 | 0.320258 | `15cb7869271d` |
| 单图 | ResNet50 | VOC | 1 | 1.262 | 125.932 | 292.7 | 0.647882 | `1fbc5a60478c` |
| 单图 | DenseNet121 | ImageNet | 1 | 1.436 | 159.059 | 224.2 | 0.029020 | `c2003091c7bb` |
| 单图 | DenseNet121 | VOC | 1 | 1.444 | 171.989 | 220.3 | 0.415024 | `c66231e8799b` |
| 单图 | VGG16 | ImageNet | 1 | 2.424 | 255.066 | 1354.3 | 0.023011 | `250a09c35e36` |
| 单图 | VGG16 | VOC | 1 | 2.575 | 254.896 | 1338.5 | 0.128499 | `bd5a00ee555c` |
| 40 debug | ResNet50 | ImageNet | 40 | 10.250 | 124.987 | 300.4 | 0.204214 | `f81dfe5a6d99` |
| 40 debug | ResNet50 | VOC | 40 | 9.959 | 124.843 | 292.7 | 0.591237 | `74fa2316015b` |
| 40 debug | DenseNet121 | ImageNet | 40 | 18.512 | 180.552 | 224.2 | 0.065805 | `71ca5a7f0e7d` |
| 40 debug | DenseNet121 | VOC | 40 | 18.765 | 170.825 | 220.3 | 0.395264 | `dbb8f32e0c45` |
| 40 debug | VGG16 | ImageNet | 40 | 15.561 | 256.731 | 1354.3 | 0.033195 | `748d4e8c89e6` |
| 40 debug | VGG16 | VOC | 40 | 15.622 | 256.061 | 1338.5 | 0.206547 | `06e009b1f323` |

每组 40 张门禁对应 80 条逐图 metric、40 条预测和 40 张 float32 图。
raw AUC 在 ImageNet softmax 与 VOC sigmoid 下分数语义不同，不能把跨数据集
均值解释为方法优劣。六组合 40 张只是效率和可执行性证据，不是正式统计排名。

## PR #10 / Issue #8 只读复核

复核对象：[PR #10](https://github.com/oioioi-521/GO3-XAI01-04/pull/10)
的 `ab60d54da6310cbdd4fe11f66d479e77d38196b8`，基于 `integrate/a-voc-eval`；
[Issue #8](https://github.com/oioioi-521/GO3-XAI01-04/issues/8) 仍开放。
PR 没有 review，文档仍标记候选、等待 C/D 复核；B 已在 Issue 表示候选口径
可用于 Occlusion。本轮**没有**提交 review 或评论，也没有把 PR #10 合入本分支。

1. **可接入**：`experiments/run_stability.py:487-489,580-582` 复用
   `_build_attributor` 和相同 `attribution` 块，可实例化 Occlusion 并对固定
   metadata 真值重新归因；`469-482` 使用 VOC sigmoid 或 ImageNet softmax。
   `381-407` 要求原始图是有限 float32 NPY，`580-582` 将候选图转 float32。
   本门禁六配置均保存原始 float32 图，可供未来候选稳定性单图联调。
2. **固定扰动和退化状态**：`experiments/stability.py:28-63` 用
   `dataset/image_id/repeat` 派生 seed，在 RGB `[0,1]` 加噪裁剪后重新归一化；
   `88-111` 对常量、非有限、未定义 Spearman 给状态和 0 分；
   `run_stability.py:597-634` 输出逐重复 trace 及每图两项稳定性指标。
   `268-334` 用两项 metric 加五条 trace 判断完成、清理部分结果。
3. **五张扰动图并未落盘**：代码仅在内存构造五次
   `A(x+delta)`（`run_stability.py:569-620`）；没有五个图文件名，也不会覆盖
   原图。`TRACE_COLUMNS` 的 `repeat/seed/config_hash/status` 区分五次记录。
   这与 Issue #8 当前候选的“不保存全部扰动图”一致；若最终要求审计五个
   NPY 文件，应先明确文件名包含 repeat 和配置身份并补实现。
4. **需 C 复核的续跑/追溯风险**：原始图读取只检查路径、dtype、形状和有限性
   （`run_stability.py:381-407`），未证明该 NPY 对应当前 base 配置哈希或权重；
   历史同名图可能被误读。默认 trace 共享 `results/stability_trace.csv`，
   state 文件名仅含 method/model/dataset、不含 split
   （`run_stability.py:229-265,493`，`configs/stability_protocol_v1.yaml`）；
   同一单元交替跑单图、40 debug、460 eval 可清除另一规模的 trace。
   未来应为每阶段独立配置 trace/state 目录，或把 split/运行身份纳入隔离键。
5. **共用汇总表的顺序风险**：PR #10 自己只 upsert 两项稳定性汇总并保留
   原始行哈希（`run_stability.py:351-378`）；但若补跑后再次调用现有
   `run_unit.py:289-321,494-495`，基础 runner 会重汇总该单元的所有 metric，
   将稳定性汇总的 `config_hash` 改成基础配置哈希。应由 C 冻结调用顺序或
   在最终集成修复，勿把被改写的稳定性哈希当成正确追溯。
6. **测试缺口**：`tests/test_run_stability.py:163-172` 用替身
   `SpatialAttributor` 代替真实 Captum；现有测试证明表结构/续跑逻辑，
   还未证明 PR #10 在真实 Occlusion GPU 单图上的完整执行。

依照 40 张归因均值做**纯归因计算量推算**，每图五次扰动重归因约
0.62–1.28 秒，460 张六组合累计约 43 分钟；这是线性预算，**不是**
稳定性实测墙钟时间，未包括图像、模型、数据传输、相似度、存储及争用。
只有 Issue #8 协议最终冻结、上述追溯/目录风险处理并完成真实 Occlusion
稳定性单图与 40 张复核后，才适合启动 460 张稳定性补跑。

## 本轮验收边界

`preprocessing/verify_data_version.py` 通过；`.venv-gpu` 全量 pytest
`74 passed`、`pip check` 无冲突。门禁 1 张→40 张
六组合均通过；未运行 460 eval、未生成正式稳定性指标或排名、未改写
IG/Grad-CAM 正式结果、未更改 checkpoint 二进制。进入正式 460 归因前，
仍需组内确认 Occlusion 的窗/步长/扰动批参数、结果目录与排名口径；
进入稳定性补跑还需 Issue #8 的 C/D 冻结及 PR #10 集成复核。
