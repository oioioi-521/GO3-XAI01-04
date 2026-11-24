# 成员 B：Occlusion 交付说明

审查基线：`feat/b-w2`，`ef8604aae63062270925693f3dc49d05d0b58da6`（与
`origin/feat/rise-pipeline` 相同）。本说明只覆盖成员 B 未提交工作区中的
Occlusion 交付物；没有提交、推送或合并。

## 实现内容

- `experiments/attribution/occlusion.py`：基于 Captum 的独立 Occlusion
  适配器。输入为归一化 RGB 图像和真值类别 `target`，输出为 CPU 上有限的
  `(H, W)` 热力图，范围 `[0, 1]`。默认 baseline 是 ImageNet 预处理下的
  RGB 黑色（`-mean/std`）。
- Captum 的三通道有符号归因先作 RGB 算术均值，再作每图 min--max 归一化；
  不取绝对值。归一化后的 `0` 表示该图的最低有符号贡献，并不总表示原始
  零贡献。现有 MoRF 只按降序排序，非恒定图的排序不会改变；恒定图输出全零。
- `experiments/run_unit.py`：保留 RISE 路径，增加 `method: occlusion`
  分发、方法化日志/热力图目录和私有续跑状态文件。公共 CSV 列完全未变：
  `per_image.csv` 仍以 `method` 列区分，`units.csv` 仍有 `config_hash`。
- 续跑状态默认在每个 `per_image_csv` 同级的 `run_state/`。其哈希包含完整
  YAML；同一 method/model/dataset 改变参数时会先删除该单元旧行和旧汇总，再
  重算，不会错误复用。续跑前还会删除运行中断留下的、不含全部请求指标的残缺行。
- `requirements.txt` 显式声明 `captum>=0.7,<1`。RISE 不在导入时加载 Captum：
  未安装 Captum 时仍可导入并运行 RISE；只有实际使用 Occlusion 才报出清晰错误。

## 对 C 的公共接口改动

- `experiments.attribution` 新导出 `Occlusion`，原有 `RISE` 未改。
- runner 的顶层 YAML `method` 由仅接受 `rise` 扩展为 `rise` 或 `occlusion`；
  `attribution` 块原样传给相应类。Occlusion 参数为
  `window_height`、`window_width`、`stride_height`、`stride_width`、
  `perturbations_per_eval`。
- 没有修改 RISE 算法、模型加载器、指标接口或既有 CSV schema。

## 运行方式

在已安装依赖、且本地 ImageNet metadata/debug 图片可用时：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe experiments\run_unit.py --config configs\occlusion_smoke_imagenet.yaml
```

`configs/occlusion_smoke_imagenet.yaml` 是离线工程通路配置：随机权重、最多一张
debug 图、CPU，**不能作为课程实验结论**。正式配置应给出与数据预处理一致的
已验证模型权重、正确类别数和真实数据 metadata。

## 已验证

在 Python 3.12 `.venv`（torch 2.11.0+cpu、torchvision 0.26.0+cpu、
captum 0.9.0）中执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q -ra
.\.venv\Scripts\python.exe -m pip check
```

结果：`15 passed, 0 failed, 0 skipped`；`pip check` 为
`No broken requirements found.` 测试包括真实 Captum 的热力图形状/范围、符号
通道归约、归一化黑色 baseline、非法参数与 target、恒定归因图及 MoRF，未装
Captum 的 RISE 子进程回归，以及合成数据的 YAML → 模型 → Occlusion → MoRF →
CSV 续跑/参数变更/中断恢复链路。合成结果没有写入项目 `results/`。

## 限制与待对接

- 尚未在真实 ImageNet 或 VOC 数据、预训练权重、GPU 上做实验；没有下载数据或
  权重，也没有训练 VOC 模型。
- `target` 必须是当前模型输出维度内的真值类别索引。接入 VOC 前需确认其 20 类
  标签顺序与 checkpoint 分类头一致。
- 缺失图像会由数据集读取直接抛错，不会静默写入虚假结果；已完成行可续跑，当前
  图片若只写入部分指标会在下一次续跑前清理并重算。
- 若 C 同时改动 runner，合并时应保留 `SUPPORTED_METHODS` 分发、`method` 列
  区分和 hash sidecar 逻辑；配置路径/结果根目录如有调整，`state_dir` 可显式配置。
