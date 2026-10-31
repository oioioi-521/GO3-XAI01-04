# C：VOC20 checkpoint 接收验收

验收日期：2026-10-09（Asia/Shanghai）。

接收源为工作区 `muti-data/GO3-XAI01-04_checkpoint_transfer_2026-09-20/`。
三份权重已按 `docs/VOC20_CHECKPOINT_MANIFEST.json` 验证，并复制至项目 Git 忽略的
`models/checkpoints/` 标准路径；没有改变正式 YAML 或配置哈希。

| 模型 | 字节数 | SHA-256 |
| --- | ---: | --- |
| ResNet50 | 94518331 | `1e54867cf18ed02383ba329c944a9143871fc37fe268bec6e769a8d364ae134f` |
| DenseNet121 | 28512970 | `ab61772eaf3459842f4b1e34be9ec15062e02a940d521390d80df0371168feca` |
| VGG16 | 537383181 | `61640717641bfc1e9ae5d6172de667ce45755e6f046f458c848f70cb4a6c9d58` |

大小和哈希均完全匹配。三个模型均通过 `models.factory.load_model` 的严格加载，
分类头为 20 类、使用 sigmoid；CUDA 前向输出 `(1, 20)` 且 logits/分数有限。
冻结数据基线 `xai01-04-data-v1` 的四份元数据哈希，以及 ImageNet/VOC 各 500 张原图
检查通过。

三个模型在冻结 VOC debug 首图 `8173`、metadata 真值 target=18 上完成 4000-mask
RISE 检查：输出均为有限、非恒定的 `224×224` float32。
ResNet50、DenseNet121、VGG16 实测分别约 10.331、10.182、20.416 秒，CUDA allocated
峰值约 441.96、367.92、1722.01 MiB。这是首图工程检查，不是正式统计。

三个 VOC 正式单元采用原六份配置中的 VOC YAML，按 ResNet50 → DenseNet121 → VGG16
顺序执行。每单元结束后立即续跑，并核对 460 个 eval ID、920 行基础指标、460 条
预测、460 张 NPY/PNG、目标类别、预测身份/权重、汇总重算和 canonical-JSON 聚合哈希。
入口为 `analysis/rise_formal_batch.py`，输出日志和验收回执位于
`results/rise_voc_batch_20261009_python/`，由 Git 忽略。

首次 PowerShell 包装器因将进度条 stderr 当作 `NativeCommandError` 而退出；该记录保留
在 `results/rise_voc_batch_20261009/`。Python 包装器合并 stdout/stderr 后按原配置安全
续跑。全部单元完成前，不把启动记录写成正式完成结果。
