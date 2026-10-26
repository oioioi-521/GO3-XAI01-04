# E / M5 本地交付说明

状态：工作稿 v4。正式案例地图已接入；候选选图仍待 B 复核，独立环境 Demo 演练仍待完成。所有修改只在本地，没有 commit、push、创建 PR 或发送 GitHub 评论。

## 交付文件

- `GO3-XAI01-04_E_M5_展示初稿_20261026_v4.pptx`：11 页可编辑演示稿。保留三张案例候选页，新增 image_id 215 同图六方法归因预览页。
- `GO3-XAI01-04_E_M5_展示初稿_20261026_v4.pdf`：对应的 11 页静态 PDF；页面为高清栅格图，文字不可搜索或选择。后续修改请编辑 PPTX 再导出。
- `demo/`：六方法固定案例页面、JSON 来源清单、启动说明，以及 18 张正式归档 PNG 预览和 18 份原始 provenance sidecar。
- `CASE_CANDIDATES.md`、`case_candidates.csv`：三张候选的选择理由、逐图指标和地图/NPY/sidecar/来源哈希。
- `REPRODUCE_DEMO.md`：独立环境启动、图片恢复和复现范围记录。

## 数据范围与完整性

- 新地图包 `dcu_cases_voc_resnet50_215_3803_7612.zip`，SHA-256 `a46fa18f447e104fc707451de4a03e8b2c2eb56679c15a675f6cc5aaacde9eb3`。包内 113 项 SHA-256 全部通过。
- 215、3803、7612 三个 VOC eval / ResNet50 候选各有六方法正式地图，共 18 张 224×224 灰度 PNG。PNG、float32 NPY、provenance sidecar、config、protocol 和 stability config 哈希都与地图索引一致；target、checkpoint 和输入图哈希也与现有逐图结果绑定。
- Demo 保留正式 PNG 和轻量 provenance sidecar；原始 float32 NPY、完整数据集、模型权重和完整归档包不复制到交付目录。NPY 哈希与完整来源链记录在 `demo/cases.json`。指标继续引用冻结结果，本次没有重跑归因或 MoRF。
- 三张输入 JPG 仍由本机授权数据包恢复，并被 `.gitignore` 排除。克隆后运行 Demo 前按数据许可恢复这些图片。
- PNG 是档案提供的灰度预览，只用于可视化；跨方法的数值比较使用正式指标和 float32 NPY。三张图是候选示例，不支持总体方法排名。

## 检查记录

- PPTX：11 页；页面尺寸与字体沿用 v3。结构和布局检查通过，未发现布局错误；已逐页渲染检查。
- PDF：11 页，已用 Poppler 渲染并逐页检查。
- 来源 ZIP：113 项包内文件哈希全部通过；18 张地图的索引身份和逐文件哈希通过。
- Demo 的输入图片、target、checkpoint、config snapshot 与地图索引匹配。

## 剩余工作

1. 由 B 在 215、3803、7612 中确认最终案例。
2. 按 README 在独立环境恢复 VOC 图片并运行 Demo，记录复现结果。
3. 审阅后由用户自行决定是否将 `deliverables/e_m5/` 纳入 commit/push。
