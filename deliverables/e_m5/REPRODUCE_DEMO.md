# E / M5 六方法 Demo 独立复现记录

## 复现范围

本记录复现固定 Demo 中的正式案例图、逐图指标、来源哈希与页面呈现；不重跑模型、归因、MoRF 或稳定性评测。页面的定量字段来自已冻结的逐图结果，原始 attribution 数值由 float32 NPY 支撑。Demo 中的 PNG 是来源包提供的 224×224 灰度预览。

## 环境

- Python 3，用于启动本地静态 HTTP 服务。
- 现代桌面浏览器。
- 不需要 GPU、checkpoint 或 Python ML 依赖。

## 运行

1. 克隆仓库后，在仓库根目录运行：

   ```bash
   python3 -m http.server 8000 --directory deliverables/e_m5/demo
   ```

2. 打开 `http://localhost:8000`，逐一选择案例 `215`、`3803`、`7612`。
3. 每个案例应显示六张方法 PNG、对应的 raw MoRF AUC / noise Spearman / 记录耗时；右上状态应为 `正式地图预览 6/6`。
4. 展开任一方法卡片的“文件与运行来源 SHA-256”，查看 PNG、NPY、provenance sidecar、配置和 source archive 哈希。完整哈希、路径、target、checkpoint 和输入图绑定记录见 `demo/cases.json`。

## 输入图恢复

原始 VOC JPG 因数据分发约束被 `.gitignore` 排除。若需要在另一环境同时展示原图，请从获准的数据包恢复 `voc/raw/000215.jpg`、`voc/raw/003803.jpg`、`voc/raw/007612.jpg` 至 `demo/assets/cases/<image_id>/image.jpg`。复制后按 `demo/cases.json` 中的 `input_image_sha256` 校验。缺少 JPG 时页面仍会显示地图和指标，但输入图区域会提示缺失。

## 来源与限制

- 案例地图包：`dcu_cases_voc_resnet50_215_3803_7612.zip`，SHA-256 `a46fa18f447e104fc707451de4a03e8b2c2eb56679c15a675f6cc5aaacde9eb3`。
- 来源是正式 DCU 评测归档；18 张 PNG 与 float32 NPY、sidecar、配置、协议、稳定性配置的哈希在该包索引中绑定。完整归档不随 Demo 分发。
- 页面只呈现已保存的正式 PNG 和已冻结指标，不执行模型推理或算法计算。PNG 灰度强度不作为跨方法的共同量尺。
- image_id 215、3803、7612 均为候选，B 尚未最终确认选图；示例不用于总体方法排名。
- 本地完成显示复现后，应由复现者记录机器/浏览器、日期、三个案例的加载状态和异常，再由 B 确认案例选择。
