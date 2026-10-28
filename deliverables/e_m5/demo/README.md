# E M5 六方法固定 Demo（正式地图已接入）

## 启动

在仓库根目录运行：

```bash
python3 -m http.server 8000 --directory deliverables/e_m5/demo
```

浏览器打开 `http://localhost:8000`。页面只依赖浏览器，不访问外网。

## 内容与边界

独立环境复现步骤见上级 [`REPRODUCE_DEMO.md`](../REPRODUCE_DEMO.md)。

- 页面提供 3 个 VOC eval / ResNet50 展示案例：215、3803、7612。每个案例包含六种方法的正式归档 PNG 预览，共 18 张；所有 PNG 均按原字节复制，未重算或修改。
- 原图来自授权数据包，保存在本地 `assets/cases/*/image.jpg`，被 `.gitignore` 排除；克隆后按数据许可从 VOC 数据包恢复。
- PNG 为 224×224 灰度预览，只用于展示。定量 attribution 使用 float32 NPY；本轻量 Demo 不包含 NPY。每张图的 PNG、NPY、provenance sidecar、配置、协议、稳定性配置、checkpoint、输入图和来源归档 SHA-256 都记录在 `cases.json`。侧车 JSON 随 PNG 提供。
- 不跨方法比较 PNG 灰度强度的绝对值；正式指标以 handoff 的逐图表和原始 NPY 为准。三张图片只用于案例说明，不用于总体方法排名。
- E 已按 B 提供的可复现规则选定这三例：215 代表中位名次画像，3803 展示忠实性与稳定性名次取舍，7612 是 387 张命中图中置信度最低的一例，但仍预测命中真值。此为事后展示选图，不是实验前预注册。
- 没有正式逐图 LIME weighted R² 时，不根据热图外观声称 LIME 局部拟合失败；输入噪声稳定性也不等同于 LIME 采样随机性。

## 地图包与完整性

地图来源：`dcu_cases_voc_resnet50_215_3803_7612.zip`，SHA-256 `a46fa18f447e104fc707451de4a03e8b2c2eb56679c15a675f6cc5aaacde9eb3`。

映射索引 SHA-256：JSON `f3db52507ba59fa80594301dae8120d9f3c6cbccf124404c7e17a0ce848364a7`；CSV `3ed36ba2afaf7c326d24b06624452362424d2b479f79be21d16f8458e1d7b329`。包内 113 项文件 SHA-256 和 18 张图对应的 NPY/PNG/sidecar/config/protocol/stability-config 哈希均已核对。案例 ID、VOC eval split、真值 target、ResNet50 checkpoint、输入图 SHA 与逐图统计记录一致。完整来源和每张图 SHA 见 `cases.json`。

五种方法使用来源归档 SHA-256 `852eb7c8c4eb74124b6c599d2fd54bf1f067f9c0579ce9a0542dfada040dbede`；LIME-8000 使用来源归档 SHA-256 `38510f8eeda54ae38ee77acaf04580666779954ff7565b59d9fbd58438fe1cd3`。完整归档未复制到仓库。

## 复现来源

- 逐图指标、预测、cohort 和统计配置快照：`dcu_team_handoff_20261014.zip`，SHA-256 `14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc`。
- 输入原图：`data_share_v1(1).zip`，SHA-256 `8fbd37057da207d2f737d92953c4277e0e4ba1636e2752df71e7c5685a685792`。单图 hash 记录在 `cases.json`。
- 来源包不需要重跑；本次只做归档文件校验、Demo 接入和 PPT 展示案例状态更新。独立环境演练仍待记录。
