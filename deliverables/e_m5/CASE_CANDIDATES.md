# E M5：真实 eval 六方法案例候选（地图已核验，待 B 复核）

本候选表从冻结 handoff 的 VOC / ResNet50 `model_correct` cohort 中筛选，保证同一输入、同一目标、同一模型权重，对应六种方法各自的正式逐图指标。现已收到并校验三个候选 image_id 的正式归因预览。**这些仍是候选，B 尚未最终确认选图。**

| image_id | 真值 target | 置信度 | 选择规则 |
| --- | --- | ---: | --- |
| 215 | cat (7) | 0.9662 | 六方法四项指标的名次画像与该单元中位画像一致；高置信代表性候选。 |
| 3803 | bird (2) | 0.8687 | VOC / ResNet50 命中组中，六方法 raw MoRF AUC 与噪声 Spearman 的方法名次差异最大，展示质量与稳定性取舍。 |
| 7612 | dog (11) | 0.5393 | 正确组中置信度最低；作为低置信困难样本候选。 |

## 已核对内容

- 三张原图都来自 `data_share_v1(1).zip` 中的 VOC eval 子集；仓库 metadata 路径分别为 `voc/raw/000215.jpg`、`voc/raw/003803.jpg`、`voc/raw/007612.jpg`。
- 三者均在 `model_correct` cohort，split 为 `eval`，模型 ResNet50，checkpoint SHA-256 `1e54867cf18ed02383ba329c944a9143871fc37fe268bec6e769a8d364ae134f`。
- 每个 image_id 有 IG、Grad-CAM、Occlusion、RISE、KernelSHAP、LIME 六张归档 PNG 预览，共 18 张。每张 PNG 为 224×224 灰度图，原字节复制到 `demo/assets/cases/<image_id>/maps/`；未重算、未做覆盖叠图。
- 原始 float32 NPY 未复制进轻量 Demo；其 SHA-256 和 provenance sidecar、配置快照、协议、稳定性配置、来源归档哈希写入 `demo/cases.json`，provenance sidecar 也随 PNG 保留。量化 PNG 只用于可视化，不代替 NPY 计算指标。
- `case_candidates.csv` 保留 18 行逐图指标，并补充每张地图 PNG、NPY、sidecar 与源归档 SHA-256。

地图提取包：`dcu_cases_voc_resnet50_215_3803_7612.zip`，SHA-256 `a46fa18f447e104fc707451de4a03e8b2c2eb56679c15a675f6cc5aaacde9eb3`。包内 113 项 SHA-256 全部通过；18 条索引对应的 NPY、PNG、sidecar、config、protocol、stability config 内容哈希均与索引相符。输入图、target、checkpoint 和 config snapshot 也与既有候选 manifest 匹配。

## 当前限制 / 下一步

- B 仍需从 215、3803、7612 中确认最终案例；不得把候选当作已签收案例。
- 这批图只能支持案例层面的解释，不能据此推断总体方法排名。跨方法性能结论继续依据完整六方法评测表。
- 独立复现时在仓库根目录运行 `python3 -m http.server 8000 --directory deliverables/e_m5/demo`；本机图片被排除在 Git 外，演示前需按许可恢复数据包中的三张原图。
- 统计和定量归因源仍为正式 handoff 与 float32 NPY；本次没有重跑归因或 MoRF。

统计来源 handoff SHA-256：`14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc`。
