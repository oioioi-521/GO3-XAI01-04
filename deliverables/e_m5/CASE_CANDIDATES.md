# E M5：六方法真实 eval 展示案例

**本地选择记录（2026-10-28）**：E 选择 image_id **215、3803、7612** 作为固定展示案例。B 已复现并支持这三例及其选择依据，并请 E 确认最终组合。此记录用于本地交付稿；尚未在 GitHub 更新 Issue #22 或创建展示 PR。

三例均来自冻结 handoff 的 VOC / ResNet50 `model_correct` cohort，使用同一输入图、metadata 真值 target、同一模型权重，并有六种方法各自的正式逐图指标和归因图。选择规则基于已有结果，属于可复现的事后选图，不是实验前预注册。

| image_id | 真值 target | 置信度 | 展示作用与选择规则 |
| --- | --- | ---: | --- |
| 215 | cat (7) | 0.9662 | 代表性命中样本：六方法四项逐图指标的名次画像与该 cohort 的中位画像完全一致（L1 距离 0）；并列时先取置信度较高者，再取较小 image_id。 |
| 3803 | bird (2) | 0.8687 | 质量与稳定性的名次取舍：raw MoRF AUC 与输入噪声 Spearman 的方法名次差平方和为 70，与 4675 并列，按较小 image_id 选 3803。 |
| 7612 | dog (11) | 0.5393 | 低置信命中样本：387 张命中样本中预测置信度最低；预测仍命中真值类别，不是误分类。 |

## 数据与来源核验

- 三张原图来自 `data_share_v1(1).zip` 中的 VOC eval 子集；metadata 路径分别为 `voc/raw/000215.jpg`、`voc/raw/003803.jpg`、`voc/raw/007612.jpg`。
- 三者均为 `eval` split、ResNet50、`model_correct`；checkpoint SHA-256 为 `1e54867cf18ed02383ba329c944a9143871fc37fe268bec6e769a8d364ae134f`。
- 每个 image_id 有 IG、Grad-CAM、Occlusion、RISE、KernelSHAP、LIME 六张正式归档 PNG 预览，共 18 张 224×224 灰度图。原 PNG 字节复制到 `demo/assets/cases/<image_id>/maps/`，没有重算或制作叠图。
- float32 NPY 本体保留在完整归档中；NPY、PNG、provenance sidecar、配置快照、协议、稳定性配置和来源归档哈希记入 `demo/cases.json`。PNG 只用于展示，定量计算使用 float32 NPY 和正式逐图指标。
- 地图包 `dcu_cases_voc_resnet50_215_3803_7612.zip`，SHA-256 `a46fa18f447e104fc707451de4a03e8b2c2eb56679c15a675f6cc5aaacde9eb3`。包内 113 项 SHA-256 及 18 组地图、NPY、sidecar、配置和协议哈希均已核验；A 于 10/27 复核了四例资产的身份与哈希。
- 统计来源 handoff SHA-256：`14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc`。

## 解释边界

- 7612 是低置信度但预测命中的样本，不能称为误分类。
- 当前没有正式逐图 LIME weighted R² 证据证明局部拟合失败，不根据热图外观作此判断。
- 协议中的固定方法 seed 下输入噪声稳定性，不等于 LIME 采样随机性测试。
- 可选的 VOC / VGG16 / 298 暂不加入固定 Demo：其 Grad-CAM 被标记为常量退化，但仍需补齐并目视核验六方法案例图后，才适合作为展示样例。
- 这三例只支持案例层面的说明，不代表总体方法排名；总体结论仍依据完整六方法评测表。

## Demo 复现

本机原图被 `.gitignore` 排除。按数据许可从获准包恢复三张 VOC 图片后，运行 `python3 -m http.server 8000 --directory deliverables/e_m5/demo`，打开 `http://localhost:8000`。另一位组员的独立环境演练和回执仍待完成，详见 [`REPRODUCE_DEMO.md`](REPRODUCE_DEMO.md)。
