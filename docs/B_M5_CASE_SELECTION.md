# B→E：可复现选图规则与候选交接

2026-10-26 · Refs #19 / #22。已阅读 #18–#22 正文及全部评论（五项均为零评论）、A/C 草稿 PR 和 E 分支 `6b4c4ec980fd3ea354a46851fa052877f21255ee` 的 CASE_CANDIDATES、CSV、Demo JSON/sidecar。E 已提出 215、3803、7612，注明待 B 复核；本轮复现并支持这三个候选，补充精确并列规则，**最终面板及第四例选择待 E 确认**。未向 E 发送消息，不能称共同确认。

## 规则 v1：先保留 E 的设计，再固定歧义

范围先固定为 VOC / ResNet50 / eval 的 `model_correct=1`，387 张，每张必须有六方法四指标，共 24 值。不因热图是否好看删图。仅解释这个条件下的案例，不称跨模型/数据集代表样本。主展示三例不更换 E 已准备的素材。

1. 对每张图，每个指标在六方法内排名；raw AUC/time 越小名次越好，stability/valid_rate 越大名次越好。并列取平均名次。由六方法×四指标构成 24 维名次画像，对 387 张逐维取中位数。选择与中位画像 L1 距离最小者，若并列，预测置信度降序，再按 VOC 数值 ID 升序。得到 **215**：距离 0，置信度 .966232；899 等也距离 0，故不是唯一典型图。高置信的并列优先是 E 既有目标，不将 215 的好看程度当准则。
2. 质量取舍例：同一候选池计算六方法 AUC 好名次与稳定性好名次的平方差之和，取最大者；并列按数值 ID 升序。得到 **3803**：70；4675 也是 70，因此明确披露 tie，取较小 ID。
3. 困难例：同一池按预测置信度升序、数值 ID 升序，取 **7612**（.539305）。这是低置信命中，不能叫预测失败、误分类或多标签漏检。
4. 可选异常例：扩展到 **VOC 三模型的各自命中组**，限 Grad-CAM trace 已标记常量的图，按无效重复数降序、model 字符串升序、数值 ID 升序，取 **298 / VGG16**，5/5 无效。其六方法指标齐全，原图 hash 已核对；六方法 map 仅有目录 hash、未收到本体，视觉核验未完成。若只展示三例，常量现象可用 trace 文字讲述，不用旧 GPU 图补位。

逐规则若选到已入选图，按同一排序取下一未选图；不以人工观感替换。规则细化是 B 的可审查提案，不追溯声称 E 当初使用了本版每个并列细节。完整 387 行筛选得分见 [selection_scores.csv](../analysis/b_m5_20261026/selection_scores.csv)。

## 复现与材料身份

在具备 pandas、NumPy、PyYAML、Pillow 的分析环境，从仓库根目录执行：

```powershell
python -m analysis.b_m5_review --handoff 'C:\Users\hp\Downloads\dcu_team_handoff_20261014.zip' --analysis-handoff 'C:\Users\hp\Downloads\dcu_analysis_handoff_20261026.zip' --output-dir results/b_m5_recheck
```

可选加 `--image-root <已有data目录>` 核验原图；输出目录必须新建，已有回执不覆盖。脚本读取固定 E 提交，需已 fetch `feat/e-m5-final-deliverables`；脚本不读取模型、不生成归因。新逻辑测试使用人工标量 fixture，仅验证筛选/拒绝行为，不是实验数据。

[24 行候选表](../analysis/b_m5_20261026/case_candidates.csv)逐图逐方法保留：image_id、dataset、model、target、预测类别/置信度、model_correct/common_correct、四项真实数值、base/stability config hash 与快照路径/hash、checkpoint hash、原图路径/hash、NPY/PNG/sidecar 路径/hash、source archive 路径/hash和本轮收到状态。map hash 来自组长新包资产目录；三例 PNG/sidecar 同时与 E 的 Git 字节核对。NPY 列明确 `indexed_hash_only_bytes_not_received`，不是本轮已验二进制。

E 已有三例六方法 PNG。本轮逐字节核对 18 张 PNG、18 份 sidecar、配置绑定及本地三张原图，并查看真实灰度图和原图；没有用 PNG 重算任何指标。第四例原图可读，map 未收到。案例解释见 [§8 初稿](B_M5_CASE_ANALYSIS.md)，最终同图面板、PPT 与 Demo 由 E 负责。

## 最小补件

若 E 要第四例，提供 VOC / VGG16 / eval / image_id **298** 的六方法基础 NPY/PNG/原 sidecar。前五方法来自 `dcu_full_five_20261013_complete.tar.gz`（`852eb7c8c4eb74124b6c599d2fd54bf1f067f9c0579ce9a0542dfada040dbede`）；LIME 来自 `lime_vectorized_formal_complete_20261014_v2.tar.gz`（`38510f8eeda54ae38ee77acaf04580666779954ff7565b59d9fbd58438fe1cd3`）。

前五方法精确源成员为 `run/formal/<method>-vgg16-voc/maps_float/<method>_vgg16_voc/298.npy` 及 `.npy.provenance.json`，PNG 为相应 `maps/<method>_vgg16_voc/298.png`；LIME 为 `run/formal/lime-vgg16-voc__00/maps_float/lime_vgg16_voc/298.npy` 及 sidecar，PNG 在对应 `maps/`。各文件预期 hash 和完整包路径均已在候选 CSV，不需要重新归因。

若需 B 进一步核验前三例 NPY，最小补件是 E 已收到的 `dcu_cases_voc_resnet50_215_3803_7612.zip`，预期 SHA-256 `a46fa18f447e104fc707451de4a03e8b2c2eb56679c15a675f6cc5aaacde9eb3`。本轮未找到该小包，不把 E 的 113 项验收记录称为 B 自己执行。
