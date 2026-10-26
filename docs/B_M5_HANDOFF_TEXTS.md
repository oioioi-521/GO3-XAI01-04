# B 侧 GitHub 交接发布记录

2026-10-26 · YaoYang-123456。本轮经用户明确授权，已发布三条 Issue 评论及一次正式 **Comment review**，全部成功读回；不是 Approve 或 Request changes。没有重复发布、关闭 Issue、合并 PR 或发布 main；PR #25 保持 Draft。

发布前读取 #19/#21/#22、#24/#25 正文和全部评论/reviews，均未发现相同交接或组员回复。#24 HEAD仍为 `507981464a770ee92dc43b00567f9b42f81071f4`；#25发布时HEAD为 `1e71afde70b4ed1222d891646abea90b74ccc376`。重新读取C三项对应源码/文档，未发现后续修复提交；协议已标明当前/历史范围，正式review据此收窄第三项措辞。

## 发布回执

以下时间直接取 GitHub 的 `created_at` / `submitted_at`，同时列 America/New_York 当地时间（EDT，UTC−04:00）。

| 位置 | 类型 | 发布链接 | 实际发布时间（EDT） | GitHub UTC |
| --- | --- | --- | --- | --- |
| #19 B完成说明 | Issue comment | [已发布](https://github.com/oioioi-521/GO3-XAI01-04/issues/19#issuecomment-6417574200) | 2026-10-26T09:46:26-04:00 | 2026-10-26T13:46:26Z |
| #22 给E | Issue comment | [已发布](https://github.com/oioioi-521/GO3-XAI01-04/issues/22#issuecomment-6417575729) | 2026-10-26T09:46:29-04:00 | 2026-10-26T13:46:29Z |
| #21 给D | Issue comment | [已发布](https://github.com/oioioi-521/GO3-XAI01-04/issues/21#issuecomment-6417577319) | 2026-10-26T09:46:33-04:00 | 2026-10-26T13:46:33Z |
| PR #24 B→C复核 | COMMENTED review | [已发布](https://github.com/oioioi-521/GO3-XAI01-04/pull/24#pullrequestreview-5652946501) | 2026-10-26T09:46:39-04:00 | 2026-10-26T13:46:39Z |

[机器可读发布回执](../analysis/b_m5_20261026/publication_receipts.json)记录ID、作者、正文SHA-256与review commit_id。评论中的成果链接固定到实际交付提交1e71afd；本次记录提交只更新文档/回执，不改变实验或审查对象。

## 仍待回复

- E / @onlysonder：确认215、3803、7612最终组合与规则；是否加入298。298尚缺map视觉核验，前三例NPY本体未由B核验；补件仍按[精确清单](B_M5_CASE_SELECTION.md)走现成归档，不重跑。
- D / @lanshuheng9-oss：确认KernelSHAP均值点估计较低但均值差CI跨零、配对秩方向偏IG的分开表述，尚不代表D已确认。
- C / @oioioi-521：回复或修订文本hash语义、四图总passed门禁及旧PR/基线状态说明。未重新执行C全量统计或代签验收。

## 实际发布正文

发布前草稿作为历史保留在[原交付提交](https://github.com/oioioi-521/GO3-XAI01-04/blob/1e71afde70b4ed1222d891646abea90b74ccc376/docs/B_M5_HANDOFF_TEXTS.md)。下列为正式已发布版本。

### Issue #19

B 的当前可执行交付已提交到草稿 [PR #25](https://github.com/oioioi-521/GO3-XAI01-04/pull/25)，[成果入口 B_M5_DELIVERY.md](https://github.com/oioioi-521/GO3-XAI01-04/blob/1e71afde70b4ed1222d891646abea90b74ccc376/docs/B_M5_DELIVERY.md)（交付提交 `1e71afde70b4ed1222d891646abea90b74ccc376`；分析基线 `dafb655`）。

已完成 §2.1 三类样本口径复核、§2.2 模型/权重与逐方法响应量、§2.4 指标/计时边界、§8 真实案例初稿、Occlusion 原理/正式参数/结果解释/局限/答辩，以及对 C / PR #24 的环形复核。

支持 E 已有的215、3803、7612三例，并提供可复现规则及六方法逐图清单；前三例原图和18张归档PNG/sidecar已核验、查看。298/VGG16为可选常量Grad-CAM例，map视觉核验尚未完成；未收到的NPY只保留目录hash绑定，不冒称B已验本体。

正式输入仍是SHA-256 `14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc` 的统一DCU包，10/26补充包中的原包内容逐字节一致。未新增正式实验。

仍待 E 确认最终选图/规则，D 回复均值CI与秩结论核对，C 处理环形审查意见；这些不标为组员已确认。#19保持开放，PR #25保持Draft。

<!-- B-M5-HANDOFF-19-20261026 -->

### Issue #22

@onlysonder B已复现并支持你现有的VOC/ResNet50命中组候选 **215、3803、7612**；请确认最终组合与选图规则。材料在 [PR #25](https://github.com/oioioi-521/GO3-XAI01-04/pull/25)，固定交付提交 `1e71afde70b4ed1222d891646abea90b74ccc376`。

- 215：六方法四指标名次画像与池内中位画像的L1距离为0；并列优先高置信度，再按数值ID。
- 3803：AUC/输入噪声稳定性的方法名次平方差和70，与4675并列，按较小ID选取。
- 7612：387张命中组中置信度最低（.539305），**仍是预测命中，不是误分类**。

原图及18张正式PNG/sidecar已按hash核对并查看。没有正式逐图LIME weighted R²证据可证明这些案例局部拟合失败，不能据外观贴此标签；固定方法seed的输入噪声测试也不是采样随机性测试。

**可选第四例：VOC/VGG16/298**，Grad-CAM五次均为常量相关退化，尚缺map视觉核验。若采用，请从既有归档提供六方法基础NPY/PNG/原sidecar，不重跑。前三例NPY本体也尚未由B核验。

[案例解释](https://github.com/oioioi-521/GO3-XAI01-04/blob/1e71afde70b4ed1222d891646abea90b74ccc376/docs/B_M5_CASE_ANALYSIS.md) · [选择规则及精确补件清单](https://github.com/oioioi-521/GO3-XAI01-04/blob/1e71afde70b4ed1222d891646abea90b74ccc376/docs/B_M5_CASE_SELECTION.md) · [24行来源/hash清单](https://github.com/oioioi-521/GO3-XAI01-04/blob/1e71afde70b4ed1222d891646abea90b74ccc376/analysis/b_m5_20261026/case_candidates.csv)。最终面板、PPT和Demo由E整合，本评论不代替你确认最终选图。

<!-- B-M5-HANDOFF-22-20261026 -->

### Issue #21

@lanshuheng9-oss 请复核ImageNet/DenseNet121中IG与KernelSHAP的均值/秩例外。B的[三口径统计核对](https://github.com/oioioi-521/GO3-XAI01-04/blob/1e71afde70b4ed1222d891646abea90b74ccc376/docs/B_M5_D_STATISTICAL_CHECK.md)已随 [PR #25](https://github.com/oioioi-521/GO3-XAI01-04/pull/25)交付（固定提交 `1e71afde70b4ed1222d891646abea90b74ccc376`）。

主分析 `model_correct`，n=371；优势方向定义为 **KernelSHAP raw AUC − IG raw AUC**（正数表示IG的AUC较低）：

- 均值优势 −0.0031269003；配对均值差95% CI [−0.0084232795, 0.0020708717]。
- IG胜259图、负112图；秩二列效应 +0.3115961；Wilcoxon Holm p=5.15549e−6（该质量检验族180项）。

请确认采用：**KernelSHAP均值点估计较低，但均值差CI跨零；配对秩方向偏IG。** 两者针对不同统计量，不能把秩显著写成任一方法均值显著更低，也不能把边际95% CI写成Holm校正后的同时区间。共同命中组n332、全eval n460的数值亦在链接表中分开列出。

B已从正式CSV核对n、均值差和胜图数，CI/效应量/p逐项对照既有JSON及报告；未重新运行bootstrap/Wilcoxon/置换。本评论是请求D确认，尚不表示D已回复或验收。

<!-- B-M5-HANDOFF-21-20261026 -->

### PR #24 / C Comment review

B / A→B→C→D→E→A 环形复核，**Comment review**。本次重新读取PR正文、评论及最新HEAD `507981464a770ee92dc43b00567f9b42f81071f4`；与B先前审查的提交相同，未发现后续修复提交或已有相同review。

1. **文本身份与严格字节hash：仍待协调。** [verify_data_version.py](https://github.com/oioioi-521/GO3-XAI01-04/blob/507981464a770ee92dc43b00567f9b42f81071f4/preprocessing/verify_data_version.py)实际在CSV/JSON哈希前CRLF→LF；因此换行字节变化仍可通过。它可验证规范化文本身份，但不等于严格原字节身份，与当前#20要求存在口径差异。请恢复严格字节比较，或明确命名/记录规范化语义并与要求协调。冻结ZIP的严格逐文件校验未被改成这种规则，未据此发现正式输入已变。
2. **四图门禁：仍未纳入总passed。** [reproduce_analysis.py](https://github.com/oioioi-521/GO3-XAI01-04/blob/507981464a770ee92dc43b00567f9b42f81071f4/scripts/reproduce_analysis.py)的`passed`要求图数为4，但`figure_byte_matches_published`在之后回执中单独计算，未参与总状态。即使四图齐全但一图字节不一致，总状态仍可能passed。C现有回执四项均true，不能据此说当前四图实际不一致；建议将all(图字节比较)纳入通过条件并加针对性回归。
3. **状态/基线：部分已有说明，仍需同步。** [STABILITY_PROTOCOL](https://github.com/oioioi-521/GO3-XAI01-04/blob/507981464a770ee92dc43b00567f9b42f81071f4/docs/STABILITY_PROTOCOL.md)开头已写10/26当前完成状态，并将10/1补跑/#8文字标作历史，认可这部分更新；历史段可进一步用标题/日期避免单独摘录歧义。实际#8/#14/#15/#16均已关闭，但[整合说明](https://github.com/oioioi-521/GO3-XAI01-04/blob/507981464a770ee92dc43b00567f9b42f81071f4/docs/C_M5_INTEGRATION.md)及PR正文仍称旧PR开放/待处理。README仍以2989537为整合起点；作为历史起点有效，但需另列当前分析基线dafb655及#23报告尚待整合，避免误认为已跟进全部现行交付。无需删除已保留历史材料或重开旧PR。

**验证归属：** B此前在此同一HEAD亲自执行191受保护blob/祖先审计（#14六项基线已有、三十项来源内容保留），并运行35项Occlusion/工具/复现测试及3项数据版本测试，共38 passed。本次HEAD未变，执行的是源码/状态重新核对，没有新跑实验或重复声称全量测试。C的全仓262项、独立CPU16项及七表/四图完整统计复现仅查阅C回执，未由B重跑；不包含B重新验算bootstrap/置换或噪声图。

[详细B复核文档](https://github.com/oioioi-521/GO3-XAI01-04/blob/1e71afde70b4ed1222d891646abea90b74ccc376/docs/B_M5_C_RING_REVIEW.md) · [B实际审计/测试回执](https://github.com/oioioi-521/GO3-XAI01-04/blob/1e71afde70b4ed1222d891646abea90b74ccc376/analysis/b_m5_20261026/c_review.json)。以上限于固定HEAD的B相关工具、三项门禁/文档及已列证据，不代签D统计验收；未提交Approve或Request changes。

<!-- B-M5-REVIEW-24-20261026 -->
