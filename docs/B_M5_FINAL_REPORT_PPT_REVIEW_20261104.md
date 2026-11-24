# B 最终报告 / PPT 共同口径核对

2026-11-04 · B / YaoYang-123456 · 对应 [Issue #19](https://github.com/oioioi-521/GO3-XAI01-04/issues/19)。

**结论：报告 v4 的 B 责任口径与 PPT v7 可见正文、所查数值相容；PPT 第 1、6、11、12 页备注有四项必要修订，暂不确认整套展示最终通过。** 无需重跑实验。问题限于图源与交接状态，不据此推翻已核对的统计数值或已完成的组员核验。

## 版本、材料与范围

| 对象 | 本轮实际固定版本 | 阅读 / 查看范围 |
| --- | --- | --- |
| A / PR #23 | [`3cfd4a8`](https://github.com/oioioi-521/GO3-XAI01-04/commit/3cfd4a8250b7f820e241bbcae3df93aa374f6c8a)；[报告 v4](https://github.com/oioioi-521/GO3-XAI01-04/blob/3cfd4a8250b7f820e241bbcae3df93aa374f6c8a/docs/FINAL_REPORT_DRAFT.md) | 全文，重点 §2、§3.1–3.3、§7–§8；合稿责任表、A 接收记录 |
| E / PR #26 | [`d00553f`](https://github.com/oioioi-521/GO3-XAI01-04/commit/d00553f9ca7fc41208e24296619c1406a339d8da)；[v7 目录](https://github.com/oioioi-521/GO3-XAI01-04/tree/d00553f9ca7fc41208e24296619c1406a339d8da/deliverables/e_m5) | 12 页 PPTX 正文及全部演讲者备注；导出 PDF 全部 12 页经 Poppler 渲染后逐页查看；案例 CSV / Demo JSON |
| B / PR #25 | [`f224484`](https://github.com/oioioi-521/GO3-XAI01-04/commit/f224484ea1edab646ee31a1fcaa51c1dac3bef7f)；[已批准交付入口](https://github.com/oioioi-521/GO3-XAI01-04/blob/f224484ea1edab646ee31a1fcaa51c1dac3bef7f/docs/B_M5_DELIVERY.md) | A 已批准版本，保留事后选图修正；沿用其冻结源码索引与 24 行案例清单 |

三个 HEAD 与用户给出的版本相同，无需重审已被替代的 v3/v6。#18/#19 与 #23/#25/#26 正文、全部评论/reviews 已刷新；#19 仅有 B 10/26 的旧交接，不是本次最终口径回执。#25 仍 open/Draft，工作树干净后仅快进接收 A 的单文件图注修订，未覆盖组员分支。

分析分支实际已更新为 [`7f9db4fd972a21fbd22ff724deedfe671cc995d8`](https://github.com/oioioi-521/GO3-XAI01-04/commit/7f9db4fd972a21fbd22ff724deedfe671cc995d8)，比 #18/#19 文字中的 `a4fe4fb` 多出 D 的 11/2 最终展示复核；统计数值基线仍为 `272ca25fccb30cc1511bbd3684f2c99397e66e93`。已读取 [D v7 / A v4 确认](https://github.com/oioioi-521/GO3-XAI01-04/blob/7f9db4fd972a21fbd22ff724deedfe671cc995d8/docs/D_E_V7_FINAL_REVIEW_20261102.md)，不再把 D 的 v7 统计确认列为首次待办；其指出的第 12 页过期备注仍存在。

v7 文件：`GO3-XAI01-04_E_M5_展示初稿_20261101_v7.pptx` / 同名 PDF。PPTX SHA-256 `ae017a7883ee8fc4baf1e53f280cb0d30fdefa1a373f2f4f4a51aa97e1e39ff9`，PDF `4c5993864e95cf7936e989dfab40867b67eb6e44eafe49a324e96a6e31db0af8`，与 D 记录一致。PDF 为位图，正文不可搜索；本轮实际查看渲染页，没有用“文本提取为空”代替阅读，也没有在 PowerPoint 应用中打开并验收放映。

## 七项核对

| 检查项 | 报告位置 / PPT 页码 | 本轮结论与证据边界 |
| --- | --- | --- |
| 1. 模型、权重、输入、target / 响应 | §2.1–2.3；P2、P9–11 | 相容。ResNet50 / DenseNet121 / VGG16；ImageNet1000、VOC20，RGB→224×224→ImageNet mean/std；metadata target。IG / Grad-CAM / Occlusion / KernelSHAP 解释 logit，RISE / LIME 解释 softmax / sigmoid 目标概率。模型工厂返回 logits，不把同 target 写成同响应。六权重 SHA / VOC 顺序由报告固定链接至 [B 模型表](https://github.com/oioioi-521/GO3-XAI01-04/blob/f224484ea1edab646ee31a1fcaa51c1dac3bef7f/docs/B_M5_MODEL_AND_METRICS.md)；PPT 是摘要，未逐项列六套权重及预处理常量，也未作相反陈述。 |
| 2. MoRF、噪声、退化、计时 | §2.4、§7；P3–5、P9、P11–12及备注 | 相容。raw AUC 为目标概率21点、0/.05/…/1、黑色替换、梯形积分、越低越好，不除原分数。RGB sigma=.005、固定方法seed、每图5次先聚合；未定义相关为空、score0，valid_rate=有效次数/5。wrapper预热/同步、内部校验/构图/拟合/CPU返回与外部预测/MoRF/I/O分开；两轮共享DCU不作受控硬件排名。PPT不展开全部公式，细节按报告/B材料，不将省略误认成另一套参数。 |
| 3. 三类样本与VOC | §2.1、§2.5；P2–5、P7、P9–10 | 相容。全eval460；本模型命中 ImageNet408/371/363、VOC387/388/400；共同命中332/353。P2备注完整列n，阶段/三因素用共同组。VOC为冻结单目标sigmoid argmax，不是exact match/mAP；跨任务raw AUC不作统一质量排名。 |
| 4. 三例、298、选图与LIME限制 | §8.1–8.2；P10–11及备注 | 数值/身份/可见图注通过，P11备注状态待修。215/3803/7612均VOC/ResNet50命中，target7/2/11、confidence .966232/.868719/.539305；7612非误分类。按已有结果可复现事后选图，非预注册；298/VGG16/person14仅留报告，Grad-CAM退化不是分类失败，Demo仍只有三例。无逐图正式LIME R²，不宣称局部拟合失败。 |
| 5. IG / KernelSHAP局部例外 | §3.1、§6；P3、P6、P12及备注 | 数值/统计语义通过，P6备注图源待修。ImageNet/DenseNet121，KS−IG均值优势−.0031269003，95%CI[−.0084232795,.0020708717]跨0；秩二列+.3115961，Holm p=5.15549e−6偏IG。正文没有把秩显著变成均值显著。P6实际图内标题和PDF均为ImageNet，错误只在备注的来源文字。 |
| 6. 三因素、相关与机制 | 摘要、§2.5、§3.2–3.3、§5、§9；P4、P8、P12及备注 | 相容。AUC七效应显著；稳定性只数据集主效应未显著（共同组名义p=.121215、WTS=.1148），不写全主效应显著、不显著不等于等效。名义/WTS分列、9999次bootstrap及Holm .0014分辨率保留；总体交互不替代局部比较，VOC DID分层/IID区间和秩p=.8374不省略。相关不当因果，机制只作假设。仅核对既有JSON，不重做统计。 |
| 7. Occlusion、答辩及批次 | §2.3、§7、§8；P11 Occlusion数值及 [B答辩](https://github.com/oioioi-521/GO3-XAI01-04/blob/f224484ea1edab646ee31a1fcaa51c1dac3bef7f/docs/B_M5_OCCLUSION_AND_DEFENSE.md) | 相容。重叠窗口黑基线替换后的target logit差，正式32×32、stride16、每批16；通道均值后min-max，非原始贡献幅度。保留窗口/基线/分布变化、固定预算与共享计时局限。PPT只展示六方法与案例，无独立Occlusion原理页；不能称12页PPT已单独展开全部B问答，答辩细节沿报告固定入口。未混入旧RTX/Windows值。 |

## 必要修订：E / PR #26 备注，A 合稿接口同步

仅改备注即可解决以下问题；本轮未修改 A/E 文件。PDF 可见正文、统计数值及案例资产无需因此重做。E 修订后记录新 PPTX SHA，再按新版本核对四处备注，不能沿用本轮文件 SHA 宣称未变。

| 位置（PPT包内） | 当前问题 / 影响 | 可直接替换的文字 |
| --- | --- | --- |
| P1，`ppt/notesSlides/notesSlide1.xml` | 开场备注称“v7（本地修订）”“v7尚未提交或推送”，与实际远端v7及可见封面状态不符，演讲者会报错版本。 | `E/M5 展示稿 v7 已进入 Draft PR #26，包含10/29混合三因素结论。215/3803/7612组合和事后选图规则已确认；独立设备Demo、E→A复核与全组实际计时仍按各自回执验收。` |
| P6，`ppt/notesSlides/notesSlide6.xml`首句 | 写成“paired_quality_effects.png的VOC面板裁切”，但本页可见图及标题实际是ImageNet；会把DenseNet/ImageNet的均值/秩例外讲错任务。 | `图源为 analysis/dcu_20261015/paired_quality_effects.png 的 ImageNet 面板裁切。正值偏向行方法，星号为质量族Holm校正的配对秩检验，不是均值检验。以IG与KernelSHAP为例，ImageNet/DenseNet121的均值差CI跨零，配对秩方向偏IG；二者分开表述。` |
| P11，`ppt/notesSlides/notesSlide11.xml`首句/末句 | 仍称“候选”“候选尚待B最终复核”，与P10、§8.1及既有B/E三例确认混淆。不能把已完成的选图确认重新列为首次待办；本次最终共同口径仍有备注修改意见。 | 首句改为 `本页展示已确认的 image_id 215（VOC eval，真值cat/7，ResNet50）的六方法正式PNG预览。`；末句改为 `215/3803/7612的组合及选图规则已获B/E确认；最终报告/PPT共同口径核对见Issue #19，独立设备Demo另行记录，不由本页代签。` 中间来源/hash/数值说明保留。 |
| P12，`ppt/notesSlides/notesSlide12.xml`“限制”段 | 仍称“v6已在Draft PR #26；v7尚未commit/push”。D在11/2已明确指出，当前固定版本未修；可见正文已正确，不将其重新报为统计结论未完成。 | 将该状态句替换为 `v7 已进入 Draft PR #26；独立设备演练和全组实际计时仍待完成。` 其余三因素、计时、VOC和退化边界保留。 |

## 已完成协作与本轮实际检查

- [A 10/27补验](https://github.com/oioioi-521/GO3-XAI01-04/blob/3cfd4a8250b7f820e241bbcae3df93aa374f6c8a/docs/A_M5_REVIEW_20261027.md)：24组NPY及配套来源、298基础Grad-CAM全零。是A的核验，本轮B未重验NPY，不再索取同一资产作为口径核对前提。
- [D统计例外确认](https://github.com/oioioi-521/GO3-XAI01-04/blob/7f147c35d8afc6424a06e47a8fd7d5fed3a406aa/docs/D_M5_STATISTICAL_CONFIRMATION_20261027.md)、[E三例确认](https://github.com/oioioi-521/GO3-XAI01-04/issues/22#issuecomment-6464441650)、A对B的批准及 [C两项门禁修复回执](https://github.com/oioioi-521/GO3-XAI01-04/blob/eb275ed535a1ba660608a9606fd3847f83460405/analysis/a_pr24_gate_review_20261031.json)均已查阅，保留执行人和日期，不重新列为首次待办。
- 本轮B实际完成：12页正文/全部备注提取阅读、12页PDF逐页目视、报告全文阅读；报告案例48个值、E案例CSV72个值、Demo JSON72个值、第11页18个值/12个hash前缀核对；第11页6张嵌入PNG与正式E预览及B索引逐字节一致；15份既有惰性冻结源码副本与原SHA索引一致。读取两cohort共28项既有因子结果的显著性分列，不重新运行检验。
- 原轻量ZIP当前在此前Downloads路径已不存在；现有惰性提取副本/已提交索引足以复核执行语义。本轮未重新验整包、模型权重二进制或NPY，不将历史校验说成本轮重做。
- [机器可读核对回执](../analysis/b_m5_20261104/final_review_receipt.json)保存实际版本、来源hash、检查范围与四项意见。未执行训练、归因、MoRF、完整统计、案例筛选，也没有运行独立设备Demo或全组计时。

本次只确认所列报告正文、PPT可见内容及静态数值范围。E清理四处备注后再形成该新版的共同口径确认；C新增统计独立复现、E→A、独立设备Demo与全组实际演练继续由各责任人记录，本记录不代签这些事项。Issue #19保持开放。
