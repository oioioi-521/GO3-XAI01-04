# 未发送的交接/复核文本

2026-10-26。本文件仅保存待用户转发的草稿，本轮没有向Issue、成员或PR发布这些文字。提交分支 `docs/b-m5-review-20261026`，统一入口 `docs/B_M5_DELIVERY.md`。

## 给 #19

B已基于最新dafb655完成模型/指标复核、三cohort连接与权重绑定、4个真实CSV候选、§8解释、Occlusion答辩和C环形审查。正式输入仍为SHA256 14b6ea…303dc；组长10/26包是同一数据的分析补充，722项原包内容逐字节一致。前三例真实原图和18张归档PNG/sidecar已核验，NPY尚未收到；298为有trace证据的第四候选，map视觉待补件。待E确认最终选图、D回复统计例外、C处理审查项；未新增正式实验。交付见docs/B_M5_DELIVERY.md，#19保持开放。

## 给 E / #22

B复现并支持你现有215、3803、7612候选，规则已明确平均名次、中央画像L1、质量名次平方差、低置信及并列处理：215距离0（并列取高置信）；3803差异70（与4675并列取小ID）；7612置信最低。18张PNG/sidecar与组长目录hash一致，原图也一致且已查看。最终组合待你确认；可选第四例VOC/VGG16/298的Grad-CAM 5/5常量，需从现成归档补六方法map，不重跑。解释及精确资产路径/hash见docs/B_M5_CASE_ANALYSIS.md和B_M5_CASE_SELECTION.md。不能将7612写为预测错误，也没有正式逐图LIME R²支持局部拟合失败标签。

## 给 D / #21

B核对ImageNet/DenseNet121的IG/KernelSHAP：主组n371，优势定义KS−IG，均值−.0031269003，95%CI[−.0084232795,.0020708717]；IG胜259/371，秩二列+.3115961，Wilcoxon Holm p=5.15549e−6。共同组/全eval也呈CI跨0、秩偏IG。请确认采用“KS均值点估计低，均值优势未证实；秩方向偏IG”，不把秩显著说成均值显著。三口径摘录与具体核对项见docs/B_M5_D_STATISTICAL_CHECK.md。本轮只核对既有统计，未重跑检验；尚待你的回复。

## 给 PR #24 / C（review草稿，不是approval）

B已在5079814候选执行191受保护文件审计，#14的6项基线已有、30项历史工具/材料按blob保留；38项相关CPU测试通过。262项/16项及完整统计复现只查阅你的回执，没有冒称B重跑。需处理：①verify_data_version实际新增CRLF规范化，与最新#20严格字节口径有出入；②reproduce_analysis的passed未纳入figure_byte_matches_published（当前回执四图均true）；③更新旧PR/#8状态，补dafb655和#23的待合稿差异。详见docs/B_M5_C_RING_REVIEW.md。不建议删除已保留历史证据，不重开#14，不启动归因重跑。
