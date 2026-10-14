# D 的图表与结果交付入口

更新日期：2026-10-14。请在 `feat/kernelshap-analysis` 分支查看；未合并 main。
本页汇总可直接阅读/下载的报告图，小规模示例与正式结果分别标记。

## 中期要求的两张图和两句结论

- [四方法 ImageNet 忠实性图](../analysis/midterm_four_method_faithfulness.png)：
  IG/Grad-CAM/Occlusion/RISE，三模型每单元 460 张；RISE 已按接收到的基础 CSV 重算。
  **全样本描述图中 IG 在三模型的 raw MoRF AUC 均最低，但不是预测正确组配对最终排名。**
- [A 同机 IG / Grad-CAM 双目标 Pareto 图](../analysis/midterm_a_ig_gradcam_pareto.png)：
  仅 A 的 RTX 2080 Ti、逐模型/数据集成对比较。
  **IG 的 AUC 更低，Grad-CAM 的耗时更短，六组各自成对比较均存在双目标取舍。**

来源、样本差异和复现方式见 [中期比较说明](MIDTERM_D_COMPARISON.md)。

## D 正式结果与 E 的中期示例

- [KernelSHAP 六单元正式汇总图](../analysis/kernelshap_formal_summary.png)：
  2,760 张单元内图像、13,800 条有效 trace；每单元立即续跑 0 processed / 460 skipped。
  2026-10-12 独立校验再次通过。[测量值与来源](KERNELSHAP_GATE_RESULTS.md)。
  图中的效率来自单次本机运行，VGG16/VOC 存在极大计时标准差，不用于方法选择。
  [计时离群审计](../analysis/kernelshap_timing_audit.json)保留全部记录：该单元中位数
  约 9.776 s，最大一条约 7609.331 s；没有据此删除数据或重跑归因。
- [E 提供的同图五方法定性示例](../analysis/lime_midterm_same_image.png)：
  VOC debug 8173 / ResNet50 / train，原面板不修改；仅定性，非忠实性或稳定性排名。
  [D 的受限接收记录与缺项](D_LIME_MIDTERM_RECEIPT.md)。

## 可以交付与仍需等待的边界

当前 D 已验收自己的六个 KernelSHAP 正式单元、B 的六个 Occlusion 正式稳定性单元、
C 的六个 RISE 基础 CSV 单元。A 的 12 单元正式稳定性在 [PR #16](https://github.com/oioioi-521/GO3-XAI01-04/pull/16)
报告完成，但 D 当前工作区没有完整逐图交接输入用于统一配对验收。
RISE 正式稳定性与 LIME 正式 eval 未在本机接收。全组 36 单元统计没有完成。

后续需要 A 的逐图指标/预测/trace，C 的稳定性与原图来源，E 的原始门禁、缺口修复和正式预算。
核对共同正确样本、权重/预测上下文后再执行限定范围或全组正式统计。
不同 GPU 耗时不能直接排序，ImageNet softmax 与 VOC sigmoid 分开分析。

2026-10-13：E 的代码提交 `64d2ffd` 已由 D 做独立工程复测（26 项 LIME 专项、
94 项全仓通过）。附加探针复现了续跑完整性/记录校验及输入图片身份缺口，预算暂不批准。
见 [D 的 LIME 代码审查与 E 待交清单](D_LIME_CODE_REVIEW_20261013.md)。
此处的 94 项是 E 分支隔离测试，不替代下面 D 分支既有 108 项复验记录。
同日 D 分支收尾复验仍为 108 passed、2 条既有 warning，依赖检查通过；审查副本及
输出目录已从默认测试搜集排除，原实验环境和结果未修改。

2026-10-14：D 已补 [原始 LIME 门禁独立校验工具与交接步骤](D_LIME_GATE_VALIDATION.md)，
用 NPY 重算 comparisons/seed variance 并核对 records/summary；只读数据、不执行 E 的脚本。
尚未收到新服务器原始包，也未批准预算。新增软件测试不等于真实门禁验收。
新增校验器 25 项测试通过，D 全仓 133 passed、2 条既有 warning，`pip check` 正常。

## 本次复验

2026-10-12：KernelSHAP 六单元独立正式校验、RISE 六单元基础 ZIP 校验、LIME 中期
样例受限校验与三张报告图重生成均通过；四张交付图（含既有 A 双目标图）已目视检查。
完整测试 `108 passed, 2 warnings`，两条是既有 `torch.load` FutureWarning；
`pip check` 无依赖冲突，`git diff --check` 通过。没有启动新 GPU 实验，没有删改原始结果。

详细进度见 [D 交接记录](W2_D_HANDOFF.md)。原始数据、weights、NPY、全量实验输出和 ZIP
仍通过受控外部交接，不入 Git；本页链接的是可公开给组内复核的图表与摘要。
