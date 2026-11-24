# B 的 VOC 多标签与 MoRF 指标契约

VOC-20 checkpoint 使用 20 维 logits、multi-hot 标签、BCEWithLogitsLoss 和 sigmoid 推理；
类别顺序固定为 preprocessing/extract_voc_labels.py 中的 VOC_CLASSES。

正式 MoRF 字段和公共函数统一为 faithfulness_morf_auc_raw：它计算删除后的真值目标
概率 AUC，值域为 [0, 1]，低值更好。ImageNet 使用 softmax，VOC 使用 sigmoid，
由配置的 output_activation 明确控制。该字段是与 PR #6 消费接口一致的正式指标。

faithfulness_morf_auc 保留为历史 ratio 指标。原始真值概率很小时它可大于 1，
因此仅用于历史可追溯，不能用于正式排名、Pareto 或 ANOVA。

冻结的 500 张 VOC debug/eval 图片不进入多标签 manifest、训练、验证、早停或调参。
