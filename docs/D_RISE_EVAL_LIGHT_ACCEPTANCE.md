# 成员 D：RISE 正式 eval 轻量包接收复核

复核日期：2026-10-09。接收包：
`D:\微信文件\xwechat_files\wxid_k52y4rqkvr8h12_9d8f\msg\file\2026-10\C_RISE_EVAL_LIGHT_20261009_214005.zip`；
SHA-256 `edf10e2b07a167e7ea939dee1fc24c7090ba7082041a8f34c9575993cdfb2c39`。

## 可接受范围

D 以 `analysis.validate_rise_formal_light` 直接只读校验 ZIP，未执行交接包代码。
46 个清单文件大小与 SHA-256 均一致，无多出/缺失/重复或路径穿越。六个
ResNet50/DenseNet121/VGG16 × ImageNet/VOC `eval` 单元各有 460 条唯一预测、
每项基础指标各 460 条唯一逐图记录；共 2,760 条预测、5,520 条逐图指标和
12 条单位汇总。每单元均与冻结 `data/metadata.csv` 的 460 个 eval 图像 ID 和目标
类别一致，且有单一 checkpoint 身份；配置哈希与结果状态、验证报告一致。
逐图均值、样本标准差、`n=460` 和正确/错误组 faithfulness 汇总均经 D 重算。
六组 run log 均有初次 460 张与立即续跑 `processed=0, skipped=460` 证据。

| 单元 | 正确预测 | raw MoRF AUC 全图均值 | 本机归因耗时均值 ms |
|---|---:|---:|---:|
| ResNet50 / ImageNet | 408/460 | 0.107078 | 11235.03 |
| ResNet50 / VOC | 387/460 | 0.541164 | 10103.93 |
| DenseNet121 / ImageNet | 371/460 | 0.054237 | 10836.08 |
| DenseNet121 / VOC | 388/460 | 0.416162 | 10157.48 |
| VGG16 / ImageNet | 363/460 | 0.052469 | 22435.32 |
| VGG16 / VOC | 400/460 | 0.268427 | 20953.30 |

以上 raw AUC 是**全图描述性数值**；faithfulness 主分析仍须按预测正确组筛选，
不能直接据全图均值排列方法。ImageNet softmax 与 VOC sigmoid 分开分析；VOC 的
`correct=1` 只表示冻结目标标签是最高分标签，不是多标签 exact-match。耗时来自 C 的
RTX 4060 Laptop GPU，不与其他 GPU 的结果直接排名。

## 尚未由该包证明

轻量包不含原始 float32 NPY、PNG 和模型权重；其中验证报告的图像哈希只是 C 的
交接声明，D 不能用这个包独立复核原图内容、图像与权重的生成时绑定。
本次只接受六组正式**基础 CSV 与汇总口径**用于后续受限统计；不接受 RISE 稳定性、
原图来源完整验收、跨方法全量排名、ANOVA、相关性或 Pareto 已完成的表述。

复核命令：

```powershell
.\.venv\Scripts\python.exe -m analysis.validate_rise_formal_light 'D:\微信文件\xwechat_files\wxid_k52y4rqkvr8h12_9d8f\msg\file\2026-10\C_RISE_EVAL_LIGHT_20261009_214005.zip'
```
