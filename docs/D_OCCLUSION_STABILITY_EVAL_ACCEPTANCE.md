# 成员 D：Occlusion 460 张正式稳定性交接验收

验收日期：2026-10-09。提供方 B 的交接分支：`origin/feat/b-occlusion-stability-gates`，
复核时头提交 `a2246ea`。这是六个 Occlusion 单元的正式 eval 接收验收，**不是**
全项目 36 单元或跨方法最终比较。

## 交接包与独立校验

- 本机接收包：`D:\微信文件\xwechat_files\wxid_k52y4rqkvr8h12_9d8f\msg\file\2026-10\occlusion_stability_eval_light.zip`；
  SHA-256 `bcfdc31373b51d103711a5fd053e488fd8a2501822a86eb38e241671dff3d0b3`。
- D 使用 `analysis.validate_occlusion_formal_delivery` **直接读取 ZIP，不执行其中代码**。
  2,862 个清单文件的大小与 SHA-256 均匹配；无清单外文件、重名或路径穿越。
- 六单元每个都有 460 条唯一预测、四项指标各 460 条唯一逐图记录、每项 `n=460`
  的单位汇总、2,300 条唯一 stability trace（460 图 × 5 repeat）。合计 2,760 图、
  13,800 条 trace、5,520 条新增稳定性逐图记录。13,800 条状态均为 `valid`，
  `stability_valid_rate=1`。
- 冻结协议 Git blob SHA-256 为
  `324315f1ecf62a807ec91e77b2cba1ea53593e9a7ad98a2a90cce09a9e32dcdf`。
  D 核对了包内协议内容、六份配置/检查点哈希、逐行 SHA 派生的 seed、元数据目标
  类别与跨模型成对 seed。逐图均值、单位均值和样本标准差由 trace 重算，与文件及
  交接摘要一致。六组立即续跑记录均为 `processed=0, skipped=460`。
- 基础 eval 的预测、faithfulness/efficiency 逐图记录及汇总与 D 此前接受的 B 基础
  轻量包逐行一致；2,760 个来源 sidecar 的图哈希与此前接受的 float32 NPY 逐文件
  SHA-256 一致。参考图绑定为 `verified_recomputation`：证明当前模型重算与旧图符合
  交接方规定容差，**不证明旧图在最初生成时使用了该权重**。

## 六单元测量值

Spearman 越高表示同一图像在冻结 RGB Gaussian 扰动协议下，归因排序越稳定。
这里仅列 Occlusion 内部的描述性结果，不做跨 GPU 耗时排序。

| 单元 | Spearman 均值 | 图间样本标准差 | valid repeats |
|---|---:|---:|---:|
| ResNet50 / ImageNet | 0.916753 | 0.070779 | 2300/2300 |
| ResNet50 / VOC | 0.915317 | 0.056971 | 2300/2300 |
| DenseNet121 / ImageNet | 0.987817 | 0.012083 | 2300/2300 |
| DenseNet121 / VOC | 0.987885 | 0.012658 | 2300/2300 |
| VGG16 / ImageNet | 0.990120 | 0.012890 | 2300/2300 |
| VGG16 / VOC | 0.992078 | 0.008619 | 2300/2300 |

验收边界：B 的交接摘要记载一次 VGG16/ImageNet 在模型加载前的失败尝试，随后安全
恢复；该次不产生结果行。收到的轻量包不包含参考 NPY 和 checkpoint 本体，D 的哈希
比对使用此前验收并留在本机 Git 忽略目录中的 B 基础包。不能由这些数据推断
KernelSHAP、RISE、IG、Grad-CAM 或 LIME 的排序，也不能据此声称 ANOVA、相关性、
三维 Pareto 或 RQ 最终结论已完成。

复核命令：

```powershell
.\.venv\Scripts\python.exe -m analysis.validate_occlusion_formal_delivery 'D:\微信文件\xwechat_files\wxid_k52y4rqkvr8h12_9d8f\msg\file\2026-10\occlusion_stability_eval_light.zip'
```
