# W2 成员 D 交接：KernelSHAP + 统计分析骨架

更新时间：2026-09-16。

## 已完成并可独立验收

- 新建开发分支 `feat/kernelshap-analysis`。
- KernelSHAP 适配器：单图接口、规则网格特征、黑色基线、固定随机种子、二维归一化输出。
- 1 个离线 smoke 配置和 6 个主体单元配置（3 模型 × ImageNet/VOC）。
- 结果 schema 校验、聚合、三因素 ANOVA 与效应量、假设诊断、Friedman 对照、Pareto、
  Pearson/Spearman 分析骨架。
- 自动化测试使用轻量假后端验证 KernelSHAP 接口，不把假后端结果作为实验结果。

## 当前依赖与阻塞

1. C 的 `feat/rise-pipeline` 目前尚未合并到 `main`。其中已有数据版本冻结、模型加载、
   `run_unit.py` 和结果 schema；合并后需要把 runner 从只接受 `rise` 扩展为方法注册表，
   再接入 `KernelSHAP`。
2. 当前本机未安装 Captum，因此只能验证适配器逻辑和接口，不能声称真实 KernelSHAP 已运行。
   独立依赖记录在 `experiments/requirements-kernelshap.txt`，合并共享环境时需并入根依赖文件。
3. ImageNet 三单元可在 Captum 和 pipeline 就绪后开始 smoke/debug。
4. VOC 三单元必须等待 B 提供与 VOC 20 类顺序一致的三个 checkpoint；不能把 ImageNet
   1000 类模型的输出直接套用 VOC 0–19 标签。
5. 正式 ANOVA、Pareto、相关性和 RQ1–RQ3 结论必须等待真实全组结果，当前只有脚本骨架。

## 接入后的第一轮验证

```bash
pytest -q
python experiments/run_unit.py --config configs/kernelshap_smoke_imagenet.yaml
python -m analysis.validate_results --per-image results/smoke/kernelshap_per_image.csv --units results/smoke/kernelshap_units.csv
```

smoke 配置使用随机模型、1 张图片和 8 次采样，只证明管线能运行，不可写入报告的性能结论。
正式 debug 配置当前使用 `n_samples=256`、`feature_grid_size=14`；跑批前需由 A 协助做成本曲线，
再锁定正式采样数和是否需要限制 KernelSHAP 的评估子集。
