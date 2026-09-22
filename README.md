# XAI01-04:归因方法模式发现——什么条件下什么方法最优

《机器学习综合实践》课程项目(5 人小组)。

## 快速入口

- 任务书要求:见课程项目池 `project-pool.pdf` §4.1.4(XAI01-04)
- **小组工作指导(角色分工 / 技术方案 / 实验矩阵 / 里程碑计划 / 协作规范)——全组必读:
  [XAI01-04_小组工作指导.md](XAI01-04_小组工作指导.md)**
- **2026-09-16 已安排的成员任务包、实验分配、复核关系与协作图:
  [docs/TEAM_AND_ROADMAP.md](docs/TEAM_AND_ROADMAP.md)**

## 目录结构

见指导文件 §7.1。`data/` 与 `results/` 不提交到 git,原始数据与中间结果通过网盘共享。

## 环境与运行

需要 Python 3.10+。在项目根目录创建环境并安装依赖：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

运行 C 的 W2 RISE 管线：

```bash
# 离线冒烟测试（随机权重，仅验证管线）
python experiments/run_unit.py --config configs/rise_smoke_imagenet.yaml

# 可提交的 ImageNet debug 结果
python experiments/run_unit.py --config configs/rise_resnet50_imagenet.yaml

pytest -q
```

实现、输出位置、6 个单元配置及 VOC checkpoint 对接要求见
[docs/W2_C_HANDOFF.md](docs/W2_C_HANDOFF.md)。

成员 A 的 IG / Grad-CAM 实现、12 个正式单元配置和方法约定见
[docs/A_IG_GRADCAM_HANDOFF.md](docs/A_IG_GRADCAM_HANDOFF.md)；2026-09-20 完成的 12 单元正式结果、验收记录、汇总图与结论见
[docs/A_IG_GRADCAM_RESULTS.md](docs/A_IG_GRADCAM_RESULTS.md)。这里的“完成”仅指忠实性和效率；稳定性协议尚未冻结，因此按任务书要求的三指标口径尚无单元完全闭环。现有正式结果使用 raw MoRF、21 点删除曲线、预热计时和 float32 归因图：

```bash
pytest -q tests/test_gradient_methods.py tests/test_gradient_runner.py
```
