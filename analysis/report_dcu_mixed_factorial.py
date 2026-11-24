"""Render mixed-design tables and an observed-task difference figure, CPU only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .plot_dcu_statistics import LABEL, MODEL_LABEL
from .report_dcu_statistics import table
from .validate_dcu_team_handoff import METHODS, MODELS, METRICS

EFFECTS = {"dataset": "数据集", "method": "方法", "dataset_method": "数据集×方法",
           "model": "模型", "dataset_model": "数据集×模型", "method_model": "方法×模型",
           "dataset_method_model": "数据集×方法×模型"}
NAMES = dict(zip(METRICS, ("raw AUC", "噪声 Spearman", "log1p(ms) 探索性")))


def number(x):
    return "NA" if x is None else f"{x:.6g}"


def omnibus_table(rows):
    return table(["指标", "效应", "F", "GG ε", "GG df1, df2", "名义 GG-Holm p", "偏η²", "WTS", "异方差 bootstrap-Holm p", "对比 RMS"], [
        [NAMES[r["metric"]], EFFECTS[r["effect"]], number(r["nominal_anova"]["F"]),
         number(r["nominal_anova"]["epsilon_gg"]),
         f"{number(r['nominal_anova']['df1_gg'])}, {number(r['nominal_anova']['df2_gg'])}",
         number(r["nominal_anova"]["p_holm"]), number(r["nominal_anova"]["partial_eta_squared"]),
         number(r["heteroscedastic_wts"]["WTS"]), number(r["heteroscedastic_wts"]["p_holm"]),
         number(r["contrast_rms_per_cell"])] for r in rows])


def figures(data, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
    for j, metric in enumerate(METRICS[:2]):
        rows = {(r["model"], r["method"]): r for r in data["observed_dataset_differences"]
                if r["cohort"] == "common_correct" and r["metric"] == metric}
        grid = np.array([[rows[model, method]["difference_imagenet_minus_voc"] for method in METHODS] for model in MODELS])
        limit = max(float(np.abs(grid).max()), 1e-12)
        ax = axes[j]
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(6), [LABEL[m] for m in METHODS])
        ax.set_yticks(range(3), [MODEL_LABEL[m] for m in MODELS])
        for k in range(3):
            for m in range(6):
                ax.text(m, k, f"{grid[k,m]:+.4f}", ha="center", va="center",
                        color="white" if abs(grid[k,m]) > .58*limit else "black")
        ax.set_title(["Raw target-probability MoRF AUC", "Input-noise Spearman"][j])
        fig.colorbar(im, ax=ax, fraction=.025, pad=.02, label="ImageNet mean - VOC mean")
    fig.suptitle("Observed task-score differences | common-correct images (332 / 353)", fontsize=14)
    fig.text(.5, .025, "Descriptive cell means; separate panel scales. Not a significance map or a common-quality ranking.\n"
             "Dataset also changes class space, images and softmax/sigmoid scoring. Both marginal 95% CI methods are in the tables.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .95))
    fig.savefig(folder / "observed_task_differences.png", dpi=170)
    plt.close(fig)


def report(data):
    primary = [r for r in data["tests"] if r["cohort"] == "common_correct"]
    return """# D：数据集×方法×模型混合设计正式统计

2026-10-29；基于原冻结 DCU 观测，无新 GPU 归因实验。七项因子效应现均有实际计算结果，
与 10/28 按数据集分开的重复测量阶段相互补充。**计算覆盖不等于 A/教师接受该统计口径，
也不等于 C 独立复现、E 最终 PPT/Demo 验收或全项目完成。**

## 可用于合稿的结论

1. 共同命中组 ImageNet332 / VOC353 的 AUC 七项效应均通过名义 GG-ANOVA 和异方差 WTS 的各自 Holm 检验。
   稳定性的数据集主效应未显著（名义 p=0.121215；WTS bootstrap p=0.1148）；其余六项显著。
   因而不能写“所有主效应均显著”。全 eval 各460张的显著/未显著模式一致，稳定性数据集主效应仍未显著（0.084957 / 0.0842）。
2. 数据集×方法×模型交互在两质量指标及两 cohort 均显著。它说明冻结评估任务之间的
   方法/模型对比结构有差异，**不能由总体交互推导任意局部方法对都显著**；继续引用
   [10/28 的直接差中差和区间](D_M5_REPEATED_MEASURES_20261028.md)。
3. 跨数据集检验只描述原始评分的任务差异：ImageNet softmax / VOC sigmoid、类别空间、
   图片总体与预处理同时变化，raw AUC 不是已校准的共同潜在解释质量尺度。
   不据数据集效应宣布哪个数据集“解释质量更好”或因果效应。方法选择仍使用同数据集内的质量、时间及例外证据。

## 主分析：共同命中组质量族

每个检验均以原图为独立单位；方法6与模型3为图像内因子，数据集2为图像间因子。
subject=(dataset,image_id)，图像嵌套于数据集。两数据集不配对，五次噪声不当成五倍独立 n。
名义 ANOVA 与 WTS 分别对14项质量假设（2指标×7效应）做 Holm，不能把两种检验拼成同一个 p。

""" + omnibus_table([r for r in primary if r["metric"] in METRICS[:2]]) + """

WTS 使用9,999次 bootstrap，p=(超越次数+1)/10,000。38/42项原始 p 在0.0001网格下界，
质量 Holm 下界0.0014，时间下界0.0007；这些数不是更精确的极小 p，也不是 p=0。
JSON 保留每项超越数、Monte Carlo 标准误、协方差秩与条件数；42项正式协方差均满秩。
偏η²是名义混合 ANOVA 的 SS_H/(SS_H+SS_error)，不是广义η²或因果解释方差；
RMS为正交对比均值向量的范数/√18，使用原指标单位（时间用log1p单位），不是配对秩效应量。

## 探索性记录时间：共同命中组

逐图先计算 log1p(recorded wrapper ms)，再建模，绝非 log1p(平均耗时)。单独七项 Holm 族。
全部七项显著不消除两轮共享 DCU、负载、实现/预算与 LIME 轮次混杂；不写硬件/算法因果加速比。

""" + omnibus_table([r for r in primary if r["metric"] == METRICS[2]]) + """

## 方法、区间和可解释边界

- 使用正交对比投影到 grand(1)、method(5)、model(2)、method×model(10) 维子空间。
  组均值 μ_g；图像内主效应对比 β=(μ_IN+μ_VOC)/2，两数据集等权；数据集及其交互 β=μ_IN−μ_VOC。
  这给出不等 n 下的 equal-cell Type III estimand，不按332/353自动加权，也不逐数据集标准化。
- 名义 ANOVA：v=Σw_g²/n_g，SS_H=||β||²/v，SS_error=Σ||z_ig−μ_g||²，
  F=SS_H(N−2)/SS_error，df=(d,(N−2)d)；每个效应用其原图×效应误差层。
  用池化残差协方差 S 的 ε=tr(S)²/[d·tr(S²)] 做 GG df 修正。
  这是依赖同质协方差/球形性近似的名义检验，不掩盖这些限制。
- 异方差敏感性 WTS=β′[Σw_g² S_g/n_g]⁻¹β；每组单独估计协方差。
  零假设 Gaussian parametric bootstrap 在每次抽样重新估计均值与协方差，
  以正态均值和独立 Wishart 样本散布的等价抽样加速。bootstrap 样本为正态不代表宣称真实观测正态；
  理论有效性是渐近的，仍需独立原图、有限四阶矩和满秩组对比协方差。
  无奇异伪逆丢维或任意 jitter；不可估计则显式 NA。
- 全部108个跨任务单元均值差附两种10,000次原图 bootstrap 边际95%CI：
  各数据集独立抽样，图像内模型/方法保持配对；类别分层与 IID 原图两种口径均保留。
  CI不是同时区间，不因全局 Holm 自动获得家族覆盖。主组 ImageNet194类中111个单例，
  VOC18类中1个单例；全 eval 为259/144和19/0。类分层 CI 条件于固定类别计数，
  不能泛化到未采样类别/随机训练权重。
- 保留 B 的 ImageNet/DenseNet121 IG/KernelSHAP“均值CI跨0、秩偏向IG”例外；
  保留 VOC/AUC IG–KS、ResNet50–DenseNet121 的直接 DID +0.014942：
  类分层CI[+0.003176,+0.026916]，IID CI[−0.006141,+0.035535]，秩Holm p=0.8374。
  不用新全局显著性抹去原有局部证据差异。

## 数值验证与复现

1. StataCorp/Winer表7.13公开混合设计的七项SS、对应误差、GG ε及可用p对照通过；
   不等n另与分组OLS等权对比复算，标量WTS与Welch t²一致，另测尺度不变和奇异处理。
2. 正式数据独立复核程序从冻结长表构造另一QR正交基，用组虚拟变量OLS复算42项
   SS、误差、F、GG ε、偏η²和观察WTS，核对108个单元均值差及校正族。
   [独立复核记录](../analysis/dcu_20261029/validation_receipt.json)。
3. 固定seed合成诊断：不等协方差正态零假设18/300=.0600，中心化指数零假设20/300=.0667，
   二项95%CI分别[.03594,.09317]和[.04119,.10109]；注入效应300/300。
   这只是小规模软件/校准诊断，不证明真实数据恰有5%检验水平；合成数据不混入课程观测。
4. [最终版本47项相关测试回执](../analysis/dcu_20261029/test_receipt.json)：混合设计6项、既有统计/交接41项；不是全GPU测试套件。

```powershell
python -m analysis.dcu_mixed_factorial results/dcu_delivery_20261027/dcu_team_handoff_20261014.zip
python -m analysis.calibrate_dcu_mixed_factorial
python -m analysis.verify_dcu_mixed_factorial results/dcu_delivery_20261027/dcu_team_handoff_20261014.zip
python -m analysis.report_dcu_mixed_factorial
```

需 NumPy/Pandas/SciPy/PyYAML/Matplotlib；版本与源码SHA见JSON。冻结输入SHA256仍为
`14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc`。
未执行归档代码、未改原始CSV/NPY/trace，10/15及10/28结果按原日期保留。

## 文件与交付

- [42项完整结果和108个单元差/CI](../analysis/dcu_20261029/mixed_factorial.json)
- [全eval敏感性及完整单元差表](../analysis/dcu_20261029/mixed_factorial_tables.md)
- [观察任务评分差图](../analysis/dcu_20261029/observed_task_differences.png)：描述均值差，不是显著性热图或跨数据集质量排名。
- [合成诊断](../analysis/dcu_20261029/mixed_synthetic_diagnostics.json)

A/E 可据此补实验4的实际结果与限制；C 的独立复现、教师/组长对口径的接受和最终合稿仍待确认。
不代签、不关闭 Issue，不要求重跑 GPU。新派生表/图/代码可走 GitHub，原图/NPY/大包继续外部共享。

## 方法依据

- [StataCorp repeated-measures ANOVA reference](https://www.stata.com/support/faqs/statistics/repeated-measures-anova/)：
  一图像间、两图像内因子设计，嵌套subject误差层及公开数值对照。
- [Friedrich、Konietschke、Pauly (2019), MANOVA.RM](https://journal.r-project.org/articles/RJ-2019-051/)：
  异方差WTS与零假设Gaussian parametric bootstrap。这里为Python重实现，未运行R包，也不声称与R逐位一致。
- [MANOVA.RM官方说明](https://stat.ethz.ch/CRAN/web/packages/MANOVA.RM/vignettes/Introduction_to_MANOVA.RM.html)：
  重复测量/混合设计、bootstrap和协方差奇异限制。
- [SciPy Wishart](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wishart.html)：
  正态样本散布的分布抽样实现。
"""


def appendix(data):
    chunks = ["# 完整混合因子表与观察任务差（2026-10-29）", "## 全 eval 各460张敏感性", omnibus_table([
        r for r in data["tests"] if r["cohort"] == "all"]),
        "## ImageNet − VOC 单元均值差与边际95% CI\n\nlogtime为逐图log1p(ms)后平均；不反变换成平均ms。两种CI均不是同时区间。"]
    for cohort in ("common_correct", "all"):
        for metric in METRICS:
            rows = [r for r in data["observed_dataset_differences"] if r["cohort"] == cohort and r["metric"] == metric]
            chunks.extend([f"### {cohort} / {NAMES[metric]}", table(["模型", "方法", "IN均值", "VOC均值", "差", "类分层CI", "IID原图CI"], [
                [r["model"], LABEL[r["method"]], number(r["mean_imagenet"]), number(r["mean_voc"]),
                 number(r["difference_imagenet_minus_voc"]), str([round(x,6) for x in r["conditional_ci95"]]),
                 str([round(x,6) for x in r["iid_image_ci95"]])] for r in rows])])
    return "\n\n".join(chunks) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("analysis/dcu_20261029/mixed_factorial.json"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    figures(data, args.input.parent)
    Path("docs/D_M5_MIXED_FACTORIAL_20261029.md").write_text(report(data), encoding="utf-8", newline="\n")
    (args.input.parent / "mixed_factorial_tables.md").write_text(appendix(data), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
