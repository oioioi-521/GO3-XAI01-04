"""Render the repeated-measures result, two supplemental figures, and full tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .plot_dcu_statistics import COLORS, DATA_LABEL, LABEL, MODEL_LABEL
from .report_dcu_statistics import table
from .validate_dcu_team_handoff import DATASETS, METHODS, MODELS
from .dcu_repeated_measures import QUALITY


def display_p(p):
    if p is None:
        return "NA"
    return "<1e-300" if p < 1e-300 else f"{p:.4g}"


def figures(data, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    titles = ["Raw MoRF AUC (lower better)", "Input-noise Spearman (higher better)"]
    primary = {(r["dataset"], r["metric"], r["effect"]): r for r in data["omnibus"] if r["cohort"] == "common_correct"}
    means = {(r["dataset"], r["metric"], r["model"], r["method"]): r for r in data["cell_means"] if r["cohort"] == "common_correct"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for i, dataset in enumerate(DATASETS):
        for j, metric in enumerate(QUALITY):
            ax = axes[i, j]
            for m, method in enumerate(METHODS):
                rows = [means[dataset, metric, model, method] for model in MODELS]
                values = np.array([r["mean"] for r in rows])
                limits = np.array([r["mean_ci95"] for r in rows]).T
                x = np.arange(3) + (m - 2.5) * .018
                ax.plot(x, values, marker="o", markersize=4, color=COLORS[m], linewidth=1.5)
                ax.vlines(x, limits[0], limits[1], color=COLORS[m], linewidth=1.1, alpha=.8)
            r = primary[dataset, metric, "method_model"]
            ax.set_title(f"{DATA_LABEL[dataset]} | {titles[j]}\nn={r['n']}; interaction GG-Holm p={display_p(r['p_holm'])}", fontsize=11)
            ax.set_xticks(range(3), [MODEL_LABEL[m] for m in MODELS])
            ax.set_xlim(-.15, 2.15)
            ax.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], color=COLORS[i], marker="o", label=LABEL[m]) for i, m in enumerate(METHODS)]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(.5, .055), frameon=False)
    fig.suptitle("Paired method-by-model profiles | same common-correct images", fontsize=15)
    fig.text(.5, .025, "Bars: marginal class-stratified image-bootstrap 95% CIs, conditional on class counts. IID-image CI sensitivity is in JSON.\nNot SD or simultaneous intervals; lines are descriptive, separate dataset scales.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .95))
    fig.savefig(folder / "method_model_profiles.png", dpi=170, metadata={"Software": "D repeated-measures statistics"})
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.8))
    for i, dataset in enumerate(DATASETS):
        for j, metric in enumerate(QUALITY):
            ax = axes[i, j]
            rows = {(r["model"], r["method"]): r for r in data["centered_interactions"]
                    if (r["cohort"], r["dataset"], r["metric"]) == ("common_correct", dataset, metric)}
            grid = np.array([[rows[model, method]["centered_cell_effect"] for method in METHODS] for model in MODELS])
            scale = max(float(np.max(np.abs(grid))), 1e-12)
            im = ax.imshow(grid, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
            ax.set_xticks(range(6), [LABEL[m] for m in METHODS], rotation=25, ha="right")
            ax.set_yticks(range(3), [MODEL_LABEL[m] for m in MODELS])
            ax.set_title(f"{DATA_LABEL[dataset]} | {titles[j]}", fontsize=11)
            for k in range(3):
                for m in range(6):
                    ax.text(m, k, f"{grid[k,m]:+.3f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(grid[k,m]) > .6 * scale else "black")
            fig.colorbar(im, ax=ax, fraction=.045, pad=.025)
    fig.suptitle("Double-centered method-by-model interaction | raw metric units", fontsize=14)
    fig.text(.5, .025, "Cell mean minus method mean minus model mean plus grand mean. Separate panel color scales; not a method ranking or a significance map.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(folder / "method_model_interaction.png", dpi=170, metadata={"Software": "D repeated-measures statistics"})
    plt.close(fig)


def report(data):
    selected = [r for r in data["omnibus"] if r["cohort"] == "common_correct"]
    effects = {"method": "方法", "model": "模型", "method_model": "方法×模型"}
    rows = []
    for r in selected:
        rows.append([r["dataset"], "AUC" if r["metric"] == QUALITY[0] else "稳定性", effects[r["effect"]], r["n"],
                     f"{r['F']:.4f}" if r["F"] is not None else "NA", f"{r['epsilon_gg']:.5f}" if r["epsilon_gg"] is not None else "NA",
                     f"{r['df_effect_gg']:.3f}, {r['df_error_gg']:.3f}" if r["df_effect_gg"] is not None else "NA",
                     display_p(r["p_holm"]), f"{r['partial_eta_squared']:.5f}" if r["partial_eta_squared"] is not None else "NA"])
    result_table = table(["数据集", "指标", "效应", "n", "F", "GG ε", "GG df1, df2", "GG-Holm p", "偏η²"], rows)
    interaction = [r for r in selected if r["effect"] == "method_model"]
    descriptions = "\n".join(f"- {r['dataset']} / {'AUC' if r['metric'] == QUALITY[0] else '稳定性'}：F={r['F']:.4f}，GG ε={r['epsilon_gg']:.5f}，Holm p={display_p(r['p_holm'])}，偏η²={r['partial_eta_squared']:.5f}。" for r in interaction)
    sensitivity = "\n".join(f"- {r['dataset']} / {'AUC' if r['metric'] == QUALITY[0] else '稳定性'}：n={r['n']}，F={r['F']:.4f}，GG-Holm p={display_p(r['p_holm'])}，偏η²={r['partial_eta_squared']:.5f}。" for r in data["omnibus"] if r["cohort"] == "all" and r["effect"] == "method_model")
    example_rows = [r for r in data["direct_did"] if r["cohort"] == "common_correct" and r["method_a"] == "ig" and r["method_b"] == "kernelshap" and r["model_u"] == "resnet50" and r["model_v"] == "densenet121"]
    examples = table(["数据集", "指标", "IG相对KS优势的差中差", "类分层95%均值CI", "IID原图95%均值CI", "DID秩 Holm p"], [
        [r["dataset"], "AUC" if r["metric"] == QUALITY[0] else "稳定性", f"{r['mean_advantage_a']:+.6f}",
         f"[{r['mean_did_ci95'][0]:+.6f}, {r['mean_did_ci95'][1]:+.6f}]",
         f"[{r['iid_image_ci95'][0]:+.6f}, {r['iid_image_ci95'][1]:+.6f}]", display_p(r["p_holm"])] for r in example_rows])
    text = f"""# D：方法/模型重复测量主效应与交互结果（2026-10-28）

本次交付对应 #21 的按数据集分开的阶段分析。使用已完成的六方法×三模型×两数据集冻结结果，
没有重跑 GPU、修改原 CSV 或替代全组最终验收。**数据集因子和完整混合三因素 ANOVA 仍未完成。**

## 1. 设计和检验

主分析为三模型共同命中组：ImageNet n=332、VOC n=353；全 eval 各 n=460 为敏感性。
每张原图在三模型六方法上形成完整重复测量块，五次噪声先按原协议汇总。
不同模型各自的命中组未拼入本设计，旧逐单元主分析继续保留。

对每个数据集和质量指标分别实施双图像内因子 ANOVA。
用正交 Helmert 对比投影到方法（5维）、模型（2维）、交互（10维）空间；
各效应用其自身的“原图×效应”误差，不用所有行的 pooled OLS 残差。
对比矩阵 Z 的效应平方和为 n‖平均Z‖²，误差平方和为 Σ‖Zᵢ−平均Z‖²，
F=SS效应/df效应 ÷ SS误差/df误差，原自由度为 d、(n−1)d。
在对应对比协方差 S 中计算 ε=(tr S)²/[d tr(S²)]，始终用 GG 修正后的两个自由度计算 p，
再在每 cohort 的 12 项质量全局族（两数据集×两指标×三效应）做 Holm 校正。
数值尾概率下限为1e-300，Holm使用该保守数值，避免将数值下溢导出为精确零。
不以 Mauchly 的显著/不显著来选择是否校正；记录协方差秩，不凭秩诊断悄悄删对比方向。

偏η²=SS效应/(SS效应+SS原图×效应)，不是广义η²或因果解释比例。
软件中的零误差/不可估计状态不伪造无穷 F 或 p=0。F/GG 是有分布假设的近似重复测量检验，
校正球形性自由度不代表已消除非正态、类别不均衡或命中组选择偏差。

## 2. 共同命中组正式结果

{result_table}

交互结果：

{descriptions}

全局交互检验问的是方法相对差异是否随这三套冻结模型/权重改变，不能从交互显著推出某方法
对所有架构都更优，也不能仅凭图上线条交叉或不同模型的 p 值不同证明交互。
主效应平均了另一因子的固定水平；存在交互时需结合逐单元配对和具体差中差解释。

## 3. 效应、不确定性与直接差中差

输出含 72 个共同命中组单元均值和边际 95% CI（含全 eval 共144行）、模型/方法边际均值、
双中心化交互单元值及其区间。采用类别分层的原图 bootstrap，{data['protocol']['bootstrap_resamples']} 次，
同一抽样索引用于全部模型/方法/指标。区间条件于冻结类别比例和样本口径，不是同时区间；
单图类别在 bootstrap 中没有类内变化，原始类别不均衡及其不确定性限制继续保留。

本次明确发现：ImageNet 共同命中组194类中111类只有一张，全 eval 259类中144类只有一张；
VOC 共同命中组18类中1类只有一张、全 eval 19类中没有单图类。
因此同时另做同样次数的普通原图配对 bootstrap，允许类别比例变化，输出 `iid_image_ci95`。
两种区间都保留；类分层区间不能外推为新类别/总体比例变化或训练不确定性的区间。

差中差定义为：[a相对b的优势]在模型u − 同一优势在模型v。
AUC 优势取 −(AUCₐ−AUCᵦ)，稳定性取 Spearmanₐ−Spearmanᵦ。
正值表示 a 相对 b 的优势在 u 更大，不是 a 在所有条件中获胜。
每 cohort 有180个直接差中差；对原图的差中差做配对秩检验及符号敏感性，分别 Holm。
均值 CI 和秩 p 对应不同统计量，不将秩显著改写为均值显著。

以下示例固定 u=ResNet50、v=DenseNet121、a=IG、b=KernelSHAP；全部结果见附表：

{examples}

其中 VOC/AUC 这一具体差中差的类分层区间不跨零，但普通 IID 原图区间跨零，
且直接配对秩 Holm p 不显著。应同时保留这项不确定性敏感性，不能从全局交互显著
推导所有方法对/模型对都显著，也不能仅凭其中一个区间宣布 IG/KernelSHAP 的普遍优势。

## 4. 全 eval 敏感性

{sensitivity}

完整两 cohort 主效应表、单元/边际均值、交互值和直接差中差均在 JSON 中；
全 eval 不取代共同命中组，主分析也不取代原报告的各模型命中组结论。

## 5. 图与合稿用语

![同图配对的方法/模型曲线](../analysis/dcu_20261028/method_model_profiles.png)

图1：共同命中组的六方法均值曲线；误差线为原图分层 bootstrap 的边际95% CI，不是SD。
两数据集分面且量纲/输出含义分别解释；线条为描述，图中交互 p 为效应专属误差+GG+Holm。

![双中心化交互](../analysis/dcu_20261028/method_model_interaction.png)

图2：单元均值减去模型均值与方法均值再加总体均值，保留原始指标单位；
每个面板独立色标，非显著性地图、方法排名或可跨数据集相减的质量量表。

可合稿：在冻结预算、共同命中图像及三套固定模型下，直接重复测量分析补充了方法/模型主效应
和方法×模型交互证据，需结合具体差中差及原逐单元结果解释条件优势。
原四图、IG/KernelSHAP 的均值/秩例外、Grad-CAM 退化、LIME预算及共享计时边界保持不变。

## 6. 未完成项与复现

本结果只包含按数据集分开的双因子重复测量分析，没有检验数据集主效应、数据集×方法/
模型或三重交互，不能称为完整三因素已完成或直接关闭 #21。
下一项需明确 ImageNet/VOC 作为图像间因子的尺度/语义、混合设计误差和异方差处理，
补充完整结果或经教师确认的替代处理；不需要为此重跑归因。
E 的最终 PPT/Demo 图注审核与全组冻结仍是独立事项。

```powershell
python -m analysis.dcu_repeated_measures 'dcu_team_handoff_20261014.zip'
python -m analysis.report_dcu_repeated_measures
python -m pytest -q tests/test_dcu_repeated_measures.py
```

输入 SHA：`{data['receipt']['zip_sha256']}`；主种子 {data['protocol']['seed']}；
来源/环境/代码 SHA、误差平方和、自由度、GG epsilon、全部 p 与效应均在
[repeated_measures.json](../analysis/dcu_20261028/repeated_measures.json)。
完整差中差：[附表](../analysis/dcu_20261028/direct_did_tables.md)。

实现经过 Stata 官方公开双重复因子示例的平方和、误差项、GG epsilon 与修正 p 值对照，
并检查两水平配对t²等价、加性无交互、水平置换/图像截距不变性、缺失/重复键及退化情况。
合成/参考数据测试不是本项目正式实验，不能代替 C 的独立分析复现。
另提供 [固定种子合成诊断](../analysis/dcu_20261028/synthetic_diagnostics.json)：
球形正态、异方差正态、中心化指数噪声的零交互及注入交互各1000次、每次n=60。
这只检查有限情景下的软件行为，不证明真实数据满足 ANOVA 假设或精确控制5%错误率。

方法依据：[Stata 官方重复测量示例](https://www.stata.com/support/faqs/statistics/repeated-measures-anova/)、
[Pingouin 官方双因素说明](https://pingouin-stats.org/generated/pingouin.rm_anova.html)。
[statsmodels 官方说明](https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html)
明确其 AnovaRM 未实现球形性校正；本结果不把未经校正的 AnovaRM 输出冒充 GG 检验。

## 7. 验证与交付

2026-10-28：新增重复测量测试与既有统计/交接/案例测试合计 **41 passed**；
两幅新增图已渲染目视检查。旧 `analysis/dcu_20261015/` 的 JSON、配对表和四图保留，
所有新结果单独放在 `analysis/dcu_20261028/`。源码/派生摘要可通过 GitHub 交付，
原始 ZIP、原图、NPY、权重和 `outputs/` 不提交。
这次复算、软件诊断与目视检查不代替 C 的独立复现、A 的最终报告冻结或 E 的最终展示验收。
另从原CSV用独立边际均值分解核对24项效应/误差平方和、144个旧摘要均值及360个直接差中差，
均一致；[验证回执](../analysis/dcu_20261028/validation_receipt.json)记录范围和最终图像哈希。
"""
    lines = ["# 直接差中差完整表", "", "正值：a 相对 b 的优势在模型 u 大于模型 v；AUC使用负向收益，稳定性使用正向收益。",
             "CI为边际均值区间；同时保留固定类别比例和普通IID原图两种bootstrap。p为差中差的直接配对秩检验，不是均值检验。每cohort各180项，分别Holm。", ""]
    for cohort in ("common_correct", "all"):
        lines += ["## " + cohort, "", table(["数据集", "指标", "方法a / b", "模型u / v", "n", "均值差中差", "类分层95% CI", "IID原图95% CI", "秩二列", "秩Holm p", "符号Holm p"], [
            [r["dataset"], "AUC" if r["metric"] == QUALITY[0] else "稳定性", r["method_a"] + " / " + r["method_b"],
             r["model_u"] + " / " + r["model_v"], r["n"], f"{r['mean_advantage_a']:+.6f}",
             f"[{r['mean_did_ci95'][0]:+.6f}, {r['mean_did_ci95'][1]:+.6f}]",
             f"[{r['iid_image_ci95'][0]:+.6f}, {r['iid_image_ci95'][1]:+.6f}]", f"{r['rank_biserial_a']:+.4f}",
             display_p(r["p_holm"]), display_p(r["sign_p_holm"])] for r in data["direct_did"] if r["cohort"] == cohort]), ""]
    return text, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("analysis/dcu_20261028/repeated_measures.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/D_M5_REPEATED_MEASURES_20261028.md"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    figures(data, args.input.parent)
    text, appendix = report(data)
    args.report.write_text(text, encoding="utf-8")
    args.input.with_name("direct_did_tables.md").write_text(appendix, encoding="utf-8")
    print("Rendered " + str(args.report))


if __name__ == "__main__":
    main()
