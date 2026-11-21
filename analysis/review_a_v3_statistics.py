"""D's fixed A-v3 report audit; no bootstrap/GPU rerun and no teammate edits."""
from __future__ import annotations

import argparse
import copy
import csv
from decimal import Decimal
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .validate_dcu_team_handoff import DATASETS, METHODS, METRICS, MODELS, EXPECTED_SHA

A_HEAD = "7e2821e2050a9cb3171f4a9587d210e670b2dba2"
LABELS = dict(ig="IG",gradcam="Grad-CAM",occlusion="Occlusion",rise="RISE",kernelshap="KernelSHAP",lime="LIME")
EFFECTS = {"数据集":"dataset","方法":"method","模型":"model","数据集×方法":"dataset_method",
           "数据集×模型":"dataset_model","方法×模型":"method_model","数据集×方法×模型":"dataset_method_model"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rounded(text, value):
    """Enforce the printed precision, including tiny p's; no blanket atol."""
    text=text.strip()
    x=Decimal(str(value))
    if text.startswith("<"):
        assert x < Decimal(text[1:]), (text,value)
    else:
        q=Decimal(text)
        tolerance=Decimal("0.501")*(Decimal(10)**q.as_tuple().exponent)
        assert abs(q-x) <= tolerance, (text,value,str(tolerance))


def md_table(text, header):
    lines=text.splitlines()
    i=next(i for i,line in enumerate(lines) if line.startswith(header))
    rows=[]
    for line in lines[i+2:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def numeric_tables(text, stage, mixed):
    rows=md_table(text,"| 数据集 | 指标 | F |")
    assert len(rows)==4
    for r in rows:
        dataset={"ImageNet":"imagenet","VOC":"voc"}[r[0]]
        metric=METRICS[0] if r[1]=="raw AUC" else METRICS[1]
        a=next(a for a in stage["omnibus"] if a["cohort"]=="common_correct" and a["dataset"]==dataset
               and a["metric"]==metric and a["effect"]=="method_model")
        for s,key in zip(r[2:],("F","p_holm","partial_eta_squared")):
            rounded(s,a[key])
    rows=md_table(text,"| 效应 | AUC GG-Holm p |")
    assert len(rows)==7 and {r[0] for r in rows}==set(EFFECTS)
    for r in rows:
        for j,metric in enumerate(METRICS[:2]):
            a=next(a for a in mixed["tests"] if a["cohort"]=="common_correct" and a["metric"]==metric and a["effect"]==EFFECTS[r[0]])
            nominal,robust=a["nominal_anova"],a["heteroscedastic_wts"]
            for s,x in zip(r[1+3*j:4+3*j],(nominal["p_holm"],nominal["partial_eta_squared"],robust["p_holm"])):
                rounded(s,x)
    return dict(stage_table_rows=4,stage_numeric_cells=12,mixed_table_rows=7,mixed_numeric_cells=42)


def front(rows, use_time):
    points=[(r["faithfulness"],-r["stability"],r["recorded_ms"])[:3 if use_time else 2] for r in rows]
    return [not any(all(a<=b for a,b in zip(other,p)) and any(a<b for a,b in zip(other,p)) for other in points) for p in points]


def pareto_tables(text, data):
    rows=md_table(text,"| 数据集 | 模型 | 质量双目标前沿 |")
    assert len(rows)==6
    checks=[]
    for i,dataset in enumerate(DATASETS):
        for j,model in enumerate(MODELS):
            r=rows[3*i+j]
            assert r[:2]==[{"imagenet":"ImageNet","voc":"VOC"}[dataset],{"resnet50":"ResNet50","densenet121":"DenseNet121","vgg16":"VGG16"}[model]]
            part=sorted([x for x in data["pareto"] if x["cohort"]=="model_correct" and x["dataset"]==dataset and x["model"]==model],key=lambda x:METHODS.index(x["method"]))
            assert len(part)==6 and {x["method"] for x in part}==set(METHODS)
            q,t=front(part,False),front(part,True)
            assert q==[x["quality_front"] for x in part]
            assert t==[x["recorded_time_3d_front"] for x in part]
            for cell,flags in zip(r[2:],(q,t)):
                assert set(cell.split("、"))=={LABELS[x["method"]] for x,f in zip(part,flags) if f}
            logged=copy.deepcopy(part)
            for x in logged:
                x["recorded_ms"]=float(np.log10(x["recorded_ms"]))
            assert front(logged,True)==t
            checks.append(dict(dataset=dataset,model=model,n=part[0]["n"],quality=[x["method"] for x,f in zip(part,q) if f],
                               recorded_three_objective=[x["method"] for x,f in zip(part,t) if f]))
    return checks


def audit(snapshot,handoff,project):
    report=snapshot/"docs/FINAL_REPORT_DRAFT.md"
    text=report.read_text(encoding="utf-8")
    stage=json.loads((project/"analysis/dcu_20261028/repeated_measures.json").read_text(encoding="utf-8"))
    mixed=json.loads((project/"analysis/dcu_20261029/mixed_factorial.json").read_text(encoding="utf-8"))
    stats=json.loads((project/"analysis/dcu_20261015/statistics.json").read_text(encoding="utf-8"))
    counts=numeric_tables(text,stage,mixed)
    panels=pareto_tables(text,stats)
    did=next(r for r in stage["direct_did"] if (r["cohort"],r["dataset"],r["metric"],r["method_a"],r["method_b"],r["model_u"],r["model_v"])==
             ("common_correct","voc",METRICS[0],"ig","kernelshap","resnet50","densenet121"))
    did_line=next(line for line in text.splitlines() if "DID=+" in line).replace("−","-")
    tail=did_line[did_line.index("DID="):]
    values=[did["mean_advantage_a"],*did["mean_did_ci95"],*did["iid_image_ci95"],did["p_holm"]]
    tokens=[re.search(r"DID=([+-]?[0-9.]+)",tail).group(1)]
    for interval in re.findall(r"CI=\[([^]]+)\]",tail):
        tokens.extend(interval.split(","))
    tokens.append(re.search(r"Holm p=([0-9.]+)",tail).group(1))
    assert len(tokens)==6
    for token,x in zip(tokens,values):
        rounded(token,x)
    counts["direct_did_numeric_cells"]=6
    # Recheck descriptive foundations for the six method-choice rules.
    groups=[[r for r in stats["pareto"] if (r["cohort"],r["dataset"],r["model"])==("model_correct",ds,model)] for ds in DATASETS for model in MODELS]
    rules=dict(IG_lowest_auc_in_five_of_six=sum(min(g,key=lambda r:r["faithfulness"])["method"]=="ig" for g in groups)==5,
        RISE_highest_stability_in_six=all(max(g,key=lambda r:r["stability"])["method"]=="rise" for g in groups),
        GradCAM_lowest_recorded_time_in_six=all(min(g,key=lambda r:r["recorded_ms"])["method"]=="gradcam" for g in groups),
        quality_front_sizes_2_to_4=all(2<=sum(r["quality_front"] for r in g)<=4 for g in groups),
        LIME_quality_front_only_ImageNet_ResNet50=[(r["dataset"],r["model"]) for g in groups for r in g if r["method"]=="lime" and r["quality_front"]]==[("imagenet","resnet50")])
    for metric,direction in ((METRICS[0],1),(METRICS[1],-1)):
        pairs=[r for r in stats["method_pairs"] if (r["cohort"],r["a"],r["b"],r["metric"])==("model_correct","ig","gradcam",metric)]
        assert len(pairs)==6
        rules["IG_GradCAM_"+metric]=all(r["p_holm"]<.05 and min(direction*x for x in r["mean_advantage_ci95"])>0 for r in pairs)
    assert all(rules.values())
    assert sha(handoff.read_bytes())==EXPECTED_SHA
    with ZipFile(handoff) as z:
        path=next(n for n in z.namelist() if n.endswith("/data/per_image_model_correct.csv"))
        source=list(csv.DictReader(io.StringIO(z.read(path).decode())))
    cells=0
    for dataset in DATASETS:
        for model in MODELS:
            for method in METHODS:
                p=next(x for x in stats["pareto"] if (x["cohort"],x["dataset"],x["model"],x["method"])==("model_correct",dataset,model,method))
                for metric,key in zip(METRICS,("faithfulness","stability","recorded_ms")):
                    raw=[r for r in source if (r["dataset"],r["model"],r["method"],r["metric"])==(dataset,model,method,metric)]
                    assert len(raw)==p["n"] and len({r["image_id"] for r in raw})==p["n"]
                    np.testing.assert_allclose(np.mean([float(r["value"]) for r in raw]),p[key],rtol=1e-12,atol=1e-12)
                    cells+=1
    # Fail-closed controls: numerical mistranscription and an altered front flag.
    rejected=[]
    try:
        numeric_tables(text.replace("0.7224","0.8224",1),stage,mixed)
    except AssertionError:
        rejected.append("altered_mixed_eta_rejected")
    bad=copy.deepcopy(stats)
    x=next(r for r in bad["pareto"] if r["cohort"]=="model_correct")
    x["recorded_time_3d_front"]=not x["recorded_time_3d_front"]
    try:
        pareto_tables(text,bad)
    except AssertionError:
        rejected.append("altered_front_flag_rejected")
    assert len(rejected)==2
    return dict(date="2026-11-01",actor="D",reviewed_A_head=A_HEAD,D_source="272ca25fccb30cc1511bbd3684f2c99397e66e93",
        status="numeric_and_source_checks_passed",**counts,method_choice_numeric_checks=rules,pareto_table_sets_checked=12,
        three_objective_flags_checked=36,raw_model_correct_means_checked=cells,subject_counts_checked=36,
        log10_display_dominance_preserved=True,negative_controls=rejected,panels=panels,
        report_sha256=sha(report.read_bytes()),plot_code_sha256=sha((snapshot/"analysis/plot_final_pareto_3d.py").read_bytes()),
        published_plot_sha256=sha((snapshot/"analysis/a_m5_20261031/recorded_time_pareto_3d.png").read_bytes()),
        frozen_handoff_sha256=EXPECTED_SHA,review_code_sha256=sha(Path(__file__).read_bytes()),
        limits="D report/source/numeric audit; visual findings recorded separately. No full bootstrap/GPU rerun, no C/teacher acceptance or actual rehearsal.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("handoff",type=Path)
    p.add_argument("--snapshot",type=Path,default=Path("results/a_v3_review_20261101"))
    p.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--output",type=Path,default=Path("analysis/dcu_20261101/a_v3_review_receipt.json"))
    a=p.parse_args()
    receipt=audit(a.snapshot,a.handoff,a.project)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print("Passed: 54 table statistics, 6 DID values, method-choice checks, 12 Pareto sets, 36 front flags, 108 raw means and 2 negative controls")


if __name__=="__main__":
    main()
