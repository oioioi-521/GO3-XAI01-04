"""生成全组共用的评估集与 debug 集清单(metadata.csv)。

规则:
    - 每个数据集单独划分:评估集 300-500 张 + debug 集 40 张(开发期快速调试用)
    - 按类别分层采样,random_state=42,只运行一次,全组共用
    - CHNCXR 的标签 CSV 尚未就绪时会跳过并提示

用法:
    python preprocessing/sample_eval_set.py

输出:
    data/metadata.csv  (image_id, dataset, class_id, class_name, image_path, split)
"""
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_CSV = os.path.join(DATA_DIR, "metadata.csv")

SEED = 42
DEBUG_PER_DATASET = 40

# 每个数据集的标签表;chncxr_labels.csv 由数据负责人下载完整数据后生成
LABEL_CSVS = {
    "imagenet": "imagenet_labels.csv",
    "voc": "voc_labels.csv",
    "chncxr": "chncxr_labels.csv",
}


def sample_dataset(dataset, labels_path, target_eval):
    """按类别分层划分 eval/debug。target_eval 为 None 时评估集取全部剩余。"""
    df = pd.read_csv(labels_path)
    # debug 集:每类按比例抽,共 DEBUG_PER_DATASET 张
    debug = (
        df.groupby("class_id", group_keys=False)
        .apply(lambda g: g.sample(n=max(1, int(round(len(g) / len(df) * DEBUG_PER_DATASET))),
                                  random_state=SEED), include_groups=False)
        .sample(n=DEBUG_PER_DATASET, random_state=SEED)
    )
    debug_ids = set(debug["image_id"])
    rest = df[~df["image_id"].isin(debug_ids)]
    if target_eval is not None:
        # 每类最多取 target_eval // 类别数 + 1,总量控制在 target_eval 附近
        per_class = max(1, target_eval // df["class_id"].nunique())
        eval_df = (
            rest.groupby("class_id", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), per_class), random_state=SEED),
                   include_groups=False)
        )
    else:
        eval_df = rest
    debug = debug.copy()
    eval_df = eval_df.copy()
    debug["split"] = "debug"
    eval_df["split"] = "eval"
    out = pd.concat([eval_df, debug], ignore_index=True)
    out["dataset"] = dataset
    keep = ["image_id", "dataset", "class_id", "class_name", "image_path", "split"]
    out = out[keep + ["bbox_x1y1x2y2"] if "bbox_x1y1x2y2" in out.columns else keep]
    return out


def main():
    # ImageNet 500 / VOC 500 全量进评估集(符合 300-500 规划),CHNCXR 平衡抽样 450
    plans = {"imagenet": None, "voc": None, "chncxr": 450}
    frames, skipped = [], []
    for dataset, target in plans.items():
        path = os.path.join(DATA_DIR, LABEL_CSVS[dataset])
        if not os.path.exists(path):
            skipped.append(dataset)
            continue
        frames.append(sample_dataset(dataset, path, target))
        print(f"{dataset}: 评估 {sum(frames[-1]['split']=='eval')} + debug {sum(frames[-1]['split']=='debug')}")

    meta = pd.concat(frames, ignore_index=True)
    meta.to_csv(OUT_CSV, index=False)
    print(f"\n输出 -> data/metadata.csv,共 {len(meta)} 行")
    if skipped:
        print(f"跳过(标签表缺失): {skipped},生成 chncxr_labels.csv 后重跑本脚本即可")


if __name__ == "__main__":
    main()
