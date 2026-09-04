"""从 ILSVRC2012 val 的 XML 标注中提取评估子集图片的类别标签。

用法:
    python preprocessing/extract_imagenet_labels.py

输出:
    data/imagenet_labels.csv  (image_id, class_id, class_name, image_path)
"""
import glob
import json
import os
import xml.etree.ElementTree as ET

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "imagenet", "raw")
XML_DIR = os.path.join(DATA_DIR, "imagenet", "annotations", "val")
OUT_CSV = os.path.join(DATA_DIR, "imagenet_labels.csv")

# 标准 ImageNet-1k 映射: class_id -> [wnid, class_name]
with open(os.path.join(DATA_DIR, "imagenet_class_index.json")) as f:
    CLASS_INDEX = json.load(f)
WNID2ID = {v[0]: int(k) for k, v in CLASS_INDEX.items()}


def extract_label(xml_path):
    """取 XML 中第一个 object 的 wnid,映射到 0-999 的 class_id。"""
    tree = ET.parse(xml_path)
    wnid = tree.find(".//object/name").text
    class_id = WNID2ID.get(wnid, -1)
    name = CLASS_INDEX[str(class_id)][1] if class_id >= 0 else None
    return wnid, class_id, name


def main():
    rows, missing = [], []
    for img in sorted(glob.glob(os.path.join(RAW_DIR, "*.JPEG"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        xml = os.path.join(XML_DIR, stem + ".xml")
        if not os.path.exists(xml):
            missing.append(stem)
            continue
        _, class_id, class_name = extract_label(xml)
        rows.append({
            "image_id": stem,
            "class_id": class_id,
            "class_name": class_name,
            "image_path": os.path.relpath(img, DATA_DIR),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"提取完成: {len(df)} 张有标签, 输出 -> {os.path.relpath(OUT_CSV)}")
    print(f"类别数: {df['class_id'].nunique()}")
    if missing:
        print(f"警告: {len(missing)} 张图片找不到对应 XML,如 {missing[:3]}")
    print(df.head())


if __name__ == "__main__":
    main()
