"""从 CHNCXR 文件名后缀生成标签表(文件名即标签,无需外部标注文件)。

命名约定(见 data/README.md,NLM 官方约定):
    CHNCXR_XXXX_1.png -> 结核病(tuberculosis)
    CHNCXR_XXXX_0.png -> 正常(normal)

用法:
    1. 将完整 CHNCXR 原始图(662 张)放入 data/chncxr/raw/
    2. python preprocessing/extract_chncxr_labels.py
    3. python preprocessing/sample_eval_set.py   # 重新生成 metadata.csv

输出:
    data/chncxr_labels.csv  (image_id, class_id, class_name, image_path)
"""
import glob
import os
import re

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "chncxr", "raw")
OUT_CSV = os.path.join(DATA_DIR, "chncxr_labels.csv")

CLASS_NAMES = {0: "normal", 1: "tuberculosis"}
PATTERN = re.compile(r"^(CHNCXR_\d+)_([01])\.png$", re.IGNORECASE)


def main():
    rows, skipped = [], []
    for img in sorted(glob.glob(os.path.join(RAW_DIR, "*.png"))):
        m = PATTERN.match(os.path.basename(img))
        if not m:
            skipped.append(os.path.basename(img))
            continue
        image_id, class_id = m.group(1), int(m.group(2))
        rows.append({
            "image_id": image_id,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "image_path": os.path.relpath(img, DATA_DIR),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"提取完成: {len(df)} 张, 输出 -> data/chncxr_labels.csv")
    if len(df):
        print(df.groupby("class_name").size().to_string())
    if skipped:
        print(f"警告: {len(skipped)} 个文件不符合命名约定,如 {skipped[:3]}")


if __name__ == "__main__":
    main()
