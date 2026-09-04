# 数据目录说明(data/)

维护人:数据负责人 · 更新:2026-09-04

## 数据总览

| 数据集 | 规模(当前) | 类别 | 图像特征 | 标签来源 |
|--------|-----------|------|---------|---------|
| ImageNet(子集) | 500 张 | 284 类(ImageNet-1k 的 0–999) | JPEG,已缩放 500×500,RGB | `imagenet_labels.csv`(由 XML 标注提取) |
| VOC(子集) | 500 张 | 20 类全覆盖,分布不均衡 | JPEG,原始尺寸(如 500×333) | `voc_labels.csv`(含 bbox,由官方 Annotations 提取) |
| CHNCXR | 662 张(下载中,当前本地 16 张原始图) | 2 类:结核(_1)/ 正常(_0) | PNG 灰度(L),约 3000×3000 | **文件名后缀即标签**(无需额外标签表) |

三个子集均来自课程网盘 resources 目录(学长毕设项目整理),规模恰好符合本项目的评估集规划(每数据集 300–500 张)。

## 目录结构

```
data/
├── imagenet/
│   ├── raw/               # 500 张子集图片(评估集来源)
│   └── annotations/       # ILSVRC2012 val 完整 XML 标注(5.5 万个,仅数据负责人需要)
├── voc/
│   ├── raw/               # 500 张子集图片(评估集来源)
│   └── VOCdevkit/         # 官方 VOC2007(Annotations/JPEGImages,仅数据负责人需要)
├── chncxr/
│   ├── raw/               # 原始胸片(当前 16 张;完整 662 张下载中)
│   └── mask_overlaid_不可用于评估/  # 学长叠加病灶mask的图,禁止用作评估数据!
├── imagenet_labels.csv    # ImageNet 标签表(image_id, class_id, class_name, image_path)
├── voc_labels.csv         # VOC 标签表(image_id, class_id, class_name, bbox, image_path)
└── imagenet_class_index.json  # ImageNet-1k 标准映射(class_id -> [wnid, 类名])
```

## 各数据集说明

### ImageNet 子集(500 张)

- **来源**:学长从 ILSVRC2012 val 集筛选——仅单目标标注、bbox 面积 < 图幅一半、ResNet50 预测正确(413 张)+ `err/` 目录 87 张(我们已合并进 raw/)。
- **标签**:`imagenet_labels.csv`(500/500 张有标签,284 类)。提取脚本 `preprocessing/extract_imagenet_labels.py`,依赖 `annotations/` 的 XML 与 `imagenet_class_index.json`(WNID→class_id 映射)。
- **注意**:图像已被学长缩放到 500×500;若报告提及图像分辨率,写明这一点。

### VOC 子集(500 张)

- **来源**:学长从 VOC2007 trainval 选出——仅单目标、bbox 面积 ≤ 0.5 图幅(readme.txt 原文)。原始尺寸,未缩放。
- **标签**:`voc_labels.csv`(500/500 张,**含 bbox**——选做的定位性指标(Pointing Game)可直接用)。类别分布不均衡:bird 76 张 / diningtable 1 张,采样与报告时注意。
- **官方包**:`VOCdevkit/` 由 `preprocessing/download_voc.py` 下载(460MB tar 已解压),仅数据负责人维护,组员不需要拷贝。

### CHNCXR(胸部 X 光)

- **来源**:NLM 深圳胸片数据集(Jaeger et al., 2014;见 prior/毕设论文.pdf §4.1.2),公开数据。662 张后前位胸片:336 结核 + 326 正常。
- **标签约定**:文件名 `CHNCXR_XXXX_1.png` = 结核,_`0` = 正常。
- **重要**:`mask_overlaid_不可用于评估/` 里的 330 张是学长把病灶 mask 叠加进图像后的版本(已像素比对证实),**不得用于归因评估**(叠加的 mask 就是"标准答案",会污染结果)。
- **状态**:任务书要求数据集 ≥2,主体实验用 ImageNet + VOC 即可。CHNCXR 为**选做第三数据集**:
  拿到完整 662 张原始图后放入 `raw/`,运行 `extract_chncxr_labels.py` + 重新运行 `sample_eval_set.py` 即可入矩阵。
  当前 16 张原始图只够定性演示,不进主矩阵。

## 数据共享方式

- **图像数据不进 GitHub**(被 .gitignore 忽略),通过网盘数据包分发:`data_share_v1.zip`(已满足组员需求;若拿到 CHNCXR 完整集再发 v2)。
- 标签表(CSV/JSON)很小,**进 GitHub 版本管理**。
- 更新流程:数据负责人更新本地 → 重打数据包 → 群通知 → 组员替换对应目录。

## 全组使用约定

1. **输入统一**:224×224 缩放到 tensor 后,用 ImageNet 均值方差归一化(`preprocessing/dataset.py` 提供入口)。
2. **CHNCXR 是灰度图**:加载时 L→RGB 三通道复制,再做归一化(pretrained 模型吃 RGB)。
3. **评估集只采样一次**:`preprocessing/sample_eval_set.py` 生成 `metadata.csv`(seed=42,按类别分层),全组共用同一评估集,保证 54 个实验单元可比。
4. 评估集目标规模:每数据集 300–500 张;另建 30–50 张 debug 集供开发期调试。

## 待办 / 状态

- [x] 运行 `sample_eval_set.py` 生成 `metadata.csv`(ImageNet/VOC 已入矩阵,各 460 eval + 40 debug)
- [x] `preprocessing/dataset.py` 统一加载入口(已联调测试)
- [ ] (选做)CHNCXR 完整集:向学长/老师索取 662 张原始图 → 生成 `chncxr_labels.csv` → 重跑采样 → 发布 data_share_v2
