# 配对质量结果全表（2026-10-15）

来自同目录 statistics.json。正效应有利于 a；AUC↓、稳定性↑。
CI 是配对均值优势的边际 95% 区间；p 是配对秩/符号检验，不是均值检验。
dz 是配对 Cohen dz；NA 表示常量非零差的标准差为零，不写成无穷。

## 主分析：六方法对比

180 项质量族，均双侧 Holm 校正；符号检验另成同规模族。

| 数据集 | 模型/方法 | 指标 | a | b | n | a 均值优势 | 95% CI | 秩二列 | dz | Holm p | 符号 Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ImageNet | ResNet50 | AUC | IG | Grad-CAM | 408 | 0.05390 | [0.04984, 0.05793] | 0.801 | 0.744 | 8.31e-43 | 1.04e-55 |
| ImageNet | ResNet50 | AUC | IG | Occlusion | 408 | 0.17258 | [0.16463, 0.18050] | 0.978 | 1.286 | 1.5e-63 | 3.79e-90 |
| ImageNet | ResNet50 | AUC | IG | RISE | 408 | 0.07557 | [0.07022, 0.08113] | 0.862 | 0.744 | 1.93e-49 | 1.36e-64 |
| ImageNet | ResNet50 | AUC | IG | KernelSHAP | 408 | 0.08264 | [0.07667, 0.08861] | 0.866 | 0.822 | 5.68e-50 | 1.28e-51 |
| ImageNet | ResNet50 | AUC | IG | LIME-8000 | 408 | 0.02941 | [0.02542, 0.03331] | 0.593 | 0.398 | 1.67e-23 | 1.43e-28 |
| ImageNet | ResNet50 | AUC | Grad-CAM | Occlusion | 408 | 0.11868 | [0.11172, 0.12574] | 0.872 | 1.036 | 1.16e-50 | 4.2e-36 |
| ImageNet | ResNet50 | AUC | Grad-CAM | RISE | 408 | 0.02167 | [0.01781, 0.02563] | 0.143 | 0.308 | 0.157 | 1 |
| ImageNet | ResNet50 | AUC | Grad-CAM | KernelSHAP | 408 | 0.02874 | [0.02469, 0.03285] | 0.354 | 0.446 | 1.82e-08 | 0.53 |
| ImageNet | ResNet50 | AUC | Grad-CAM | LIME-8000 | 408 | -0.02449 | [-0.02748, -0.02150] | -0.627 | -0.405 | 2.89e-26 | 2.92e-34 |
| ImageNet | ResNet50 | AUC | Occlusion | RISE | 408 | -0.09701 | [-0.10403, -0.08997] | -0.794 | -0.848 | 5.17e-42 | 1.16e-33 |
| ImageNet | ResNet50 | AUC | Occlusion | KernelSHAP | 408 | -0.08994 | [-0.09686, -0.08309] | -0.810 | -0.857 | 9.69e-44 | 2.67e-39 |
| ImageNet | ResNet50 | AUC | Occlusion | LIME-8000 | 408 | -0.14317 | [-0.15059, -0.13584] | -0.941 | -1.157 | 7.59e-59 | 1.8e-68 |
| ImageNet | ResNet50 | AUC | RISE | KernelSHAP | 408 | 0.00708 | [0.00260, 0.01154] | 0.130 | 0.100 | 0.278 | 1 |
| ImageNet | ResNet50 | AUC | RISE | LIME-8000 | 408 | -0.04616 | [-0.05060, -0.04167] | -0.698 | -0.557 | 1.56e-32 | 1.16e-33 |
| ImageNet | ResNet50 | AUC | KernelSHAP | LIME-8000 | 408 | -0.05324 | [-0.05782, -0.04873] | -0.737 | -0.664 | 2.82e-36 | 6.61e-32 |
| ImageNet | ResNet50 | 稳定性 | IG | Grad-CAM | 408 | -0.32169 | [-0.33030, -0.31323] | -1.000 | -1.881 | 2.43e-66 | 5.45e-121 |
| ImageNet | ResNet50 | 稳定性 | IG | Occlusion | 408 | -0.24731 | [-0.25466, -0.24000] | -0.999 | -1.677 | 3.17e-66 | 2.04e-118 |
| ImageNet | ResNet50 | 稳定性 | IG | RISE | 408 | -0.32732 | [-0.33607, -0.31863] | -1.000 | -1.856 | 2.43e-66 | 5.45e-121 |
| ImageNet | ResNet50 | 稳定性 | IG | KernelSHAP | 408 | -0.32499 | [-0.33378, -0.31630] | -1.000 | -1.851 | 2.43e-66 | 5.45e-121 |
| ImageNet | ResNet50 | 稳定性 | IG | LIME-8000 | 408 | -0.00860 | [-0.01825, 0.00113] | -0.040 | -0.049 | 1 | 1 |
| ImageNet | ResNet50 | 稳定性 | Grad-CAM | Occlusion | 408 | 0.07438 | [0.07138, 0.07744] | 1.000 | 1.163 | 2.43e-66 | 5.45e-121 |
| ImageNet | ResNet50 | 稳定性 | Grad-CAM | RISE | 408 | -0.00562 | [-0.00623, -0.00501] | -0.634 | -0.422 | 7.86e-27 | 6.04e-25 |
| ImageNet | ResNet50 | 稳定性 | Grad-CAM | KernelSHAP | 408 | -0.00330 | [-0.00388, -0.00271] | -0.254 | -0.271 | 0.000181 | 0.273 |
| ImageNet | ResNet50 | 稳定性 | Grad-CAM | LIME-8000 | 408 | 0.31310 | [0.30686, 0.31937] | 1.000 | 2.778 | 2.43e-66 | 5.45e-121 |
| ImageNet | ResNet50 | 稳定性 | Occlusion | RISE | 408 | -0.08001 | [-0.08324, -0.07685] | -1.000 | -1.139 | 2.43e-66 | 5.45e-121 |
| ImageNet | ResNet50 | 稳定性 | Occlusion | KernelSHAP | 408 | -0.07768 | [-0.08094, -0.07447] | -1.000 | -1.125 | 2.43e-66 | 5.45e-121 |
| ImageNet | ResNet50 | 稳定性 | Occlusion | LIME-8000 | 408 | 0.23871 | [0.23247, 0.24495] | 0.994 | 1.979 | 1.57e-65 | 6.75e-105 |
| ImageNet | ResNet50 | 稳定性 | RISE | KernelSHAP | 408 | 0.00233 | [0.00200, 0.00266] | 0.602 | 0.369 | 3.51e-24 | 1.43e-28 |
| ImageNet | ResNet50 | 稳定性 | RISE | LIME-8000 | 408 | 0.31872 | [0.31238, 0.32515] | 1.000 | 2.777 | 2.43e-66 | 5.45e-121 |
| ImageNet | ResNet50 | 稳定性 | KernelSHAP | LIME-8000 | 408 | 0.31639 | [0.31006, 0.32283] | 1.000 | 2.763 | 2.43e-66 | 5.45e-121 |
| ImageNet | DenseNet121 | AUC | IG | Grad-CAM | 371 | 0.01412 | [0.00830, 0.01985] | 0.520 | 0.161 | 1.69e-16 | 2.76e-25 |
| ImageNet | DenseNet121 | AUC | IG | Occlusion | 371 | 0.04360 | [0.03483, 0.05268] | 0.507 | 0.300 | 1.07e-15 | 4.37e-18 |
| ImageNet | DenseNet121 | AUC | IG | RISE | 371 | 0.01025 | [0.00459, 0.01591] | 0.440 | 0.116 | 7.69e-12 | 4.37e-18 |
| ImageNet | DenseNet121 | AUC | IG | KernelSHAP | 371 | -0.00313 | [-0.00842, 0.00207] | 0.312 | -0.037 | 5.16e-06 | 6.58e-13 |
| ImageNet | DenseNet121 | AUC | IG | LIME-8000 | 371 | 0.01252 | [0.00674, 0.01809] | 0.536 | 0.150 | 1.72e-17 | 4.91e-28 |
| ImageNet | DenseNet121 | AUC | Grad-CAM | Occlusion | 371 | 0.02948 | [0.02168, 0.03751] | 0.059 | 0.241 | 1 | 1 |
| ImageNet | DenseNet121 | AUC | Grad-CAM | RISE | 371 | -0.00387 | [-0.00609, -0.00166] | -0.209 | -0.095 | 0.00814 | 0.0997 |
| ImageNet | DenseNet121 | AUC | Grad-CAM | KernelSHAP | 371 | -0.01725 | [-0.01982, -0.01477] | -0.477 | -0.388 | 7.31e-14 | 7.29e-12 |
| ImageNet | DenseNet121 | AUC | Grad-CAM | LIME-8000 | 371 | -0.00160 | [-0.00435, 0.00104] | 0.002 | -0.036 | 1 | 1 |
| ImageNet | DenseNet121 | AUC | Occlusion | RISE | 371 | -0.03335 | [-0.04130, -0.02579] | -0.182 | -0.265 | 0.0351 | 1 |
| ImageNet | DenseNet121 | AUC | Occlusion | KernelSHAP | 371 | -0.04673 | [-0.05441, -0.03932] | -0.470 | -0.405 | 1.76e-13 | 1.49e-07 |
| ImageNet | DenseNet121 | AUC | Occlusion | LIME-8000 | 371 | -0.03108 | [-0.03921, -0.02333] | -0.053 | -0.259 | 1 | 1 |
| ImageNet | DenseNet121 | AUC | RISE | KernelSHAP | 371 | -0.01338 | [-0.01559, -0.01123] | -0.286 | -0.273 | 4.68e-05 | 0.00617 |
| ImageNet | DenseNet121 | AUC | RISE | LIME-8000 | 371 | 0.00227 | [-0.00010, 0.00461] | 0.157 | 0.046 | 0.122 | 0.565 |
| ImageNet | DenseNet121 | AUC | KernelSHAP | LIME-8000 | 371 | 0.01565 | [0.01358, 0.01772] | 0.470 | 0.430 | 1.74e-13 | 6.58e-10 |
| ImageNet | DenseNet121 | 稳定性 | IG | Grad-CAM | 371 | -0.23995 | [-0.24827, -0.23184] | -1.000 | -1.523 | 2.01e-60 | 6.03e-110 |
| ImageNet | DenseNet121 | 稳定性 | IG | Occlusion | 371 | -0.22902 | [-0.23699, -0.22123] | -1.000 | -1.512 | 2.01e-60 | 6.03e-110 |
| ImageNet | DenseNet121 | 稳定性 | IG | RISE | 371 | -0.23997 | [-0.24830, -0.23184] | -1.000 | -1.519 | 2.01e-60 | 6.03e-110 |
| ImageNet | DenseNet121 | 稳定性 | IG | KernelSHAP | 371 | -0.23698 | [-0.24523, -0.22894] | -1.000 | -1.511 | 2.01e-60 | 6.03e-110 |
| ImageNet | DenseNet121 | 稳定性 | IG | LIME-8000 | 371 | 0.02711 | [0.01746, 0.03661] | 0.236 | 0.173 | 0.00149 | 4.59e-05 |
| ImageNet | DenseNet121 | 稳定性 | Grad-CAM | Occlusion | 371 | 0.01093 | [0.01037, 0.01150] | 0.999 | 1.063 | 2.44e-60 | 1.95e-107 |
| ImageNet | DenseNet121 | 稳定性 | Grad-CAM | RISE | 371 | -0.00002 | [-0.00016, 0.00011] | 0.128 | -0.008 | 0.331 | 0.378 |
| ImageNet | DenseNet121 | 稳定性 | Grad-CAM | KernelSHAP | 371 | 0.00297 | [0.00278, 0.00316] | 0.936 | 0.839 | 5.55e-53 | 5.92e-91 |
| ImageNet | DenseNet121 | 稳定性 | Grad-CAM | LIME-8000 | 371 | 0.26706 | [0.26107, 0.27325] | 1.000 | 2.584 | 2.01e-60 | 6.03e-110 |
| ImageNet | DenseNet121 | 稳定性 | Occlusion | RISE | 371 | -0.01095 | [-0.01151, -0.01039] | -1.000 | -1.014 | 2.01e-60 | 6.03e-110 |
| ImageNet | DenseNet121 | 稳定性 | Occlusion | KernelSHAP | 371 | -0.00796 | [-0.00847, -0.00745] | -0.971 | -0.797 | 5.34e-57 | 7.38e-75 |
| ImageNet | DenseNet121 | 稳定性 | Occlusion | LIME-8000 | 371 | 0.25613 | [0.25016, 0.26232] | 1.000 | 2.510 | 2.01e-60 | 6.03e-110 |
| ImageNet | DenseNet121 | 稳定性 | RISE | KernelSHAP | 371 | 0.00299 | [0.00280, 0.00318] | 0.963 | 1.038 | 3.63e-56 | 5.66e-88 |
| ImageNet | DenseNet121 | 稳定性 | RISE | LIME-8000 | 371 | 0.26708 | [0.26108, 0.27327] | 1.000 | 2.575 | 2.01e-60 | 6.03e-110 |
| ImageNet | DenseNet121 | 稳定性 | KernelSHAP | LIME-8000 | 371 | 0.26409 | [0.25811, 0.27027] | 1.000 | 2.556 | 2.01e-60 | 6.03e-110 |
| ImageNet | VGG16 | AUC | IG | Grad-CAM | 363 | 0.26566 | [0.25186, 0.27962] | 0.997 | 1.142 | 8.36e-59 | 9.19e-99 |
| ImageNet | VGG16 | AUC | IG | Occlusion | 363 | 0.02158 | [0.01779, 0.02563] | 0.699 | 0.374 | 5.37e-29 | 1e-28 |
| ImageNet | VGG16 | AUC | IG | RISE | 363 | 0.03723 | [0.03365, 0.04099] | 0.901 | 0.593 | 4.4e-48 | 7.82e-64 |
| ImageNet | VGG16 | AUC | IG | KernelSHAP | 363 | 0.01346 | [0.01178, 0.01514] | 0.682 | 0.441 | 1.43e-27 | 1.01e-31 |
| ImageNet | VGG16 | AUC | IG | LIME-8000 | 363 | 0.03527 | [0.03253, 0.03797] | 0.908 | 0.733 | 7.94e-49 | 8.5e-62 |
| ImageNet | VGG16 | AUC | Grad-CAM | Occlusion | 363 | -0.24408 | [-0.25861, -0.22963] | -0.959 | -1.048 | 1.76e-54 | 1.14e-72 |
| ImageNet | VGG16 | AUC | Grad-CAM | RISE | 363 | -0.22843 | [-0.24272, -0.21427] | -0.948 | -1.011 | 2.95e-53 | 7.14e-65 |
| ImageNet | VGG16 | AUC | Grad-CAM | KernelSHAP | 363 | -0.25220 | [-0.26606, -0.23865] | -0.986 | -1.096 | 1.63e-57 | 1.73e-81 |
| ImageNet | VGG16 | AUC | Grad-CAM | LIME-8000 | 363 | -0.23040 | [-0.24419, -0.21670] | -0.925 | -1.019 | 1.19e-50 | 1.51e-51 |
| ImageNet | VGG16 | AUC | Occlusion | RISE | 363 | 0.01565 | [0.01128, 0.01998] | 0.459 | 0.252 | 1.25e-12 | 4.12e-11 |
| ImageNet | VGG16 | AUC | Occlusion | KernelSHAP | 363 | -0.00813 | [-0.01135, -0.00525] | -0.202 | -0.191 | 0.0138 | 0.821 |
| ImageNet | VGG16 | AUC | Occlusion | LIME-8000 | 363 | 0.01368 | [0.00967, 0.01740] | 0.525 | 0.253 | 2.07e-16 | 2.26e-16 |
| ImageNet | VGG16 | AUC | RISE | KernelSHAP | 363 | -0.02377 | [-0.02702, -0.02071] | -0.604 | -0.463 | 9.73e-22 | 2.69e-21 |
| ImageNet | VGG16 | AUC | RISE | LIME-8000 | 363 | -0.00197 | [-0.00531, 0.00136] | 0.052 | -0.040 | 1 | 1 |
| ImageNet | VGG16 | AUC | KernelSHAP | LIME-8000 | 363 | 0.02181 | [0.01963, 0.02401] | 0.655 | 0.568 | 1.56e-25 | 5.84e-19 |
| ImageNet | VGG16 | 稳定性 | IG | Grad-CAM | 363 | -0.14106 | [-0.14861, -0.13364] | -0.958 | -1.112 | 2.11e-54 | 1.73e-81 |
| ImageNet | VGG16 | 稳定性 | IG | Occlusion | 363 | -0.19811 | [-0.20475, -0.19148] | -1.000 | -1.587 | 3.76e-59 | 1.41e-107 |
| ImageNet | VGG16 | 稳定性 | IG | RISE | 363 | -0.20487 | [-0.21172, -0.19805] | -1.000 | -1.582 | 3.76e-59 | 1.41e-107 |
| ImageNet | VGG16 | 稳定性 | IG | KernelSHAP | 363 | -0.20368 | [-0.21058, -0.19686] | -1.000 | -1.571 | 3.76e-59 | 1.41e-107 |
| ImageNet | VGG16 | 稳定性 | IG | LIME-8000 | 363 | 0.06964 | [0.06212, 0.07726] | 0.559 | 0.534 | 1.25e-18 | 2.26e-16 |
| ImageNet | VGG16 | 稳定性 | Grad-CAM | Occlusion | 363 | -0.05705 | [-0.06277, -0.05167] | -0.984 | -0.664 | 2.43e-57 | 8.81e-92 |
| ImageNet | VGG16 | 稳定性 | Grad-CAM | RISE | 363 | -0.06381 | [-0.06962, -0.05824] | -1.000 | -0.721 | 3.76e-59 | 4.84e-105 |
| ImageNet | VGG16 | 稳定性 | Grad-CAM | KernelSHAP | 363 | -0.06262 | [-0.06836, -0.05706] | -1.000 | -0.708 | 3.76e-59 | 4.84e-105 |
| ImageNet | VGG16 | 稳定性 | Grad-CAM | LIME-8000 | 363 | 0.21070 | [0.20304, 0.21837] | 0.977 | 1.665 | 1.52e-56 | 8.81e-92 |
| ImageNet | VGG16 | 稳定性 | Occlusion | RISE | 363 | -0.00676 | [-0.00731, -0.00623] | -0.921 | -0.628 | 2.74e-50 | 2.37e-70 |
| ImageNet | VGG16 | 稳定性 | Occlusion | KernelSHAP | 363 | -0.00556 | [-0.00603, -0.00512] | -0.921 | -0.620 | 3.18e-50 | 5.15e-56 |
| ImageNet | VGG16 | 稳定性 | Occlusion | LIME-8000 | 363 | 0.26775 | [0.26177, 0.27378] | 1.000 | 2.491 | 3.76e-59 | 1.41e-107 |
| ImageNet | VGG16 | 稳定性 | RISE | KernelSHAP | 363 | 0.00119 | [0.00089, 0.00146] | 0.561 | 0.234 | 9.89e-19 | 8.52e-24 |
| ImageNet | VGG16 | 稳定性 | RISE | LIME-8000 | 363 | 0.27451 | [0.26850, 0.28062] | 1.000 | 2.509 | 3.76e-59 | 1.41e-107 |
| ImageNet | VGG16 | 稳定性 | KernelSHAP | LIME-8000 | 363 | 0.27332 | [0.26733, 0.27942] | 1.000 | 2.494 | 3.76e-59 | 1.41e-107 |
| VOC (single target) | ResNet50 | AUC | IG | Grad-CAM | 387 | 0.10669 | [0.09981, 0.11345] | 0.943 | 1.278 | 4.12e-56 | 4.65e-63 |
| VOC (single target) | ResNet50 | AUC | IG | Occlusion | 387 | 0.19269 | [0.18128, 0.20355] | 0.960 | 1.446 | 3.34e-58 | 5.12e-66 |
| VOC (single target) | ResNet50 | AUC | IG | RISE | 387 | 0.11112 | [0.10287, 0.11928] | 0.915 | 1.162 | 6.92e-53 | 4.65e-63 |
| VOC (single target) | ResNet50 | AUC | IG | KernelSHAP | 387 | 0.05877 | [0.05097, 0.06642] | 0.655 | 0.666 | 3.53e-27 | 2.03e-19 |
| VOC (single target) | ResNet50 | AUC | IG | LIME-8000 | 387 | 0.06870 | [0.06154, 0.07576] | 0.767 | 0.831 | 3.59e-37 | 1.84e-29 |
| VOC (single target) | ResNet50 | AUC | Grad-CAM | Occlusion | 387 | 0.08601 | [0.07700, 0.09486] | 0.727 | 0.805 | 2.12e-33 | 3.59e-22 |
| VOC (single target) | ResNet50 | AUC | Grad-CAM | RISE | 387 | 0.00443 | [0.00021, 0.00875] | 0.052 | 0.099 | 1 | 1 |
| VOC (single target) | ResNet50 | AUC | Grad-CAM | KernelSHAP | 387 | -0.04792 | [-0.05208, -0.04376] | -0.854 | -0.991 | 4.43e-46 | 2.72e-52 |
| VOC (single target) | ResNet50 | AUC | Grad-CAM | LIME-8000 | 387 | -0.03799 | [-0.04179, -0.03422] | -0.837 | -0.934 | 2.62e-44 | 7.66e-50 |
| VOC (single target) | ResNet50 | AUC | Occlusion | RISE | 387 | -0.08157 | [-0.09042, -0.07257] | -0.717 | -0.773 | 1.57e-32 | 1.19e-22 |
| VOC (single target) | ResNet50 | AUC | Occlusion | KernelSHAP | 387 | -0.13393 | [-0.14236, -0.12527] | -0.988 | -1.364 | 1.51e-61 | 2.81e-88 |
| VOC (single target) | ResNet50 | AUC | Occlusion | LIME-8000 | 387 | -0.12399 | [-0.13296, -0.11498] | -0.929 | -1.201 | 1.96e-54 | 3.74e-61 |
| VOC (single target) | ResNet50 | AUC | RISE | KernelSHAP | 387 | -0.05235 | [-0.05672, -0.04818] | -0.911 | -1.106 | 2.26e-52 | 1.04e-55 |
| VOC (single target) | ResNet50 | AUC | RISE | LIME-8000 | 387 | -0.04242 | [-0.04670, -0.03804] | -0.823 | -0.923 | 8.31e-43 | 4.28e-38 |
| VOC (single target) | ResNet50 | AUC | KernelSHAP | LIME-8000 | 387 | 0.00993 | [0.00581, 0.01409] | 0.243 | 0.222 | 0.000698 | 0.017 |
| VOC (single target) | ResNet50 | 稳定性 | IG | Grad-CAM | 387 | -0.41895 | [-0.43215, -0.40592] | -1.000 | -2.577 | 5.44e-63 | 9.77e-115 |
| VOC (single target) | ResNet50 | 稳定性 | IG | Occlusion | 387 | -0.35030 | [-0.36332, -0.33740] | -1.000 | -2.224 | 5.44e-63 | 9.77e-115 |
| VOC (single target) | ResNet50 | 稳定性 | IG | RISE | 387 | -0.43298 | [-0.44673, -0.41925] | -1.000 | -2.575 | 5.44e-63 | 9.77e-115 |
| VOC (single target) | ResNet50 | 稳定性 | IG | KernelSHAP | 387 | -0.41510 | [-0.42847, -0.40199] | -1.000 | -2.537 | 5.44e-63 | 9.77e-115 |
| VOC (single target) | ResNet50 | 稳定性 | IG | LIME-8000 | 387 | -0.19971 | [-0.21352, -0.18599] | -0.926 | -1.297 | 4.25e-54 | 2.71e-59 |
| VOC (single target) | ResNet50 | 稳定性 | Grad-CAM | Occlusion | 387 | 0.06864 | [0.06371, 0.07382] | 0.985 | 1.259 | 4.18e-61 | 3.78e-97 |
| VOC (single target) | ResNet50 | 稳定性 | Grad-CAM | RISE | 387 | -0.01403 | [-0.01587, -0.01235] | -0.954 | -0.756 | 2.23e-57 | 1.53e-92 |
| VOC (single target) | ResNet50 | 稳定性 | Grad-CAM | KernelSHAP | 387 | 0.00385 | [0.00200, 0.00565] | 0.356 | 0.200 | 3.83e-08 | 5.16e-12 |
| VOC (single target) | ResNet50 | 稳定性 | Grad-CAM | LIME-8000 | 387 | 0.21923 | [0.20997, 0.22876] | 1.000 | 2.128 | 5.44e-63 | 9.77e-115 |
| VOC (single target) | ResNet50 | 稳定性 | Occlusion | RISE | 387 | -0.08268 | [-0.08819, -0.07745] | -1.000 | -1.438 | 5.44e-63 | 9.77e-115 |
| VOC (single target) | ResNet50 | 稳定性 | Occlusion | KernelSHAP | 387 | -0.06480 | [-0.06976, -0.06017] | -0.997 | -1.269 | 1.27e-62 | 8.21e-108 |
| VOC (single target) | ResNet50 | 稳定性 | Occlusion | LIME-8000 | 387 | 0.15059 | [0.14063, 0.16067] | 0.986 | 1.410 | 3.22e-61 | 1.14e-89 |
| VOC (single target) | ResNet50 | 稳定性 | RISE | KernelSHAP | 387 | 0.01788 | [0.01636, 0.01954] | 0.986 | 1.076 | 2.8e-61 | 6.48e-110 |
| VOC (single target) | ResNet50 | 稳定性 | RISE | LIME-8000 | 387 | 0.23327 | [0.22359, 0.24318] | 1.000 | 2.178 | 5.44e-63 | 9.77e-115 |
| VOC (single target) | ResNet50 | 稳定性 | KernelSHAP | LIME-8000 | 387 | 0.21539 | [0.20608, 0.22492] | 1.000 | 2.099 | 5.44e-63 | 9.77e-115 |
| VOC (single target) | DenseNet121 | AUC | IG | Grad-CAM | 388 | 0.12412 | [0.11438, 0.13377] | 0.822 | 0.890 | 8.5e-43 | 5.61e-39 |
| VOC (single target) | DenseNet121 | AUC | IG | Occlusion | 388 | 0.14220 | [0.12708, 0.15715] | 0.611 | 0.613 | 1.08e-23 | 2.78e-18 |
| VOC (single target) | DenseNet121 | AUC | IG | RISE | 388 | 0.09946 | [0.08892, 0.10992] | 0.634 | 0.654 | 1.51e-25 | 1.24e-16 |
| VOC (single target) | DenseNet121 | AUC | IG | KernelSHAP | 388 | 0.04495 | [0.03467, 0.05511] | 0.261 | 0.237 | 0.000178 | 0.000399 |
| VOC (single target) | DenseNet121 | AUC | IG | LIME-8000 | 388 | 0.09303 | [0.08276, 0.10350] | 0.547 | 0.535 | 4.79e-19 | 3.83e-12 |
| VOC (single target) | DenseNet121 | AUC | Grad-CAM | Occlusion | 388 | 0.01807 | [0.00673, 0.02995] | 0.129 | 0.104 | 0.302 | 0.821 |
| VOC (single target) | DenseNet121 | AUC | Grad-CAM | RISE | 388 | -0.02466 | [-0.03039, -0.01887] | -0.321 | -0.284 | 1.2e-06 | 0.000587 |
| VOC (single target) | DenseNet121 | AUC | Grad-CAM | KernelSHAP | 388 | -0.07917 | [-0.08501, -0.07328] | -0.618 | -0.596 | 2.91e-24 | 1.4e-19 |
| VOC (single target) | DenseNet121 | AUC | Grad-CAM | LIME-8000 | 388 | -0.03109 | [-0.03762, -0.02439] | -0.403 | -0.330 | 2e-10 | 6.58e-10 |
| VOC (single target) | DenseNet121 | AUC | Occlusion | RISE | 388 | -0.04273 | [-0.05365, -0.03212] | -0.242 | -0.224 | 0.000698 | 0.0566 |
| VOC (single target) | DenseNet121 | AUC | Occlusion | KernelSHAP | 388 | -0.09724 | [-0.10826, -0.08652] | -0.865 | -0.860 | 1.97e-47 | 4.84e-41 |
| VOC (single target) | DenseNet121 | AUC | Occlusion | LIME-8000 | 388 | -0.04916 | [-0.06111, -0.03744] | -0.378 | -0.329 | 3.58e-09 | 7.57e-06 |
| VOC (single target) | DenseNet121 | AUC | RISE | KernelSHAP | 388 | -0.05451 | [-0.06089, -0.04816] | -0.333 | -0.352 | 3.81e-07 | 0.00137 |
| VOC (single target) | DenseNet121 | AUC | RISE | LIME-8000 | 388 | -0.00643 | [-0.01386, 0.00096] | -0.093 | -0.049 | 1 | 1 |
| VOC (single target) | DenseNet121 | AUC | KernelSHAP | LIME-8000 | 388 | 0.04808 | [0.04136, 0.05489] | 0.408 | 0.434 | 1.16e-10 | 0.000159 |
| VOC (single target) | DenseNet121 | 稳定性 | IG | Grad-CAM | 388 | -0.29389 | [-0.30568, -0.28205] | -1.000 | -1.991 | 3.98e-63 | 5.2e-115 |
| VOC (single target) | DenseNet121 | 稳定性 | IG | Occlusion | 388 | -0.28346 | [-0.29463, -0.27215] | -1.000 | -2.006 | 3.98e-63 | 5.2e-115 |
| VOC (single target) | DenseNet121 | 稳定性 | IG | RISE | 388 | -0.29506 | [-0.30688, -0.28323] | -1.000 | -1.992 | 3.98e-63 | 5.2e-115 |
| VOC (single target) | DenseNet121 | 稳定性 | IG | KernelSHAP | 388 | -0.29064 | [-0.30233, -0.27894] | -1.000 | -1.987 | 3.98e-63 | 5.2e-115 |
| VOC (single target) | DenseNet121 | 稳定性 | IG | LIME-8000 | 388 | -0.02910 | [-0.04104, -0.01662] | -0.269 | -0.212 | 0.000108 | 0.000399 |
| VOC (single target) | DenseNet121 | 稳定性 | Grad-CAM | Occlusion | 388 | 0.01043 | [0.00931, 0.01168] | 0.979 | 0.837 | 1.43e-60 | 4.64e-99 |
| VOC (single target) | DenseNet121 | 稳定性 | Grad-CAM | RISE | 388 | -0.00117 | [-0.00147, -0.00091] | -0.866 | -0.393 | 1.72e-47 | 9.93e-58 |
| VOC (single target) | DenseNet121 | 稳定性 | Grad-CAM | KernelSHAP | 388 | 0.00325 | [0.00282, 0.00369] | 0.920 | 0.708 | 1.43e-53 | 5.55e-76 |
| VOC (single target) | DenseNet121 | 稳定性 | Grad-CAM | LIME-8000 | 388 | 0.26479 | [0.25523, 0.27484] | 1.000 | 2.255 | 3.98e-63 | 5.2e-115 |
| VOC (single target) | DenseNet121 | 稳定性 | Occlusion | RISE | 388 | -0.01160 | [-0.01286, -0.01045] | -1.000 | -0.915 | 3.98e-63 | 5.2e-115 |
| VOC (single target) | DenseNet121 | 稳定性 | Occlusion | KernelSHAP | 388 | -0.00718 | [-0.00821, -0.00623] | -0.919 | -0.693 | 2.07e-53 | 1.24e-58 |
| VOC (single target) | DenseNet121 | 稳定性 | Occlusion | LIME-8000 | 388 | 0.25436 | [0.24507, 0.26410] | 1.000 | 2.236 | 3.98e-63 | 5.2e-115 |
| VOC (single target) | DenseNet121 | 稳定性 | RISE | KernelSHAP | 388 | 0.00442 | [0.00407, 0.00481] | 1.000 | 1.118 | 3.98e-63 | 5.2e-115 |
| VOC (single target) | DenseNet121 | 稳定性 | RISE | LIME-8000 | 388 | 0.26596 | [0.25636, 0.27605] | 1.000 | 2.256 | 3.98e-63 | 5.2e-115 |
| VOC (single target) | DenseNet121 | 稳定性 | KernelSHAP | LIME-8000 | 388 | 0.26154 | [0.25207, 0.27160] | 1.000 | 2.242 | 3.98e-63 | 5.2e-115 |
| VOC (single target) | VGG16 | AUC | IG | Grad-CAM | 400 | 0.25182 | [0.23214, 0.27188] | 0.894 | 0.999 | 3.22e-52 | 8.11e-60 |
| VOC (single target) | VGG16 | AUC | IG | Occlusion | 400 | 0.15780 | [0.14331, 0.17218] | 0.625 | 0.586 | 1.54e-25 | 3.6e-21 |
| VOC (single target) | VGG16 | AUC | IG | RISE | 400 | 0.17168 | [0.15516, 0.18867] | 0.830 | 0.846 | 5.46e-45 | 1e-43 |
| VOC (single target) | VGG16 | AUC | IG | KernelSHAP | 400 | 0.07639 | [0.06652, 0.08627] | 0.452 | 0.375 | 1.94e-13 | 3.96e-11 |
| VOC (single target) | VGG16 | AUC | IG | LIME-8000 | 400 | 0.11044 | [0.09857, 0.12245] | 0.609 | 0.530 | 2.62e-24 | 5.37e-24 |
| VOC (single target) | VGG16 | AUC | Grad-CAM | Occlusion | 400 | -0.09402 | [-0.11129, -0.07708] | -0.506 | -0.407 | 8.67e-17 | 2.67e-17 |
| VOC (single target) | VGG16 | AUC | Grad-CAM | RISE | 400 | -0.08013 | [-0.09740, -0.06307] | -0.374 | -0.365 | 2.77e-09 | 4.01e-06 |
| VOC (single target) | VGG16 | AUC | Grad-CAM | KernelSHAP | 400 | -0.17543 | [-0.19308, -0.15783] | -0.777 | -0.774 | 1.76e-39 | 3.03e-38 |
| VOC (single target) | VGG16 | AUC | Grad-CAM | LIME-8000 | 400 | -0.14138 | [-0.15939, -0.12325] | -0.644 | -0.637 | 4.11e-27 | 1.28e-21 |
| VOC (single target) | VGG16 | AUC | Occlusion | RISE | 400 | 0.01389 | [-0.00065, 0.02836] | 0.068 | 0.049 | 1 | 1 |
| VOC (single target) | VGG16 | AUC | Occlusion | KernelSHAP | 400 | -0.08141 | [-0.09146, -0.07161] | -0.745 | -0.666 | 2.59e-36 | 3.31e-29 |
| VOC (single target) | VGG16 | AUC | Occlusion | LIME-8000 | 400 | -0.04736 | [-0.06041, -0.03448] | -0.075 | -0.244 | 1 | 1 |
| VOC (single target) | VGG16 | AUC | RISE | KernelSHAP | 400 | -0.09529 | [-0.10853, -0.08221] | -0.420 | -0.395 | 1.23e-11 | 3.58e-08 |
| VOC (single target) | VGG16 | AUC | RISE | LIME-8000 | 400 | -0.06124 | [-0.07395, -0.04883] | -0.262 | -0.291 | 0.000131 | 0.203 |
| VOC (single target) | VGG16 | AUC | KernelSHAP | LIME-8000 | 400 | 0.03405 | [0.02480, 0.04312] | 0.383 | 0.233 | 1.05e-09 | 7.89e-11 |
| VOC (single target) | VGG16 | 稳定性 | IG | Grad-CAM | 400 | -0.18570 | [-0.19625, -0.17459] | -0.977 | -1.362 | 3.52e-62 | 2.97e-104 |
| VOC (single target) | VGG16 | 稳定性 | IG | Occlusion | 400 | -0.22000 | [-0.22948, -0.21063] | -1.000 | -1.803 | 4.63e-65 | 1.32e-118 |
| VOC (single target) | VGG16 | 稳定性 | IG | RISE | 400 | -0.22624 | [-0.23589, -0.21669] | -1.000 | -1.827 | 4.63e-65 | 1.32e-118 |
| VOC (single target) | VGG16 | 稳定性 | IG | KernelSHAP | 400 | -0.22398 | [-0.23367, -0.21447] | -1.000 | -1.806 | 4.63e-65 | 1.32e-118 |
| VOC (single target) | VGG16 | 稳定性 | IG | LIME-8000 | 400 | 0.06540 | [0.05452, 0.07638] | 0.555 | 0.532 | 3.42e-20 | 5.84e-15 |
| VOC (single target) | VGG16 | 稳定性 | Grad-CAM | Occlusion | 400 | -0.03430 | [-0.04091, -0.02889] | -0.955 | -0.521 | 1.72e-59 | 6.02e-77 |
| VOC (single target) | VGG16 | 稳定性 | Grad-CAM | RISE | 400 | -0.04054 | [-0.04715, -0.03506] | -0.998 | -0.609 | 8.86e-65 | 1.21e-111 |
| VOC (single target) | VGG16 | 稳定性 | Grad-CAM | KernelSHAP | 400 | -0.03829 | [-0.04487, -0.03284] | -0.998 | -0.579 | 8.14e-65 | 8.68e-108 |
| VOC (single target) | VGG16 | 稳定性 | Grad-CAM | LIME-8000 | 400 | 0.25110 | [0.23906, 0.26315] | 0.988 | 1.643 | 1.72e-63 | 1.11e-109 |
| VOC (single target) | VGG16 | 稳定性 | Occlusion | RISE | 400 | -0.00624 | [-0.00700, -0.00555] | -0.935 | -0.753 | 5.41e-57 | 2.51e-85 |
| VOC (single target) | VGG16 | 稳定性 | Occlusion | KernelSHAP | 400 | -0.00398 | [-0.00466, -0.00337] | -0.805 | -0.558 | 2.36e-42 | 1.32e-41 |
| VOC (single target) | VGG16 | 稳定性 | Occlusion | LIME-8000 | 400 | 0.28540 | [0.27465, 0.29655] | 1.000 | 2.047 | 4.63e-65 | 1.32e-118 |
| VOC (single target) | VGG16 | 稳定性 | RISE | KernelSHAP | 400 | 0.00226 | [0.00185, 0.00264] | 0.828 | 0.480 | 9.39e-45 | 1.92e-70 |
| VOC (single target) | VGG16 | 稳定性 | RISE | LIME-8000 | 400 | 0.29164 | [0.28077, 0.30278] | 1.000 | 2.097 | 4.63e-65 | 1.32e-118 |
| VOC (single target) | VGG16 | 稳定性 | KernelSHAP | LIME-8000 | 400 | 0.28938 | [0.27850, 0.30057] | 1.000 | 2.059 | 4.63e-65 | 1.32e-118 |

## 共同正确组：跨模型

72 项质量族，均双侧 Holm 校正；符号检验另成同规模族。

| 数据集 | 模型/方法 | 指标 | a | b | n | a 均值优势 | 95% CI | 秩二列 | dz | Holm p | 符号 Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ImageNet | IG | AUC | ResNet50 | DenseNet121 | 332 | 0.00651 | [0.00146, 0.01146] | 0.194 | 0.089 | 0.011 | 3.78e-08 |
| ImageNet | IG | AUC | ResNet50 | VGG16 | 332 | -0.02175 | [-0.02533, -0.01825] | -0.242 | -0.409 | 0.000779 | 0.951 |
| ImageNet | IG | AUC | DenseNet121 | VGG16 | 332 | -0.02826 | [-0.03355, -0.02316] | -0.578 | -0.360 | 1.32e-18 | 5.09e-12 |
| ImageNet | IG | 稳定性 | ResNet50 | DenseNet121 | 332 | -0.08984 | [-0.09247, -0.08729] | -1.000 | -2.237 | 2.18e-54 | 1.49e-98 |
| ImageNet | IG | 稳定性 | ResNet50 | VGG16 | 332 | -0.12454 | [-0.12778, -0.12134] | -1.000 | -2.201 | 2.18e-54 | 1.49e-98 |
| ImageNet | IG | 稳定性 | DenseNet121 | VGG16 | 332 | -0.03470 | [-0.03706, -0.03244] | -0.964 | -0.928 | 1.11e-50 | 9.38e-72 |
| ImageNet | Grad-CAM | AUC | ResNet50 | DenseNet121 | 332 | -0.03410 | [-0.03773, -0.03039] | -0.649 | -0.537 | 3.37e-23 | 5.54e-21 |
| ImageNet | Grad-CAM | AUC | ResNet50 | VGG16 | 332 | 0.19623 | [0.18073, 0.21132] | 0.737 | 0.824 | 9.14e-30 | 4.37e-18 |
| ImageNet | Grad-CAM | AUC | DenseNet121 | VGG16 | 332 | 0.23033 | [0.21479, 0.24561] | 0.877 | 0.985 | 4.94e-42 | 8.51e-34 |
| ImageNet | Grad-CAM | 稳定性 | ResNet50 | DenseNet121 | 332 | -0.00868 | [-0.00934, -0.00802] | -0.988 | -0.728 | 3.63e-53 | 6.67e-90 |
| ImageNet | Grad-CAM | 稳定性 | ResNet50 | VGG16 | 332 | 0.05588 | [0.05020, 0.06178] | 0.996 | 0.654 | 5.31e-54 | 6.67e-90 |
| ImageNet | Grad-CAM | 稳定性 | DenseNet121 | VGG16 | 332 | 0.06456 | [0.05866, 0.07061] | 1.000 | 0.729 | 2.18e-54 | 4.49e-96 |
| ImageNet | Occlusion | AUC | ResNet50 | DenseNet121 | 332 | -0.13727 | [-0.14823, -0.12592] | -0.744 | -0.777 | 2.54e-30 | 1.09e-28 |
| ImageNet | Occlusion | AUC | ResNet50 | VGG16 | 332 | -0.19039 | [-0.19912, -0.18119] | -0.958 | -1.382 | 4.56e-50 | 2.04e-63 |
| ImageNet | Occlusion | AUC | DenseNet121 | VGG16 | 332 | -0.05312 | [-0.06284, -0.04360] | -0.595 | -0.399 | 1.27e-19 | 4.74e-15 |
| ImageNet | Occlusion | 稳定性 | ResNet50 | DenseNet121 | 332 | -0.07305 | [-0.07643, -0.06978] | -1.000 | -1.174 | 2.18e-54 | 1.49e-98 |
| ImageNet | Occlusion | 稳定性 | ResNet50 | VGG16 | 332 | -0.07558 | [-0.07906, -0.07219] | -1.000 | -1.207 | 2.18e-54 | 1.49e-98 |
| ImageNet | Occlusion | 稳定性 | DenseNet121 | VGG16 | 332 | -0.00253 | [-0.00315, -0.00192] | -0.484 | -0.234 | 3.23e-13 | 2.86e-14 |
| ImageNet | RISE | AUC | ResNet50 | DenseNet121 | 332 | -0.06624 | [-0.07163, -0.06072] | -0.786 | -0.675 | 8.18e-34 | 2.79e-29 |
| ImageNet | RISE | AUC | ResNet50 | VGG16 | 332 | -0.06670 | [-0.07270, -0.06057] | -0.759 | -0.670 | 1.5e-31 | 1.09e-28 |
| ImageNet | RISE | AUC | DenseNet121 | VGG16 | 332 | -0.00046 | [-0.00382, 0.00304] | -0.073 | -0.009 | 0.248 | 0.28 |
| ImageNet | RISE | 稳定性 | ResNet50 | DenseNet121 | 332 | -0.00307 | [-0.00335, -0.00281] | -0.945 | -0.634 | 1.13e-48 | 1.57e-64 |
| ImageNet | RISE | 稳定性 | ResNet50 | VGG16 | 332 | -0.00174 | [-0.00205, -0.00142] | -0.617 | -0.334 | 4.61e-21 | 4.18e-24 |
| ImageNet | RISE | 稳定性 | DenseNet121 | VGG16 | 332 | 0.00133 | [0.00113, 0.00157] | 0.701 | 0.354 | 4.7e-27 | 2.79e-29 |
| ImageNet | KernelSHAP | AUC | ResNet50 | DenseNet121 | 332 | -0.08744 | [-0.09274, -0.08202] | -0.919 | -0.990 | 4.59e-46 | 1.88e-43 |
| ImageNet | KernelSHAP | AUC | ResNet50 | VGG16 | 332 | -0.09858 | [-0.10459, -0.09246] | -0.933 | -1.021 | 1.72e-47 | 1.55e-48 |
| ImageNet | KernelSHAP | AUC | DenseNet121 | VGG16 | 332 | -0.01114 | [-0.01293, -0.00934] | -0.426 | -0.381 | 2.18e-10 | 2.05e-08 |
| ImageNet | KernelSHAP | 稳定性 | ResNet50 | DenseNet121 | 332 | -0.00222 | [-0.00250, -0.00195] | -0.549 | -0.376 | 8.19e-17 | 3.97e-13 |
| ImageNet | KernelSHAP | 稳定性 | ResNet50 | VGG16 | 332 | -0.00264 | [-0.00290, -0.00240] | -0.694 | -0.457 | 1.63e-26 | 9.22e-26 |
| ImageNet | KernelSHAP | 稳定性 | DenseNet121 | VGG16 | 332 | -0.00042 | [-0.00060, -0.00024] | -0.273 | -0.111 | 0.000141 | 3.5e-05 |
| ImageNet | LIME-8000 | AUC | ResNet50 | DenseNet121 | 332 | -0.00994 | [-0.01322, -0.00672] | -0.115 | -0.162 | 0.211 | 0.747 |
| ImageNet | LIME-8000 | AUC | ResNet50 | VGG16 | 332 | -0.01540 | [-0.01849, -0.01230] | -0.249 | -0.240 | 0.000597 | 0.146 |
| ImageNet | LIME-8000 | AUC | DenseNet121 | VGG16 | 332 | -0.00546 | [-0.00790, -0.00302] | -0.165 | -0.130 | 0.0359 | 0.161 |
| ImageNet | LIME-8000 | 稳定性 | ResNet50 | DenseNet121 | 332 | -0.05941 | [-0.06689, -0.05193] | -0.509 | -0.488 | 1.67e-14 | 5.65e-10 |
| ImageNet | LIME-8000 | 稳定性 | ResNet50 | VGG16 | 332 | -0.04934 | [-0.05734, -0.04134] | -0.433 | -0.363 | 1.06e-10 | 1.33e-07 |
| ImageNet | LIME-8000 | 稳定性 | DenseNet121 | VGG16 | 332 | 0.01006 | [0.00329, 0.01674] | 0.099 | 0.096 | 0.235 | 0.951 |
| VOC (single target) | IG | AUC | ResNet50 | DenseNet121 | 353 | -0.11052 | [-0.11871, -0.10242] | -0.712 | -0.754 | 1.41e-29 | 6.22e-26 |
| VOC (single target) | IG | AUC | ResNet50 | VGG16 | 353 | -0.33566 | [-0.34279, -0.32845] | -1.000 | -3.645 | 9.48e-58 | 2.66e-102 |
| VOC (single target) | IG | AUC | DenseNet121 | VGG16 | 353 | -0.22514 | [-0.23327, -0.21706] | -0.998 | -1.851 | 1.31e-57 | 4.87e-98 |
| VOC (single target) | IG | 稳定性 | ResNet50 | DenseNet121 | 353 | -0.14071 | [-0.14473, -0.13678] | -1.000 | -3.412 | 9.48e-58 | 7.85e-105 |
| VOC (single target) | IG | 稳定性 | ResNet50 | VGG16 | 353 | -0.20643 | [-0.21216, -0.20076] | -1.000 | -3.299 | 9.48e-58 | 7.85e-105 |
| VOC (single target) | IG | 稳定性 | DenseNet121 | VGG16 | 353 | -0.06572 | [-0.06932, -0.06223] | -1.000 | -1.731 | 9.48e-58 | 7.85e-105 |
| VOC (single target) | Grad-CAM | AUC | ResNet50 | DenseNet121 | 353 | -0.09163 | [-0.09831, -0.08489] | -0.684 | -0.731 | 2.33e-27 | 7.26e-25 |
| VOC (single target) | Grad-CAM | AUC | ResNet50 | VGG16 | 353 | -0.17462 | [-0.19382, -0.15580] | -0.672 | -0.717 | 2.01e-26 | 2.81e-22 |
| VOC (single target) | Grad-CAM | AUC | DenseNet121 | VGG16 | 353 | -0.08299 | [-0.10174, -0.06407] | -0.359 | -0.313 | 4.89e-08 | 1.81e-10 |
| VOC (single target) | Grad-CAM | 稳定性 | ResNet50 | DenseNet121 | 353 | -0.01321 | [-0.01480, -0.01173] | -0.983 | -0.883 | 7.53e-56 | 4.87e-98 |
| VOC (single target) | Grad-CAM | 稳定性 | ResNet50 | VGG16 | 353 | 0.02595 | [0.02034, 0.03285] | 0.758 | 0.404 | 1.96e-33 | 5.39e-35 |
| VOC (single target) | Grad-CAM | 稳定性 | DenseNet121 | VGG16 | 353 | 0.03916 | [0.03353, 0.04602] | 0.999 | 0.607 | 1.08e-57 | 2.66e-102 |
| VOC (single target) | Occlusion | AUC | ResNet50 | DenseNet121 | 353 | -0.16401 | [-0.17857, -0.14930] | -0.802 | -0.930 | 2.08e-37 | 3.67e-31 |
| VOC (single target) | Occlusion | AUC | ResNet50 | VGG16 | 353 | -0.37117 | [-0.38635, -0.35551] | -0.937 | -1.447 | 6.89e-51 | 8.83e-54 |
| VOC (single target) | Occlusion | AUC | DenseNet121 | VGG16 | 353 | -0.20716 | [-0.22235, -0.19126] | -0.817 | -0.934 | 8.51e-39 | 1.91e-38 |
| VOC (single target) | Occlusion | 稳定性 | ResNet50 | DenseNet121 | 353 | -0.07230 | [-0.07792, -0.06687] | -0.984 | -1.319 | 6.64e-56 | 4.5e-100 |
| VOC (single target) | Occlusion | 稳定性 | ResNet50 | VGG16 | 353 | -0.07686 | [-0.08259, -0.07139] | -0.999 | -1.380 | 1.22e-57 | 2.66e-102 |
| VOC (single target) | Occlusion | 稳定性 | DenseNet121 | VGG16 | 353 | -0.00456 | [-0.00565, -0.00353] | -0.571 | -0.430 | 3.01e-19 | 2.62e-16 |
| VOC (single target) | RISE | AUC | ResNet50 | DenseNet121 | 353 | -0.12285 | [-0.13099, -0.11461] | -0.796 | -0.925 | 8.25e-37 | 9.16e-32 |
| VOC (single target) | RISE | AUC | ResNet50 | VGG16 | 353 | -0.27301 | [-0.28701, -0.25888] | -0.962 | -1.583 | 1.55e-53 | 8.19e-67 |
| VOC (single target) | RISE | AUC | DenseNet121 | VGG16 | 353 | -0.15016 | [-0.16334, -0.13679] | -0.794 | -0.876 | 1.07e-36 | 9.16e-32 |
| VOC (single target) | RISE | 稳定性 | ResNet50 | DenseNet121 | 353 | -0.00155 | [-0.00186, -0.00128] | -0.936 | -0.507 | 9.22e-51 | 1.02e-65 |
| VOC (single target) | RISE | 稳定性 | ResNet50 | VGG16 | 353 | -0.00055 | [-0.00100, -0.00008] | -0.492 | -0.115 | 1.93e-14 | 2.81e-22 |
| VOC (single target) | RISE | 稳定性 | DenseNet121 | VGG16 | 353 | 0.00100 | [0.00066, 0.00140] | 0.481 | 0.256 | 7.55e-14 | 0.000133 |
| VOC (single target) | KernelSHAP | AUC | ResNet50 | DenseNet121 | 353 | -0.12546 | [-0.13359, -0.11730] | -0.852 | -1.051 | 3.67e-42 | 6.97e-40 |
| VOC (single target) | KernelSHAP | AUC | ResNet50 | VGG16 | 353 | -0.31590 | [-0.32550, -0.30601] | -0.951 | -1.573 | 2.6e-52 | 4.58e-69 |
| VOC (single target) | KernelSHAP | AUC | DenseNet121 | VGG16 | 353 | -0.19044 | [-0.19877, -0.18179] | -0.933 | -1.296 | 1.72e-50 | 1.62e-61 |
| VOC (single target) | KernelSHAP | 稳定性 | ResNet50 | DenseNet121 | 353 | -0.01459 | [-0.01618, -0.01310] | -0.983 | -0.950 | 1.14e-55 | 2.72e-84 |
| VOC (single target) | KernelSHAP | 稳定性 | ResNet50 | VGG16 | 353 | -0.01566 | [-0.01729, -0.01417] | -0.982 | -0.990 | 9.73e-56 | 1.33e-87 |
| VOC (single target) | KernelSHAP | 稳定性 | DenseNet121 | VGG16 | 353 | -0.00108 | [-0.00144, -0.00072] | -0.448 | -0.278 | 5.49e-12 | 1.59e-13 |
| VOC (single target) | LIME-8000 | AUC | ResNet50 | DenseNet121 | 353 | -0.08384 | [-0.09251, -0.07541] | -0.619 | -0.632 | 1.82e-22 | 2.62e-16 |
| VOC (single target) | LIME-8000 | AUC | ResNet50 | VGG16 | 353 | -0.28943 | [-0.30083, -0.27791] | -0.958 | -1.609 | 3.67e-53 | 1.39e-63 |
| VOC (single target) | LIME-8000 | AUC | DenseNet121 | VGG16 | 353 | -0.20558 | [-0.21556, -0.19515] | -0.914 | -1.064 | 2.18e-48 | 1.26e-55 |
| VOC (single target) | LIME-8000 | 稳定性 | ResNet50 | DenseNet121 | 353 | 0.02947 | [0.02041, 0.03875] | 0.387 | 0.319 | 3.29e-09 | 8.42e-07 |
| VOC (single target) | LIME-8000 | 稳定性 | ResNet50 | VGG16 | 353 | 0.06044 | [0.04968, 0.07147] | 0.574 | 0.520 | 1.97e-19 | 1.01e-14 |
| VOC (single target) | LIME-8000 | 稳定性 | DenseNet121 | VGG16 | 353 | 0.03097 | [0.02179, 0.04062] | 0.342 | 0.316 | 2.29e-07 | 2.26e-05 |
