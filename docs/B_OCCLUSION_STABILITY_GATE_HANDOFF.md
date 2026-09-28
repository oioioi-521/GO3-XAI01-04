# B：Occlusion 稳定性候选协议 GPU 门禁交接（2026-09-28）

## 范围与代码身份

这是三模型 × ImageNet/VOC 的 **单图 → 40 张 debug 工程门禁**，不是 460 张 eval，也不是正式稳定性排名。基础 Occlusion 六组 eval 已另行完成；本轮未重跑它们或 MoRF。B 分支的 A 稳定性实现来自保留提交的 merge `a69df80`，其 PR #10 修复源提交为 `9891eecc4ec869c0bc14a054a2cd85a0beca4266`；B 本轮新增隔离拷贝、测量、校验和交接工具，不把 A 的算法实现算作 B 贡献。

设备为 RTX 5080 Laptop GPU；项目 `.venv-gpu/Scripts/python.exe`，Python 3.12.14，torch 2.11.0+cu128、torchvision 0.26.0+cu128、Captum 0.9.0、SciPy 1.18.1、Matplotlib 3.11.2。用户手动关闭 Smart App Control 后，在**新 Python 进程**中实际导入 `matplotlib._image`、`scipy.stats.spearmanr`、`captum.attr.Occlusion`，Spearman 与 CUDA 矩阵计算有限、真实小型 Captum GPU Occlusion 成功。未更改其它安全策略或驱动。此前为何同一官方 wheel 从可加载变为被拦截仍未知；历史 Code Integrity 证据保存在 `docs/B_OCCLUSION_WINDOWS_CODE_INTEGRITY.md`。

协议采用 RGB `[0,1]` 域高斯噪声，`sigma=0.005`，裁剪后再按冻结预处理归一化；每图 5 次，配对 seed 由数据集、image_id 和重复号确定；固定 metadata 真值 target；Spearman，相同 rank 退化按协议记 0 且进入均值，另记 valid_rate。原图和扰动图的 Occlusion 均用 32×32 窗口、16×16 步长、`perturbations_per_eval=16`。ImageNet 为 softmax，VOC 为 sigmoid。稳定性单独写 `stability_spearman`、`stability_valid_rate` 和逐重复 trace；旧基础 `efficiency_time_ms` 与 raw MoRF 不含来源重算及本轮稳定性耗时。Issue #8 的 C/D 协议和汇总验收仍待最终确认。

## 输入隔离与验证

源自 `results/occlusion_gates/debug40/<unit>/` 的既有 40 张门禁结果。`scripts/prepare_occlusion_stability_gates.py` 只复制冻结 debug 子集的 CSV、预测表与 float32 NPY 到被 Git 忽略的 `results/occlusion_stability_gates/{single,debug40}/<unit>/`，单图与 40 张使用独立配置、trace、state 和输出；不覆盖已存在目录。`input_manifest.json` 记录源与副本的 SHA-256。旧 NPY 无 sidecar，PR #10 runner 在**副本**上重算一次 `A(x)`，要求 float32 形状/有限性相同且 `np.allclose(rtol=1e-4, atol=1e-5)`，才写入 `verified_recomputation`；六组 1+40 张均通过。这只证明复核时与当前模型/配置在该容差内一致，不能反向证明最初生成时的权重身份。

验证工具逐图核对冻结 debug image_id、预测 target、activation、checkpoint SHA、配置/协议哈希、NPY 字节哈希、sidecar、trace 重复号和 seed、状态、指标取值、汇总及立即续跑。源文件哈希持续匹配 `input_manifest.json`；原 debug 结果与六组基础 eval、checkpoint 和既有交付 ZIP 均未修改。

## 实测结果

以下均为每组 **40 张 debug**；`Spearman` 是稳定性逐图均值的组均值，仅用于门禁描述，不作方法优劣排名。`归因 ms` 是 200 次扰动归因的平均计时；`墙钟 s` 是单次稳定性进程实测，含模型加载、来源重算和验证等，因此不能与基础 `efficiency_time_ms` 混用。峰值为该进程的 CUDA allocated，并非整机显存占用。

| 单元 | 稳定性 config_hash | Spearman | valid_rate | 平均扰动归因 ms | 墙钟 s | CUDA 峰值 MiB |
|---|---|---:|---:|---:|---:|---:|
| ResNet50 / ImageNet | `fe95e8ebf249` | 0.918921 | 1.000 | 126.369 | 34.97 | 300.98 |
| ResNet50 / VOC | `e3a1a989ce70` | 0.906132 | 1.000 | 126.690 | 35.16 | 293.24 |
| DenseNet121 / ImageNet | `39583bfa0212` | 0.986859 | 1.000 | 155.336 | 43.11 | 224.78 |
| DenseNet121 / VOC | `3326f4e8ac0e` | 0.988125 | 1.000 | 151.745 | 42.55 | 220.88 |
| VGG16 / ImageNet | `4c984ea35645` | 0.989010 | 1.000 | 260.569 | 69.79 | 1354.83 |
| VGG16 / VOC | `235cc51b7aca` | 0.991460 | 1.000 | 260.065 | 70.12 | 1339.07 |

六组合计 **240 个单元内图片、1,200 条 trace、480 行新增稳定性逐图指标**；每组 40 个唯一 debug ID、200 个互异 `(image_id, repeat)`、80 行新指标，六组均无失败/退化/常量图状态。各组立即 resume 均为 `processed=0, skipped=40`，trace 不重复。六组单图也各为 1 张、5 条有效 trace，立即 resume 为 `processed=0, skipped=1`。首个单图直接运行原 runner，故该单图没有包装器峰值显存记录；其余有。

隔离 `probe/resnet50_imagenet` 先真实运行一次基础 runner 建立其自身配置身份，再运行稳定性。基础 runner 的正常 resume 为 `processed=0, skipped=1`，稳定性行与 hash 保留；基础 `--force` 在写入前抛出拒绝错误，per_image/units/trace 的三个 SHA-256 前后相同。稳定性 `--force` 重新产生 5 条 trace、无基础 MoRF 重跑，再次 resume 为 `processed=0, skipped=1`；五个 Spearman 与独立单图对应值逐项差异为 0。数值相同是本次实测，**不预设所有 GPU/驱动必须逐字节一致**。probe 允许改写其副本，不影响原始 debug 或正式结果。

## 复现与本地交付

在项目根目录以 `.venv-gpu` 运行，逐组先 `python -m scripts.prepare_occlusion_stability_gates --stage single --unit resnet50_imagenet`，再 `python -m scripts.run_occlusion_stability_gate --gate-dir results/occlusion_stability_gates/single/resnet50_imagenet`，然后同命令验证立即 resume，最后 `python -m scripts.validate_occlusion_stability_gate --gate-dir ...`。40 张将 stage 换为 `debug40`；必须串行，不能复用已有目标目录。每组实际 `base.yaml`、`protocol.yaml`、CSV、trace、sidecar、state、日志和 `validation_report.json` 位于相应隔离目录。

本地轻量包 `results/occlusion_stability_gates/delivery/occlusion_stability_gates_light.zip`，**373,591 字节**，SHA-256 `d35a6d47532dae15c063d94a6bb5e22708652076bd2e14b2b9f2eca2d7994174`；包内 407 个轻量文件，逐文件 `package_manifest.json` SHA-256 `cb0e63ac598b7d35676a9382e807d1e1b498ad972d342369beed8c204718d10f`，打包后已逐项回读核验。包**不含 NPY 或 checkpoint**，float32 参考 NPY 仍保存在隔离门禁目录；该包尚未传输，不能把本机路径当成组员下载链接。接收时先验 ZIP SHA，再核包内 manifest，最后核配置、trace 和 CSV；若需要复核参考 NPY，应另行使用组内批准渠道传递或从已验收的基础包重建并重新跑来源校验。

全量 `.venv-gpu/Scripts/python.exe -m pytest -q -ra`：**93 passed、0 failed、0 skipped**；`python -m pip check`：无依赖冲突。结果目录被 `.gitignore` 忽略，未纳入 Git。

## 仍待组内确认

1. C/D 在 Issue #8 最终冻结候选协议、退化计分和统一汇总口径；本门禁不代签统计验收。
2. PR #10 的 A 实现需按其自身审查/合并流程接入公共基线；B 的 PR 仅叠加门禁工具和证据。
3. 组内确认是否接收本地轻量包及是否还需参考 float32 NPY 的受控传输。当前没有已确认的可用传输渠道，**未上传**。
4. 待上述确认后才考虑独立的 460 张稳定性 eval；本轮未启动，亦未生成正式排名。
