# 前五种方法的 DCU 全量运行

**最终状态：2026-10-13 19:44:20（Asia/Shanghai）完成全部 30 组，正式队列墙钟用时 4 小时 21 分 6 秒。已通过最终审计与本地归档校验，详见 [完整结果汇总](DCU_FULL_RESULTS_20261013.md)。下文保留原始部署记录。**

## 启动状态

2026-10-13 15:23:14（Asia/Shanghai）完成 30/30 个迁移检查，随即进入正式 eval 队列。15:29:24 的启动快照显示 8 个任务运行中、22 个排队、0 个失败；已完成 704 张原图的基础评估，稳定性阶段尚未开始。**这是部署与启动记录，不能作为全量实验已完成的证明。**

节点 node62，DAPJobs 实例 #78，容器 `dev-59d5af49`，56 CPU / 256 GiB / 8 DCU。按用户已有授权暂停 #75，供本次八卡运行使用。任务使用 `docker exec -d` 启动，浏览器和终端断开不会终止任务。

## 运行范围与口径

| 项目 | 本次设置 |
| --- | --- |
| 方法 | Grad-CAM、IG、Occlusion、RISE、KernelSHAP；不含 LIME |
| 模型 | ResNet50、DenseNet121、VGG16 |
| 数据集 | ImageNet、VOC，各 460 张冻结 eval 图片 |
| 矩阵 | 5 × 3 × 2 = 30 个单元 |
| 原图评估 | 13,800 个图像/方法/模型组合 |
| 稳定性 | 每张图 5 次噪声归因，共 69,000 条重复记录 |
| 归因总量 | 原图 + 稳定性，共 82,800 次，不含预热及迁移检查 |
| 基础指标 | attribution efficiency、raw MoRF AUC（21 个删除比例） |
| 稳定性指标 | Spearman、valid rate；既有 sigma=0.005、种子协议 |
| 数值与模型输出 | FP32；ImageNet softmax、VOC sigmoid |

保留正式实验参数：IG 50 步、内部 batch 10；Occlusion 32×32 窗口、stride 16、batch 16；RISE 4,000 masks、7×7、p=0.5、seed 42、batch 32；KernelSHAP 2,048 samples、7×7 grid、batch 16。DenseNet Grad-CAM 层为 `features.norm5`。未启用改变上述口径的加速参数。

正式 RISE 配置及 sigmoid 支持来自 `origin/feat/rise-stability-eval`（`782f103a5fde0736b75668239b16200d6d6b565a`）；Occlusion 六个正式配置来自 `origin/feat/b-occlusion-stability-gates`（`a2246ea66efb58815ef324cb3938162fba941c7f`）。其余使用当前分支已有正式实现。

## 调度与时间预算

控制器 `scripts/run_dcu_formal_queue.py` 为每张物理卡启动一个独立任务，预计耗时最长的单元优先；完成后自动领取下一个单元。每个任务依次运行 460 图基础评估及其五次稳定性重复，输出彼此隔离。CPU / BLAS 线程每进程限制为 2。

本轮采用八个单元并行，不是八卡切分同一个单元。沿用此前容量评估，暂预留 **6–8 小时**，从 15:23 起约为当天 **21:30–23:30**；这是规划区间，并非全量实测或完成承诺。VOC 性能、稳定性阶段及队列末尾负载不均衡都会影响最终墙钟时间。

## 验证与启动证据

- 本地 52 项相关既有测试全部通过。首轮有两项因缺少本地 scikit-learn 失败，在隔离依赖目录补齐后重跑通过。
- 服务端预检核对了发布代码/配置哈希、1,000 张冻结图片及元数据、六个模型权重、八卡可用性和磁盘空间。
- 每个单元先用一张 debug 图片验证基础评估、五次稳定性、输出条数、有限值、原图 NPY 及权重溯源，并调用实际 runner 验证续跑为 processed=0 / skipped=1。30/30 通过后才进入 eval。
- 该一图检查是迁移兼容性和续跑检查，不替代完整科学结果验收。正式单元结束后还会检查 460 图结果并再次验证续跑。
- 启动硬件采样显示所有八卡有工作负载；瞬时 HCU 利用率约 10%–100%，其中 KernelSHAP 会在设备计算与 CPU 拟合之间切换，不能把某次采样当作持续利用率。

本地证据目录：`results/formal-server-launch-evidence/`。包含 30 个 debug 检查的完整输出、预检、启动进度快照、硬件采样、实例截图。已通过浏览器正常下载并核对服务端/本地 SHA-256：

```text
five_run_launch_evidence_20261013.tar.gz
560e6839440d9511d63944858b4c6b37f90ff17f1538470e9f1c284e718ef7b1
```

部署发布清单 `release-manifest.json` SHA-256：

```text
d2483f8b8b5e645309755a269ce4269fdc30d96c140e809ff78d1aff57e7724c
```

## 服务端目录与观察方法

独立发布目录：

```text
容器：/workspace/xai-formal-five-20261013
宿主：/opt/dapjobs-dev/cas_mfwzan/dev-59d5af49/xai-formal-five-20261013
运行根目录（相对发布目录）：results/dcu-full-five-20261013
```

旧试跑目录 `/workspace/xai` 保留；新发布复用其中的依赖和已校验模型缓存，结果写入新目录。运行期间不要修改发布源文件、正式配置、图片或模型缓存。

在 node62 的网页终端读取最新进度：

```sh
python3 -m json.tool /opt/dapjobs-dev/cas_mfwzan/dev-59d5af49/xai-formal-five-20261013/results/dcu-full-five-20261013/status.json
```

`status.json` 约每 5 秒更新，包含 running / pending / completed / failed，及每张卡的 base_images、stability_images。单元日志位于 `formal/<method>-<model>-<dataset>/worker.log`。异常写入单元 `failure.json` 或根目录 `controller_failure.json`，失败任务不会被静默计作完成。

全量成功要求根目录出现 `COMPLETE.json`、`status.json` 为 complete、`controller.exit` 为 0。仅有进度达到 460 或进程消失不足以认定成功。

所有单元完成并验收后自动合并：

| 文件 | 预期条数 |
| --- | ---: |
| `merged/per_image.csv` | 55,200 |
| `merged/units.csv` | 120 |
| `merged/stability_trace.csv` | 69,000 |

原图归因、预测和溯源信息保存在各单元目录中。汇总完成标记记录合并文件哈希。稳定性使用既有 trace 输出策略，不额外保存每次噪声归因 NPY。

## 启动与恢复

本次实际启动命令（在 node62 宿主执行）：

```sh
docker exec -d dev-59d5af49 bash -lc 'cd /workspace/xai-formal-five-20261013; mkdir -p results/dcu-full-five-20261013; PYTHONPATH=.deps:. PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u scripts/run_dcu_formal_queue.py > results/dcu-full-five-20261013/controller.log 2>&1; echo $? > results/dcu-full-five-20261013/controller.exit'
```

若中断，先确认原控制器和它的工作进程均已退出，保留异常日志，解决原因后使用同一发布目录及同一配置续跑。控制器有独占锁，已验收单元会跳过，未完成单元由既有 runner 按图恢复。不要在旧进程仍运行时重复执行上述重定向命令，否则可能截断日志；外部终止控制器也不能假定所有工作进程同时退出。

本次交付时任务仍在运行，尚未回收 #78 资源、恢复 #75 或下载完整正式结果。

## 定期检查与最终验收

用户于部署后授权定期检查、异常通知和完成后的结果校验汇总。已创建当前会话的自动检查 `dcu`（DCU 全量实验进度与结果验收），按用户后续要求调整为每小时检查一次，正常推进时保持安静。异常、监测中断或完成时通知用户；完成验收、归档和汇总后暂停自动检查。

初始监测基线位于 `results/dcu-full-five-monitor-state.json`。2026-10-13 15:36:22 的观察中，8 个任务的基础评估分别推进至 170–221 / 460，队列未报告失败。后续检查结合阶段、日志与进程识别卡住，不能把一次低 DCU 利用率当作异常。

验收后将形成 `docs/DCU_FULL_RESULTS_20261013.md`，汇总正式指标、真实用时和可比的既有试跑/2080 Ti 证据。本自动检查依赖本机与 Codex 应用运行；服务器计算任务本身不依赖浏览器持续连接。

## 最终验收与资源收尾

30/30 单元完成，`status complete`、`controller.exit=0`、`COMPLETE.json` 及各单元 `completion.json` / `resume_verified` 均通过。最终审计复核发布、数据与权重溯源、图像 ID、数值、预测、全部 NPY、五次稳定性重复及种子；合并结果为 55,200 行 `per_image.csv`、120 行 `units.csv`、69,000 行 `stability_trace.csv`。

完整归档 `results/formal-final-evidence/dcu_full_five_20261013_complete.tar.gz` 已下载并校验服务端 SHA-256，已解压核验全部 13,800 个 NPY 和 13,800 个 PNG。精简指标日志包与完整审计记录保存在同一目录。

已停止本轮八卡实例 #78，并恢复此前临时暂停的单卡实例 #75；页面状态确认完成，实例和结果均保留。自动检查 `dcu` 已暂停，未启动 LIME。
