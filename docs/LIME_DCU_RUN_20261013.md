## 2026-10-14 最新执行授权：优化版 8000、八卡、每小时主动汇报

用户明确要求“按照新的把8000的跑起来，8卡并行，每一个小时汇报一次进度”。这取代下文历史暂停、四/六并发及正常时安静的约定。旧发布和旧结果保持原样，本次使用独立执行层 `VectorizedLIME`，batch32，采样/分割/距离/核/Ridge/目标和五次RGB噪声协议均保持。8000是已完整审阅的固定预算，不宣称完全收敛。

本次容器脚本：`/workspace/lime-vectorized-formal-20261014/run_lime_vectorized_full.py`；冻结基础发布：`/workspace/xai-lime-20261013`；新输出：`results/dcu-lime-vectorized-8000-20261014`。宿主结果目录为 `/opt/dapjobs-dev/cas_mfwzan/dev-59d5af49/xai-lime-20261013/results/dcu-lime-vectorized-8000-20261014`。全部服务器操作经 DAPJobs 网页终端，由 6.1-sol 浏览器子代理执行。

正式矩阵仍是3模型×2数据集、每组460张eval、每图5次稳定性，共2760原图组合和16560次归因。每组按冻结metadata顺序分为10份×46张，共60分片，动态调度到卡0–7，VGG优先；每进程显存分配上限10%，CPU/BLAS两线程。每份配置明确记录image_ids、预算审阅SHA和独立源码manifest SHA；只在进程内替换LIME实现与dataset分片入口，原发布源码和metadata不改动。六组合汇总按原图重算均值/标准差，不平均分片标准差。输出NPY保留在分片中，以merged/maps_index.json溯源。

部署包本地 `results/lime-vectorized-formal-deploy-20261014/lime_vectorized_formal_8000_20261014.tar.gz`，22114字节，SHA-256 `c02a4ce16a4802fbea51dd8323d67ce35f72bdca998c562c69f3917ded78ccb3`。独立manifest SHA-256 `78d1f2c799e073d6e8ae87981d8ec71c40f44a30456b9c578cc563e25049a5f6`，控制器 `bfbf0e16052dd5401b35c62f4cbaf4596767a245c65172bac5b88cba384d3ae9`。优化实现沿用已验证原型，SHA-256 `c48e5b6b18298cd4ebf5736d4fda46c265011662f29c9a3207bb823f159ab314`。原8000预算审阅副本及优化独立审计均被manifest绑定；原审阅中的六整组并发估时是历史记录，不能用作本轮实际排程。

本地42项测试通过（6.21s），覆盖原LIME/向量化逐值等价、明确非首段ID选取、两张分片真实base→stability→resume、五次trace、无重复预测和汇总。服务器流水线顺序：

1. `--prepare-only`：复核原125源文件、1000图、metadata和六权重，以及完整高预算预算证据绑定，生成不可改配置和计划。
2. `regression`：六组各40debug×3采样seed，720张新map，要求与旧8000同batch32的map和weighted R²逐值一致；24个10图分片。
3. `gate`：六组各首/末两张debug，12个单图分片，使用真实runner生成原图与五次噪声结果，并真实续跑检查processed0、skipped1。
4. `formal`：全部门禁通过后自动60分片，8并发；每分片完成时核验完整性并实际续跑，processed0、skipped46；独立汇总及运行后源/数据/权重复核。

本轮控制器拒绝自动重启已有运行。检查新根目录 `status.json`、`controller.exit`、`controller_failure.json` 和 `regression|gate|formal/<shard>/failure.json`，避免把旧smoke主动中止130或probe收尾问题当作本轮错误。控制器状态约5秒刷新，包含phase、running/pending/completed/failed、base_images/stability_images和regression_maps；formal总数2760，两种图像进度应分别报告。`formal_start.json`才证明正式eval已开始。状态超过5分钟未更新或一小时无进度须结合实际日志与进程确认；不因瞬时低利用率判卡死。

完成必须核验 `COMPLETE.json/status complete/controller.exit=0`、60分片completion+resume_verified、六逻辑单元 `units/<unit>/completion.json`，全部ID唯一完整、2760 NPY+预测、13800稳定性trace及5次种子、source/weights/overlay/review/config/output哈希。合并：per_image11040行，units24行，stability_trace13800行，predictions2760行。缺项不能算验收通过。

完整40图回归的每组120次新归因平均用时：ResNet50 / ImageNet 4.374s、VOC 4.345s；DenseNet121 / ImageNet 6.631s、VOC 6.577s；VGG16 / ImageNet 7.948s、VOC 7.921s。按60分片八工作槽模拟，纯归因约3.74h；计入MoRF、预热、模型装载与验收，暂估正式4–5h，随eval实测速率更新，不当作已完成墙钟。

每小时在本会话主动报告正常进度、阶段、已完成量、预计剩余时间（基于实测并注明估计）；异常及时报告。已确认平台下载URL为 http://42.228.13.157:40527/api/v1/nodes/1/download?username=cas_mfwzan&path=<已在/home/cas_mfwzan确认的文件名>（文件名需URL编码）。页面下载按钮会打开新标签，不能用原标签监听后直接点击按钮。通过验收后使用同标签先监听download且立即catch、再导航已确认下载URL、await path、本地/服务端SHA一致的流程归档，形成最终六组对比汇总。完成归档后暂停自动检查，停止本轮#78并恢复此前临时暂停#75，同时核对平台和Docker/进程；不删除实例/结果、不操作其他服务。

2026-10-14 14:29:46.898（Asia/Shanghai）回归阶段实际启动，prepare.exit=0、preflight passed。14:31:27快照：八卡各9张，共72/720对照map，failed为空，formal_started=false。自动检查dcu已ACTIVE，每小时主动汇报。14:40:47完成720张回归，全部map与weighted R²逐值一致，墙钟661.020s；随后12项真实runner+稳定性+续跑门禁通过。14:44:18核验已进入formal，卡0–7八个worker各base_images=2，共16张，failed为空；这是已实际启动的正式eval。formal_start.json确认正式启动时间14:43:28.144。后续JSON快照八卡各7张（总56），截图八卡各26张（总208），均无失败；稳定性阶段尚未开始。gate实际墙钟160.223s。原始快照 results/lime-vectorized-formal-deploy-20261014/launch-snapshot.json，SHA-256 b0a85c1a21c03c0f24651c112de74eecf6e20d38aba0184e2433c208efcdf49c；截图同目录formal-launch.png。JSON包已从Downloads找到并核对服务器SHA一致，没有重复启动下载。

### 2026-10-14 15:38 每小时检查：监测连接中断

Browser Use 初次看到终端“连接中”，随后出现 ERR_EMPTY_RESPONSE；同标签一次终端恢复和一次实例页访问均为空响应，没有进入CAS登录页。未执行服务器命令，未取得本轮status/进程/日志，不能判断实验停止或失败，也不能刷新完成量和ETA。保留此前成功快照，监测状态明确记为中断；不修改或重启实验。

证据：`results/lime-vectorized-formal-deploy-20261014/monitor-20261014-1532.json`（检查时间15:38:20 +08）。截图工具因生成的data:错误页被Browser Use URL policy拦截，未生成截图。自动检查dcu仍ACTIVE、每小时一次，下一轮继续尝试并报告监测状态。

### 2026-10-14 15:51 连接恢复，正式任务正常推进

用户要求再次检查后，node62网页终端恢复。15:51:45快照中status仅落后4.4秒，正式已运行68分17秒，八个worker均存活且处于stability阶段；原图736/2760、完整五次稳定性664/2760，分片完成8、运行8、排队44、失败0。各活动分片base46/stability37；无controller.exit、controller_failure.json、正式failure.json，尚未出现COMPLETE.json。较上次向用户报告的208张原图增加528；较上次原始JSON快照增加680张原图、664张稳定性、8份已校验分片。

已完成八份VGG16/ImageNet分片平均墙钟2197.331s（36分37秒），原图归因平均7.812s、噪声归因平均7.821s。用已观测VGG时间、其余模型完整40图回归均值和当前八槽排程外推，计算尚需约2.7h，沟通范围2.5–3.5h；其他模型正式耗时与后续负载尚未实测，最终独立验收/下载归档另计。每小时监测继续ACTIVE，本轮未改动或重启实验。

原始可见终端快照：`results/lime-vectorized-formal-deploy-20261014/monitor-recheck-20261014.json`，同前缀PNG；ETA计算记录 `eta-recheck-20261014.json`。注意formal阶段status中的regression_maps是控制器派生计数，不代表新回归；实际回归完成固定720，正式监测只使用base_images/stability_images及分片状态。

### 2026-10-14 16:38 每小时检查：24分片完成，八卡转入DenseNet

16:38:46快照，status仅落后0.21秒，八个worker全部存活；完成24、运行8、排队28、失败0，原图1410/2760、完整五次稳定性1126/2760。较15:51原始快照增加674张原图、462张稳定性、16个已校验分片。20份VGG分片与4份DenseNet121/ImageNet分片已完成；当前八个任务均为DenseNet。没有controller.exit/controller_failure/COMPLETE标记，也没有formal failure.json。

所有24个完成分片均有46条原图效率记录、230条稳定性trace。四份DenseNet121/ImageNet分片平均墙钟1868.242s，原图归因6.559s/张、噪声归因6.571s/次。按实测/尚未运行模型的40debug均值与当前八槽排程估算，还需约1.95h完成计算，沟通范围1.5–2.5h，最终独立校验和归档另计；后续负载变化仍会影响时间。

证据：`results/lime-vectorized-formal-deploy-20261014/monitor-20261014-1636.json`及同前缀PNG；ETA为`eta-20261014-1636.json`。未改动或重启实验，每小时主动汇报继续ACTIVE。

### 2026-10-14 18:00 每小时检查：48分片完成，仅余ResNet50

18:00:18快照，status仅落后1.76秒，八个worker均存活；完成48、运行8、排队4、失败0。原图2454/2760（88.9%），完整五次稳定性2238/2760（81.1%）；较16:38快照增加1044张原图、1112张稳定性、24个完成分片。VGG16和DenseNet121的四个模型/数据集组合全部分片已完成，当前仅余ResNet50。无controller.exit、controller_failure.json、COMPLETE.json或formal failure.json；未改动或重启队列。

48份完成分片的效率记录均为46条、稳定性trace均为230条。已完成8份ResNet50/ImageNet分片平均墙钟1212.271s（20分12秒），原图归因4.263s/张、噪声归因4.271s/次。按当前八槽及剩余4份排队任务、正式实测和ResNet50/VOC的40debug实测估算，约35分钟完成计算，沟通范围30–50分钟；完整独立验收和下载归档另计。此时只完成分片监测，尚未宣布全量验收通过。

证据：`results/lime-vectorized-formal-deploy-20261014/monitor-20261014-1758.json`及同前缀PNG；ETA为`eta-20261014-1758.json`。主监测状态中的evidence_sha256已按本次快照实际内容计算，自动检查继续每小时主动汇报。

### 2026-10-14 19:04 完成、验收和归档

正式在18:35:34完成，controller.exit=0、60/60分片和6/6单元verified/resume_verified。正式含合并/运行后复核墙钟13926.715秒（3小时52分6.7秒）。本地完整复核720回归、2760原图/预测、13800稳定性、各层哈希和均值/样本标准差通过。服务器独立重核125源/1000数据/6权重通过。较18:00增加306张原图、522张完整稳定性、12分片。

完整实文件包43,356,655字节，服务端/本地SHA-256为38510f8eeda54ae38ee77acaf04580666779954ff7565b59d9fbd58438fe1cd3，已通过同标签下载并解压到results/lime-vectorized-formal-complete-20261014。汇总见docs/LIME_DCU_RESULTS_20261013.md及analysis/lime_dcu_full_results_20261014.json。19:10:17收尾完成：#78平台/Docker已停止；#75平台/原Docker容器运行，仅tail/bash，未重启。平台停止未实际停容器的差异已通过网页终端仅停止本轮dev-59d5af49修复。容器退出137/OOMKilled=false是完成后的清理，实验controller.exit仍0。宿主结果与归档保留。任务完成后已删除自动检查dcu，避免空转。

---

# LIME DCU 运行与验收记录

## 当前状态

2026-10-13（Asia/Shanghai）按用户要求接收仓库新增 LIME 实现并部署到 node62。当前是 **高预算比较已完成并通过独立完整性审计；按用户要求暂停后续运行，正式 eval 尚未启动**。六个组合均已完成一图基础归因、五次稳定性及真实续跑验证，随后进入预算比较。用户要求适当保留计算资源，因此已将并发从六个任务降到四个任务；此前已落盘的输出保留。

LIME 来源：`origin/feat/e-lime-local`，提交 `64d2ffde8a1c206d6b5bac9cc8b2c83622e702cb`。接入现有 `feat/dcu-throughput-20261013` 工作区，保留 KernelSHAP 及此前 RISE/Occlusion 适配，没有提交或推送这些本地集成修改。

## 实现检查与验证

- 官方 `lime==0.2.0.1`，quickshift（scikit-image 0.25.2），指定 metadata 真值 target，ImageNet softmax / VOC sigmoid。
- 隐藏超像素还原为传入的归一化黑色 baseline；所有有符号回归系数投影后 min-max 归一化。保持采样 seed 与分割 seed 独立、逐次调用可复现。
- 六份仓库配置仍是 `num_samples=1000`、`budget_frozen=false`；保留正式评测保护，没有把候选预算直接认定为正式预算。
- 本地 LIME 专项 26 项、受影响 runner 19 项，共 45 项通过；实际 DCU 容器再次运行同样的 45 项，全部通过（22.42 秒，5 条既有 Pillow 弃用警告）。
- 新增 `scripts/run_dcu_lime_queue.py` 负责分卡、预检、状态与异常记录；归因实现、预算比较算法、基础/稳定性 runner 均使用仓库代码。调度器完成语法检查，实际运行作为部署验证。

## 资源与并发

- 复用 DAPJobs #78、容器 `dev-59d5af49`，容器配额 56 CPU / 256 GiB / 8 DCU；此前用户允许临时停止的 #75 已再次暂停。
- 实际同时运行 **4 个 LIME 任务，只使用逻辑卡 0、1、2、3**。卡 4–7 不运行本轮 LIME；每个工作进程使用 2 个 CPU/BLAS 线程。
- 每个实际模型进程设置 PyTorch 显存分配上限为单卡 10%（约 6.4 GiB），包括独立预算比较子进程。此限制不代表所有驱动分配，也不限制其他容器。
- 部署时已有 vLLM 服务占用全部八卡，每卡约 43.4 GiB / 64 GiB，采样计算利用率为 0%；未停止或修改该服务。LIME 历史独立试跑峰值约 1.8 GiB，此时每卡仍有约 21 GiB 可用。
- 这是共享服务器运行，耗时必须注明共享负载，不能直接当作独占 DCU 性能数据。若出现显存或竞争异常，先检查并保留证据，不操作他人服务。

原六并发控制器通过 SIGINT 正常进入清理逻辑，退出码 130，已核对全部本轮旧子进程退出。首次真实 smoke 输出保留；当时预算逐图记录均为 0，尚无完整预算图记录。历史控制器、日志、退出码、发布清单和调整原因保存在 `history/six-worker-attempt/`。随后更换的仅为并发默认值与预算子进程显存限制，LIME 实现和实验配置未变；继续使用原目录安全续跑。

## 阶段与数量

1. 资产预检：125 个发布文件、1,000 张冻结图片、metadata、三份 ImageNet 权重、三份 VOC20 checkpoint、八卡可见性及磁盘空间。
2. 六单元真实 smoke：各一张 debug 原图、五次 sigma=0.005 稳定性重复，核验 NPY/provenance、目标类别、指标，并验证基础和稳定性续跑均 `processed=0 / skipped=1`。
3. 六单元预算比较：各自固定 40 张 debug 图片，1000/2000 samples × seed 42/43/44，分割 seed 固定 42；每单元 240 次计时归因、六次预热。写入逐图 map 哈希、跨预算相似度、跨 seed 方差、退化状态、加权 R² 和耗时。
4. 预算结束写 `status.state=budget_ready`，不会自动跳过审阅。检查六组完整性、低相似图片、退化与方差后，形成明确的审阅 JSON；仅在结论支持时选取已测试预算，绑定六个 summary SHA-256，冻结本次运行配置，再启动正式评测。用户已授权“检查没问题后部署运行”，无需重复请求普通运行许可；若证据表明存在未解决问题，先报告问题，不虚构通过标准。
5. 正式阶段：3 模型 × 2 数据集 × 460 eval 图片，五次稳定性重复；2,760 个原图组合、16,560 次归因。保留真值 target、21 点 raw MoRF、FP32、既有稳定性与种子协议、NPY/provenance、预测和续跑检查。

在 2000 samples 下，历史容量试跑对应四卡理想均摊约 9 小时；这里按六个完整单元排队，会有尾部负载不均衡。初始正式阶段预估曾为 10–12 小时；本轮 40 图实测修正后，若固定 2000 samples，四并发完整单元队列约需 **14–16 小时**（仅归因部分约 13.51 小时）。尚未批准该预算；更高预算正式耗时需按追加比较的实测重新估计。共享服务负载也是不确定因素。

## 目录与发布标识

- 容器发布：`/workspace/xai-lime-20261013`
- 宿主发布：`/opt/dapjobs-dev/cas_mfwzan/dev-59d5af49/xai-lime-20261013`
- 运行目录：发布目录下 `results/dcu-lime-20261013`
- 本地部署证据：`results/lime-deploy-20261013/`
- 本地监测快照：`results/dcu-lime-monitor-state.json`
- 四并发发布清单 SHA-256：`c1d82ec2b6f5a179db32fbb3042d07853f01bea88b7fb8ab4f10777169eb0526`
- 初始完整代码包 SHA-256：`c29aec114fca163cde58bcf985013e24bfdfcc387fa963503a1f459883064dcd`
- 四并发调度补丁包 SHA-256：`0f80f3c2fb51e29eaad050370bebf428f2c6d488a3c4833284bcc7562195c529`

数据和权重通过只读使用的目录链接复用此前部署材料。前五种方法的发布和正式结果仍在原 `xai-formal-five-20261013` 中，本轮没有修改。

## 检查、审阅与后续正式启动

必须通过既有 DAPJobs 网页终端操作，不改用 SSH/直接平台 API：
`http://42.228.13.157:40527/node.html?id=1&host=node62`。

原 1000/2000 预算启动命令（已完成，保留溯源，不重复执行）：

```sh
docker exec -d dev-59d5af49 bash -lc 'cd /workspace/xai-lime-20261013 && PYTHONPATH=.lime-deps:.deps:. PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u scripts/run_dcu_lime_queue.py --phase budget --devices 0,1,2,3 > results/dcu-lime-20261013/budget_controller.log 2>&1; echo $? > results/dcu-lime-20261013/budget_controller.exit'
```

读取 `status.json`、`budget_controller.exit`、`budget_controller_failure.json`、单元 `failure.json` 及必要日志。历史目录中的退出码 130 是已处理的资源调整，不当作新失败重复告警。预算工作目录 `budget/lime-<model>-<dataset>/`；单图迁移检查目录 `gate/`。控制器约每五秒更新状态，`budget_images` 记录完整落盘图片数。

原控制器的审阅文件格式如下，仅供溯源；它读取原 `budget/` 且只允许原有预算。追加高预算审阅完成前不得使用此示例启动正式任务，也不能单独填 8000 绕过绑定校验：

```json
{
  "status": "accepted_for_formal",
  "num_samples": 2000,
  "reason": "实际逐图证据支持的预算选择理由与局限",
  "units": {
    "lime-resnet50-imagenet": {"summary_sha256": "实际哈希"}
  }
}
```

`units` 必须包含全部六个单元。保存为 `results/dcu-lime-20261013/budget_review.json`，保留本地副本。正式启动使用相同环境、同一根目录及 `--phase formal --devices 0,1,2,3 --review results/dcu-lime-20261013/budget_review.json`，日志和退出码分别为 `formal_controller.log`、`formal_controller.exit`。控制器会重新核验所有预算 map、计划及 summary，再生成带审阅文件哈希的正式配置；不得修改已运行正式单元的配置或参数。

正式成功要求 `COMPLETE.json`、`status complete`、`formal_controller.exit=0`、六个单元 `completion.json` 与 `resume_verified`。合并无遗漏/重复：`per_image.csv` 11,040 行、`units.csv` 24 行、`stability_trace.csv` 13,800 行；2,760 份原图 NPY 和预测。还要复核发布/权重哈希、图像 ID、有限数值、稳定性种子、均值/有效率及输出哈希。仅进度达到 460 或进程消失不算验收通过。

沿用每小时检查、正常时安静的偏好；预算需要审阅时由后续检查继续完成，发现实际问题再通知。完成后形成 `docs/LIME_DCU_RESULTS_20261013.md`，通过浏览器同标签下载并核对 SHA-256；工具超时先核对已落盘文件，不重复启动下载。最后暂停监测、按部署前状态恢复 #75，保留实例与所有结果。释放 #78 前同时核对平台与 Docker/进程实际状态，不只依赖页面状态。

## 最新已核实快照

2026-10-13 20:42:22（Asia/Shanghai）：预检通过，六组真实 smoke 与续跑均通过；4 个任务运行、2 个排队、0 个失败。ResNet50 / DenseNet121 的两个数据集各已完成 2/40 张预算记录，两个 VGG16 单元排队。当前无活动 failure.json，正式 eval 尚未开始。每小时自动检查 `dcu` 已恢复为 ACTIVE，任务提示已改为本轮 LIME 预算审阅、四并发正式运行和最终验收；前五种方法不重跑。

## 21:38 预算完成后的审阅

六组各 40 张预算比较全部完成，控制器退出 0，四并发阶段墙钟 58 分 24.8 秒。原始证据已下载并核对 SHA-256；本地独立复核 1,440 张 NPY 和 2,160 组配对、方差、ID/target、发布及六份权重溯源，完整性通过。

2000 samples 在 DenseNet121 / VGG16 的 ImageNet 上跨 seed 平均 Spearman 分别为 0.7756 / 0.7766，VGG16 最差为 -0.2227，因此没有冻结预算或启动正式 eval。四个模型/图片组合的 4000/8000 样本诊断已在卡 0、1 完成并校验，显示采样增多可改善种子一致性，但低局部 R² 仍存在；用户随后明确选择先补完整高预算 40 图检查。完整审阅见 [预算审阅记录](LIME_DCU_BUDGET_REVIEW_20261013.md)。

## 22:06 追加高预算完整比较

用户明确选择先补高预算 40 图检查，再决定正式预算。2026-10-13 22:06:27（Asia/Shanghai）启动新队列，22:08:52 核对四个实际模型子进程已进入采样计算，卡号为 0–3，CPU/BLAS 线程为 2，启动参数均设置单卡 PyTorch 显存分配上限 10%。无新 failure.json，控制器每五秒更新。首次完整图记录要等各预算/seed 的九次预热完成后产生。

- 六组合各 40 张冻结 debug，2000/4000/8000 samples × 42/43/44 三个采样 seed，分割 seed 42。
- 每组 360 张计时 map + 九次预热，共 2160 张计时 map、54 次预热；含预热共 10,332,000 个遮挡样本前向。
- 2000 档重新运行作为共同对照，旧预算和诊断全部保留。
- 按实测单图耗时做四工作槽排程，预估约 4–5 小时；这是共享 vLLM 负载下的外推，需以实际结束时间为准。
- 先运行耗时较长的 VGG16 / ImageNet、DenseNet121 / VOC、VGG16 / VOC、DenseNet121 / ImageNet；ResNet50 两组排队。
- 启动前再次校验发布源 125 项、1,000 图、metadata、六份权重、八卡可见性和磁盘空间；`high_budget/prepare.exit=0`、`preflight.status=passed`。

独立调度器位于容器 `/workspace/run_lime_high_budget_queue.py`（本地 `scripts/run_lime_high_budget_queue.py`），SHA-256 `152c4a3fe5d6af03e17619dd17f7715c9234a338eb7964b51920815df9ef3844`。原冻结发布源未改动，发布清单 SHA-256 仍为 `c1d82ec2b6f5a179db32fbb3042d07853f01bea88b7fb8ab4f10777169eb0526`。

当前检查入口改为运行根目录下：

- `high_budget/status.json`：state/running/pending/failed/completed；运行中每个任务记录 `budget_images`，完成时 state 为 `high_budget_ready`。
- `high_budget/controller.log`、`controller.exit`、`controller_failure.json`。
- `high_budget/lime-<model>-<dataset>/{plan.json,records/,maps/,worker.log,completion.json,failure.json}`；实际 map 相对路径以 records 的 `map_path` 为准。
- `high_budget/controller_manifest.json` 绑定六个计划、脚本及发布哈希、四卡上限和候选预算。每组完成后 `completion.status=verified_integrity_only` 只代表完整性校验，不等于预算审阅通过。

启动命令如下（已执行，不重复执行）：

```sh
docker exec -d dev-59d5af49 bash -lc 'cd /workspace/xai-lime-20261013 && PYTHONPATH=.lime-deps:.deps:. PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u /workspace/run_lime_high_budget_queue.py --release /workspace/xai-lime-20261013 --output /workspace/xai-lime-20261013/results/dcu-lime-20261013/high_budget > results/dcu-lime-20261013/high_budget/controller.log 2>&1; echo $? > results/dcu-lime-20261013/high_budget/controller.exit'
```

现有每小时自动检查已切换跟进追加阶段。待全部完成，下载并独立复核全部 2160 maps、九种预算/seed 设置与逐图配对/方差，再审阅全组分布、低分尾部、weighted R² 与真实时间。不能只凭小范围改善或平均分上升宣布收敛。

**后续正式控制器注意事项：**旧 `scripts/run_dcu_lime_queue.py` 绑定原 `budget/` 和 1000/2000 候选。若新证据支持 4000/8000，需在冻结发布源以外适配独立正式控制器，绑定 `high_budget/` 六个 summary 与计划哈希，验证所选采样预算实际完成，再生成本轮正式配置；不能只切换 `budget_frozen` 或套用旧审阅 JSON。正式 eval 当前仍未启动。

22:10:41 复核：四任务均已结束预热并开始写入归因图，分别已有 2/3/3/3 张 map，控制器状态新鲜（2.13 秒），无失败。单图需九种预算/seed 全部落盘后才计入 `budget_images`，故此时仍为 0/40。

资源状态补充：初次 LIME 部署时 vLLM 每卡约 43 GiB 的占用是历史快照。22:10 左右只读检查发现其原容器名 `qf-r436-3f63b0-serve1` 已不存在，当前 Docker 列表仅见开发容器；容器内 rocm-smi 显存快照为卡 0–3 各 2%–3%、卡 4–7 为 0%。本轮没有停止、修改或删除其他服务，无法从该快照判断服务移除时间/原因，也不能据此宣称独占服务器。仍严格维持四并发，后续耗时应注明负载会变化。

22:11:41 再次确认四任务各已写入 6 张 map，均较上一快照推进，0 失败、2 排队；首张图九种设置尚未全部完成。启动截图与快照保存在 `results/lime-deploy-20261013/high-budget-launch.*` 和 `high-budget-launch-state.json`。

## 2026-10-14 完整高预算验收与用户暂停

高预算控制器于 2026-10-14 02:06:32（Asia/Shanghai）结束，`high_budget_ready`、退出码 0，无活动失败；真实墙钟 14,405.554 秒（4 小时 5.554 秒）。10:36 恢复本地监测时网页终端已断开，刷新成功后读到最终状态；此前最后一次本地快照为 00:43，期间没有伪称持续在线监测。正式任务没有自动启动。

服务器结束后再核验 125 个发布文件、1,000 图与 metadata、六份权重，`high_budget/postrun_integrity/check.exit=0`。下载 `lime_high_budget_20261014.tar.gz` 共 10,353,464 字节，本地/服务器 SHA-256 `88645ef4b66908c54e4b599805c9868261e160739cdcb5bff11e764781621596`。独立审计 `scripts/audit_lime_high_budget.py` 通过 2160 NPY、40唯一ID/targets、3600配对/方差复算、三类CSV摘要和全部溯源检查；本轮720张2000档归因图与旧档逐字节相同。证据位于 `results/lime-high-budget-evidence-20261014/`；审计 `analysis/lime_dcu_high_budget_audit_20261014.json`。

8000 samples 已形成带局限的固定预算审阅，绑定六单元 summary/plan 哈希及独立审计；这不表示完全收敛或每张图都具有高局部拟合质量。文件 `results/lime-deploy-20261013/budget_review_8000.json`，SHA-256 `f7438b43dfee29bc5d0b86bb2cca3a9af7cc434a3434670ccda6a1178d0521df`。独立正式调度入口 `scripts/run_lime_reviewed_formal.py` 部署于 `/workspace/run_lime_reviewed_formal.py`，SHA-256 `b39c721729bb10a26ef5605df53945be8363226b11b7505e0a9425a41826c403`；原发布源保持不变。

用户先明确放宽至最多8卡，六组合按现有单元粒度可六并发。随后又明确要求“先别急着跑”，先分析LIME为何慢。六组8000档短验证已经收到 SIGINT 并清理所有本轮子进程，退出130、KeyboardInterrupt为主动暂停，不能当作新故障。实际模型进程余数0，`formal_start.json`不存在，正式 eval未启动。自动检查 `dcu` 已设 PAUSED；没有新的明确继续指令，不得启动或恢复队列。

当前短验证中止材料保留在 `formal_smoke/`、`formal_smoke_controller.log/.exit`、`smoke_reviewed_controller_failure.json`，此前所有高预算证据完整保留。预算审阅不等于执行授权；用户暂停要求优先。若后续用户同意恢复，先核对当前状态并保留本次主动中止记录，不覆盖冻结配置或重复启动工作队列。

## 2026-10-14 优化探索授权与限定对照

用户随后要求先探索优化，若仍不可行再考虑减少 samples 或换方法，并要求 Browser Use 使用 6.1-sol 子代理。当前仅授权本次优化原型和有规模上限的性能/等价性对照；正式 eval 与自动检查继续暂停。

原型 `scripts/lime_vectorized.py` 保留官方 LIME 的随机采样序列、分割、距离、核函数和 Ridge，只以设备上的超像素索引批量构造邻域图片。它保留原适配器的 RGB 截断/重新归一化与输入布局，未接入正式 runner，也未替换冻结发布的 `experiments/attribution/lime.py`。本地原有与新增测试合计 40 项通过，包含随机状态不污染、扰动输入逐元素相同、归因图/R²相同、两种输出激活、空间基线和非方形输入；这是 CPU 软件检查，DCU 数值和性能另测。

服务器独立脚本目录 `/workspace/lime-optimization-20261014-v1/`，原型 SHA-256 `c48e5b6b18298cd4ebf5736d4fda46c265011662f29c9a3207bb823f159ab314`；测速入口 `benchmark_lime_vectorized.py` SHA-256 `108c7af116c7950fefdbcd10c7f6754509834a2d3365d8baa33cb866fbeeec3c`。本次输出只新增 `results/dcu-lime-20261013/optimization_v1/`，脚本拒绝覆盖已有输出目录，运行前后核对冻结源/数据/六份权重。

六组合各三张冻结 debug，采样预算2000/4000/8000；原版batch32与原型batch32/64/128配对，另各组一张图按既有协议做五次2000档噪声归因配对。至多276次计时归因、1,128,000个计时模型输入，另外6144个预热输入；最多六个worker，逻辑卡0–5，单模型显存分配上限10%、CPU/BLAS两线程。较大batch若OOM，只记录并停用对应可选变体，不改变原版及同batch32对照。预检与启动通过Browser Use完成，确认六组running、无failed、formal_started=false；不能把小规模probe当成正式结果或完整40图门禁。

限定对照已全部完成，从调度至最后worker结束395.89秒。六组各46条记录、32组比较；独立复核276maps/192配对及54张旧高预算锚点通过，batch32的84组全部逐元素一致，全部预算的提速为3.79–7.36倍。无OOM。原控制器在全部worker完成后因缺少postrun目录退出1，独立收尾脚本只补目录并核验冻结源、数据、权重及结果，退出0；原始退出码与日志保留。新包已按同标签download流程落盘，文件/哈希/详细时间表见预算审阅记录末节。网页终端确认无本轮benchmark/finalize进程，正式eval未启动，自动检查保持PAUSED。

本地证据 `results/lime-optimization-20261014/`，审计 `analysis/lime_optimization_20261014.json`，规划外推 `analysis/lime_optimization_estimate_20261014.json`。优化32的六并发全量纯归因外推为2000档1.83h、4000档3.20h、8000档6.04h；这些不是已完成正式运行的墙钟。下一步若继续，需要形成绑定优化源码的新发布并做完整debug回归与runner/续跑校验，不能让原8000正式控制器悄悄使用不同实现。
