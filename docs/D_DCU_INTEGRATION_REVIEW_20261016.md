# D：PR #17 集成复核与 LIME 门禁修复

日期：2026-10-16。工作分支 `feat/kernelshap-analysis`；不修改 `main`。

## 固定对象与合并处理

- 原 D 统计交付：`9c474599332ae9164323fc9758a7d412451c7b39`。
- PR #17 来源：`feat/dcu-throughput-20261013`，固定提交
  `7eb0a316b097073fe43da1f95e2ec9c8897c356a`。
- 在 D 分支集成该提交，保留 A 的 DCU/LIME 实现、配置、队列和证据，以及 D 的统计源码、
  四图、结果 JSON、配对表和历史审查。不向对方分支推送，也不改写历史。
- `docs/D_DELIVERY_INDEX.md` 的文本冲突手工解决：保留双方交接链接，并以 10 月 15 日
  D 已完成表格统计的事实替代过时的“D 尚未接收/统计”。同时修正自动合并后 README 中的旧状态。
- `outputs/`、冻结原始结果、ZIP、NPY、权重、协议、预算和测量值不作本次改动。

## 复核范围

复核普通 LIME 的 metadata target、softmax/sigmoid、局部 RNG、baseline、符号系数投影；
检查 VectorizedLIME 的采样矩阵与输入等价路径、分片 ID 选择及按原图重算标准差。
运行相应自动化回归，包括合成输入上的真实参考 LIME/向量化逐值对照、runner/稳定性/续跑链路。
这不是八卡部署环境、厂商 torch 或归档中 720 张真实 debug map 的再次独立运行。
队列脚本作为固定运行证据保留，没有执行其模型队列、下载或远端操作。

## 修复 P1：不能只看已有 map 的 SHA 就跳过

新门禁采用 schema 2，在加载模型/预热或改写 plan、记录、CSV、summary 前，验证所有已存在记录：

- metadata target、image ID、计划身份；预算×seed 完整唯一组合；
- 固定到当前图像/预算/seed 的 map 路径、文件大小、SHA、float32、形状、有限值和 [0,1]；
- num_samples、超像素数、非负有限计时、有限 weighted R²（负值允许）；
- 从 NPY 重算所有预算 Spearman/RMSE、跨 seed 样本方差与 seed-pair 比较，拒绝不一致记录；
- 摘要之前再次检查全 40 图、所有输入身份与全部记录，再从完整 runs 导出 CSV/汇总。

完整续跑无需加载模型或执行预热：`processed=0, skipped=40, warmup_calls=0`。
两预算×三 seeds 对应 240 runs、120 预算比较、80 条 variance、240 seed pairs，
计数按配置动态计算，不对其他预算硬编码。部分续跑只重算缺少完整记录的图像。

损坏记录会报错而非自动“修补”旧统计；不会把缺 runs 的 JSON 当完整图像计数。
输出目录有 OS 管理的非阻塞独占锁，同目录并发启动被拒绝，异常退出释放锁，
不依赖删除旧 PID 文件或删除结果目录恢复。

## 修复 P2：输入与执行身份

新 plan 和逐图记录绑定源图片字节 SHA、实际预处理 CHW float32 张量的 dtype/shape/字节 SHA。
计算前后再次核对。计划同时绑定明确列出的执行源码（LF 规范化 SHA）、依赖版本、
机器标识哈希、平台/Python、device、CUDA/HIP、设备信息和线程设置；代码或机器变化不能
默默复用同一目录的计时与 map。

这些字段是软件生成的可审计记录，不是签名证明或完整供应链快照，也不覆盖所有驱动/共享
负载变化。不能由哈希一致反推历史实际执行一定正确，不能据此制造硬件加速比。

## 兼容性与复现

- schema 1 的旧门禁一律只读，不自动迁移/回填来源；继续使用独立校验工具核对旧包，
  其报告仍不批准正式预算、不证明原生成时来源。
- schema 2 必须选择新目录。先 `--plan-only` 检查资产与预算，再按既定首图成本确认执行。
  本次没有执行真实 40 图门禁，也没有开启 460 张模型归因。
- `analysis.validate_lime_budget_gate` 继续支持旧包；对新格式额外独立核对 plan/record
  输入绑定、当前源图片/受控预处理张量和动态 summary 计数。它不执行包内代码。
- 旧正式 DCU 批次对应原代码/配置归档；本次修复不会追认旧门禁缺口已经不存在，
  也不会否定已有外部验收并经 D 接收的表格统计。

复验命令（项目环境安装 `requirements.txt`；本次使用既有隔离 LIME 依赖，不升级原 `.venv`）：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q --basetemp=results/.pytest_dcu_integration_recheck
.\.venv\Scripts\python.exe -m pip check
```

测试使用显式合成 fixture，不是新的正式结果。最终测试数量、依赖及提交证据见
[交付入口](D_DELIVERY_INDEX.md)。给 E/A 的整合清单见 [D→E 交接](D_TO_E_HANDOFF_20261016.md)。

## 最终复验结果

- 最终代码完整测试：**227 passed，2 条既有 `torch.load` FutureWarning，91.87 s**；
  此前集成后的首轮同为 227 passed。没有 failed、skipped 或 xfailed 冒充通过。
- 新门禁 CLI/完整与部分续跑/损坏记录/身份变更/输出锁回归共 31 项；独立门禁校验器
  29 项（原 25 项 + 新 schema 2 的 4 项）。另包含参考 LIME、向量化等价、分片链路等测试。
- 项目 Python 3.10.20，torch 2.5.1+cu121、NumPy 2.2.6、pandas 2.3.3、SciPy 1.15.3；
  LIME 0.2.0.1、scikit-image 0.25.2 来自既有 ignored 隔离依赖。`pip check` 无冲突；
  没有升级原 `.venv`，没有运行真实 GPU 归因。
- 原始轻量 ZIP SHA256 `14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc`
  再验通过：720 份文件、721 个 checksum 项、36 单元、90 配置、66,240 指标、82,800 trace。
- 相对原 D 统计交付，`analysis/dcu_20261015/`、metadata 和稳定性协议无改动。
  派生统计 JSON 的 LF 规范化 SHA256 仍为
  `6c098e27cc36e98c46172845ca5d5c5f7477cb0df29b23325dbe27d7ecbbe473`。
- 新交付/交接入口的本地链接可解析，冲突标记已清理；提交前检查不包含
  `outputs/`、`results/`、原始 CSV/ZIP/NPY 或 checkpoint。
