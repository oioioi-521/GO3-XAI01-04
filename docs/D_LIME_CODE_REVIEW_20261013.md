# D 对 E 的 LIME 代码与预算门禁审查

2026-10-16 后续：固定提交 `64d2ffd` 的下述发现按原样保留。D 在集成 PR #17 后已修复
通用门禁的 P1/P2 并增加 schema 2 续跑回归，见
[修复/兼容性记录](D_DCU_INTEGRATION_REVIEW_20261016.md)。
该修复不回写或追认旧门禁来源，不代表重新运行了真实预算/正式实验；
新 DCU 表格统计另见 [10 月 15 日统计报告](DCU_STATISTICAL_ANALYSIS_20261015.md)。

审查日期：2026-10-13。结论：**适配器与现有工程测试通过；预算门禁的续跑验收需要修复，暂不批准冻结预算或启动正式 eval。**

## 固定审查对象与范围

- E 分支 `feat/e-lime-local`，提交 [`64d2ffde8a1c206d6b5bac9cc8b2c83622e702cb`](https://github.com/oioioi-521/GO3-XAI01-04/commit/64d2ffde8a1c206d6b5bac9cc8b2c83622e702cb)。
- 基线 `integrate/a-voc-eval`：`00d73b4d10f9fbea32656951be05ddc336aab893`；E 在该基线上增加一个提交。
- 在 D 工作区的 ignored `results/e_lime_review_20261013/` 中导出独立副本；没有切换 D 分支、合并 E、修改 E 的源代码或运行真实 GPU 门禁。
- Windows Git 默认导出可能转换 CRLF。最终复核使用 `git -c core.autocrlf=false archive` 的 `source_lf/`，适配器、门禁、runner、测试、七份 LIME YAML 与 Git blob 逐字节一致。
- 冻结稳定性协议未改动，Git blob 及 `source_lf/` 字节 SHA-256 均为 `324315f1ecf62a807ec91e77b2cba1ea53593e9a7ad98a2a90cce09a9e32dcdf`。普通 CRLF 工作副本的字节哈希不能冒充这个冻结哈希。

源码 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `experiments/attribution/lime.py` | `f9b8b900c1f0ea86274d8b7def5ca6277cd2987783770e0c4bc1d4e8dc574b9a` |
| `experiments/lime_budget_gate.py` | `26d388d3425fbca5727f4a3ecc4728a766f1230451c4170eed1e4261ac5079e0` |
| `experiments/run_unit.py` | `4342a917f4c5ce01588c58121eba8bc3b2ccdb9d9fc18da63ef388c8fc63f781` |
| `tests/test_lime.py` | `9a6779e6368e9549c2757f11916b9ea3e90fa918fa2a57bf073a1be7cce7e8b8` |

## 已独立验证的工程内容

- 接受 metadata 真值 target，官方 LIME 调用使用 `labels=(target,)`、`top_labels=None`，不自动解释预测类别。
- ImageNet 使用 softmax，VOC20 使用 sigmoid；六份候选配置与模型输出语义一致。
- quickshift 固定独立分割 RNG，每次 attribute 新建采样 RNG；warm-up、调用顺序与续跑不消耗后续图像的采样流。
- 所有超像素的有符号 Ridge 系数投影后 min-max，输出有限 CPU float32 `(H,W)`；不是绝对系数图，也不是仅正贡献区域。
- 六份正式候选 YAML 均为 eval、完整清单、21 点 raw MoRF、一次 warm-up、float32 保存和 resume；仍为 1000 samples、`budget_frozen: false`。未冻结预算的直接 eval 会被拒绝。
- E 原有 LIME 专项：**26 passed**。最终 LF 副本全仓：**94 passed，2 条既有 torch.load FutureWarning**。
- 审查依赖通过 `pip check`：`No broken requirements found`。LIME 相关依赖仅安装到 ignored `results/.../deps/`，未更新 D 的 `.venv` 或现有实验包。
- 两个独立 CPU 进程、两个隔离目录执行 E 的合成 tiny-model smoke，各首次 `1 processed / 0 skipped`，立即续跑 `0 / 1`；两个 float32 map 字节 SHA-256 相同：`1e1115c8b1080c8e536dc2fa6762d7f57a9f114c1f8b939059603a4c1ac2a45d`。仅验证工程链路，不证明真实模型性能或预算门禁通过。

审查环境：Python 3.10，torch `2.5.1+cu121`、torchvision `0.20.1+cu121`、lime `0.2.0.1`、scikit-image `0.25.2`、scikit-learn `1.7.2`、NumPy `2.2.6`、SciPy `1.15.3`、pandas `2.3.3`、pytest `8.4.2`。工程 smoke 与附加探针均指定 CPU，不是 E 的 GPU 环境独立复跑。

D 分支收尾复验：**108 passed，2 条既有 warning，28.14 s**，`pip check` 无冲突。
`pytest.ini` 保留默认目录排除并新增 results/outputs 排除，避免将隔离副本和依赖包的测试误当本项目测试；没有删改这些目录。
证据保存在 `results/e_lime_review_20261013/d_branch_tests.log`，`git diff --check` 通过。

## 需要 E 修复的发现

### P1：续跑只检查已列出的 map，不检查完整记录与统计一致性

位置：[门禁第 152–161 行](https://github.com/oioioi-521/GO3-XAI01-04/blob/64d2ffde8a1c206d6b5bac9cc8b2c83622e702cb/experiments/lime_budget_gate.py#L152)，以及第 200–222 行的汇总。

当前只核对 `config_hash`、`image_id` 和 `runs` 中已有 map 的 SHA，然后直接跳过该图。没有验证 runs 是否完整覆盖预算 × seeds、target 是否匹配 metadata，也没有从 maps 重算已保存的 comparisons/variance。

在合成 fixture 中先产生完整 40 图、240 runs、120 预算比较，再作两种互相独立的损坏探针：

1. 清空一张记录的 runs/comparisons/variance，map 文件不动。续跑不拒绝、也不补算，仍写 `n_images=40`，实际只有 **234 runs、117 预算比较**。
2. 将一张记录的 target 从 1 改为 0，将一个 Spearman 改为 -123，map SHA 和计划不变。续跑仍接受，并把非法比较值纳入均值（完整 fixture 的均值由 1 变成约 -0.033333）。

这不证明 E 已有真实记录损坏，但证明“40 条 JSON 文件”不能自动视为完整门禁。原子写入也不能替代读取时的内容验收。

修复要求：验证每图 target、预算/seed 唯一完整组合、有限非负 time/RMSE、有限 R²（允许负值）、Spearman 在 [-1,1] 内及 map shape/dtype/range；从通过 SHA 核验的 maps 重算预算比较和跨 seed 方差，拒绝不一致记录。输出最终摘要前，核验 40 个唯一 ID、240 runs、120 预算比较、80 条 budget variance、每预算 120 个 seed pairs。该计数适用于两预算 × 三 seeds；其他设置应由实际参数推导，不硬编码。

### P2：门禁计划身份没有绑定输入图片字节

位置：[门禁第 68–85 行](https://github.com/oioioi-521/GO3-XAI01-04/blob/64d2ffde8a1c206d6b5bac9cc8b2c83622e702cb/experiments/lime_budget_gate.py#L68)。

计划记录 metadata、权重上下文、配置、版本、ID，却只对图片做 `is_file()`；未记录图片字节 SHA。复现探针使用固定 metadata/权重上下文，替换同一路径的输入字节后，计划 `config_hash` 不变。中断期间图片被替换时，续跑可能混合旧 map 与新输入，而不报身份变化。

修复要求：将每图源文件字节 SHA 和实际预处理输入 SHA 写入计划/逐图记录，在新计算和续跑时核对。门禁包还应绑定运行代码提交和实际机器/GPU，以避免不同源码或不同机器计时混入同一回执；本次未用真实 GPU 复现跨机混用。

## 回归探针与复现

D 新增 [`analysis/lime_review_probe_tests.py`](../analysis/lime_review_probe_tests.py)，不会被正常 `pytest` 的默认命名规则收集；仅手动针对 E 的隔离副本运行。

固定提交上的结果为 **1 passed、3 strict xfailed**：正常完整续跑成功；缺记录、篡改 target/统计、输入字节变更的三个期望拒绝检查均失败。**xfailed 是已知缺陷的复现，不是三项验收通过**；E 修复后会出现 XPASS(strict)，应更新标记并使正常断言通过。

本机复现命令（从仓库根目录，已有归档与隔离依赖）：

```powershell
$reviewRoot = 'C:\Users\Administrator\Desktop\GO3-XAI01-04'
$env:LIME_REVIEW_SOURCE = "$reviewRoot\results\e_lime_review_20261013\source_lf"
$env:PYTHONPATH = "$reviewRoot\results\e_lime_review_20261013\deps"
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
Push-Location $env:LIME_REVIEW_SOURCE
try {
  & "$reviewRoot\.venv\Scripts\python.exe" -m pytest -q --basetemp=results/d_review_repeat_tmp
  & "$reviewRoot\.venv\Scripts\python.exe" -m pytest -q "$reviewRoot\analysis\lime_review_probe_tests.py" --basetemp=results/d_review_repeat_probe_tmp -rx
} finally { Pop-Location }
```

本地证据：`results/e_lime_review_20261013/all_tests_lf.log`、`review_probes_final.log`、`lime_tests.log`、`synthetic_a.log`、`synthetic_b.log`；合成 map/log 存在 `source/results/d_review_synthetic_{a,b}/`。这些输出不入 Git，必要时受控交接。

## 正式预算与结果仍待接收

E 的 [2026-10-13 进度](https://github.com/oioioi-521/GO3-XAI01-04/blob/64d2ffde8a1c206d6b5bac9cc8b2c83622e702cb/docs/E_LIME_PROGRESS_20261013.md)报告：服务器三个门禁完成 40 张，ResNet50/VOC 12/40、VGG16/ImageNet 8/40、VGG16/VOC 8/40；六个 eval 单图 smoke 和六个 debug pilot 有回执，**六组 460 张正式 eval 与正式稳定性未启动**。这些真实运行描述仍是 E 的报告，D 本次没有服务器原始回执，不作独立认证。

下一次交接需要：

1. 修复上述发现后的固定代码提交、六份 YAML 及变更说明；记录候选预算，暂不要设 `budget_frozen: true`。
2. 新服务器及旧 4080 门禁分别打包 plan、40 图 records、maps SHA/原始 float32 maps、comparisons、variance、timings、summary、failures 和运行日志；带环境、机器/GPU、代码提交及配置/资产 SHA。旧耗时异常需说明计时分母、预热、暂停/重试/恢复范围。
3. 三个中断单元需完成真正的 40 图门禁；不能用 1 图 eval smoke 或低预算 pilot 替代。D 收到原始包后独立重算，再与 E 确认预算是否足够。
4. 审查通过后才冻结六份正式预算，并使用无来源混杂的隔离输出目录启动 460 张 eval，再按 Issue #8 补正式 RGB-noise 稳定性。

既有 [VOC 8173 中期定性面板](D_LIME_MIDTERM_RECEIPT.md)的受限接收不受本次新发现否定；E 新上传的 synthetic 示例也不能替代该真实图面板或成为方法排名证据。

本次完成的是 D 的固定提交代码审查和独立工程复测，**不是 E 正式实验验收或全组 36 单元统计**。没有替 E 修改分支、冻结预算、发送消息、发布 GitHub review/comment 或合并代码。
