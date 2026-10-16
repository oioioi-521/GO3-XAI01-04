# E / onlysonder：LIME 本地实现与验收交接

记录日期：2026-10-02（Asia/Shanghai）。仅本地修改、验证；未 commit、push、创建 PR、评论或合并。

## 基线与最新远端状态

- 本次工作区初始为空；重新克隆到 `work/GO3-XAI01-04`，没有沿用旧工作区。仓库及上级工作目录未发现 `AGENTS.md`。
- 读取 README、全组工作指导、TEAM_AND_ROADMAP、数据版本、VOC checkpoint manifest、统一 runner/interface、稳定性说明及 Issue #11/#8 的实时 GitHub API。
- `integrate/a-voc-eval` 最新 SHA：`00d73b4d10f9fbea32656951be05ddc336aab893`。该分支含 PR #10 的稳定性补跑、安全续跑、模型/数据和 raw MoRF 接口，符合 Issue #11 指定基线。创建本地 `feat/e-lime-local`；main 并非本任务指定的集成基线。
- [Issue #11](https://github.com/oioioi-521/GO3-XAI01-04/issues/11) 仍 open，assignee 为 `onlysonder`，无评论、无可验证远端 LIME 实现。
- [Issue #8](https://github.com/oioioi-521/GO3-XAI01-04/issues/8) 最新正文确认协议已冻结；[PR #10](https://github.com/oioioi-521/GO3-XAI01-04/pull/10) 已于 2026-09-30 合入指定基线。基线 `docs/STABILITY_PROTOCOL.md` 的“候选”状态文字已过时，本次未改动其他成员的文档。
- 稳定性 YAML SHA-256 实测为 `324315f1ecf62a807ec91e77b2cba1ea53593e9a7ad98a2a90cce09a9e32dcdf`，与 Issue #8 冻结值相同。PR #16/#14/#15/#13 仍 open；未合入其他成员未合并分支。未重复评论 PR #6。

## 实现与约定

`experiments/attribution/lime.py` 用官方 `lime==0.2.0.1` 的 `LimeImageExplainer`；适配其已发布 API。官方 master 额外提供的 `progress_bar` 参数在该版本不存在，不能使用。参考：[官方 image LIME 实现](https://github.com/marcotcr/lime/blob/master/lime/lime_image.py)、[官方回归实现](https://github.com/marcotcr/lime/blob/master/lime/lime_base.py)、[scikit-image quickshift API](https://scikit-image.org/docs/0.25.x/api/skimage.segmentation.html#skimage.segmentation.quickshift)。实际已安装源码与依赖版本已验证。

- 接受单张 `(3,H,W)` 或 `(1,3,H,W)`，解释调用者提供的 metadata 真值 target，不使用 `top_labels` 自动选择预测类别。
- 反归一化 RGB `[0,1]` 上 quickshift：`kernel_size=4, max_dist=200, ratio=0.2, sigma=0, convert2lab=true, channel_axis=-1, rng=42`。使用 `scikit-image==0.25.2` 的显式 RNG API；输出标签重排为连续整数。实际 superpixel 数由图像决定，门禁逐次记录。
- 采样为官方 Bernoulli(0.5) 开/关超像素，第一行为全可见。每次 attribute 新建局部采样 RNG，seed=42；分割 seed 独立固定为 42。调用顺序、warm-up、resume 不消耗后续图像的 RNG；预算门禁改采样 seed 为 42/43/44，分割不变。
- 隐藏区域使用传入 normalized baseline，默认归一化 RGB 黑色；支持空间变化的 baseline。通过内部负值 sentinel 标记隐藏像素，送入模型前还原为 baseline；sentinel 不进入模型。推理重新归一化，ImageNet softmax / VOC sigmoid 明确写入 attribution YAML，并与模型设置核对。
- 余弦距离，核 `sqrt(exp(-d²/0.25²))`；sample-weighted Ridge，`alpha=1, fit_intercept=true, solver=cholesky`。保留所有超像素，`feature_selection=none`，避免仅选前若干正贡献区域。
- signed coefficient 投影回原空间，再 min-max 为 `[0,1]`，保留系数的排序；常量图置零。此图不代表原始系数大小或仅正贡献。输出 CPU float32 `(H,W)`；非有限输入、baseline、logits/map、无效 target 或 train 模式快速失败。
- 显式参数及 target、mask、kernel、baseline、map、seed 策略均写入配置快照。安装版本完整记录于交付环境清单；应在 GPU 机器再次记录实际版本。
- `__init__.py` 和 `run_unit.py` 注册 LIME，稳定性入口复用同一 builder。现有基础 runner 保存 NPY/provenance、预测表、长表、配置哈希、一次不计时 warm-up 和续跑日志。

## 配置与运行边界

六份 `configs/lime_{resnet50,vgg16,densenet121}_{imagenet,voc}.yaml` 覆盖 3×2，使用 `eval`、`max_images: null`、真值 target、21 点 raw MoRF、`warmup_runs: 1`、`resume: true` 和 float32 map。VOC 指向 B 的三个冻结 20 类多标签 checkpoint，输出 sigmoid。

这些是**待门禁的配置**，1000 samples 只是第一个候选预算，尚未冻结。`runtime.budget_frozen: false` 使直接 `run_unit.py` 正式 eval 快速拒绝；不得为绕过门禁而改成 true。预算通过并完成审阅后，更新六份配置的 `num_samples`、budget 状态、证据引用，再冻结配置哈希。正式输出隔离在 `results/lime_eval/`，真实 smoke 在 `results/lime_smoke/`，预算比较目录由 `--output` 指定。不要让 debug/eval 共享 CSV 或状态目录。

本次未启动真实 40/460 或正式稳定性。不使用合成图时间估计正式模型速度，不填写不存在的结果。

## 当前机器与实际验证

隔离 `.venv` 使用 Python 3.12.14。实际依赖包括 torch 2.14.1、torchvision 0.29.1、lime 0.2.0.1、scikit-image 0.25.2、scikit-learn 1.9.1、NumPy 2.5.3、SciPy 1.18.1。CUDA=false、MPS=false，只能 CPU。

- 四份冻结 metadata/labels/index SHA 校验均 OK；图片目录 `data/imagenet/raw`、`data/voc/raw` 缺失，故数据整体校验未通过。三个 VOC checkpoint 及 ImageNet 预训练权重缓存缺失。
- 自动化测试覆盖有限值、shape/dtype、确定性、局部 RNG、target 与 softmax/sigmoid、空间 baseline、常量图、错误输入、正式预算阻断、统一 runner warm-up/NPY/provenance/续跑/force、稳定性不重算 MoRF及原始指标/哈希保留。
- 门禁工具测试使用显式合成 fixture 验证 40 条记录、预算/seed 比较、续跑去重、归因文件哈希核验。这是软件测试，不是冻结 40 图实验结果。
- 单图工程 smoke 使用自建 24×32 RGB 图和无训练的确定性 tiny 模型、64 samples、quickshift max_dist=10（仅加速接口测试）。target=1，预测为0；21 点 raw MoRF、一次 warm-up、float32 NPY 与 sidecar 均生成；首次 1 processed/0 skipped，立即续跑 0/1。
- 两个独立 Python 进程、两个独立输出目录计算；NPY SHA-256 相同：`1e1115c8b1080c8e536dc2fa6762d7f57a9f114c1f8b939059603a4c1ac2a45d`。仅证明本机合成场景复现，不等同第三方真实评测独立复现。
- `experiments/lime_synthetic_smoke.py` 生成标注 SYNTHETIC ONLY 的 input/map/overlay，可用作布局与管线样例；尚未提供真实冻结图可视化。
- 最终全套测试：94 passed；pip check 无依赖冲突。具体运行时保存在交付 `all-tests.log`、`smoke/run_log.jsonl`、`smoke/units.csv`；这些时间与 MoRF 数字不进入正式矩阵。

## 恢复实验的步骤

在本地仓库根目录执行；建议在 A 已授权的 GPU 环境运行，但本次没有访问或发消息给他人。

```bash
python -m venv .venv
# 激活对应平台的环境后
python -m pip install -r requirements.txt
python -m pytest -q
python preprocessing/verify_data_version.py
```

先取得冻结 `data_share_v1.zip` 图片并按 metadata 的原始路径安放，不得重新抽样或替换图片；取得三个 VOC checkpoint，并按 `docs/VOC20_CHECKPOINT_MANIFEST.json` 核验 SHA-256。在项目缓存准备 torchvision 默认权重。先用 `--plan-only` 查看资产清单及输出位置：

```bash
python experiments/lime_budget_gate.py \
  --config configs/lime_resnet50_imagenet.yaml \
  --output results/lime_gate/resnet50_imagenet --plan-only
python experiments/run_unit.py --config configs/lime_smoke_imagenet.yaml
# 立即续跑，核对 0 processed / 1 skipped、NPY/provenance与CSV
python experiments/run_unit.py --config configs/lime_smoke_imagenet.yaml
```

真实 smoke 前记录配置、输出与预期耗时；首次缺少实测时明确记为待测，不使用合成速度。完成后用同机、同模型、最小候选预算下的归因耗时（CSV efficiency_time_ms，非 MoRF 耗时）填入门禁计划。每个模型/数据集先做对应单图 smoke，独立输出；不要使用正式 YAML 加 `--max-images 1` 写入正式目录。

```bash
python experiments/lime_budget_gate.py \
  --config configs/lime_resnet50_imagenet.yaml --budgets 1000 2000 \
  --seeds 42 43 44 --single-image-ms <真实单图归因毫秒> \
  --output results/lime_gate/resnet50_imagenet_ready --plan-only
# 检查 plan.json、线性估算、资产与输出，再去掉 --plan-only 执行
```

预算工具固定 debug，严格校验冻结 metadata 哈希及40个唯一ID；禁止用 eval 或少量子集代替。预算 1000/2000，seed 42/43/44，每设置一次独立 warm-up：共246次归因、369,000个masked输入（不含回归/分割/I/O）。逐图存预算间配对 Spearman/RMSE、跨 seed sample pixel variance、seed-pair 相似度、退化状态、superpixel 数、weighted R²、耗时和float32图哈希；原子 JSON 记录支持中断续跑，失败记录在 failures.jsonl。文件被修改/丢失或计划身份变化时拒绝复用。

预计耗时以真实单图耗时线性外推并写入 plan.json，明确不含启动/I/O、需用40图墙钟校验。不自动把任何相似度阈值当作全组验收标准：对比均值/分位数、低相似度图片、退化比例、方差和速度，与 D 复核后冻结预算；若2000也不稳定，增加预算再比较。建议六个单元分别完成门禁并记录，以确认预算跨模型、数据集适用。

门禁通过后再决定是否启动6×460，运行前保存冻结配置、环境、资产哈希、40图推算的单元耗时及 `results/lime_eval/` 输出位置。评测每单元验收460唯一ID、920条基础逐图指标、460份有限float32图/sidecar，立即续跑0/460。checkpoint 哈希和预测/配置来源保留；不同预算不能混用结果目录。

稳定性协议现已冻结，基础门禁与正式参考图齐全后直接复用 `experiments/run_stability.py`：先独立debug单图/40图门禁，正式补跑从原NPY/sidecar读取参考，只计算5次RGB sigma=.005扰动归因；不重复MoRF，不改变原忠实性/效率及其配置哈希。正式每单元应有2300条trace、920条新增稳定性指标，续跑0/460。协议未被本次修改。

最后补真实图可视化、第三方干净环境复现记录、六组真实实验验收及结果解释。此次交付只完成安全可做的本地实现与工程验证。
