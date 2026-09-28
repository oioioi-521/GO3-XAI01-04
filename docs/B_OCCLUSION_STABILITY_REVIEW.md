# B：PR #10 修复复审与 Occlusion 稳定性门禁状态（初审 2026-09-27，GPU 复核 2026-09-28）

> 后续状态更新：用户于 2026-09-28 手动关闭 Smart App Control 后，新的 Python
> 进程已通过 Matplotlib、SciPy Spearman、Captum 和真实 CUDA Occlusion；
> 全量 93 passed，六组合单图及 40 张 debug 均已完成。以下旧“阻塞/尚未执行”
> 章节保留当时的证据，不再代表当前状态。完整实测与限制见
> `docs/B_OCCLUSION_STABILITY_GATE_HANDOFF.md`。

## 来源和结论边界

- B 基线：`feat/b-occlusion-gates` 的 `5672ccfe38f8f045818b8eedf0496d1216e74d8e`。
- A 来源：`origin/feat/stability-pilot` 的
  `9891eecc4ec869c0bc14a054a2cd85a0beca4266`，本分支以普通 merge 保留
  A 的原始提交；`experiments/map_provenance.py`、`run_stability.py`、
  `stability.py` 和基础 runner 修复均归属 PR #10，不计作 B 新增算法贡献。
- 本记录是**代码与测试源码复审**，不是本机 GPU 稳定性门禁通过记录。
  由于下面的 Windows 应用程序控制拦截，本轮未运行单图、40 张或 460 张
  稳定性，也未调用 force。基础 eval 六组、交付 ZIP、旧 debug 和 checkpoint
  均未修改。

## 对 B 三项 review 的逐项复核

1. **原图 float32 来源**：`experiments/map_provenance.py` 的
   `write_map_provenance` 将 `image_id`、单元、split、基础配置哈希、预测上下文
   哈希、checkpoint SHA-256、NPY 字节 SHA-256 和 `binding` 写入 sidecar；
   `verify_map_provenance` 逐项拒绝不匹配。`experiments/run_unit.py` 保存新 NPY
   后写 `base_runner` sidecar。`experiments/run_stability.py::_load_reference_map`
   对没有 sidecar 的旧图用当前模型、target、Occlusion 参数重算一次原图，
   仅当 float32 形状/有限性相同且
   `np.allclose(old, fresh, rtol=1e-4, atol=1e-5)` 时写
   `verified_recomputation`；否则拒绝，不运行 MoRF。
   `tests/test_run_stability.py` 含错误基础配置、NPY 被替换和 checkpoint
   字节变化的拒绝回归。**限制**：对旧图的 sidecar 只能证明在复核时与当前
   推理结果在该容差内一致，不能追认原始生成时的权重；真实 RTX 5080 上
   旧 Occlusion 图能否全部通过该容差尚未实测。sidecar 写入会改变被测目录，
   因此真实门禁必须先复制已有 debug 图与基础表到独立目录。
2. **split/trace/state 隔离**：`run_stability.py::_state_path` 纳入单元、split
   和基础 per-image/units 输出路径身份；`_trace_path` 加 split 和稳定性配置哈希；
   `_check_split_isolation` 在清理前拒绝 debug/eval 共用基础表。
   `test_debug_eval_debug_preserves_independent_trace_and_state` 覆盖
   debug → eval → debug 和共用表拒绝。**限制**：这是源码/合成测试证据，
   本机仍需在独立的真实门禁目录核对两阶段记录与恢复。
3. **基础 runner 保护**：`run_unit.py::_prepare_resume_state` 在稳定性行存在时
   拒绝基础 `force` 或配置哈希变化，且拒绝发生在移除基础行前；
   `_upsert_unit_summary` 仅重汇总当前基础配置的指标，保留稳定性行原哈希。
   `test_base_resume_preserves_stability_and_force_fails_closed` 核对 resume、
   force 拒绝及基础 CSV 不变。**限制**：本机真实 Occlusion 的独立副本上
   尚未实测 resume/force，不能把合成测试等同于 GPU 验收。

综上，PR #10 的 `9891eec` 在源码和新增回归测试中回应了三项 review；
未发现必须绕过 provenance 才能继续的设计缺口。C 的 runner/schema 集成复核和
D 的稳定性聚合/退化计分最终确认仍未完成；Issue #8 保持候选协议状态。

## 本机测试阻塞：Windows Smart App Control

2026-09-28 的后续只读诊断见
`docs/B_OCCLUSION_WINDOWS_CODE_INTEGRITY.md`：已对照设备实际状态、
`VerifiedAndReputableDesktop` 策略 ID、事件 3077/3099/3118、文件签名、
PyPI 官方 wheel SHA 与已安装 `.pyd` 字节。现有证据支持当前由该策略拦截，
而**此前可运行、现在不能运行的具体状态变化仍未知**。重复安装同一官方
未签名 wheel 没有修复依据；未更改安全策略。

在 `.venv-gpu` 上执行全量 `pytest -q -ra` 时，初次收集出现 2 个错误
（SciPy 缺失）和 3 个 Captum 相关 skip（Matplotlib `_image` DLL 导入被拦截），
**没有获得全量通过结果**。随后仅按 `requirements.txt` 安装
`scipy 1.18.1`；`pip check` 为 `No broken requirements found`，但
`from scipy.stats import spearmanr` 又因 SciPy `_zeros` DLL 被同一策略拦截。
系统 `Microsoft-Windows-CodeIntegrity/Operational` 事件 ID 3077 明确记载
`.venv-gpu` 下 `matplotlib/_image.cp312-win_amd64.pyd` 与
`scipy/optimize/_zeros.cp312-win_amd64.pyd` 未满足代码完整性签名策略。
原 CPU `.venv` 的 Matplotlib 同样被拦截。CUDA 版 torch 本身仍可导入并
报告 GPU 可用；这不等于 Captum 与 Spearman 可执行。

安装 SciPy 后重新执行**完整** pytest，实际为 **66 passed、3 failed、
3 skipped**：三个失败均为 `tests/test_run_unit_occlusion.py` 中真实 Captum
调用触发 Matplotlib `_image` 拦截；三个 skip 是其它 Captum 测试模块的
导入跳过。若干合成稳定性测试能通过，并不证明 `scipy.stats` 的真实
Spearman 或 GPU Occlusion 可运行；单独导入 `spearmanr` 已因 `_zeros` 被拦截。

已独立运行不依赖上述导入的
`tests/test_data_version.py tests/test_occlusion_eval_configs.py`：**10 passed**；
`pip check` 无冲突，`git diff --check` 通过。不可将 3 skipped 计为 passed，
也不能用合成结果代替真实 GPU 门禁。没有修改或关闭 Windows 安全策略，
没有尝试规避代码完整性控制。需由设备管理员/用户提供可获批准的 Python
原生扩展运行方式，并在相同 `.venv-gpu` 验证 Captum 与 SciPy 导入后，重新
运行全量测试。未解除前不得启动单图或 40 张稳定性。

## 恢复后的门禁顺序（尚未执行）

1. 先确认 GPU 空闲及上述导入、全量测试、依赖检查。把源 debug 门禁的
   CSV、预测表、NPY、状态及哈希复制到独立
   `results/occlusion_stability_gates/`，单图和 40 张分目录，原目录只读。
   旧图通过重算验证后才允许写副本的 sidecar。
2. 串行六组真实 Captum 单图；逐组立即 resume；在另一个副本上验证
   稳定性 force 与基础 runner resume/force 的保护，记录比较容差与哈希。
3. 单图全通过后串行六组各 40 张 debug，核对 240 个单元内图片、
   1,200 个 repeat、480 条稳定性逐图指标、seed/status/退化/有效率、耗时和
   峰值显存。此阶段只验证候选协议，不运行 460 张，不作正式排名。

基础 eval 的接收回执与 A 参数确认见
[PR #13 讨论](https://github.com/oioioi-521/GO3-XAI01-04/pull/13)；D 的
`docs/D_OCCLUSION_EVAL_ACCEPTANCE.md` 接受 LIGHT 包的 faithfulness/同环境
efficiency 输入并冻结正确/错误分组分析口径，随后 A 又校验了 FLOAT32 包。
这几项不等于稳定性协议已冻结或 C 已完成统一集成。
