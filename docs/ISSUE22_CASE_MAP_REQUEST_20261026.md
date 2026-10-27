# Issue #22：三张 VOC 案例的归因文件补件清单

**2026-10-27 更新：18 组文件已收到并通过验收，不再需要重发。**
新包 `dcu_cases_voc_resnet50_215_3803_7612.zip` 包含三张原图及六方法 18 组 NPY/PNG/原 sidecar，
身份、target、配置、文件哈希及逐图表子集均核对通过。
见 [当前交付状态](D_M5_HANDOFF_20261027.md) 和
[验收记录](../analysis/dcu_20261027/delivery_receipt.json)。B/E 最终选图保持待确认。
下文是 10/26 仅收到统计轻量包时的历史补件记录；`case_request_20261026.json` 保留原申请状态，不冒充验收回执。

2026-10-26 检查 `dcu_team_handoff_20261014(1).zip`：13,707,240 bytes，722 个条目，ZIP CRC 通过。
SHA256：`14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc`。
这是已验收的统计轻量包副本；**0 个 NPY、1 个 PNG、0 个 maps_index.json**。
唯一 PNG 是 `team-handoff-20261014/source/lime/release/docs/assets/lime_synthetic_example.png`，
属于合成示例，不能充当真实 eval 案例。包内交接文档也注明原始归因图按需另发。

## 当前已核对和仍缺的内容

范围：VOC / eval / ResNet50，六方法 IG、Grad-CAM、Occlusion、RISE、KernelSHAP、LIME。

| image_id | 冻结 metadata target | 类别 | 原图相对路径 |
|---|---:|---|---|
| 215 | 7 | cat | voc/raw/000215.jpg |
| 3803 | 2 | bird | voc/raw/003803.jpg |
| 7612 | 11 | dog | voc/raw/007612.jpg |

18 组身份、配置和预测上下文已整理到
[`analysis/case_request_20261026.json`](../analysis/case_request_20261026.json)。
其中 `packaged_groups=0`、`complete=false`；这只是补件清单，不是归因图交付。
已有标量 CSV、配置和摘要哈希不能还原真实归因 NPY/PNG。
旧 GPU 批次的本地文件不混入 DCU 正式批次案例。

## 可直接发给 A 的补件文字

> 我这里只有 dcu_team_handoff_20261014 统计轻量包，没有真实归因图。
> 请从 DCU 完整归档中只提取 VOC eval / ResNet50 的 image_id 215、3803、7612，
> 六方法 IG、Grad-CAM、Occlusion、RISE、KernelSHAP、LIME，共 18 组。
> 每组请带原始 float32 NPY、对应保存的 PNG、原 provenance/sidecar 或 maps_index 记录，
> 以及可核对 image_id、真实 target、checkpoint/config 和逐文件 SHA256 的清单。
> 不需要重跑，也不必发整个归档；若 B 确认后改了案例 ID，请同步。

如 E 需要叠加原图而未持有图片，再单独按数据授权提供这 3 张原图。
本页只准备转发文字，没有代用户发送群消息或代表他人确认案例。

## 有完整归档时的选择性打包工具

仓库的 `analysis.package_dcu_cases` 只读取已有结果，不调用模型。
它固定核对这两份原归档的 SHA256，再选择性读取所需成员；不解压整个包。

| 原归档 | SHA256 |
|---|---|
| dcu_full_five_20261013_complete.tar.gz | 852eb7c8c4eb74124b6c599d2fd54bf1f067f9c0579ce9a0542dfada040dbede |
| lime_vectorized_formal_complete_20261014_v2.tar.gz | 38510f8eeda54ae38ee77acaf04580666779954ff7565b59d9fbd58438fe1cd3 |

```powershell
python -m analysis.package_dcu_cases --handoff '轻量包路径.zip' --five-archive '五方法完整包路径.tar.gz' --lime-archive 'LIME完整包路径.tar.gz' --output 'results/issue22_cases.zip'
```

工具检查 NPY float32/224×224/有限值/归一化范围、PNG 格式/尺寸、原 sidecar 的图像/方法/
数据集/模型/划分、checkpoint/config/预测上下文绑定及 map SHA；LIME 另核对冻结 completion 哈希。
不修改原 sidecar，不伪造其 target 字段；target 单独关联冻结 metadata。
全部通过才创建 ZIP，附逐文件 SHA256 和来源成员路径，并拒绝覆盖已有输出。
PNG 保留原保存结果，不新画热图。实际归档尚未收到，不能声称这 18 组已完成验收。
测试使用合成内存数据，只证明选择/绑定/拒绝错误/打包读回行为，不是正式案例证据。
2026-10-26 本地检查：`tests/test_package_dcu_cases.py` 与
`tests/test_dcu_statistics.py` 合计 **27 passed**；使用独立 CPU 分析环境。

原始归档、实验 NPY/PNG、案例 ZIP 仍走受控交接，不提交到 Git 或 `outputs/`。
