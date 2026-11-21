# D：修订版备注确认与剩余验收

2026-11-20。D 对 A 已修订的 E v7 做补充确认，并同步 D 交付入口。
本次没有修改 A/B/C/E 分支，也没有代签他们的验收。

## 已完成的收尾

E 固定提交为 [`e2003bf78b8f15cbf65294580ad453d67d7da2e1`](https://github.com/oioioi-521/GO3-XAI01-04/tree/e2003bf78b8f15cbf65294580ad453d67d7da2e1/deliverables/e_m5)，
A 报告固定提交为 [`138b865e161d4808ab696864ce68b8daf5edafd5`](https://github.com/oioioi-521/GO3-XAI01-04/blob/138b865e161d4808ab696864ce68b8daf5edafd5/docs/FINAL_REPORT_DRAFT.md)。
这些分别是 A 在 11/4 修订备注、11/5 更新状态的交付；下列是 D 在本次实际执行的核对。

| 核对项 | 本次结果 |
|---|---|
| PPTX 身份、ZIP CRC 与 XML | 新旧文件逐字节读取；CRC、全部 XML/关系解析通过；12 页正文与 12 页备注 |
| 包内差异 | 仅第 1、6、11、12 页的四份 notesSlide XML 改变，其余 87 项字节相同；可见正文、媒体、版式与关系未变 |
| 第 1/12 页备注 | 已写明 v7 进入 PR #26；D 在 11/2 指出的“v7 尚未提交”已删除，仍保留异机 Demo 和真实计时待验收 |
| 第 6 页备注 | 图源为 ImageNet 面板；均值差区间跨零、配对秩偏 IG 分开解释，没有将秩显著写成均值显著 |
| 第 11 页备注 | 三例与选图规则已确认；来源包、PNG/NPY/sidecar、配置及哈希说明保留，独立 Demo 另行验收 |
| 统计限制 | 名义/WTS、9999 次 bootstrap 与 .0014 分辨率、跨任务评分限制、VOC DID 的 IID/分层区间与秩 p=.8374 均保留 |
| PDF / Demo | PDF 字节相同；40 个受管 Demo 文件的 Git 对象及文件模式相同；继承 11/2 的已审范围 |
| A 报告 | 相对已审 v4 仅 4 行状态/来源段落修订，82 行表格完全相同；10/15、10/28、10/29 分析目录及三轴导出代码未变 |
| 修订方回执 | 实测新旧 PPTX/PDF SHA 与 A 的 11/4 回执匹配 |

**D 提出的第 12 页过期备注问题已解决。** 本轮阅读全文提取出的四页备注并检查报告差异；
没有重新渲染未变的可见页面，没有打开 PowerPoint，没有重算统计或重跑模型。
D 的这份补充确认不冒充 B/E 的修订后确认、C 的独立复现或课程最终验收。

新 PPTX SHA-256：`69081383bcace59d4301c3eb4e665118a3b01e7abfe4838f6a13456d904b8158`。
PDF SHA-256：`4c5993864e95cf7936e989dfab40867b67eb6e44eafe49a324e96a6e31db0af8`。
11/2 的旧 PPTX SHA 仅标识历史文件，不继续用作当前文件身份。

核对程序：[review_notes_closeout_20261120.py](../analysis/review_notes_closeout_20261120.py)。
机器回执：[notes_closeout_receipt.json](../analysis/dcu_20261120/notes_closeout_receipt.json)。
在本地已有上述固定提交及旧 E `d00553f9ca7fc41208e24296619c1406a339d8da`、
旧 A `3cfd4a8250b7f820e241bbcae3df93aa374f6c8a` 的 Git 对象后，从仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m analysis.review_notes_closeout_20261120
```

程序只用 Python 标准库及 Git 读取固定对象，回执写入本轮 analysis 目录；
提取文字写入忽略的 results 目录。它不执行交接包或队友的代码。

## 当前真正剩余的事

本次读取的 [D Issue #21](https://github.com/oioioi-521/GO3-XAI01-04/issues/21)
已接收 D 最终统计确认；[E Issue #22](https://github.com/oioioi-521/GO3-XAI01-04/issues/22)
已记录四页备注修订。这两项不再作为“D 未交统计或必须重跑”的待办。

| 剩余项 | 实际完成条件 | 当前责任边界 |
|---|---|---|
| 新增统计独立复现与最终整合 | C 从冻结输入复现 10/28、10/29，交本人执行回执并接入各分支成果 | PR #24 head 仍为 `eb275ed535a1ba660608a9606fd3847f83460405`；D 的已有复算不能代替 C |
| 四处备注修订后确认、E→A / LIME 问答 | B/E 记录新 PPTX 身份与四处确认，E 审核最终报告及本人方法问答 | D 本次已完成自己的补充核对，未代签 B/E |
| 独立设备 Demo | 在另一设备恢复 215/3803/7612 原图，逐例 6/6 地图、切换、来源展开和静态备用都实际检查 | 由 C/E 或另一组员记录设备、浏览器、日期与异常；本机字节核对不替代 |
| 全组计时与最终提交 | 实际朗读与投影、记录真实时长和问题，冻结版本；负责人评审 main 与课程平台提交 | D 的[三分钟讲稿](D_M5_REHEARSAL_20261101.md)已备，实际时长仍未测 |

这次可独立完成的 D 文案/文件核对已收尾。没有收到新的实质统计问题或重跑要求，
不新增 GPU 归因、完整 bootstrap 或重复旧测试集合。原始 ZIP、NPY、权重及 outputs/ 不进入本次提交；
不新增必须从微信重发的数据文件。
