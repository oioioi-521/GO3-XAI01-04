# A 向 C 转交 VOC20 checkpoint 的校验说明

2026-10-01：A 已重新读取交接 ZIP，CRC 通过，三份 checkpoint 的大小与 SHA-256
逐项匹配原冻结 manifest。用户将通过微信转交此已存在的 ZIP；本记录只证明发送前
校验完成，不代表 C 已收到或完成加载验收。

交接文件：`GO3-XAI01-04_checkpoint_transfer_2026-09-20.zip`

- ZIP 字节数：`613060967`
- ZIP SHA-256：`3ca6a0affab5efaae4eaea4cf0508b7dd29ccf8c0b6e926cbf7baf91f400d79f`
- ZIP 顶层目录：`GO3-XAI01-04_checkpoint_transfer_2026-09-20/`

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `resnet50_voc20.pt` | 94518331 | `1e54867cf18ed02383ba329c944a9143871fc37fe268bec6e769a8d364ae134f` |
| `densenet121_voc20.pt` | 28512970 | `ab61772eaf3459842f4b1e34be9ec15062e02a940d521390d80df0371168feca` |
| `vgg16_voc20.pt` | 537383181 | `61640717641bfc1e9ae5d6172de667ce45755e6f046f458c848f70cb4a6c9d58` |

C 接收后先核 ZIP SHA-256，再解包。将三份 `.pt` 放入项目
`models/checkpoints/`，逐文件核对上表和包内 `VOC20_CHECKPOINT_MANIFEST.json`。
三份权重属于 VOC20、20 类 sigmoid 多标签接口；保持冻结类别顺序，严格加载并执行
有限的 `(1,20)` 前向检查后，再按既有 RISE 配置先做单图检查、随后运行三个 VOC 单元。

Windows PowerShell 校验示例：

```powershell
Get-FileHash .\GO3-XAI01-04_checkpoint_transfer_2026-09-20.zip -Algorithm SHA256
Get-FileHash .\models\checkpoints\*_voc20.pt -Algorithm SHA256
```

checkpoint 和 ZIP 保持在 Git 忽略的本地/共享位置。完整训练与类别身份见
[`VOC20_CHECKPOINT_MANIFEST.json`](VOC20_CHECKPOINT_MANIFEST.json) 和
[`B_VOC20_COMPLETED_HANDOFF.md`](B_VOC20_COMPLETED_HANDOFF.md)。
