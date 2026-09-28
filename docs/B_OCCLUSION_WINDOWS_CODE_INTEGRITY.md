# B：Windows 原生扩展加载阻塞诊断（2026-09-28）

## 结论与边界

本机 `.venv-gpu` 的 Matplotlib 和 SciPy wheel **未见安装损坏或来源不明的证据**；
Windows Code Integrity 正在以 `VerifiedAndReputableDesktop` 策略阻止两个
未签名的 `.pyd` 加载。`Get-MpComputerStatus` 报告
`SmartAppControlState: On`，同时间的 Code Integrity 3118 事件描述为
“Smart App Control Block Details”。所以这不是仅凭 3077 事件编号推断的
Smart App Control；策略名称、策略 ID、设备状态和 3118 共同支持该判断。

此结论只解释当前明确出现的两个加载失败，不证明系统中所有 Python 扩展均受阻。
2026-09-26 的 Occlusion 基础 eval 与测试曾能运行；当前 Matplotlib 文件创建于
2026-09-19，策略激活日志为 2026-09-21，而该文件首次可见拦截为
2026-09-27 23:26 本地时间。**为什么同一文件此前可加载、后来被拒绝，
现有日志无法确定**；可能涉及当时信誉判断或其它状态，不能当作已证实根因。
没有改动安全策略、驱动、文件标记或系统 Python，也没有启动稳定性门禁。

## 完整异常与策略证据

- `import captum` 经 `captum.attr` → `matplotlib.pyplot` →
  `matplotlib.image` 导入 `_image` 时抛出
  `ImportError: DLL load failed ... 应用程序控制策略已阻止此文件`。
- `from scipy.stats import spearmanr` 经 `scipy.optimize` 导入 `_zeros` 时抛出
  同一错误。`pip check` 的“无依赖冲突”不能推翻原生模块加载失败。
- Code Integrity `Operational` 日志的事件 3077 指向两份文件，包含
  `Policy ID {0283ac0f-fff1-49ae-ada1-8a933130cad6}`。事件 3099 在
  2026-09-21 22:03 本地时间记录该 ID 的 `VerifiedAndReputableDesktop`
  策略刷新并激活。事件 3118 附带 Matplotlib 文件的完整 SHA-256 和
  Smart App Control 阻止详情；设备查询显示 `SmartAppControlState: On`。
  `CiTool.exe -lp` 在当前非管理员会话返回 `0x80070005`，因此未取得其
  `Is Currently Enforced` 列表；这一限制不应隐瞒。
- Windows `Get-AuthenticodeSignature` 对下列两份 `.pyd` 均为 `NotSigned`：

| 已安装文件（仓库相对路径） | 字节 | SHA-256 |
|---|---:|---|
| `.venv-gpu/Lib/site-packages/matplotlib/_image.cp312-win_amd64.pyd` | 446,976 | `2259c5cc02e86b3f0efccde8b31fce7cae06ea1515ab16da148df40cc7f35705` |
| `.venv-gpu/Lib/site-packages/scipy/optimize/_zeros.cp312-win_amd64.pyd` | 23,040 | `8d1797a4cf71c48e1c22d38dafeb4cdd9bf882d7c59a16b2d9670a1a0a1808c2` |

## 官方 wheel 与安装一致性

两包的 `.dist-info/INSTALLER` 均为 `pip`，wheel tag 为
`cp312-cp312-win_amd64`；安装的 `.pyd` 大小及 SHA-256 与各自
`.dist-info/RECORD` 完全一致。另从 `https://pypi.org/simple` 下载相同
版本的 wheel 到**被 Git 忽略**的
`results/occlusion_stability_gates/diagnostics/wheels/`；本机 wheel SHA 与
PyPI 官方 JSON 公布值一致，wheel 内对应 `.pyd` 与已安装文件逐字节相同：

| wheel | 字节 | wheel SHA-256（本机＝PyPI JSON） |
|---|---:|---|
| `matplotlib-3.11.2-cp312-cp312-win_amd64.whl` | 9,349,409 | `c5c1c68ee401fc98271263410f0e5ce88285abacf7627132914e8adf3d70ff43` |
| `scipy-1.18.1-cp312-cp312-win_amd64.whl` | 36,658,278 | `5e4d44984abc0020154ea81b247adeddcc3ac5527b975ff798bd1ba0adc513c2` |

据此无依据对同一 wheel 反复重装，或在新环境重复安装同样被拦截的文件。
SciPy 1.18.1 是上轮依仓库 `requirements.txt` 新增到原有 `.venv-gpu` 的
唯一必要依赖；原环境和 CPU `.venv` 均保留。CPU `.venv` 的 Matplotlib
`_image` 也被当前策略拦截。

## 当前依赖快照和测试

解释器为项目 `.venv-gpu/Scripts/python.exe`，Python 3.12.14；其 venv 基础
解释器位于 Codex bundled Python 运行时。关键版本：
`torch 2.11.0+cu128`、`torchvision 0.26.0+cu128`、
`captum 0.9.0`、`matplotlib 3.11.2`、`scipy 1.18.1`、
`numpy 2.5.2`、`pandas 2.3.3`、`pytest 8.4.2`。
CUDA 12.8 和 RTX 5080 Laptop GPU 对 torch 可见，但真实 Captum 和
SciPy Spearman 的导入失败。最后一次完整 pytest 为
**66 passed、3 failed、3 skipped**；三个失败来自真实 Captum 路径，
跳过模块不能视为通过。独立数据/配置测试 10 passed，`pip check` 正常。

## 最小恢复条件

由设备管理员/用户提供**符合当前系统策略且获批准**的原生模块发行或运行环境，
或由发行方处理签名/信誉问题。微软文档说明 Smart App Control 不提供单个应用
的本地放行开关；本任务不会关闭、削弱或绕过它。获得合规环境后，先单独验证
`import matplotlib._image`、`from scipy.stats import spearmanr` 的小型计算、
`from captum.attr import Occlusion`，再做真实 GPU tensor/Captum 冒烟、全量
pytest 和 `pip check`。全部通过后才允许在隔离目录做 Occlusion 稳定性
单图 → 40 张门禁；不得用空导入、skip 或模拟结果代替。

参考：[Microsoft Smart App Control FAQ](https://support.microsoft.com/en-us/windows/security/threat-malware-protection/smart-app-control-frequently-asked-questions)、
[Microsoft 策略与事件说明](https://learn.microsoft.com/en-us/windows/apps/develop/smart-app-control/test-your-app-with-smart-app-control)。
