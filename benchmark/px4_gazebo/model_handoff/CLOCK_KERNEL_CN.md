# 只读内核校时上下文：真实分支采集仍被拒绝

主线不变：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型
预测能力能否组合迁移到新任务。这里处理真实候选分支的环境前置阻塞，不是新的
模型或迁移成绩；任务没有永久发现者/确认者分配。

## 为什么补这项

此前已有多轮host clock拒绝及原始记录，见CLOCK_DIAGNOSIS_CN.md与
STARTUP_EVIDENCE_CN.md。空载通过或服务显示active/已同步，都不能豁免启动全程
10ms门控。本轮不追加仿真、起飞、候选动作或暂停同步服务，只做一次20秒空载
只读诊断，补每次时钟采样附近的内核校时参数，避免仅凭服务状态推断时钟稳定。

## 实现与安全边界

`clock_kernel_probe.py` 保留原clock_probe四路读取区间和startup_clock连续质量判定，
不修改原函数，不拟合/扣除速率差、不挑选最佳子窗口、不校正旧数据。RAW相对
MONOTONIC偏移幅度超过10ms仍拒绝，实际采样间隔超过250ms同样拒绝。
探针quality_passed仅适用于空载探针窗口，startup_quality_verified始终false。

每条记录在host clock读完后调用Linux `adjtimex(modes=0)` 查询；每次用全新、全字节
置零的ctypes结构，没有设置模式/offset/frequency的入口、不请求CAP_SYS_TIME。
支持的实时ABI限定为Linux x86_64 LP64，并检查结构大小208及关键字段偏移。
其他平台拒绝实时查询，Windows可离线复核，不自动尝试未知ABI。

原始字段保留：返回状态、status、tick微秒、freq的scaled ppm整数、offset及其单位、
maxerror/esterror微秒、单调时钟查询前后界。freq除65536才是ppm；offset按
STA_NANO（0x2000）决定ns/us，不把所有字段共用一种单位。
TIME_ERROR或STA_UNSYNC会另行拒绝；未看到这些标志不代表物理时钟稳定。
查询错误停止采样且保存可得记录，不重试、不改参数；中断也保存部分证据并拒绝质量。

仅在探针前后用带2秒上限的 `systemctl show systemd-timesyncd` 读取状态和InvocationID。
不会stop/start/enable/disable服务，不修改NTP配置、Windows时间、内核参数或网络。
端点命令可能失败，其原始输出保留；它不是服务全程连续监测，也不是核心时钟门控的
替代。inspect重算时钟/内核摘要，不认证服务来源或操作者没有同时修改全部证据。

## 本次固定真实诊断（2026-10-18）

一次新进程、目标50ms采样、20秒单调时钟窗口，无隐含重试或系统干预。
私有目录：`D:/AeroWAM-data/clock_kernel_readonly_20261018_v1/`。
起止boot均为 `fb8ca0dc-0072-4480-b30b-40513017f7ad`，clocksource为tsc。
前后uptime约51.08/71.09秒。服务端点均active/running、enabled，同一InvocationID；
不能由端点状态推断每个时刻都已稳定。此前另一命令中的状态查询boot不同，不混作本窗口证据。

| 检查项 | 实际记录 |
| --- | --- |
| 时钟/内核成对样本 | 400条 |
| monotonic跨度 | 20.000136756秒 |
| 最大实际采样间隔 | 50.666065ms，满足250ms覆盖规则 |
| RAW相对MONOTONIC偏移幅度下界 | 552.226629ms，超过10ms，拒绝 |
| RAW相邻最大区间外偏移变化 | 4.562434ms；仅相邻检查不足以发现整段幅度 |
| REALTIME、BOOTTIME相对幅度下界 | 均0，不表示物理时钟绝对不变 |
| 内核freq | -53.93658447265625至+0.000244140625ppm |
| tick | 9167至10000微秒 |
| status / adjtimex返回值 | 全部8192（STA_NANO）/0；没有UNSYNC/TIME_ERROR |
| 最大内核读取区间 | 0.080522ms；400条查询无错误 |

tick的原始变化为10000→9899→9167→10000微秒，对应窗口内约
0/5.219002208/13.194814639/18.407896204秒。出现次数分别为137/159/104条。
频率补偿不能只看freq而忽略tick；本次只记录两者，不反推修正值、不积分后消除漂移。
同窗口内核参数变化与RAW相对时钟变化是进一步调查校时链路的具体线索，
但不唯一证明timesyncd、Windows、WSL或其组合是根因，也不解释所有DDS/ACK故障。
本轮没有在仿真负载下采样，不把空载结果称为候选分支或曝光状态证据。

结果issues仅clock_offset_excursion，quality_passed=false。保留失败，不晋升训练或
配对分支。Windows/WSL分别从原始记录重算，saved_summary_consistent=true，
独立inspection文件逐字节一致；两端检查均保留拒绝，不重采样。

- probe.json SHA256：`45d9d0bd4f75b7deb181a1148dcc0318e2e1e41b71efc3f8580bb174cb091cc4`。
- 两端inspection SHA256：`8385a59245d2c7c0c774780fcb7e4ae64f0b460a6da10180d1688326125b4674`。

采样进程已结束，事后未发现本次诊断/Gazebo/PX4/Agent残留；服务端点仍active/enabled。
WSL后续命令中的自动会话/服务生命周期不等于本工具主动stop/start；不宣称永久不变。

## 复现与回归

输出必须是不存在的新文件，原始诊断和inspect均不覆盖既有文件：

```bash
# Linux x86_64，只读采样；无sudo、无仿真。
python3 -B -m benchmark.px4_gazebo.model_handoff.clock_kernel_probe \
  --duration-s 20 --interval-ms 50 --output /absolute/private/new_probe.json

# Windows/WSL均可：只重算已保存证据，不读当前内核或查询服务。
python3 -B -m benchmark.px4_gazebo.model_handoff.clock_kernel_probe \
  --input /absolute/private/new_probe.json --output /absolute/private/new_inspection.json
```

实时duration限1..60秒、interval限10..250ms；inspect拒绝混用实时参数。
CLI合格退出0，质量拒绝或摘要不一致退出2；格式/ABI/文件错误非零退出。
原始数据包含boot、服务上下文和本地环境信息，不自动上传；报告哈希是身份绑定，
不是时间先后、硬件计时或完整来源的签名证明。

新增25项合成回归；连同时钟探针、启动连续门控、启动原始证据、组合留出、候选评分，
Windows/WSL各131项通过。覆盖零字节/modes=0、ABI拒绝、freq/offset单位、查询失败、
往返漂移、250ms覆盖、10ms不放宽、只读systemctl命令、中断保留、不可覆盖及CLI复核。
合成测试不算真实系统干预；真实窗口拒绝验证的是探针与门控，不是环境修复成功。

下一步真实分支仍受环境阻塞，未完成起点匹配比较器、前缀重放或完整状态恢复。
修复系统同步链路属于新的系统干预，需要具体授权；此前临时暂停授权已用完，
本工具没有据“继续”重新暂停。即使环境通过，还须验证曝光对齐、控制器/估计器、
目标/任务/队友状态与分支容差，之后才接候选预测评分和组合迁移实验。
