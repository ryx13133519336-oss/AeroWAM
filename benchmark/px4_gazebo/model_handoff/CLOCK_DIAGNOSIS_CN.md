# 时钟诊断：累积漂移与跨启动证据边界

这是候选分支比较的工程质量前置检查，不是模型评测。主线仍是部分可观测、
队友行为变化下双机协作技能与世界模型预测的组合迁移；不固定无人机永久角色。
已发布的只读阶段未启动Gazebo/PX4、策略或训练，未更改系统时间、NTP服务或全局配置。
本文末尾另记录用户单独授权的一次临时服务暂停对照；它没有更改永久配置。

## 已有证据与新增空载记录

旧20秒预检 `branch_clock_preflight_20261018_v1.json`，boot前缀 `7a031086`，
读取时间对应11:34 CST；原始数据重新分析仍通过，raw相对偏移区间包含0。
后续r1/r2启动记录没有直接保存boot ID，不能倒填；历史journal的启动区段与时间戳
显示它们处于后一个 `4504142f` 区段。该区段内timesyncd也有多次启停/初始同步。
因此旧预检通过不能作为后续运行环境不变的证据。journal只辅助关联，不能把
跨时钟时间戳关联当精确的曝光/运行时状态认证。

2026-10-18新增一次不间断WSL会话内连续两段20秒、50ms间隔的只读检查。
按 experimental-design 的分组原则固定同一boot、记录分段而不是跨boot比较；
未随机化任何服务干预，没有对照停NTP，不给出因果效应或独立重复数量推断。
三次环境读取boot均为 `dd1959e4-395a-4efc-aa95-eb3e08290b13`，
clocksource均tsc，开始uptime约65.91秒，结束约105.92秒。

私有文件保存在 `D:/AeroWAM-data/`：

- `clock_same_boot_context_20261018_v1.json`：环境和三次内核校时读取。
- `clock_same_boot_20261018_v1_segment0.json` / `segment1.json`：原始时钟采样与旧分析。
- 对应 `_analysis_v2_wsl.json`：新版本只读重分析，包含源文件SHA256。
- segment0另有 `_analysis_v2_windows.json`，源哈希及report与WSL完全一致。

| 同boot分段 | monotonic跨度 | raw净相对偏移 | realtime整段偏移幅度下界 | 旧相邻检查 | 新整段检查 |
| --- | ---: | ---: | ---: | --- | --- |
| 0 | 20.000139秒 | -28.423毫秒 | 0毫秒 | 通过 | 拒绝 |
| 1 | 20.000104秒 | +1596.770毫秒 | 697.898毫秒 | 拒绝 | 拒绝 |

segment0相邻raw偏移变化最大仅0.07175ms，旧检查确实漏掉了累计28.423ms变化。
segment1相邻raw变化最大4.56525ms，也低于10ms，但整段raw偏移幅度下界1600.123ms；
该段realtime相邻变化约697.897ms已被旧检查检出。不能仅比较首末两点来代替整段
幅度，因为偏移可能先变大再回原点。

内核读取使用Linux x86_64的208字节timex ABI，`adjtimex(modes=0)`，仅查询，
三次返回0。`tick_us`先为10014/10014，结束9167，frequency约+23.198/+23.198/-33.333ppm。
`tick=9167`相对通常的10000有约-8.33%的粗调量，与先前raw/monotonic约1.09的
量级相符；但这不是逐采样内核参数轨迹，不能把先前每个区间归因于这次末尾状态。
直接证据是内核校时参数在这40秒中发生变化，空载也出现相对速率差及realtime变化；
不需要Gazebo才能复现。触发来源仍不能唯一归因于timesyncd、WSL或主机时间服务。
Windows `w32tm /query /status`另显示未同步、源为Local CMOS Clock，只是主机线索。

## 分析profile 2-draft

保留旧相邻偏移/读取不确定性10ms门槛与`events`含义，另新增整段检查：每次
clock-minus-monotonic可行偏移区间为 `[value-after, value-before]`；所有区间中
最大下界减最小上界（截断至0）为 `minimum_offset_excursion_ms`。
该值超过10ms，新增 `clock_offset_excursion` issue。它检测被观测到的慢漂移/回返，
不证明没有被采样捕获的变化不存在，也不把漂移一律命名为突跳。
同时报告首末有符号偏移区间、净相对速率ppm界；读取区间交叠而无法给出速率时为null。
整数差值先计算再转换毫秒，避免浮点UNIX大数损失精度。缺失时钟仍标不可用，
不生成假值；宽读取区间只证明不确定性，不冒充时钟变化。

原始probe/1结构兼容，报告新增明确profile；`--require-stable`现在也拒绝累计漂移。
原有trace质量标准/任务验收不变，原始采样与旧report保留；新profile不是正式冻结
标准，也不是校时修复。原20秒稳定预检新分析仍退出0，两段新记录分别退出2/2。
原始SHA256均未变化。不因需要数据而放宽门槛，不反写旧通过/失败结论。

15项新增测试：慢漂移、采样稠密/稀疏、回返、正负速率、读取区间不确定性、
精确10ms边界与+1ns、缺失/非法时钟、源文件保留、独占输出及真实CLI退出码。
起止运行环境保留也有无等待mock测试。
Windows/WSL各232项相关回归通过；原collection_quality和trace_quality独立自测也通过。

只读重分析示例（不采样、不修改时间，输出须全新）：

```bash
python3 -B -m benchmark.px4_gazebo.model_handoff.clock_probe \
  --input /absolute/saved_probe.json --output /absolute/new_analysis.json --require-stable
```

## 临时服务对照之前的授权边界

当前尚未证明稳定采样或分支起点可匹配，不扩大训练数据、不安排新飞行。
可考虑临时暂停WSL时间同步服务的短时对照，并在结束/出错时恢复原活动状态，
不改enabled状态或永久配置。但暂停服务不会自动撤销已经存在的内核slew；
没有变化也不能据此排除它的历史作用。该干预须另获用户明确授权，不能把
“继续研究”解释为可以修改主机/WSL时间。Windows全局校时也不自动执行。

## 2026-10-18：获准的单次临时服务对照

用户针对“临时暂停WSL时间同步服务，采样20秒后立即恢复原状态”的具体问题回复
“继续”后，执行一次同会话ABA工程诊断：active_before → paused → active_after，
每段20秒monotonic、目标采样间隔50ms。不随机化、不把逐次采样当独立实验重复；
顺序/时段效应和既有内核校时残留仍可能混杂。此次没有飞行或训练，没有enable/disable、
修改NTP配置、调整Windows时间或写入内核校时参数。

两次早期脚本调用在Python/证据目录创建之前退出，复查服务仍active/enabled；
不能把它们计为两次暂停实验。标准输入传递脚本的无修改检查通过后，单次对照完成。
实际脚本、原始采样、每秒服务/内核上下文、汇总及只读复核脚本均保存在私有目录
`D:/AeroWAM-data/clock_service_pause_20261018_v1/`，不上传原始数据。

| 窗口 | 样本数 | monotonic跨度/秒 | realtime偏移幅度下界/ms | raw偏移幅度下界/ms | profile 2-draft |
| --- | ---: | ---: | ---: | ---: | --- |
| 正常运行 | 398 | 20.000147416 | 3685.653094 | 0.002840 | 拒绝 |
| 临时暂停 | 398 | 20.000131786 | 0 | 0 | 通过 |
| 恢复运行 | 398 | 20.000128687 | 3671.277417 | 2.191599 | 拒绝 |

幅度0表示已观测区间给出的下界为0，不表示物理时钟完全无变化。前后窗口均触发
`clock_offset_inconsistency`和`clock_offset_excursion`；暂停窗口issues为空，
realtime首末净偏移区间为[-0.001654, 0.002244]ms。仅看首末会遗漏前后窗口的往返变化。
每段22条边界/周期上下文的服务状态分别全部为active/inactive/active，enabled始终不变。
全部上下文与三段probe起止boot一致：`282eac9a-ca92-4024-a636-0a80a8599228`。
所有上下文中查询的tick均10000、frequency均0ppm；`adjtimex(modes=0)`只读，
返回非负，恢复初始瞬间返回5（TIME_ERROR），不能把服务active当作已同步。

Python finally及外层退出trap均执行恢复；汇总`original_service_restored=true`，
结束时服务active/running、enabled。后续只读复查仍active/enabled，但InvocationID已
变化；journal也显示WSL会话间启停。因此不声称跨后续命令保持同一服务进程。
journal的同boot单调时间记录暂停约45.417秒、恢复启动约65.477秒，包含服务/采样边界
开销；“20秒”是暂停窗口采样预算，不是承诺服务恰好只停止20.000秒。

Windows与WSL独立从1194条原始时钟采样重算，并核验上下文状态和恢复结果；
每段报告与原保存报告及summary一致。私有源文件SHA256：

- active_before.json：`6df803ce85181d133a06744bafc2c99546e428ba9d93c4a82335b8fbd7979e11`
- paused.json：`402cf16db7fa38b8ad414e9900698bd3045884b42bc27e3e6a37677b6e9e87f6`
- active_after.json：`71eca56d16f277471bf4b277c065dc0ec4b628e5f03ac6478752e986621c9a95`

结论仅为：这一次同boot对照中，秒级realtime相对偏移出现在服务运行窗口，暂停窗口
未检出，恢复后再次出现，值得优先调查同步链路。不能唯一归因于该服务，也不能据此
证明DDS/ACK超时、模型延迟或所有历史故障的原因；不证明长期稳定、曝光对齐、候选
分支起点匹配或组合迁移。旧dd1959e4窗口中的raw速率异常不能用此次tick轨迹倒推解释。

本次临时授权已用完，服务没有留在停用状态。不自动追加更长暂停、Windows/内核
校时修改或新仿真。下一步建议另行批准一个有上限的临时暂停窗口，仅做无解锁启动
采样并立即恢复，验证短时空载改善能否延续到真实传感器采集；若仍失败保留失败，
不放宽10ms质量门槛或400ms模型时限。只有工程门槛通过后，才继续无永久角色分工的
多任务候选分支和世界模型预测评测。

## 后续单次真实启动验证：短窗口通过，整体仍拒绝

用户另行确认最长180秒的临时暂停授权后，一次无解锁启动采样已完成，服务在约
44.148 raw秒后恢复，独立150秒恢复定时器也已停止。详见 STARTUP_EVIDENCE_CN.md
末节及私有 `startup_clock_pause_20261018_v1/`。本轮无动作、无ACK、无飞行或训练。

8组双机帧采样窗口4.510秒，raw偏移幅度下界2.304ms，按现有局部门槛通过；但
覆盖启动至退出清理的879点连续监测44.030秒，raw整段幅度下界16.821862ms，
仍被10ms门槛拒绝。realtime/boottime幅度下界均0。manifest的局部`complete=true`
不等于整段通过，另存复核报告明确`whole_interval_quality_passed=false`，未晋升
数据。完整监测原始文件经Windows/WSL重算一致，64个图像/ROS原始文件解码校验通过。

这次结果限制了把前述20秒空载暂停窗口解释为修复方案的范围：暂停服务本身不足以
保证真实启动全程稳定。未逐时刻记录内核校时状态，不能认定残留slew为唯一根因。
下一项可离线补全启动全程连续质量门控；新增系统干预或仿真仍需按其实际范围授权。

后续离线实现已完成：`startup_clock`接入无解锁启动证据模式，从传输初始化之前
监测至自有进程清理后；manifest `/2-draft`延后最终完成，持续采样失败/环境变化/
覆盖不足或清理失败均拒绝。沿用10ms时钟门槛，另有50ms目标采样/250ms最大实际
间隙的draft覆盖规则。26项新增故障/runner接线回归及原232项在Windows/WSL各
258项通过；上述879条真实旧数据被新函数正确拒绝，两端重分析一致且源文件未改。
这只是离线实现与故障拒绝验证，不是环境修复或新runner实测。详见启动证据文档末节。

## 新runner保持服务原状的单次实测

2026-10-18，`df8ceee` 新runner完成一次不解锁、不起飞启动，未暂停同步服务，
无策略调用或训练。8组双机帧局部raw幅度下界1.390176ms通过，但从传输初始化
前到自有进程清理后的717点/36.206730070秒连续记录raw幅度下界16.717833ms，
触发整段10ms门槛。最大实际采样间隔182.993974ms通过250ms覆盖规则，其余
realtime/boottime幅度下界均0；没有监测或清理错误。

最终manifest明确 `capture_complete=true`、`session_completed=true`、`complete=false`，
summary为 `simulator_error/startup_session_quality_failed`，原startup_only结果另存。
Windows/WSL离线重算完全一致，64份原始文件核验，WSL全部解码通过；无动作/ACK，
自有进程已停止，服务前后active/running、enabled。原始记录及复核脚本留在私有
`startup_session_gate_live_20261018_v1/`，详细口径及哈希见 STARTUP_EVIDENCE_CN.md。

这完成新runner真实拒绝路径验证，没有修复环境或产生模型/迁移成绩。与暂停轮次
boot不同，不能作服务效果的因果比较；没有额外逐时内核参数证据，根因仍未唯一
确定。不追加系统改动或仿真，不放宽质量门槛；新数据采集仍受该门控约束。

## 后续：只读内核上下文诊断

`clock_kernel_probe` 补充每次时钟采样附近的adjtimex(modes=0)参数及服务端点，
不写系统时间、不暂停同步服务。一次20秒空载诊断中，400条记录的RAW相对
MONOTONIC偏移幅度下界552.226629ms仍被原10ms门槛拒绝，同时tick在
9167–10000微秒间变化。不能仅据freq字段或“已同步”状态认定速率稳定。
两端独立重算一致；25项新增回归及相关测试共131项在Windows/WSL通过。
这不是环境修复、起点匹配或模型成绩，根因未唯一确定；详细口径、哈希及
复现命令见 [CLOCK_KERNEL_CN.md](CLOCK_KERNEL_CN.md)。本轮未启动仿真或训练。
