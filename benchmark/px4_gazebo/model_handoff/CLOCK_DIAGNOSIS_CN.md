# 时钟诊断：累积漂移与跨启动证据边界

这是候选分支比较的工程质量前置检查，不是模型评测。主线仍是部分可观测、
队友行为变化下双机协作技能与世界模型预测的组合迁移；不固定无人机永久角色。
本轮未启动Gazebo/PX4、策略或训练；未更改系统时间、NTP服务或任何全局配置。

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

## 下一步需要单独的系统级授权

当前尚未证明稳定采样或分支起点可匹配，不扩大训练数据、不安排新飞行。
可考虑临时暂停WSL时间同步服务的短时对照，并在结束/出错时恢复原活动状态，
不改enabled状态或永久配置。但暂停服务不会自动撤销已经存在的内核slew；
没有变化也不能据此排除它的历史作用。该干预须另获用户明确授权，不能把
“继续研究”解释为可以修改主机/WSL时间。Windows全局校时也不自动执行。
