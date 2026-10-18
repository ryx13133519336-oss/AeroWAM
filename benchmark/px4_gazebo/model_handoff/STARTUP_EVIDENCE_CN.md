# 未解锁启动证据：不是分支恢复

主线仍是：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和
世界模型预测能力能否组合迁移到新任务。无人机编号不预定义永久任务角色。
本工具只补真实分支比较之前缺失的原始起点记录，不构成技能或预测成绩。

## 范围与固定规则

`search_pilot --startup-evidence-frames N` 为 opt-in，默认0关闭，启用时允许2..20。
它只接受 `rgb_scan`、nominal response，拒绝 `--execute-flight`、
`--control-link-check` 和 checkpoint/rgb_search 策略。CLI和 `run_condition` 都有入口检查。
此模式不创建控制发布端、不给策略发送观测、不调用策略、不发送解锁/未解锁或运动命令。
仅启动已有隔离SITL实例20/21；已有进程占用时拒绝，不清理其他会话。

通过原有启动检查后，最多15单调时钟秒采样。每个采样需要两机RGB/labels共同
Gazebo时间戳，采样之间至少250仿真毫秒；两机odometry sample stamp也须各自前进。
接收年龄限制为status 2秒、odometry 0.5秒、相机0.25秒、clock 0.5秒；
图像落后sim clock最多200毫秒，拒绝未来接收时间、非有限状态、未知坐标系、
已解锁或failsafe状态。采样时冻结同锁保护的消息引用及接收时刻；消息接收回调
替换引用，不就地修改旧消息。序列化及磁盘写入不持有该锁。

私有 `evaluator/startup_evidence/` 保存：

- `manifest.json`：固定采样规则、缺失状态和完成状态。
- `samples.jsonl`：位置、速度、四元数、角速度、原始PX4时间戳、reset counter、
  质量、状态及各接收时刻；每个原始文件的路径、字节数、SHA256。
- 每帧两机各一份原始RGB/labels protobuf及odometry/status ROS CDR，合计8文件。
- 逐采样host realtime/monotonic/raw/boottime读取区间和独立时钟诊断。

使用现有 `clock_probe` 的10毫秒相邻/整段偏移变化及读取不确定性诊断门槛；
任一路可用时钟的诊断失败，则保留全部原始文件，`complete=false`，退出2。
当前runner另要求下述会话连续监测通过，使用manifest `/2-draft`；采齐帧后只标
`capture_complete=true`，此时`complete`仍为false。进程清理后才决定最终完成。
`complete=true`只表示声明范围内的接收采样/时钟及清理门槛完成，不表示状态恢复
或分支匹配。采样频率不足以排除间隔内的瞬时变化。旧`/1-draft`仅检查采样窗口，
不能作为整个启动阶段通过的证据。
目录独占创建；未采齐、IO失败、陈旧数据等不得取得完成标记。

这些仍是接收时状态，不是曝光时状态。没有physics/controller/estimator内部快照、
RNG状态、待处理控制/传感器队列、task/teammate状态。始终保留
`start_matched=false`、`full_state_restored=false`、`exposure_state_aligned=false`。
不产生 candidate_scoring 分支结果或utility；不能据此填 `start_matched=true`。
静止目标相位无变化也不能补足飞行时动态目标的恢复证据。

## 2026-10-18实测：两轮均未过质量门槛

按 experimental-design 的实验单位/重复原则，事先固定两次相同配置独立启动，
每次8组双机采样。仅一场景、无候选动作处理，不进行因果或功效推断；
16组连续采样不是16个独立场景，也不是候选分支重复。

数据分别留在 `AeroWAM-data/disarmed_startup_evidence_20261018_r1` 和 `r2`，
不提交Git。之前20秒空载时钟预检通过，不据此豁免运行中检查。
两轮场景SHA256均为
`e22a0eaa3108a4810e59e5e733c3a5180bc5e3598ef147e165f3050475799a38`。

| 记录 | ROS/图像启动等待 | 双机采样 | 图像仿真跨度 | 采样host跨度 | 最大相邻raw偏移变化 | realtime偏移变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 90.295秒 | 8 | 1.848秒 | 4.260秒 | 57.942毫秒 | 读取区间外未检出 |
| r2 | 57.515秒 | 8 | 1.848秒 | 4.059秒 | 54.506毫秒 | 2967.127毫秒 |

两轮均 `simulator_error/ValueError:Sampled host clocks inconsistent during evidence capture`，
manifest为未完成，不把采齐文件解释为质量通过。两轮各64个原始文件经独立只读
SHA256/长度校验、protobuf/CDR反序列化通过；图像stamp、odometry位置/角速度/sample stamp、
status未解锁及无failsafe与JSON一致。动作请求0、ACK为空、没有control_link.json，
contacts为空、自有进程全部停止、cleanup_errors为空。事后无相关仿真进程残留。

r1首末monotonic增量4.259714114秒、raw增量4.646961401秒，说明至少存在
相对速率差，不能把每个raw诊断事件均叫作突跳。r2另有realtime约2.97秒相对变化。
只读检查显示WSL clocksource为tsc、NTP active，timesyncd报告过秒级offset且
最近日志有服务启停/初始同步；这些是环境线索，不是对NTP/WSL根因的因果证明。
本轮未修改系统时间、NTP服务、网络或全局配置，未训练、未起飞。

## 验证与后续门槛

后续更新：时钟分析已升级为 `aerowam-host-clock-analysis/2-draft`，保留相邻10ms
检查并新增整段偏移幅度10ms检查，详细方法/空载真实反例见 CLOCK_DIAGNOSIS_CN.md。
新的manifest记录分析profile和起止boot/uptime/clocksource。上面两轮是升级前原始
记录，均已被旧门槛拒绝，未补写或覆盖；不能倒填它们没有记录的boot证据。

17项新增合成/故障测试及200项已发布相关回归，Windows/WSL各217项通过。
包含禁止运动入口、未完成/IO失败、重复图像/遥测、陈旧/未来时刻、坐标系、
非有限角速度、原始文件哈希和时钟变化拒绝。失败采样正确保留并拒绝是
记录功能的验证结果，不是仿真质量通过，更不是组合迁移成功。

接下来先定位运行期时钟变化；若需要修改系统时间服务，必须单独授权，
不自动停NTP、不放宽门槛或无限重试。不扩大世界模型训练数据。
即使时钟稳定，仍需明确飞行前缀重放/完整状态恢复的可行性及匹配容差，
覆盖动力学、控制器/估计器、目标、队友/任务状态和传感器时序。
完成真实候选配对后才接独立预测评分，之后再检验角色自由与新任务组合迁移。

复现无飞行记录（新目录；不要附加飞行或命令检查参数）：

```bash
source /opt/ros/humble/setup.bash
source /home/ryx/px4_ros2_ws/install/setup.bash
cd /mnt/d/AeroWAM
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.search_pilot \
  --px4-root /home/ryx/PX4-Autopilot \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --agent /mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent \
  --policy rgb_scan --startup-evidence-frames 8 --startup-timeout 150 \
  --instruction "Find the blue cube and have the other UAV confirm it" \
  --expected-target blue --output-dir /absolute/new/private/output
```

指令只沿用现有启动工具的必填契约；本模式不执行搜索或评分，不把旧固定角色
搜索runner包装成新角色自由任务闭环。

## 2026-10-18：临时暂停同步服务的一次启动采样

用户单独确认“临时暂停同步服务最多180秒，仅一次不解锁、不起飞启动采样，结束或
失败立即恢复”后执行本轮。使用同一场景/资源、8组双机帧、原有10ms门槛；启动等待
预算固定110秒，不重试。外部私有监督器使用raw时钟140秒上限，停止服务前先创建
150秒独立恢复定时器。仿真以普通用户ryx运行；同步服务只临时stop/start，不改enabled、
NTP配置、Windows时间或内核参数，不调用策略、不训练。

私有目录为 `D:/AeroWAM-data/startup_clock_pause_20261018_v1/`。实际服务暂停约
44.147932 raw秒，runner正常退出后立即恢复；原活动状态active/running、enabled已
核验，独立恢复定时器已停止且卸载。没有其他仿真进程残留。单次启动不能作为前两轮
不同boot的随机对照，也不能据32.743秒启动等待声称DDS稳定性或因果改善。

| 检查范围 | 样本数 | monotonic跨度/秒 | realtime偏移幅度下界/ms | raw偏移幅度下界/ms | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 双机原始帧采样窗口 | 8 | 4.510258 | 0 | 2.303655 | 通过 |
| 独立连续监测（启动至退出清理） | 879 | 44.029818 | 0 | 16.821862 | 拒绝 |

原runner返回 `startup_only/no_motion_commands_requested`，manifest的`complete=true`
只覆盖第一行的8帧采样规则，不能扩展为整个运行期通过。外部连续监测另保存
`paused_host_clocks.json`，触发`clock_offset_excursion`；私有复核报告明确
`whole_interval_quality_passed=false`。两种记录均保留，不反写原始manifest或summary，
不将此轮晋升训练数据或候选分支配对。原工具尚未把启动/清理期连续监测接入完成
门槛，这是下一项离线实现工作；不能用一次外部监督器代替该功能。

连续采样的采集前子区间702点/35.155301秒已出现13.673123ms raw偏移幅度；覆盖8帧
首末的连续子区间90点/4.462750秒约2.339031ms，退出阶段87点/4.311467秒约1.757591ms。
这些区间是事后定位，不是多次独立实验或新门槛。整段首末净raw偏移仅约4.458ms，
仍不能代替16.822ms整段幅度。下界0不表示时钟物理上绝对无变化。

所有上下文/probe及采样起止boot一致：`99acf337-f292-4a66-9892-cc13a95e46f4`。
暂停与监测上下文均记录服务inactive/enabled；恢复上下文active/enabled。
没有逐时刻内核校时参数记录，因此不能唯一归因于残留slew、WSL或某个同步来源。
本轮证明“暂停服务”不足以保证这次整个启动过程通过当前时钟门槛；不证明它对所有
时钟异常均无效，也不把此前秒级realtime异常与此次raw变化混为同一已证实机制。

64个原始文件均独立校验SHA256/长度并反序列化：32份1280×960 RGB/labels protobuf、
32份odometry/status ROS CDR；图像共同stamp、遥测字段和JSON一致，sample stamp各自
前进。接收年龄保守上界：status1.071311秒、odometry0.010525秒、camera0.070880秒、
clock0.003053秒。图像跨度1.848仿真秒。这些仍是接收时状态，不能称曝光状态对齐。
动作请求和ACK均0、没有control_link/flight_diagnostics，contacts和cleanup_errors为空，
自有进程停止。Windows另从连续原始时钟采样重算，与WSL报告完全一致。

私有核验入口为本轮目录的 `verify_evidence.py`，结果 `verification.json`；关键源哈希：

- paused_host_clocks.json：`1e40f984552ef36335b13ec056fcf66c61ffd0e6a781bae14bfc418476539090`
- startup_evidence/manifest.json：`9d6daf86bf677f8faf5b891c8fb940d999c7a702caa0ac5565f2703f6bec9cbe`
- startup_evidence/samples.jsonl：`c8dde79eb266ab583dc7cf80a9073a6c5e94f10a5ee48493e123f536ac352ce2`

本次系统干预授权已执行完，不追加暂停或仿真。下一步先离线补全整个启动期的连续
时钟记录/拒绝逻辑，再决定新环境验证；不放宽门槛、不给模型oracle输入。主线继续
保持无永久角色分工的协作技能与世界模型预测组合迁移；本轮没有产生该主线的成绩。

## 连续会话门控实现：离线阶段记录

`search_pilot --startup-evidence-frames`现已自动启用 `startup_clock`。监测在ROS/Gazebo
传输初始化和任何自有进程启动之前开始，在ROS关闭、自有进程停止及日志关闭之后
结束；不覆盖此前Python导入、静态场景/资源准备或最后报告序列化的耗时。普通飞行
和命令检查模式不启用此功能。没有新系统服务设置或仿真启动。

私有 `evaluator/startup_host_clocks.json` 保存完整原始读取区间、起止环境和报告，
保持 `aerowam-host-clock-probe/1` 原始格式，新增 `session_quality` 字段。
报告 `aerowam-startup-session-clock-quality/1-draft` 固定目标50ms间隔、同步端点读取、
最多10000条内存采样；超过容量直接失败，不截断历史。相邻10ms/整段10ms/读取宽度
10ms检查沿用profile 2-draft；另外要求实际最大采样间隔不超过250ms、realtime/raw/
boottime三路均可用、起止boot及clocksource非空且相同。这是新增的draft覆盖条件，
不能把250ms间隙解释为其中没有未观测到的变化。

manifest `/2-draft` 的 `capture_complete` 表示帧采样阶段已完成其局部检查，
`session_completed` 表示runner正常结束且自有进程停止、无清理错误；`complete`只有
在两者及连续时钟检查均通过后才为true。`session_clock_quality`包含报告文件的相对
文件名、长度及SHA256（路径相对于evaluator目录）。summary保存
`startup_clock_quality` 与 `startup_receive_quality_passed`。帧已采齐但全程门控失败
时，category为`simulator_error`、reason为`startup_session_quality_failed`，
`result_before_startup_clock_gate`保留门控前结果。已有启动失败原因不被时钟问题覆盖。

启动/后台线程读取异常、线程无法停止、样本上限、数据缺失、环境不可核实、采样
间隔过大和时钟报告落盘失败均不得通过。即使ROS或日志清理异常，也优先停止自有
进程并保存失败状态；没有创建采样manifest的早期失败仍保存连续监测结果。原始帧
和失败报告留在私有目录。direct `StartupEvidence` 的旧局部API仍支持 `/1-draft`，
runner始终显式要求会话门控，不将旧API或旧记录视为新版本通过。

新增26项无飞行回归；加上已发布232项，Windows/WSL各258项通过。包含真实runner
配内存传输/进程的接线回归：启动前开监测、清理后收尾、准备阶段失败、ROS清理
失败和退出阶段漂移；这些不是真实仿真结果。

另对上轮879条保存数据只读应用新质量函数：因16.821862ms整段raw幅度拒绝，新增
覆盖条件在这份记录中没有额外issue。Windows/WSL重分析完全一致，源文件和旧
manifest哈希未变；新报告另存私有 `session_gate_reanalysis_20261018_v1.json`。
不重写旧`complete=true`，此阶段尚未用新runner完成实际采样。新环境实测、曝光时
状态对齐、候选同起点匹配和世界模型预测成绩仍待完成。

## 2026-10-18新runner实测：局部采齐，全程门控正确拒绝

基于 `df8ceee` 运行一次无解锁、无起飞启动复验，时间同步服务保持原设置；没有
再次暂停服务或调整系统时间。启动等待预算固定110秒，8组帧的收集预算沿用15秒，
不重试，不调用策略或训练。私有目录：
`D:/AeroWAM-data/startup_session_gate_live_20261018_v1/`。

```bash
python3 -B -u -m benchmark.px4_gazebo.model_handoff.search_pilot \
  --px4-root /home/ryx/PX4-Autopilot \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --agent /mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent \
  --policy rgb_scan --startup-evidence-frames 8 --startup-timeout 110 \
  --instruction "Find the blue cube and have the other UAV confirm it" \
  --expected-target blue \
  --output-dir /mnt/d/AeroWAM-data/startup_session_gate_live_20261018_v1
```

英文指令及私有蓝色评分目标只用于满足启动入口合同，此模式没有向模型发送观测
或指令，不产生搜索/确认成绩。场景身份仍为
`e22a0eaa3108a4810e59e5e733c3a5180bc5e3598ef147e165f3050475799a38`。
该身份按XML根序列化计算，不能与包含XML声明的world.sdf文件字节哈希混用。
起止boot一致：`68447553-0d1b-4e6c-acd6-d36024ce6064`，clocksource为tsc。

| 检查范围 | 样本数 | monotonic跨度/秒 | raw偏移幅度下界/ms | 结果 |
| --- | ---: | ---: | ---: | --- |
| 双机帧采样窗口 | 8 | 4.529476837 | 1.390176 | 局部通过 |
| 传输初始化前至自有进程清理后 | 717 | 36.206730070 | 16.717833 | 拒绝 |

启动流等待27.197668601秒，图像跨度1.848仿真秒。全程最大实际采样间隔182.993974ms，
未超过250ms覆盖限制；raw相邻最大偏移差0.842253ms，realtime/boottime整段幅度下界
均0，monitor_errors为空。本轮只有 `clock_offset_excursion`，没有监测线程或清理
故障。raw首末净偏移区间约[2.343666, 2.344147]ms，仍不足以替代16.717833ms整段幅度。

manifest `/2-draft` 如实保存 `capture_complete=true`、`session_completed=true`、
`complete=false`。summary最终为 `simulator_error/startup_session_quality_failed`，
`startup_receive_quality_passed=false`；`result_before_startup_clock_gate`保留原本的
`startup_only/no_motion_commands_requested`，CLI非零退出。pair_summary同样记录
simulator_error。真实runner已验证局部采齐后不会把全程不合格记录误标完成；这不是
环境修复验证，也没有真实全程质量通过的结果。

私有 `verify_evidence.py` 独立核对全部64个文件的SHA256/长度，WSL另反序列化32份
1280×960 RGB/labels protobuf及32份ROS CDR，与JSON中的时间戳、状态/位姿等字段
一致。两机8组记录均未解锁且无failsafe，动作请求/ACK均0，无control_link或
flight_diagnostics，自有进程停止且cleanup_errors为空；退出后独立pgrep没有发现
PX4/Gazebo/Agent或其他采集任务残留。服务前后只读检查均active/running、enabled，
没有改变其设置；这些端点检查不证明服务在整个区间的内部同步状态。

接收年龄保守上界：status0.991257秒、odometry0.052065秒、camera0.110312秒、
clock0.030344秒；仍不能当作曝光状态对齐。Windows独立重算连续时钟及局部报告、
核验64份文件，除WSL额外解码的标记外，两端复核报告完全相同。关键原始哈希：

- startup_host_clocks.json：`140e6b9277892d89bbbd6485ca447de0b68af0dbef3ded7e1d3e939c52d69eb0`
- startup_evidence/manifest.json：`e1720f5994dac5ba0d6a5906c5ae6ff2b11769c9a78e558b73e9bcd4279a34fc`
- startup_evidence/samples.jsonl：`d63242f1a0aa9e070cdf7585799aedc6bd2bea9c22ec1d448376ae10405d3c31`

保留原始失败证据，不晋升训练或候选分支数据。与此前暂停服务轮次boot不同，不据此
作暂停/未暂停的因果比较，不唯一归因于NTP、WSL或残留slew。当前连续门控已完成
一次真实拒绝路径验证；可用环境、曝光状态对齐、候选起点匹配和组合迁移成绩仍待
解决。本轮不追加仿真或系统干预，不放宽10ms/400ms限制；无人机没有永久角色分工。
