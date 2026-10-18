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

使用现有 `clock_probe` 的10毫秒相邻偏移变化/读取不确定性诊断门槛；
任一路可用时钟的诊断失败，则保留全部原始文件，`complete=false`，退出2。
`complete=true`只表示此次采样与上述门槛完成，不表示状态恢复或分支匹配。
采样频率不足以排除间隔内的瞬时变化，也未覆盖整个启动阶段的时钟。
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
