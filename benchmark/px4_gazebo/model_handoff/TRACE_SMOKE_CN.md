# 发布控制记录的单场景仿真验收

日期：2026-10-14。WSL/Linux、headless Gazebo lawn、双 PX4 SITL、oracle。
场景为冻结 wall-blocking manifest 的 train_000，扰动概率 0.15。
这是接口/采集链路 smoke，不是新模型评测或完整决策 benchmark 验收。

## 发现与修复

初次运行 attempt_00 任务成功，但样本 17/18（protocol_t 85/86）相机最大时间戳
均为 43332000000 ns，严格验证拒绝。原因是专家交接事件会在普通记录步之后
额外采样，而相机新帧尚未收到。数据未晋升到成功数据目录。
原代码 retry 的 attempt_01 成功，保留在 oracle_trace_smoke_v2 / dataset_trace_smoke_v2；
它不能作为修复有效的证据。

修复后，启用 dataset recorder 时在调用有状态专家/协议前检查相机最大时间戳
是否比最近写入样本前进；没有前进就等待，保持原发布 tick 和安全/流超时检查。
专家不会在等待过程中提前消费 handoff_request，确认 pending 逻辑保留。
写入器另在写图像前拒绝重复或倒退时间戳，防止写入半条重复样本。
未启用 recorder 的模型评测路径不受门控影响；100 ms 图像同步限制未调整。

## 修复后的证据

独立结果目录：`/mnt/d/AeroWAM-data/oracle_trace_smoke_frame_fix_v2`。
独立数据目录：`/mnt/d/AeroWAM-data/dataset_trace_smoke_frame_fix_v2`。
日志：`/mnt/d/AeroWAM-data/px4_logs/trace_smoke_frame_fix_v2/train_000/attempt_00/20261014_103038_219320`。
episode：`live_handoff_safe_occlusion_v3_36859c79d3d5`。
首次 attempt_00 成功并落地：`total=1, completed=1, simulator_error=0`。

| 检查 | 结果 |
|---|---|
| 数据 schema | aerowam-model-dataset/2 |
| RGB 样本 | 28；哈希、256×256 RGB JPEG、时间及真值对应检查通过 |
| 专家标签 | none 26 / handoff 1 / confirm 1；历史一致性检查通过 |
| 发布区间 | 706；每机 353；区间连续、端点遥测非未来、样本锚点及身份检查通过 |
| 相机双视角最大时间差 | 64 ms；原 100 ms 限制通过 |
| 发布区间均值 | 每机约 50.927 ms |
| 最长发布区间 | UAV0 356.567 ms / UAV1 356.440 ms |
| 开始端点最大遥测接收年龄 | 373.080 ms（host monotonic 接收年龄，非传感器曝光年龄） |

另外独立调用 dataset 验证器，同时要求 expert supervision 和 execution trace，通过。
本次创建的 DDS agent 及仿真进程结束后已清理。
interface_self_test、repair_self_test、live_handoff_self_test、model_handoff.self_test
及 model forward 检查通过；git diff --check 通过。

## 尚未验收的部分

记录语义仍是发布 setpoint，而非 PX4 确认执行。host monotonic、host UNIX、PX4
timestamp、ROS image stamp 尚未对齐。最长发布间隔和遥测年龄说明不能把每条记录
当作理想 50 ms、严格同步的状态转移。该 smoke 没有降低质量门槛、修复基础设施
偶发 set_pose timeout，也不能证明批量运行稳定或世界模型能泛化。

下一步需要时钟映射、新鲜度与间隔质量标记、独立多任务验收、候选执行与匹配分支。
明确这些条件后才采集新的世界模型转移/候选收益监督；不需要重采旧 280 条专家数据。

## controls/2 时钟证据 smoke（2026-10-14）

新增采集 `/aerowam/sim_clock` 和两机 DDS TimesyncStatus，及样本图像/遥测接收时刻。
本机 PX4 serializer 对 timestamp/timestamp_sample 已施加 session offset；新版 trace
保留接收值，不再次加减偏移，仍明确标记 clock alignment 未验证。

独立目录：`/mnt/d/AeroWAM-data/oracle_trace_timing_smoke_v3` 和
`/mnt/d/AeroWAM-data/dataset_trace_timing_smoke_v3`。
episode：`live_handoff_safe_occlusion_v3_7e985ff09959`。
train_000 首次 attempt 成功：total=1, completed=1, simulator_error=0。
独立 dataset 验证器要求 expert supervision 与 execution trace，28 样本验证通过。
本次仿真及创建的 DDS agent 已退出。

| 检查 | 结果 |
|---|---|
| 发布区间 | 718；每机 359 |
| 最长发布区间 | UAV0 91.824 ms / UAV1 91.811 ms |
| 最大端点遥测接收年龄 | UAV0 136.130 ms / UAV1 136.236 ms |
| 最大样本图像接收年龄 | UAV0 491.642 ms / UAV1 325.033 ms |
| 仿真 clock 证据 | 584 条 |
| DDS timesync 证据 | UAV0 31 条 / UAV1 37 条 |
| offline draft 接收质量 | 不通过；两机图像年龄、一侧复用 image stamp、host 时钟配对不一致 |

一次 clock 证据的 host UNIX 增量与 monotonic 增量相差约 1800 ms，发生于 RUN 之前。
顺序读取两个 host 时钟不是原子操作，可能是系统时钟调整，也可能是读取之间进程被
暂停；现有证据无法区分。因此报告 `host_clock_pair_inconsistency`，不宣称已定位 NTP
或 WSL 故障。RUN 内 command 时间增量最大差约 0.324 ms。

任务成功与结构验证通过不代表接收质量合格，更不代表精确转移监督就绪。
上述异常保留在报告中，不降低 draft 门槛、不删除样本、不改写任务结果。
仍需严格的双帧前进/新鲜度采样策略、时钟读取不确定性测量和曝光对应状态验证。
无飞行 timing fault、interface、Handoff/FixedRole、专家修复、model forward 回归通过。

## 双帧采样与读取区间 smoke（2026-10-14）

controls/3 增加 UNIX 读取前后的 monotonic 区间，区分读取不确定性和偏移不一致。
新增 trace 采样必须两路分别前进、接收年龄 <=250 ms、相机 skew <=100 ms；
持续不满足已有 camera_timeout 则以 simulator_error 结束。旧采集/评测模式保持原规则。

独立结果/数据：`oracle_trace_bounded_sampling_smoke_v4` /
`dataset_trace_bounded_sampling_smoke_v4`，均位于 `/mnt/d/AeroWAM-data`。
日志：`/mnt/d/AeroWAM-data/px4_logs/trace_bounded_sampling_smoke_v4/train_000/attempt_00/20261014_112245_954087`。
episode：`live_handoff_safe_occlusion_v3_b184084b8fe4`。
首次 attempt 成功，28 样本、730 发布区间（每机365），独立专家/trace/dataset 验证通过。
本次创建的 DDS agent 和仿真进程已清理。

| 检查 | 结果 |
|---|---|
| 最大图像接收年龄 | UAV0 149.235 ms / UAV1 149.952 ms；两路均前进，无复用 image stamp |
| 最大端点遥测接收年龄 | UAV0 113.146 ms / UAV1 125.848 ms |
| 最大发布间隔 | UAV0 103.672 ms / UAV1 103.589 ms；仍高于原100 ms draft门槛 |
| 时钟证据 | sim clock 582；DDS timesync UAV0 34 / UAV1 40 |
| 最大时钟读取区间 | 0.013251 ms |
| 整段 draft 质量 | 不通过；发布间隔、RUN 内 host UNIX/monotonic 偏移不一致 |

两机 protocol_t 80→81 的 UNIX 与 monotonic 增量差均约 +2400 ms；对应 sim clock
证据增量为 28 ms、接收 monotonic 增量 133.244 ms、读取区间宽度仅0.000857 ms。
这次不一致不能由读取区间内约2.4秒的进程暂停解释；具体操作系统/时钟原因尚未定位。
host monotonic 发布区间不能被当作已验证的真实动作保持时间，也不跨此异常强行拟合
UNIX/仿真时钟。保留记录、失败质量标签与既有100 ms门槛，没有重采旧专家数据。
只读环境检查：WSL2 kernel 6.18.33.2、当前 clocksource 为 tsc、NTP 与同步状态均为 yes。
筛选内核时间源日志仅见启动时切换，未见能解释这次差异的明确故障记录；这些状态
不能证明异常来自 NTP 或 tsc，也不足以授权/支持修改系统时间源。

这次确认了采样门控能保留 handoff/confirm 监督并解决上一轮旧帧/复用问题；
不证明长期稳定、完整时钟对齐或世界模型监督就绪。后续需处理运行环境时钟不一致、
发布时限及曝光对应状态。无飞行单侧冻结/年龄/恢复、clock bracket/暂停/真实偏移、
旧 trace 兼容、接口、Handoff 和专家修复回归通过。

## 无仿真host时钟定位与新采集质量隔离（2026-10-14）

新增 `clock_probe`，20秒无Gazebo/PX4/模型推理检查，400个样本。
证据：`/mnt/d/AeroWAM-data/host_clock_probe_idle_v1.json`。
UNIX相对monotonic四次偏移变化：约 -1836、+1847、-1893、+1895 ms。
最大UNIX读取区间0.018025 ms；raw、boottime相对monotonic最大区间外偏移变化均为0。
最大轮询间隔50.699 ms。该证据说明实时时钟异常可独立于仿真出现；不是模型推理
才能触发的问题，不把它简单归因于set_pose。具体系统原因仍未确认。
同期WSL时间服务日志有启动/停止记录，但仅凭这些记录不足以断言NTP是原因。

collect新增可选接收质量门槛、成功尝试质量报告，以及新批次启动前host时钟探针。
任务成功但质量不合格时不晋升正式数据/结果目录，原始success和图像仍在attempts；
分类trace_quality_error，与simulator_error及模型失败分开。历史attempt递增保留，
续跑不会覆盖原异常成功结果。新门槛通过也不等于精确转移监督就绪。

无飞行测试覆盖启动前拒绝不启动sim、拒绝保留task success、报告保留、下一次晋升、
resume重验、默认未启用门槛时保留原行为、探针真实偏移/读取不确定性/文件不覆盖。
对已保存的controls/3真实异常数据只读resume检查正确拒绝，未启动仿真，未修改原结果。
本轮没有再飞行、没有扩大新数据或启动模型训练，没有修改系统时钟/NTP/clocksource。

第二次20秒真实批次启动前检查通过，证据：
`/mnt/d/AeroWAM-data/clock_preflight_check_v1/.diagnostics/host_clock_probe_42d8f0162df04f818626643d33e9024c.json`。
本次同样没有启动仿真。两次窗口不同结果说明该异常间歇出现；通过一个短窗口
不能保证后续整场稳定，启动前门槛不能替代落地后的完整trace质量报告。
