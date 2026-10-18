# 模型接入接口（2026-10-14）

2026-10-18 不预分配角色约束：新任务使用 `role_free_task` / `aerowam-language-task/2-draft`，
无固定 source/receiver、高低空或机号职责。事件发生后由私有评分归属此次发现/交接，
不向策略下发角色答案。双机动作与候选协议不变；旧 /1 任务、规则策略和运动评分门
仍仅为固定角色诊断，不能作为新主线结果。见 ROLE_FREE_MODELS_CN.md。

当前主线（2026-10-18对齐）：在部分可观测、队友行为变化的环境中，评测双无人机
协作技能和世界模型预测能力能否组合迁移到新任务。以下含通信限制的历史目标
作为后续扩展，通信暂不作为当前实现前置条件。

统一目标（2026-10-15）：在部分可观测、通信受限、队友行为变化的环境中，
评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。
当前集中式双视角接口仅是通信不受限的接入基线，不代表最终信息权限。
后续受限通信接口必须约束队友RGB/状态/消息的实际送达，不允许共享字段
绕过消息限制；还需显式记录队友行为条件，并支撑独立任务与预测/决策评测。
不得用旧接口结果直接宣称完整研究目标已完成。详见 DECISION_BENCHMARK_CN.md。

旧交接接口保持 `aerowam-model-handoff/1`。新候选预测接口单独使用
`aerowam-candidate-decision/1-draft`，目前可做无飞行接入检查，尚未连接候选仿真执行器。
这不是完整多任务 benchmark 已实现的声明。通信限制仍暂缓。

## 旧交接模型与外部 VLN/VLA 的动作接口

输入由 `ModelObservation` 定义：原文 `instruction`、episode ID、双机 256×256 RGB JPEG、
图片时间戳、本机状态、最多 4 条历史建议/请求动作和最多 16 条消息。
`ModelObservation.from_wire()` 在进入模型前检查协议、公开字段、双机信息、有限数值和
100 ms 图片同步门限。关键字隔离只是结构检查，外部适配器同样不得读取 evaluator 文件。

| 字段 | 语义 |
|---|---|
| instruction | 自然语言原文，编码器由参评方法声明；不得按 fixture ID 查答案 |
| states.position / velocity | 各机起飞原点、初始朝向的局部 FLU，单位 m / m·s⁻¹；不是同一个全局参考系 |
| states.yaw | 相对初始朝向的 yaw，rad；动作使用当前机体朝向 |
| timestamp_ns / image_timestamp_ns | ROS 图片时钟；不能直接与 host monotonic 或 PX4 时钟相减 |
| action_history | 旧 oracle 数据是无扰动专家建议；在线模型是模型请求动作，都不是 PX4 确认执行历史 |
| message_history | 当前仅记录意图，不能冒充通信阶段的真实送达消息 |
| waypoints.uav_0 / uav_1 | 当前机体 FLU 下的 [dx, dy, dz, dyaw]，m / rad；旧 runner 每 0.5 s 决策 |
| intent | none / handoff / confirm |

动作限幅为 dx/dy ±1 m、dz ±0.5 m、dyaw ±15°。runner 将增量除以 0.5 s 转为速度，
再应用原有速度/加速度限制并转换 NED，20 Hz 发布；不保证精确到达请求位置。
默认 400 ms 时限覆盖连接、发送和所有分段接收；超时处理与原有安全规则保留。

2026-10-18搜索pilot可选`--teammate-response nominal/cautious/delayed`，仅运动模式
允许非nominal。该runner的`action_history`记录响应后交控制器的请求，原模型响应
另存；仍不代表限幅后的发布或物理执行。私有条件身份和响应调度不传策略，已知
队友条件、私有移动标定和真实候选收益字段在递归结构检查中拒绝。不是完整语义
泄露检测。该变化是UAV1命令响应扰动，不是独立队友决策，也没有改变旧交接runner。
实际响应规则、版本、日志及非飞行证据见SEARCH_PILOT_CN.md的队友响应小节。

最小外部策略位于 `adapter_example.py`，只返回 hover，用来检查接入，不能作为有效模型。
自定义 factory 返回 callable：`policy(wire_request) -> ModelJointAction.to_wire()`。

```bash
python3 -B -u -m benchmark.px4_gazebo.model_handoff.adapter_server \
  --factory benchmark.px4_gazebo.model_handoff.adapter_example:create_policy \
  --host 127.0.0.1 --port 8765
```

旧 runner 的 `--policy model --policy-endpoint HOST:PORT` 可连接返回旧动作协议的适配器。
批量配置目前仍只注册三个内置方法；外部 factory 服务入口不代表任意模型已适配，
外部批处理注册、来源/权重哈希/训练预算记录仍在待办中，不能用内置方法名字冒充外部模型。

## 候选条件预测与选择

`decision_interface.py` 定义请求、响应、外部 `DecisionAdapter` 和 `DecisionClient`。
模型实现以下入口：

```python
def predict(*, history, instruction, candidates, horizon_s, task):
    # 实际模型必须对每个候选动作序列单独预测。
    return predictions  # 每项 CandidatePrediction
```

请求包含 decision ID、task（handoff/search/inspection）、原文 instruction、同一 episode 的
严格递增观测历史，以及候选 ID 和双机动作序列。每个动作声明 `duration_s`；所有候选
累计持续时间必须等于 `horizon_s`。单步增量用当前机体 FLU，每步除以该步时长；
接口拒绝静默限幅的候选。安全控制器未来仍可能进一步限制实际执行，必须通过 trace 记录。
候选不得内嵌 oracle 路线、目标真值、障碍物地图、真实收益或未来结果。
动作候选的提供者和执行器需统一版本化；技能名称到动作的执行器尚未实现。

每个 `CandidatePrediction` 返回对应候选 ID、预测 utility、分别属于两机的目标可见概率、
交接成功概率、确认概率和碰撞概率。每个 ID 必须出现一次，概率在 [0,1]，utility 有限。
本草案中 visibility 指窗口末可见；交接/确认/碰撞指窗口内对应事件。
事件标签必须在对应任务 evaluator 和正式窗口定义冻结后才能用于采集/训练，
不能将几何安全终止直接写成物理接触。非交接任务对交接字段记 0 并标注不适用。
未来视觉 latent 输出/损失及辅助评测留待预测模型阶段扩展，不影响旧动作接口。

固定 `select(request, predictions, max_collision_probability=...)` 可按预测风险筛选后选最高
utility；平局按请求候选顺序，全部被筛除时返回 None。默认没有风险阈值，utility 的定义、
阈值和 abstain 执行方式仍需冻结。预测风险不是安全保证，也不替代飞行安全控制。
接口没有使用真实候选收益，不会自动生成反事实监督，更不代表模型已经有预测能力。

2026-10-18：新增独立 `candidate_scoring.py`，绑定公开请求/预测与分离的私有分支结果，
检查计划候选×重复覆盖、起点匹配声明、故障与删失，再评分utility、排序/regret和逐机
概率。只消费外部证据，不执行候选或认证仿真恢复；无预分配角色，拒绝旧固定方向intent。
23项新测试与显式合成CLI复核通过，尚无真实配对分支成绩。定义、限制及用法见
[CANDIDATE_SCORING_CN.md](CANDIDATE_SCORING_CN.md)。动作菜单标签不能冒充候选后果预测。

通过 `DecisionAdapter(your_predictor)` 接入同一个 factory 服务入口。
`DecisionClient.predict(request)` 返回逐候选预测与调用时延；400 ms 默认预算保留。
**候选协议不能直接连接旧 live runner**，旧 runner 只接收动作；新闭环选择/执行路径后续接入。

## 发布控制记录接口

采集和 runner 新增可选 `--record-execution-trace`；runner 要求 `--record-dir`。
开启后数据为 `aerowam-model-dataset/2`，policy 目录新增 `published_controls.jsonl`，
sample 增加 `protocol_t`。默认仍输出原版 dataset/1。旧数据不能事后推算该记录，
启用新选项重用旧成功目录会被验证器拒绝，应使用独立的新目录。

trace 记录 RUN 阶段每次发布的机体 FLU 速度/yaw-rate、转换后的 NED 速度、对应策略步、
DDS 接收的 PX4 odometry 和样本锚点。首次发布发生在第一次策略计算之前，因此 protocol_t 可为 None；
终止策略步可能没有真正发布，不按 protocol 日志猜测执行。降落开始关闭 RUN 区间，落地后写盘。
区间表示相邻发布调用之间的 host monotonic 时间，不是 PX4 应答确认的保持时间。

host monotonic、host UNIX、DDS odometry timestamp 与 ROS image stamp 分别存储，尚未对齐。
本机 PX4 生成的 DDS serializer 已对 timestamp/timestamp_sample 加入 session time_offset，
不能把接收值直接称为 PX4 原始启动时间，也不能再次盲目加减 TimesyncStatus 的 offset。
锚点是采集调用开始时间，不是相机曝光时间；端点遥测是当时最近一次接收值，可能偏旧。
因此该文件支持追溯，尚不满足严格同步转移监督。日志异常不会阻止降落，异常/缺失文件会使
dataset/2 验证失败；无飞行回归与单场景仿真采集 smoke 已通过，证据见
[TRACE_SMOKE_CN.md](TRACE_SMOKE_CN.md)。仍未满足精确转移监督的时钟与新鲜度要求。

2026-10-14 smoke 发现交接事件额外记录可能复用上一组相机帧，造成重复样本时间戳。
旧 recorder 路径等待图像最大时间戳前进。启用 trace 的新版路径则要求两路各自比上次
写入的 image stamp 前进；图像接收/时间戳前进、遥测接收年龄均须在 250 ms 内，
相机时间差仍不超过 100 ms，再调用有状态专家和协议。等待期间 tick 继续发布上一条
控制并执行原有流超时/安全检查，不提前消费专家交接请求。
连续等待达到已有 camera_timeout 时以 simulator_error 结束，即使只有一路冻结或
仍不断收到重复帧也不会永久等待；条件恢复后等待计时重置。
writer 在写图像前另拒绝任一路 stamp 未前进，metadata 标记
`aerowam-dual-frame-receive-sampling/1-draft`，离线 dataset 验证同步检查该约束。
该更严格路径仅用于 trace 采集；不改旧数据、旧 recorder 模式和无记录的模型评测。

新增记录使用 `aerowam-published-controls/3`；旧 controls/1、controls/2 仍可读，dataset/2 和
策略动作协议不变。collect 在启用 trace 时额外启动只读的世界 clock ROS 桥，runner
订阅 `/aerowam/sim_clock` 与两机 TimesyncStatus。手动启动 runner 时需自行提供该桥；
缺失证据会在离线质量报告中显示，不冒充已经对齐。clock 接收证据约 20 Hz，记录倒退；
样本另存图像/遥测接收时刻、DDS 时间和 host UNIX。无同步磁盘写入发布 tick。
controls/3 在每次 UNIX 读取前后记录 monotonic，给读取时刻一个区间，而不是假定
两次读取发生在同一时刻。质量报告以 UNIX-minus-monotonic 的可行区间比较偏移，
区间变宽标记读取不确定性；区间明显分离才标记配对不一致，仍不直接断言系统原因。

`trace_quality` 是独立的离线诊断，不改任务成功率，也不删除成功/失败样本。
quality/2-draft 接收质量门槛：发布间隔 100 ms，图像/遥测接收年龄 250 ms，
host 可行时钟偏移变化/读取区间宽度 10 ms。旧 controls/2 因缺少读取区间会标记证据不足。
这些门槛用于发现问题，尚未冻结为正式转移监督验收标准；接收年龄不是传感器生成年龄。
即使 `receive_quality_pass=true`，`precise_transition_supervision_ready` 仍为 false。

collect 开启 trace 时，在通过结构验证的成功尝试的 `evaluator/trace_receive_quality.json`
保存质量报告。默认只标记问题；新世界模型数据的 pilot 可额外启用
`--require-trace-receive-quality`：质量未通过的任务成功记录仍保留在 `.attempts`，
不晋升到正式结果/数据目录，耗尽重试后分类为 `trace_quality_error`，不算模型失败。
续跑重新计算质量，不按旧报告假定合格；不覆盖此前 attempt 和报告。
缺少时间证据的旧 trace 会被这个新门槛拒绝，但旧专家采集无需启用它。

启用质量门槛且还有新场景要运行时，批次启动前默认检查20秒 host时钟，
`--clock-preflight-seconds` 可设1..60秒。异常立即退出2，不启动仿真，证据位于
`output-dir/.diagnostics/host_clock_probe_<id>.json`。通过仅证明这段检查窗口无异常；
仍需运行中的 trace 质量验收。批次有 simulator_error 或 trace_quality_error 时也退出2。
此门槛是新数据筛选用途；不得用仅保留通过质量检查的成功轨迹冒充无偏模型成功率。

独立只读探针同时对照 UNIX、monotonic raw、boottime，不需要 ROS/Gazebo：

```bash
python3 -B -m benchmark.px4_gazebo.model_handoff.clock_probe \
  --duration-s 20 --interval-ms 50 --require-stable \
  --output /mnt/d/AeroWAM-data/host_clock_probe_manual_01.json
python3 -B -m benchmark.px4_gazebo.model_handoff.collection_quality_self_test
```

输出独占创建，请使用新文件名。该工具不会调整系统时钟、停止NTP或切换clocksource。
若 UNIX 单独异常而 raw/boottime 与 monotonic 稳定，说明异常可存在于host实时时钟，
仍不能直接推断NTP/WSL具体责任。系统级修复和运行环境复验需独立处理。

```bash
python3 -B -m benchmark.px4_gazebo.model_handoff.trace_quality \
  /path/to/episode/policy/published_controls.jsonl
# 如需让脚本在 draft 接收质量不达标时退出 2，再加 --require-receive-quality。
python3 -B -m benchmark.px4_gazebo.model_handoff.trace_quality_self_test
```

仅在小规模仿真检查时使用以下命令，不需要重采旧的 280 条专家数据：

```bash
python3 -B -u -m benchmark.px4_gazebo.model_handoff.collect \
  --mode collect \
  --manifest benchmark/px4_gazebo/model_handoff/manifests/handoff_wall_blocking_v1.jsonl \
  --splits train --limit 1 --record-execution-trace \
  --perturb-probability 0.15 --simulator-retries 3 --session-startup-timeout 150 \
  --output-dir /mnt/d/AeroWAM-data/oracle_trace_smoke_v2 \
  --dataset-dir /mnt/d/AeroWAM-data/dataset_trace_smoke_v2 \
  --auto-log-dir /mnt/d/AeroWAM-data/px4_logs/trace_smoke_v2
```

## 无飞行验收

2026-10-14 新增独立语言目标/多任务草案及可选 GroundedAction 信封，
详见 LANGUAGE_TASKS_CN.md。它不修改本页旧动作或候选协议。
2026-10-15：可选 `GroundedAdapter(policy)` 已接 adapter_server；策略函数
签名为 `ModelObservation -> GroundedAction`，客户端使用独立
`PolicyClient.infer_grounded(observation)`。保持旧 `infer()` 返回动作不变。
输入仍为集中式公开双视角观测，不含私有目标/标签；响应报告的agent/stamp
必须对应当前输入。拒绝错帧、额外真值字段、错误信封和错误策略返回类型。
实际TCP回环、输入隔离和旧接口兼容测试通过；尚未接实时多任务runner。

传输示例（只返回hover且无目标报告，不是任务模型）：

```bash
python3 -B -m benchmark.px4_gazebo.model_handoff.adapter_server \
  --factory benchmark.px4_gazebo.model_handoff.adapter_example:create_grounded_policy \
  --host 127.0.0.1 --port 8766
```

新信封仅供`infer_grounded`消费，不应发给旧live runner。

2026-10-15 新增独立search_pilot：默认由用户在终端输入原文指令，另填私有评分
目标，后者不进入ModelObservation。policy_server可选`--grounded-search-pilot`
输出旧模型动作+显式RGB颜色报告器信封，旧模式不变。新场景无解锁启动通过，
飞行闭环尚未验证。报告器不是模型学习输出，限制与分阶段命令见SEARCH_PILOT_CN.md。

```bash
python3 -B -m benchmark.px4_gazebo.model_handoff.interface_self_test
python3 -B -m benchmark.px4_gazebo.model_handoff.language_self_test
python3 -B -m benchmark.px4_gazebo.model_handoff.interface_check \
  --request /path/to/request.json --response /path/to/response.json
```

接口检查核对结构、图像、身份及数值，不验收语言理解、预测准确率或闭环成功率。
上述目录已保存 smoke 结果，重新验证修改时请使用新的独立目录，避免 resume 跳过旧结果。
下一门槛是各时钟对齐与新鲜度验收、独立多任务验收和匹配起点候选分支；完成后再冻结接口。
