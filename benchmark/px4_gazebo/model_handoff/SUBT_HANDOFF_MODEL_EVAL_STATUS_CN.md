# SubT 楼梯 handoff：模型闭环与榜单缺口

## 当前已经具备

- 楼梯路线、目标颜色/形状、身份置换、队友运动变化和固定通信延迟的开发矩阵；
- RGB、公开本机状态、已送达消息、事件监督和未来观察监督的分离格式；
- 独立数据审计，能够拒绝旧目录、颜色支持错误、时序错误和私有字段泄漏；
- SubT 训练导出到统一 core episode 的 adapter；
- 外部模型的旧 handoff transport、adapter manifest 和无飞行接口自测；
- 新增的 evaluation report 和 leaderboard 聚合规则。

这些内容还不能证明模型已经在楼梯场景执行 handoff，也不能产生正式成功率。

2026-10-21 进度：开发集已独立复核 8/8 条件、1,888 双机样本和 3,776 RGB 图片。第一版共享事件头/下一观察 Ridge checkpoint `subt_fixedcomm_shared_event_20261021_161727` 已训练且独立重算通过。单条下行宏 F1=0.39493，发现/确认漏检，55 次 request 只有 1 次正确；下一 RGB RMSE=0.03681，差于 persistence 的 0.01797。该结论只适用于这一条离线教师消息回放，不代表闭环成功率。补充混淆计数、precision 和 persistence 已另行重算审计；不根据测试结果选择 checkpoint。

新增 `subt_handoff_frozen_eval.py` 提供完整 8 格方向留出矩阵的冻结权重评估与单独 audit，未齐全则拒绝。尚未执行真实全矩阵评估。蓝方块下行存在 JPEG 解码跨平台颜色支持差异；有 `batch_audit_error.txt` 的目录保持拒绝，不通过更换平台恢复准入。命令和已知限制见 `SUBT_NEXT_RUN_CN.md`。

## 已完成的模型接口 runner

`subt_handoff_model_eval.py` 使用已发布的 `core.ModelInput/ModelOutput`，读取已经独立审计的原始楼梯 episode，只发送 RGB、原始公开代理本体状态、已送达示范消息、模型动作历史和任务指令。代理状态不伪造成 PX4 速度或健康状态。响应使用逐机 `handoff.*` 事件，避免旧单个 `intent` 的固定方向语义。

这是 recorded-observation replay：相机帧不响应模型动作，消息来自示范者，不来自当前模型。因此 `success=null`、`closed_loop=false`、`flight_admitted=false`，不生成正式 evaluation report，也不能交给正式榜单聚合器。模型 manifest 在此仅绑定运行身份，尚不证明模型权重和训练资料正确。

`io_trace.jsonl` 保存完整公共请求、原始 JSON 响应、时延和失败；`subt_handoff_model_audit.py` 在单独命令中重新审计原始 episode、逐条重建请求、核对响应时间与逐机动作，并拒绝篡改摘要。审计通过仍不代表任务成功。

新增 `subt_handoff_evaluator.py` 是供后续真实执行器调用的私有事件状态机：角色来自唯一有效发现报告；接受必须晚于请求至少 0.5 秒；确认必须有原观察者至少 1 秒物理遮挡和接收者持续带框报告。零可见像素本身不能证明物理遮挡。它仍需执行器提供曝光对齐的私有证据，没有独立授予飞行权限。

## 新增的 Gazebo 动作驱动代理入口（待运行验收）

`subt_handoff_proxy.py` 启动独立 Gazebo world，私有楼梯路线只驱动标靶；两路相机在初始化后完全使用模型的逐机位移和 yaw 更新，初始 pitch 保持不变。每个动作前检查保守代理扫掠体积、两相机距离及目标距离。目标位姿服务接受不等于曝光对齐的精确执行，更不是飞行动作。

本入口复用 core.ModelInput/ModelOutput，消息来自当前模型事件并延迟 0.5 s，不使用教师事件。它保存公共请求、模型响应、服务请求、RGB、原生私有实例图和私有裁判结果。`subt_handoff_proxy_audit.py` 单独重建消息/动作历史、实例框、全目标遮挡、事件顺序及动作服务请求。

2026-10-21 已完成一次 5 秒真实 Gazebo 代理 smoke：`D:\AeroWAM-data\subt_proxy_hold_20261021_190821`，11 帧、11 次动作服务请求、进程清理无错误；独立 `subt_handoff_proxy_audit.py` 通过，`frames_verified=11`、`model_actions_traced=11`。这是接口和代理闭环验收，不是 handoff 成功率；`proxy_event_success=false`、`kinematic_proxy=true`、`px4_started=false`、`flight_admitted=false`，不进入榜单。

新增 `subt_handoff_checkpoint_server.py` 和 `subt_handoff_checkpoint_adapter.py`：Windows 端加载冻结 sklearn checkpoint，WSL 端通过 core 长度前缀协议请求。适配器重建训练时的公共 419 维特征，仅记录事件概率，输出零动作且不伪造 bbox；这是模型权重进入实时请求链的诊断入口。2026-10-21 已完成一次跨 Windows/WSL 的 5 秒 Gazebo 代理 smoke：`D:\AeroWAM-data\subt_proxy_checkpoint_20261021_192240`，`model_action_services_accepted=11`、`samples=11`、进程清理无误；独立 `subt_handoff_proxy_audit.py` 通过（`frames_verified=11`、`model_actions_traced=11`）。这证明冻结权重进入实时 Core 请求链，不证明 handoff 成功、世界模型能力或 PX4 飞行。

同日按相同固定 checkpoint 重跑的审计目录为 `D:\AeroWAM-data\subt_proxy_checkpoint_20261021_200300`（WSL：`/mnt/d/AeroWAM-data/subt_proxy_checkpoint_20261021_200300`）。独立审计通过，`run_id=subt-proxy-ee667a8486f04c568e7df12913ce043f`、`frames_verified=11`、`model_actions_traced=11`、`summary_sha256=7ea609350b4ca1f9c0ee8af9e6ea9d33b67c4ad1c418163afd9bfe6a25253839`。结果仍为 `proxy_event_success=false`、`kinematic_proxy=true`、`flight_admitted=false`、`formal_score=false`；这是重复的接口/审计证据，不是新的模型成功率或独立任务重复。

私有初始相机摆放仍使用教师计划；目前是集中式双机控制，尚无 normal/delayed 独立队友策略；不是专家采集替代品，也不能据此评测世界模型真实飞行动力学。`proxy_summary.json` 的 success 固定 null；审计可单独给出 `proxy_event_success`，不会进入正式榜单。

### Grounded 模型接入桥（2026-10-21）

新增 `subt_handoff_grounded_bridge.py`，把已有 grounded policy 的 `TargetReport` 转换为 SubT Core `ModelOutput`。桥接层逐事件保留 `agent_id`、当前相机帧时间戳和归一化 bbox，并将 `found/request/accept/confirm` 映射到 `handoff.*` 命名空间；它不根据事件推断固定 source/receiver 角色。

默认模式会把旧模型的 body-FLU 位移全部降为逐机 hold，只评测 grounded 事件能否进入 SubT proxy；这是为了避免未经验证的坐标系被当作楼梯 ENU 动作。`allow_proxy_enu_actions=True` 是单独的运动学诊断开关，只有完成 SubT 机体包络、楼梯净空和动作 trace 审计后才能使用。桥接自测命令：

```bash
python -B -m benchmark.px4_gazebo.model_handoff.subt_handoff_grounded_bridge_self_test
```

该桥仍不提供 PX4 飞行准入、动作执行确认或正式成功率；真实模型运行必须继续通过 `subt_handoff_proxy_audit.py`，而正式榜单还要求 `flight_admitted=true` 和 `action_trace_verified=true`。

命令见 [SUBT_NEXT_RUN_CN.md](SUBT_NEXT_RUN_CN.md)。长时运行由用户执行，助手不占用 Gazebo 等待。

## 还缺什么

### 1. SubT PX4 闭环适配器

当前 `subt_handoff_collect.py` 是特权专家/运动学代理采集器：它沿规划路线移动目标和相机代理，不接收模型动作。因此它只能产生训练/接口诊断数据，不能作为模型成绩。回放 runner 已经覆盖模型协议和报告生成，但还没有真实 PX4 控制链。

真实准入仍需要一个独立的 SubT PX4 adapter：

- 启动固定版本 SubT Urban 楼梯 world；
- 两架真实 PX4/Gazebo x500 由模型输出控制，或明确标记为 `diagnostic_proxy`；
- 每个决策点只构造公开 `ModelInput`，不把目标真值、路线、掩码、私有可见性或 evaluator 结果传给模型；
- 接收模型动作和事件，记录原始请求、原始响应、发布控制、接收遥测和时间戳；
- 私有 evaluator 独立判断 `found -> handoff_request -> handoff_accept -> confirm`、安全停止、超时和基础设施错误；
- 每个 episode 结束后生成 `aerowam-subt-handoff-evaluation/1` report。

代理 runner 可以先用于接口 smoke，但必须将 `kinematic_proxy=true`、`flight_admitted=false`、`model_score_eligible=false` 写入报告，不能进入正式榜单。

### 2. 可复现的模型身份

每个模型必须有 adapter manifest、代码/配置/权重 SHA256、训练数据 ID、训练 seed、输入权限、运行设备和超时设置。没有身份绑定的“某个模型输出”不能写入榜单。

### 3. 真正的队友行为变化

当前 `normal/delayed` 主要是响应时间/运动行为代理，不是独立队友策略。正式主线需要冻结至少两个可复现队友 policy：每个 policy 有自己的 manifest、动作日志和行为版本，且不把队友私有策略状态传给被测模型。

### 4. 正式测试矩阵和重复

当前楼梯下行路线是同一场景族的方向留出，不是独立场景。正式榜单前还需要：

- 固定训练、开发、同场景方向留出、独立场景留出和新任务留出；
- 每个条件多个独立 episode/repeat，按 episode 计分，不按帧计分；
- 新场景、新任务和测试队友行为不参与训练、阈值选择和 checkpoint 选择；
- 固定通信延迟为 0.5 秒，通信影响暂不作为实验因素。

warehouse 可作为第一套独立场景族，但必须先完成统一 adapter 和独立审计。

### 5. 榜单输入和排名规则

`subt_handoff_leaderboard.py` 只聚合合格 evaluation report：

- `closed_loop=true`；
- `public_input_only=true`；
- `independent_audit=true`；
- `flight_admitted=true`；
- `action_trace_verified=true`；
- `privileged_teacher=false`、`kinematic_proxy=false`；
- `status=complete` 且 `success` 是布尔值；
- 固定通信延迟为 0.5 秒；
- 同一 method/condition/repeat 不得重复。

教师、代理、规则策略、仿真错误和安全停止会保留在审计输出中，但不进入正式成功率分母，也不排名。

## 推荐执行顺序

1. 先完成 SubT evaluation runner 的无飞行 adapter smoke，确认模型请求/响应、超时、控制记录和私有 evaluator 对齐。
2. 用一个明确的外部模型和一个规则 reference 做小规模双条件闭环，验证失败分类和清理。
3. 通过 PX4/Gazebo 真实 x500 闭环准入，再开始正式重复采集。
4. 在楼梯开发矩阵上训练/调参，只在 `development_training_only` 使用专家数据。
5. 在 `stair_descent_v1` 和独立 warehouse 场景上运行冻结 checkpoint，生成 evaluation reports。
6. 运行 leaderboard 聚合器，发布成功率、有效 episode 数、覆盖条件和基础设施失败数。
7. 再加入 search/inspection 等任务，使用新的 task namespace 和 evaluator，不修改旧 handoff 字段语义。

## 现在用户还需要准备的内容

- 选定第一批要接入的开源模型及其 checkpoint/权重来源；
- 给每个模型指定 adapter manifest 和推理超时；
- 在终端执行 [SUBT_NEXT_RUN_CN.md](SUBT_NEXT_RUN_CN.md) 的长时续采和验收；SubT x500 world、动作链和审计代码由助手继续实现；
- 与 warehouse 对话冻结队友行为 manifest、独立场景 adapter 和评测起点；
- 当前正式榜单继续仅接收通过准入的真实 PX4/Gazebo 闭环，代理结果只保留为诊断，不要求重复确认。

## 检查命令

```bash
python -m benchmark.px4_gazebo.model_handoff.core.core_self_test
python -m benchmark.px4_gazebo.model_handoff.subt_handoff_self_test
python -m benchmark.px4_gazebo.model_handoff.subt_handoff_leaderboard_self_test
python -m benchmark.px4_gazebo.model_handoff.subt_handoff_model_eval_self_test
```

使用本地模型 adapter 做已审计 episode 的协议回放（不会启动 PX4）：

```bash
python -m benchmark.px4_gazebo.model_handoff.subt_handoff_model_eval \
  --episode /mnt/d/AeroWAM-data/subt_stair_blue_swapped_motion_only_v1 \
  --output /mnt/d/AeroWAM-data/subt_model_eval_$(date +%Y%m%d_%H%M%S) \
  --method-id my-model --method-kind repository_model \
  --model-manifest /path/to/adapter_manifest.json \
  --adapter my_adapter:infer
```

`my_adapter:infer` 接收 `core.ModelInput.to_wire()`，返回 `core.ModelOutput.to_wire()`。manifest JSON 至少包含匹配的 `method_id`、`method_kind`；完整权重准入后续另审。动作使用 `type=waypoint_delta`、`delta_enu_m=[dx,dy,dz]`、`delta_yaw_rad`，逐机事件包含 `agent_id/event/image_timestamp_ns/bbox_xyxy`。发现、接受和确认必须提供当前观察框，失视后的请求允许框为 null。

若使用 TCP，把 `--adapter` 替换为 `--policy-endpoint HOST:PORT`；服务端必须返回 core 协议，旧 `ModelJointAction` 服务不能直接兼容。全零 hold reference 为 `benchmark.px4_gazebo.model_handoff.subt_handoff_model_io:hold`，仅检查传输，不代表视觉策略。独立检查：

```bash
python -m benchmark.px4_gazebo.model_handoff.subt_handoff_model_audit \
  --run-dir /path/to/replay_output --output /path/to/new_replay_audit.json
```

生成榜单时输入的是独立 evaluation reports，而不是训练数据、专家标签或 Gazebo 预览目录：

```bash
python -m benchmark.px4_gazebo.model_handoff.subt_handoff_leaderboard \
  --reports /path/to/evaluation_report_*.json \
  --output /path/to/subt_handoff_leaderboard.json
```
