# AeroWAM 接口适配指南

本文给新增任务、场景和模型的贡献者使用。目标是让新的任务/场景可以接入同一个 episode、模型输入、世界模型预测和独立审计流程，而不再为每个场景重复修改执行器。

核心代码位于 `benchmark/px4_gazebo/model_handoff/core/`：

| 模块 | 作用 |
|---|---|
| `task_schema.py` | 任务 ID、事件词表、动作名和 evaluator 声明 |
| `scene_schema.py` | 场景 ID、场景族、公开通道和安全契约声明 |
| `episode_schema.py` | episode、公开观测、消息、任务监督、未来观察监督 |
| `model_io.py` | 模型输入/输出和世界模型输入/输出 |
| `audit_schema.py` | 独立审计、证据边界和 provenance |
| `core_self_test.py` | 核心协议回归测试 |

当前的 SubT 楼梯 handoff 是参考 adapter。warehouse、search、inspection 和后续同学提供的任务/场景都应沿用同样的边界。

## 1. 先区分四类内容

一次 episode 必须把以下内容分开保存：

1. **公开观测**：模型实际可以看到的 RGB、本机状态、已送达消息、动作历史和公开任务上下文。
2. **任务监督**：用于训练或离线评估的事件标签。它不进入模型输入。
3. **未来观察监督**：下一时刻的公开观察或预测目标。它不等于控制执行结果。
4. **私有审计**：目标真值、遮挡几何、路线、碰撞、安全检查、清理证据和来源摘要。

模型只能消费第 1 类。第 2 至第 4 类必须通过独立审计器处理。`target_pose`、`target_identity`、实例掩码、路线进度、未来帧、评测答案和私有可见性不能出现在模型输入；核心校验会递归拒绝这些字段。

## 2. 任务适配

任务 adapter 声明任务语义，不负责 Gazebo 连接。任务事件必须使用自己的命名空间，形式为 `task_id.event_name`。

```python
from benchmark.px4_gazebo.model_handoff.core.task_schema import TaskSpec

TASK = TaskSpec(
    task_id="search",
    task_version="1",
    event_names=("search.target_found", "search.target_lost"),
    action_names=("move", "hold", "report"),
    evaluator_id="search_evaluator_v1",
)
```

任务 adapter 还要提供：

- 原文任务指令和任务上下文的定义；
- `TaskSupervision` 生成器，明确标签来源和时间点；
- evaluator，明确成功、失败、删失和超时；
- 按 episode 定义独立重复单位，不能把连续帧当成独立实验；
- 训练、开发、同场景留出、跨场景和跨任务 split 规则；
- 任务私有字段清单和独立 audit 命令。

任务监督示例：

```python
from benchmark.px4_gazebo.model_handoff.core.episode_schema import TaskSupervision

label = TaskSupervision(
    sample_index=42,
    agent_id="uav_1",
    event="search.target_found",
    source="human",  # 也可以是独立规则审计或专家标签
)
```

不要在任务事件名中编码固定角色，例如 `uav_0_finder`。事件归属由 episode 的可见性、消息和任务状态在 evaluator 中判断。

## 3. 场景适配

场景 adapter 负责 world、设备接口、观测采集和安全契约。它不能改变任务字段的语义，也不能把私有 Gazebo 真值直接送给模型。

```python
from benchmark.px4_gazebo.model_handoff.core.scene_schema import SceneSpec

SCENE = SceneSpec(
    scene_id="warehouse_partition_v1",
    scene_family_id="warehouse_partition",
    scene_version="1",
    adapter_id="warehouse_partition_adapter_v1",
    public_channels=("dual_rgb", "local_state", "messages"),
    safety_contract="warehouse_partition_safety_v1",
    supports_tasks=("handoff", "search"),
)
```

场景适配必须说明：

- 每个公开通道的来源、频率、时间戳和单位；
- 每架无人机的本机状态坐标系；
- 消息何时算已送达；
- 目标真值、碰撞几何、PX4 状态和安全区域的私有存放位置；
- Gazebo/PX4 进程的启动、清理和失败分类；
- 场景文件、采集代码和审计报告的 SHA256。

静态场景验收、规则策略、运动学代理和 Gazebo 预览只能作为场景/接口证据，不能写成模型成绩或 PX4 飞行成绩。

## 4. 统一 episode 格式

`EpisodeRecord` 的关键字段如下：

```json
{
  "schema": "aerowam-episode/1",
  "episode_id": "episode_0001",
  "task_id": "handoff",
  "task_version": "1",
  "scene_id": "subt_urban_stair_v1",
  "scene_family_id": "subt_urban_2_story",
  "split": "development_training_only",
  "agent_ids": ["uav_0", "uav_1"],
  "role_assignment": null,
  "observations": [],
  "messages": [],
  "task_supervision": [],
  "future_observation_supervision": [],
  "private_audit": {},
  "provenance": []
}
```

`role_assignment` 必须为 `null`。每个公开 observation 必须包含所有 `agent_ids`，且 sample index 连续、时间戳递增。消息只记录模型实际收到的内容；没有送达的消息不能塞进 `messages`。

未来观察监督要写明 scope：

```python
FutureObservationTarget(
    sample_index=42,
    next_sample_index=43,
    delta_s=0.5,
    target={"next_rgb": {"uav_0": "images/43_uav_0.jpg"}},
    scope="observation_prediction_only_no_causal_action_conditioning",
)
```

如果没有记录真实执行动作、动作后的观测和反事实分支，必须保持 `action_conditioned_world_model=false`。

## 5. 模型适配

模型适配的推荐函数签名是：

```python
from benchmark.px4_gazebo.model_handoff.core.model_io import ModelInput, ModelOutput

def predict(request: ModelInput) -> ModelOutput:
    return ModelOutput(
        episode_id=request.episode_id,
        timestamp_ns=request.timestamp_ns,
        actions={
            "uav_0": {"action": "hold"},
            "uav_1": {"action": "hold"},
        },
    )
```

`ModelInput` 包含 `task_id`、`scene_id`、`scene_family_id`、原文 instruction、agent IDs、每架无人机公开 observation、已送达消息、动作历史和公开 task context。输出只按 agent ID 给动作、任务事件和预测概率；不要在输出中声明 source/receiver 或固定角色。

当前核心 schema 是统一数据和 I/O 契约，尚不代表旧 PX4 runner 已经自动接入所有新任务。新模型先通过离线 request/response 预检，再由对应 task/scene runner 连接；预检通过不等于闭环成绩或飞行准入。

## 6. 世界模型适配

世界模型使用 `WorldModelInput` 和 `WorldModelOutput`：

```python
from benchmark.px4_gazebo.model_handoff.core.model_io import WorldModelInput, WorldModelOutput

def predict_next(request: WorldModelInput) -> WorldModelOutput:
    return WorldModelOutput(
        episode_id=request.observation.episode_id,
        timestamp_ns=request.observation.timestamp_ns,
        predicted_observation={"rgb": {"uav_0": "predicted.jpg"}},
        action_conditioned=False,
    )
```

只有同时满足以下条件时才可以写 `action_conditioned=True`：

- 输入 `WorldModelInput.action` 是当前模型实际收到的候选动作；
- 输出 `WorldModelOutput.action` 原样记录该动作；
- 有动作后的真实公开观测作为监督；
- 审计报告能把动作、观测和时间戳对应起来。

仅根据连续帧预测下一帧、使用目标运动学代理、使用专家期望位姿，都只能标记为 observation-only prediction。

## 7. 证据和审计字段

每个 episode 或模型方法都要保留以下旗标：

| 字段 | 含义 |
|---|---|
| `independent_audit` | 是否经过独立审计器，而不是采集器自报成功 |
| `model_score_eligible` | 是否允许进入正式模型成绩统计 |
| `flight_admitted` | 是否满足真实飞行准入，不等于仿真成功 |
| `privileged_teacher` | 是否使用了策略不可见的教师/专家真值 |
| `kinematic_proxy` | 是否使用目标/相机/运动学代理 |
| `action_conditioned_world_model` | 是否有真实动作条件的未来观察监督 |

当前楼梯专家数据应保持：`independent_audit=true`、`privileged_teacher=true`、`kinematic_proxy=true`、`model_score_eligible=false`、`flight_admitted=false`、`action_conditioned_world_model=false`。

## 8. 数据分区

| split | 用途 |
|---|---|
| `development_training_only` | 训练、调参和接口诊断，不做正式成绩 |
| `held_out_route_test` | 同一场景族的路线/方向留出 |
| `held_out_scene` | 不同场景族的迁移测试 |
| `held_out_task` | 新任务或任务组合的迁移测试 |

按 episode 计独立重复；同一 episode 的帧、颜色置换和身份置换不能当成独立场景重复。新场景或新任务的测试数据不得参与训练和阈值选择。

## 9. 必须运行的检查

贡献者至少运行：

```bash
python -m benchmark.px4_gazebo.model_handoff.core.core_self_test
python -m benchmark.px4_gazebo.model_handoff.<task_adapter>_self_test
python -m benchmark.px4_gazebo.model_handoff.<scene_adapter>_self_test
```

SubT 参考适配器：

```bash
python -m benchmark.px4_gazebo.model_handoff.subt_handoff_core_adapter \
  --export <audited-training-export> \
  --output <new-core-export>
```

提交前检查：

- 所有公共输入通过 truth-free 递归检查；
- `role_assignment` 为 `null`；
- 任务事件使用正确 namespace；
- 独立 audit 报告和源文件 SHA256 存在；
- 失败 episode 和删失原因保留，不被静默删除；
- 没有把教师、代理、预览或规则 evaluator 写成模型成绩；
- 不提交数据盘大图、原始日志、模型权重和临时运行目录。

## 10. 提交流程和文件 ownership

提交顺序建议是：

1. 先提交 task/scene manifest 和 adapter self-test；
2. 再提交一条通过独立审计的最小 episode manifest；
3. 再提交训练/开发/留出 split 设计和数据 provenance；
4. 最后接入模型或世界模型，不修改核心 schema。

本线维护 `core/`、SubT adapter 和公共审计协议。warehouse 对话维护 warehouse world、目标运动、相机/位姿采集、PX4/Gazebo 执行器、安全门和 warehouse evaluator；warehouse 只新增 adapter 消费本协议。若需要修改核心字段，必须先更新 schema 版本和所有 self-test，不能直接改变旧字段含义。
