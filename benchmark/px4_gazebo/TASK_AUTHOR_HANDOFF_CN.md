# AeroWAM Task API v0.1 交接说明

这份接口用于把新的双 UAV 任务族接入现有 benchmark。当前参考任务是
`handoff_static_occluder`；它是接口和评测流程的参考实现，不代表最终任务族。

## 已冻结的边界

任务作者只需要新增任务插件、任务 manifest 和任务 evaluator。以下模块不应因新增任务而修改：

- `policy_api.py`：VLN、WAM、规则策略共同使用的策略接口；
- `communication.py`：带宽、延迟和丢包模型；
- 公共 episode/evaluator schema；
- PX4/Gazebo session lifecycle。

策略只能消费 `PolicyContext`，不能读取 simulator truth。目标位姿、队友真值位姿、完整地图、未来轨迹和 hidden events 只能出现在 evaluator-side episode record。

## 策略接入

每个 VLN/WAM backend 实现 `PolicyBackend`：

```python
from benchmark.px4_gazebo.policy_api import PolicyBackend, PolicyContext, PolicyOutput

class MyBackend:
    def reset(self, task_spec, agent_id, seed):
        ...

    def act(self, context: PolicyContext) -> PolicyOutput:
        ...

    def close(self):
        ...
```

`PolicyOutput` 必须返回动作；消息、预测和置信度是可选的。WAM 可以填充 `Prediction`，VLN 可以先只返回动作和消息。动作统一为 body-FLU velocity/yaw-rate：`vx, vy, vz, yaw_rate`。

## 任务接入

任务插件实现 `TaskPlugin`：

```python
from benchmark.px4_gazebo.task_api import TaskPlugin

class MyTask:
    task_id = "my_task_v1"
    simulator = "mock-or-gazebo"

    def reset(self, seed, task_spec=None): ...
    def step(self, action): ...
    def episode_record(self): ...
```

任务实现放在 `benchmark/px4_gazebo/tasks/<family>/` 下，并在组合入口显式注册 factory：

```python
from benchmark.px4_gazebo.task_registry import register_task
from benchmark.px4_gazebo.tasks.my_task.task import MyTask

register_task("my_task_v1", lambda task_spec: MyTask(task_spec=task_spec))
```

runner 通过同一个入口创建任务，不需要导入或遍历其他任务族：

```python
from benchmark.px4_gazebo.task_registry import create_task

task = create_task("my_task_v1", task_spec)
```

`reset` 和 `step` 返回给策略的 observation 不得包含 truth。任务特有的成功条件和指标放进 evaluator；公共指标保持一致，例如 `task_success`、碰撞、路径长度、通信字节和完成步数。

## 交接前验收

在仓库根目录运行：

```bash
python -m benchmark.px4_gazebo.contract_smoke
python benchmark/tools/validate_contract.py
```

新增任务至少应通过以下检查：

1. 相同 seed 能得到相同初始场景和 `reset_hash`；
2. 两个 UAV 的 observation 时钟同步；
3. observation 中没有 `target_pose`、`target_position`、`target_identity`、`teammate_pose`、`global_map`、`future_trajectory` 或 `truth`；
4. 策略可以在不修改 runner 和通信模块的情况下被替换；
5. episode JSONL 可以被公共 evaluator 重放并重新计算指标；
6. 至少有一个成功、一个超时或失败、一个通信受限的 smoke fixture；
7. train、validation、test/OOD fixture 分开，测试 fixture 不进入训练 cache。

## 当前资产

当前 Handoff 数据集位于外部数据根目录：

```text
/mnt/d/AeroWAM-data/model_handoff_dataset_wall_blocking_v1
```

其中 `train` 和 `validation` 共 280 个 fixture。它可用于验证接口和训练流程；新增任务族应建立独立的数据目录和 manifest，不要覆盖这个目录。

