# AeroWAM Task API v0.1

这是 AeroWAM 双 UAV benchmark 的任务与策略接口冻结版本，供任务族作者接入新任务，也供 VLN、WAM 和规则策略使用同一评测入口。

## 已冻结内容

公共接口位于 `benchmark/px4_gazebo/`：

- `policy_api.py`：`PolicyBackend`、`PolicyContext`、`PolicyOutput`、`Prediction`；
- `task_api.py`：`TaskPlugin` 与 observation truth-leakage 检查；
- `contract_smoke.py`：最小策略/任务契约检查；
- `TASK_AUTHOR_HANDOFF_CN.md`：任务作者接入、验收和数据隔离要求。

参考任务是 `model_handoff` 的 wall-blocking Handoff。它用于验证接口和训练流程，不限制后续任务族的任务语义。

## 契约检查

在 WSL 的 AeroWAM 工作区执行：

```bash
cd /mnt/d/AeroWAM
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
source ~/venvs/aerowam-model/bin/activate

python -m benchmark.px4_gazebo.contract_smoke
python benchmark/tools/validate_contract.py
```

预期输出包括 `policy/task contract smoke: PASS`。任务作者提交新任务前必须通过这两个检查，并为自己的任务增加独立 smoke fixture。

## 接入策略后端

策略后端只依赖 `PolicyContext`，不导入 PX4、Gazebo 或 evaluator：

```python
from benchmark.px4_gazebo.policy_api import PolicyContext, PolicyOutput

class MyBackend:
    def reset(self, task_spec, agent_id, seed):
        pass

    def act(self, context: PolicyContext) -> PolicyOutput:
        return PolicyOutput()

    def close(self):
        pass
```

动作统一为 body-FLU 的 `[vx, vy, vz, yaw_rate]`。消息、置信度和世界模型预测是可选输出；WAM 可以填充 `Prediction`，VLN 可以先只返回动作和消息。

## 接入任务族

任务实现 `TaskPlugin` 的 `reset`、`step` 和 `episode_record`。策略 observation 只能包含策略当时可获得的信息：语言、相机、UAV 自身状态、历史动作和通信消息。目标真值、完整地图、未来轨迹、队友真值位姿和隐藏事件只能进入 evaluator-side episode record。

每个任务族必须使用独立的 manifest、数据目录和结果目录，不覆盖现有 Handoff 数据。train、validation、test/OOD fixture 必须分开，测试 fixture 不得进入训练 cache。

## 数据与训练边界

数据、特征 cache、PX4/Gazebo 日志、checkpoint 和生成结果不放入代码仓库。当前 Handoff 数据位于外部路径：

```text
/mnt/d/AeroWAM-data/model_handoff_dataset_wall_blocking_v1
```

`joint_bc` 是一个可替换的策略后端，不会改变已冻结的任务接口；接口契约通过后即可生成特征并训练基线。

## 交接验收

任务作者交付前应确认：相同 seed 可复现初始场景；两个 UAV 的 observation 时钟同步；observation 不含 truth；策略后端可替换而无需修改通信和公共 runner；episode JSONL 可重放；至少有成功、失败/超时和通信受限 smoke fixture。

详细字段和示例见 [`benchmark/px4_gazebo/TASK_AUTHOR_HANDOFF_CN.md`](benchmark/px4_gazebo/TASK_AUTHOR_HANDOFF_CN.md)。
