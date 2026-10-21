# 新任务/新场景提交模板

提交者只需要新增 adapter 和声明文件，不修改 `core/` 的字段语义。大图、原始日志、模型权重和 Gazebo 数据盘留在外部数据目录，只在仓库提交代码、manifest、审计摘要和文件 SHA256。

## 任务 adapter

```python
from benchmark.px4_gazebo.model_handoff.core.task_schema import TaskSpec

TASK = TaskSpec(
    task_id="<task_id>",
    task_version="1",
    event_names=("<task_id>.event_a", "<task_id>.event_b"),
    action_names=("move", "hold", "report"),
    evaluator_id="<task_id>_evaluator_v1",
)
```

任务必须提供：

- 任务说明和事件词表，事件名必须是 `task.event`；
- 任务监督 adapter，输出 `TaskSupervision`，不混入模型输入；
- evaluator，说明成功、失败、删失和独立重复单位；
- 训练/开发/held-out split 规则；
- 私有真值字段清单和独立 audit 命令。

不得把固定 `uav_0`/`uav_1` 角色、目标位姿、实例掩码、路线进度或评测答案写进 `ModelInput`。

## 场景 adapter

```python
from benchmark.px4_gazebo.model_handoff.core.scene_schema import SceneSpec

SCENE = SceneSpec(
    scene_id="<scene_id>",
    scene_family_id="<scene_family_id>",
    scene_version="1",
    adapter_id="<scene_id>_adapter_v1",
    public_channels=("dual_rgb", "local_state", "messages"),
    safety_contract="<scene_id>_safety_v1",
    supports_tasks=("<task_id>",),
)
```

场景必须提供：

- world、话题和设备接口的 adapter；
- 公开观测生成器和私有审计分支；
- 安全契约、清理和失败分类；
- 目标真值、碰撞几何、PX4 状态的隔离说明；
- 可复现的 episode manifest 和文件摘要。

场景验收、规则策略、运动学代理和 Gazebo 预览不能写成模型成绩或 PX4 飞行成绩。

## 模型/世界模型 adapter

模型实现 `ModelInput -> ModelOutput`。世界模型实现 `WorldModelInput -> WorldModelOutput`；只有输入中带有实际候选 action 且输出明确记录该 action 时，才允许 `action_conditioned=true`。只有 observation sequence 的下一帧预测必须保持 `false`。

每个方法提交：来源、版本、权重/配置摘要、输入权限、训练数据 ID、运行时限制和公开请求/响应样本。外部模型接入预检通过不等价于闭环成绩或飞行准入。

## 验收命令

```bash
python -m benchmark.px4_gazebo.model_handoff.core.core_self_test
python -m benchmark.px4_gazebo.model_handoff.<task_adapter>_self_test
python -m benchmark.px4_gazebo.model_handoff.<scene_adapter>_self_test
```

提交说明必须写明：`model_score_eligible`、`flight_admitted`、`privileged_teacher`、`kinematic_proxy`、`action_conditioned_world_model` 五个标志，以及当前证据不能支持的结论。
