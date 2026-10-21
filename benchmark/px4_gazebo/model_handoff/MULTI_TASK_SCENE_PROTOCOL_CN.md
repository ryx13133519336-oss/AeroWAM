# AeroWAM 多场景、多任务接入协议

本协议把任务、场景、模型和证据分成四层。楼梯 handoff 是第一套参考实现；warehouse、search、inspection 以及同学后续提供的任务和场景都通过 adapter 接入，不修改核心字段语义。

## 固定边界

模型服务只接收 `core.model_io.ModelInput`：任务说明、场景标识、每架无人机的公开观测、已送达消息、动作历史和公开任务上下文。`role_assignment` 永远为 `null`，模型不能假定 `uav_0` 是发现者或 `uav_1` 是接收者。角色由可见性、消息和任务状态形成，并由任务 evaluator 在私有审计侧归因。

任务监督、未来观察监督和私有审计分开保存。楼梯当前的事件标签来自专家/运动学代理，因此 `privileged_teacher=true`、`kinematic_proxy=true`、`model_score_eligible=false`。未来帧只支持 observation-sequence prediction；没有真实执行动作和反事实分支时，`action_conditioned_world_model` 必须为 `false`。

## 新增任务

新增任务实现 `TaskSpec`，声明 `task_id`、版本、命名空间事件、动作名和 evaluator ID。事件必须形如 `task.event`，例如 `search.target_found`、`inspection.inspect_complete`。任务 adapter 负责把场景记录转为任务监督并实现任务专属 evaluator，不能把私有目标位姿、实例掩码或成功答案写入模型输入。

## 新增场景

新增场景实现 `SceneSpec`，声明 `scene_id`、`scene_family_id`、公开话题/通道和安全契约。Gazebo world、PX4 话题、目标真值、碰撞几何和安全区域只属于 scene adapter 与私有审计。warehouse 对接时新增 warehouse adapter，保留现有 warehouse 执行器和安全门，不把静态场景验收写成模型成绩。

## 数据分区和证据

`development_training_only` 只能训练或调参；`held_out_route_test` 只能做同场景族方向留出；`held_out_scene` 用于独立场景迁移；`held_out_task` 用于新任务组合迁移。独立审计必须在每个 episode 晋升前运行，且按 episode 而不是帧计独立重复。所有报告保留 `model_score_eligible`、`flight_admitted`、`privileged_teacher`、`kinematic_proxy` 和 `action_conditioned_world_model` 标志。

## 两个对话的文件 ownership

本协议和 `core/`、SubT adapter 由本线维护。warehouse 对话继续维护 warehouse world、目标运动、相机/位姿采集、PX4/Gazebo 执行器、安全门和 warehouse evaluator。warehouse 对话只新增一个 adapter 消费本协议，不修改 `core/` 字段含义；本线不改 warehouse 执行器。合并前分别运行 `python -m benchmark.px4_gazebo.model_handoff.core.core_self_test`、SubT self-test 和 warehouse self-test。

## 接入检查

```bash
python -m benchmark.px4_gazebo.model_handoff.core.core_self_test
python -m benchmark.px4_gazebo.model_handoff.subt_handoff_core_adapter --export <audited-export> --output <new-core-export>
```

新同学提交时只需提供 task/scene manifest、adapter、独立 audit 报告和文件摘要；不得提交数据盘大图、日志或模型权重到本仓库。
