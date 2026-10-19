# 给接手对话：外部仓库场景接入

研究主线：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。不得固定 UAV0 搜索 / UAV1 确认，不向策略提供真实目标坐标、标签、场景答案或裁判结果。

## 当前可用范围

实际仓库位于 `D:\AeroWAM`，WSL 对应 `/mnt/d/AeroWAM`，不是旧桌面目录。使用当前工作区；接手前查看 Git 状态并保留其他改动。

- 外部 AWS 仓库已本地离线导入并固定版本；纹理、25 个背景碰撞网格、红球/蓝块及私有标签均保留。来源、许可、哈希和静态结果见 [WAREHOUSE_SCENE_CN.md](WAREHOUSE_SCENE_CN.md)。
- 真实静态双相机遮挡/显露验收通过，原始证据在 `/mnt/d/AeroWAM-data/external_scenes/warehouse_static_check_20261019_v5/`。
- `warehouse_scene.aircraft_world()` 提供两架动态 x500、机载 RGB/私有实例相机、PX4 所需世界插件、地理参考及磁场。两个候选出生点在中央通道，没有任务角色。
- `warehouse_startup` 可以导出世界文件和**私有**配置，也可以启动两架未解锁 PX4 检查真实流。该入口不创建控制发布器，不接受飞行开关，不能当作模型闭环。
- 原 `search_pilot` 默认场景不变。**不能把旧飞行命令原样运行后声称跑的是仓库。**

启动实测保留在 `/mnt/d/AeroWAM-data/external_scenes/`：`warehouse_dual_startup_20261019_v1`（未加地理参考，流检查通过但传感器日志有缺项）、`v2`（已修地理参考，双相机正常但 ROS 遥测为零，启动超时）和 **`v3`（最终版本未解锁启动通过）**。v3 使用 ROS 域 81，各机 95 条里程计、2 条状态、8 对 RGB/标注；清理正常，零运动命令。地理参考缺失及插件加载错误均为零；两机采样时 `pre_flight_checks_pass=false`，不能当作可飞证明。DDS 稳定性仍需后续复验，不能由一次成功认定已解决。若遥测超时，先读该轮日志、核对 DDS 匹配，不重新启动系统校时排查。

Windows 与 WSL 各 38 项仓库/标注/启动工具回归通过。最终双机场景 SHA256：`dd08919837df66c3829adb83703b8db13dd7873c326a25df4da9f2b93355e409`。导出副本在 `/mnt/d/AeroWAM-data/external_scenes/warehouse_dual_export_20261019_v1/`，与 v3 启动世界哈希一致。

本机已有资源：

```text
外部场景包 /mnt/d/AeroWAM-data/external_scenes/aws_warehouse_pinned_v2
相机/机体 /mnt/d/AeroWAM-data/annotation_models_v1
PX4       /home/ryx/PX4-Autopilot
DDS Agent /mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent
ROS       /opt/ros/humble/setup.bash
PX4 msgs  /home/ryx/px4_ros2_ws/install/setup.bash
```

## 先运行这个命令

在 WSL 中运行；输出路径必须不存在。首次建议不加 GUI，避免把窗口问题和接口问题混在一起。无人机实例 20/21 被占用时应停止本次启动，不得清理其他会话。

```bash
cd /mnt/d/AeroWAM
source /opt/ros/humble/setup.bash
source /home/ryx/px4_ros2_ws/install/setup.bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.warehouse_startup \
  --package /mnt/d/AeroWAM-data/external_scenes/aws_warehouse_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --px4-root /home/ryx/PX4-Autopilot \
  --agent /mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent \
  --startup-check --timeout-s 90 --ros-domain-id 81 \
  --output-dir "/mnt/d/AeroWAM-data/external_scenes/warehouse_startup_$(date +%Y%m%d_%H%M%S)"
```

可加 `--gui`，启动检查结束后会关闭该会话的窗口。此检查只确认新鲜的未解锁遥测、相机配对和会话清理；没有要求全部起飞前检查通过，没有校验完整飞行时钟门。

只导出供查看或后续接线：

```bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.warehouse_startup \
  --package /mnt/d/AeroWAM-data/external_scenes/aws_warehouse_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --output-dir /mnt/d/AeroWAM-data/external_scenes/my_warehouse_export
GZ_PARTITION=aerowam_warehouse_preview gz sim -r -v 3 \
  /mnt/d/AeroWAM-data/external_scenes/my_warehouse_export/world.sdf
```

导出文件 `scene_contract.private.json` 含真实目标位置、模型指纹、出生位姿及话题，**只给环境和裁判使用，不能传入模型请求或公共元数据**。

## 从“能启动”到“能跑策略”还必须完成

1. **净空与接触。** 用真实 x500 外形验证起降点和航行高度；验证碰撞网格在物理引擎中能阻挡机体，并将无人机与所有仓库障碍物的接触送入安全终止逻辑。旧入口只监听两个目标的接触，不能覆盖货架/墙体。约 0.9 米的货架间隔不自动视为可飞通道；低静态相机位姿也不是飞行航路。
2. **显式场景入口。** 为策略执行器增加可选 warehouse 参数，使用 `aircraft_world()` 返回的世界名、机体名和相机话题，所有资源及场景指纹只写私有目录。保留旧默认及简单场景回归。该工作区可能有另一个对话正在改 `search_pilot.py`，须读最新差异后合并。
3. **坐标和界限。** 将旧 `private_local_goals` 中硬编码的 `SPAWNS_NED` 改为从本次场景私有出生位姿计算 ENU→NED→初始机体 FLU；目标真值只服务裁判。不得直接扩大/关闭 2.5 米运动保护；定义版本化的仓库安全区域、速度/高度/行程限制及有效终止理由，匹配策略端保护与裁判端判据。
4. **无固定分工的闭环。** 仅接入 unassigned 路径；发现者由有效观测事件决定，另一架确认。先做一轮小范围低速闭环诊断，保存请求、执行和裁判证据，分开记录模型失败、模拟器错误及标注错误。不能用写死的目标路线或裁判坐标导航补模型成绩。
5. **再扩展任务。** search 闭环可复现后再接移动目标 handoff、inspection 和队友行为变化，之后做新任务/新场景组合划分。当前这些能力未验证，不得宣布迁移成功。

实际解锁前还要检查起飞前状态以及未通过原因。v3 只是“流已接通且未解锁”，没有等待/验证完整起飞就绪；不得关闭估计器或解锁检查来绕过。另需复核静态目标与机体的接近距离规则，不能把窄过道中的取景需求等同于飞行通道可达。

现有 10 毫秒飞行时钟门、400 毫秒在线推理预算保持不变。不重启 WSL 校时深挖，不修改服务或系统时钟，不训练模型来掩盖接口故障。若时钟门失败，保留模拟器错误分类；仓库静态/未解锁成功不能替代飞行门。

## 应交回的结果

给出实际执行命令、场景 SHA256、代码提交、输出目录、起降和接触验证、角色事件来源及原始失败分类。只有经过真实策略闭环的结果才计入模型成绩。未完成的飞行门需明确列出；不要把“GUI 可见”“startup_passed”或静态显露当作闭环成功。
