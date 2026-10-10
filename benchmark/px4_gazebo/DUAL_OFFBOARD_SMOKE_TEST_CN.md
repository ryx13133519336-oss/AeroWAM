# PX4-Gazebo 双机 Offboard 冒烟测试

这个测试只验证最基础、最关键的双机 ROS 2 控制闭环：

```text
等待双机遥测
→ 5 秒安全倒计时
→ 以 20 Hz 预发布 setpoint 2 秒
→ 两机进入 Offboard
→ 两机解锁
→ 各自在本地起点上方 3 m 起飞
→ 同时悬停 10 秒
→ 发送 NAV_LAND
→ 等待两机落地并锁定
```

脚本使用每架飞机第一次 `vehicle_odometry` 的位置作为它自己的本地 NED 原点。因此，`px4_0` 和 `px4_1` 不需要共享同一个全局坐标系。

## 一、运行前检查

必须已经启动以下内容：

1. Micro XRCE-DDS Agent；
2. Gazebo Sim；
3. 两个 PX4 SITL 实例 `px4_0`、`px4_1`；
4. QGroundControl 可以打开用于监控，但不要点击任务 `Start`，也不要在脚本运行时手动切换飞行模式。

确认 QGroundControl 中两架飞机都满足：

- 已连接；
- 未解锁；
- 无红色起飞前检查错误；
- 两架飞机周围有足够空间；
- Gazebo 仿真没有暂停。

## 二、先验证 ROS 2 话题

新开一个 WSL 终端：

```bash
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash

ros2 topic list | grep -E '^/px4_[01]/fmu/(in|out)/(offboard_control_mode|trajectory_setpoint|vehicle_command|vehicle_odometry|vehicle_status_v4)$' | sort
```

应该能看到两架飞机各自的 5 个关键话题。状态话题必须是：

```text
/px4_0/fmu/out/vehicle_status_v4
/px4_1/fmu/out/vehicle_status_v4
```

## 三、运行自动测试

继续在同一个 WSL 终端运行：

```bash
cd /mnt/d/AeroWAM
python3 -m benchmark.px4_gazebo.dual_offboard_smoke_test
```

默认参数为：

```text
相对起飞高度：3 m
起飞前倒计时：5 s
Offboard 预发布：2 s
悬停：10 s
setpoint 发布频率：20 Hz
```

测试开始后，不需要在 QGroundControl 点击任何起飞按钮。正常日志顺序应包含：

```text
双机遥测和起飞前检查正常
开始以 20 Hz 预发布 Offboard setpoint
请求两架飞机进入 Offboard 并解锁
两机已进入 Offboard 并解锁
两机到达目标高度
进入安全降落流程：悬停测试完成
双机已落地并锁定；冒烟测试成功完成
```

## 四、第一次建议使用更保守参数

如果希望第一次只飞到 2 m，并只悬停 5 秒：

```bash
python3 -m benchmark.px4_gazebo.dual_offboard_smoke_test \
  --altitude 2 \
  --hover 5
```

## 五、安全停止

如果出现异常，按一次 `Ctrl+C`。脚本会先向两架飞机发送 `VEHICLE_CMD_NAV_LAND`，再退出。

按下 `Ctrl+C` 后仍要观察 Gazebo 和 QGroundControl，确认两架飞机确实下降并锁定。不要直接关闭 PX4 或 Gazebo 窗口。

脚本会在以下情况自动进入安全降落：

- 任意一架飞机遥测超时；
- 里程计中出现 NaN 或无穷大；
- PX4 报告 failsafe；
- 起飞前检查失败；
- 12 秒内未能同时进入 Offboard 并解锁；
- 起飞超时；
- Python 节点发生异常。

## 六、常见问题

### 1. 一直停在“等待双机遥测”

检查 Agent 和两架 PX4 是否仍在运行：

```bash
ros2 topic hz /px4_0/fmu/out/vehicle_odometry
ros2 topic hz /px4_1/fmu/out/vehicle_odometry
```

两个话题都应持续输出频率。按 `Ctrl+C` 退出 `topic hz` 后再运行测试。

### 2. 无法进入 Offboard

确认输入话题名称全为小写：

```text
offboard_control_mode
trajectory_setpoint
vehicle_command
```

并确认 Gazebo 没有暂停。脚本在切换模式前会先发布至少 2 秒 setpoint。

### 3. 第二架飞机没有响应

本项目固定映射为：

```text
uav_0 → namespace px4_0 → target_system 1
uav_1 → namespace px4_1 → target_system 2
```

在两个 PX4 控制台分别执行 `mavlink status`，并在 QGroundControl 中确认系统 ID 分别为 1 和 2。

### 4. QGroundControl 显示飞行模式变化

这是正常现象。测试期间模式会自动进入 `Offboard`，测试完成后自动进入降落模式。QGroundControl 只负责显示和紧急人工监控，不参与策略控制。
