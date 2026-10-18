# 搜索场景真实遮挡静态诊断

研究主线：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。

`aerowam-search-occlusion-probe/1-draft` 先验证遮挡场景与像素验收可用。
它不是飞行评测、连续重观察实验或组合迁移结果。三个条件是独立启动的静态传感器场景，
没有 PX4、Agent、模型推理、训练或策略动作。默认 `search_pilot` 飞行场景仍为开阔场景。

## 对照与信息边界

与现有搜索共享红色黑条纹球、蓝色白条纹方块的资产、标签和位置：
ENU `(4, 1, 2.4)`、`(4, -1, 2.4)`。使用已校验的原生配对相机资源，
1280×960 RGB 与私有实例图；另核对现有完整图像拉伸到 256×256 后的可见框。

| 条件 | 目标及相机 | 墙 | 验收 |
| --- | --- | --- | --- |
| `search_clear` | 两相机 ENU `(0,0,2.4)`、`(0,-3,2.4)`，朝向目标区域中心 | 无 | 两路均有红、蓝可见框及 RGB 主体颜色支持 |
| `search_blocked` | 与 clear 完全一致 | ENU `(2,-0.5,2.4)`，尺寸 `0.15×4×2.8 m` | 两路红、蓝实例像素均为零，有墙像素 |
| `search_revealed` | 只换相机到 `(0,6,2.4)`、`(0,-8,2.4)` 并朝向区域中心 | 与 blocked 完全一致 | 两路红、蓝重新有合格可见框 |

检查器核对保存世界的 SHA256、所有非干预场景内容、同位置 clear/blocked 相机、
blocked/revealed 的同一墙以及相同相机资源清单。目标不移动，不能靠删除墙或更换目标
冒充侧面观察。每路至少三对唯一、严格相同时间戳 RGB/labels_map，逐对核验原始消息戳、
原生分辨率和 256×256 的像素支持；记录消息摘要。背景与墙有明确的私有标签映射，
未知标签、边缘裁切、过小目标、相机不匹配等条件拒绝通过。

实例图产生私有可见性标签，颜色检查只作画面对照，不反过来生成真值。
叠框图、世界配置、条件标识、像素数量和检查结果只进入诊断目录；没有接入策略输入。
静态相机并非真实飞行器，检查器也不声称验证了飞行曝光对应姿态或两机联合曝光同步。

## 复现

先按现有 `annotation_models` 生成并验证本地配对相机资源。以下命令在 WSL 的
Gazebo 8 Python 环境执行，输出目录必须不存在，不覆盖以前的验证：

```bash
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.search_occlusion_probe \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --output-dir /mnt/d/AeroWAM-data/search_occlusion_static_new \
  --timeout-s 40
```

逐条件使用随机 Gazebo partition，只有静态渲染服务器；清理仅针对本次自有进程组。
失败保留各条件 `summary.json` 和 `gazebo.log`。不重试挑选成功条件。
对已存原始消息重新验收，不启动仿真：

```bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.search_occlusion_probe \
  --output-dir /mnt/d/AeroWAM-data/search_occlusion_static_20261018_v1 --inspect-only
```

产物为各条件世界、原始 protobuf、采集摘要、`inspection.json`、每路叠框图及
`visibility_controls.png`。这些运行产物留在本地，不提交原始数据。

## 2026-10-18 首次真实渲染证据

本地目录 `D:/AeroWAM-data/search_occlusion_static_20261018_v1`。三个条件各两路、
每路 8 对，共 48 对严格同戳图像，所有帧在原生及 256×256 分辨率通过。
下表为各路第一对的 256×256 实例像素数（红 / 蓝）：

| 条件 | uav_0 | uav_1 |
| --- | --- | --- |
| clear | 798 / 864 | 394 / 762 |
| blocked | 0 / 0 | 0 / 0 |
| revealed | 300 / 216 | 118 / 202 |

blocked 墙像素分别 44520 / 17288；revealed 仍有墙像素 1658 / 1054。
叠框画面目视对照通过。三次自有服务器均停止，无 cleanup error，事后进程检查无残留。
这只证明本场景可以构造与检验真实遮挡，不构成多场景成功率或模型识别能力证据。

Windows 与 WSL 各119项相关回归通过，其中10项新增测试覆盖场景混杂、相机/墙
不一致、动态实体、残留目标像素、缺少墙、RGB不符、未知标签及缩小后目标过小等拒绝条件。
保存原始消息的独立离线重查也通过，没有再次启动仿真。

## 同世界可见性序列（2026-10-18）

新增独立 `aerowam-search-occlusion-sequence/1-draft`，在一个服务器和同一 partition 中
保持目标与墙不变，只对两个静态相机 rig 下发位姿切换：侧面看见 → 正面被墙遮住 →
两相机交换侧面位置再次看见。它是有连续仿真时间的分步视点诊断，不是连续运动轨迹。
不调用 PX4，不恢复飞行器快照，不调用策略或把预设视点描述成自主路线。

命令 ACK 后，从正在发布的 `pose/info` 核对两相机及两目标、墙的位置/朝向，
然后排除至少1仿真秒的过渡期。每阶段保留至少3组双相机同时间戳 RGB/labels_map，
跨度至少1仿真秒，相邻采样不超过0.6仿真秒。核对每路原生和256分辨率可见性，
阶段结束再次读回位姿。原始命令、位姿流和图像均保存摘要，独立离线检查顺序、
单调时戳、命令/稳定边界、双机同步和原始像素。过渡期图像未纳入验收，不能将这些
分段样本直接当作完整转移监督或曝光对应位姿。

```bash
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.search_occlusion_sequence \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --output-dir /mnt/d/AeroWAM-data/search_occlusion_sequence_new --timeout-s 90

/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.search_occlusion_sequence \
  --output-dir /mnt/d/AeroWAM-data/search_occlusion_sequence_20261018_v2 --inspect-only
```

首次 `search_occlusion_sequence_20261018_v1` 在切换相机后失败：命令接受，
`scene/info` 仍回传旧位姿，随后服务读取失败；没有通过位姿门槛，记录保留、服务器停止。
修正版使用持续 `pose/info`，不再将配置场景服务当作实时位姿回执。
这两次是诊断实现修复，不是用多次模型运行挑成功结果。

`search_occlusion_sequence_20261018_v2` 通过，单服务器启动1次，每阶段5组双机观测，
共15组联合时间戳、30对RGB/实例图。各阶段的实际采样时间如下：

| 阶段 | 首/末图像时间（仿真秒） | 持续跨度 | 两路红、蓝目标 |
| --- | --- | --- | --- |
| seen | 1.13 / 2.18 | 1.05 s | 均可见 |
| hidden | 3.37 / 4.43 | 1.06 s | 像素均为0 |
| reacquired | 5.58 / 6.64 | 1.06 s | 均再次可见 |

同目录 `sequence.png` 已检查，原始消息离线重查通过。自有服务器正常停止、
cleanup error为空、进程检查无残留。Windows与WSL各131项相关回归通过，
其中12项新测试覆盖顺序/缺阶段、复用/复位时戳、不同步/单相机、短暂/稀疏帧、
过渡边界、实时位姿缺失/不符/重复、非有限值及原始证据被修改等失败条件。

输出明确保持 `pose_at_exposure_verified`、`continuous_motion_verified`、
`memory_use_verified`、`flight_verified`、`compositional_transfer_verified` 为 false。
同一个场景发生遮挡及重新可见，仍不能证明策略必须使用历史或已经使用历史。

## 后续仍需完成

同世界分步视点序列已通过；连续运动、捕获对应状态、墙接触安全与公开观测驱动路线仍未接线。
当前两路静态遮挡不能直接证明必须使用历史：还需明确任务中的历史信息、与无历史输入
可区分的决策需求，并独立检查模型是否使用了历史。
随后已构造并验收一组“当前RGB逐像素相同、历史不同、正确左右判断相反”的真实渲染
配对，详见 [SEARCH_HISTORY_CN.md](SEARCH_HISTORY_CN.md)。这只证明该历史判断需要
额外信息，尚未检验学习模型是否使用历史，也没有验证导航动作效用。
在这些前提就绪后，再与队友响应变化、独立任务验收和候选动作后果预测组合。
条件、资产及场景组必须整体管理；同一场景的帧数不能充当独立实验重复。
