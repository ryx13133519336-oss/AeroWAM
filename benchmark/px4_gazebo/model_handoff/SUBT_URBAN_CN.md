# SubT Urban 双层建筑接入与使用

主线：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。

本次把官方 SubT Urban 2 Story 接入独立 Gazebo 场景，保留楼梯、竖井、隔墙、梁柱、工业设备和原始碰撞网格。约 40 × 40 m、两层、总高约 20 m。它是有纹理的工业建筑，不是摄影测量城市，也没有新增室外复杂地形。

## 当前能直接用到什么程度

可直接打开 GUI、自由观察、导出双机场景、采集两路 RGB/私有实例图，以及执行不解锁的 PX4 双机遥测检查。已验证墙前不可见、墙后可见，二层也能真实渲染。

**还不能把原 `search_pilot --execute-flight` 命令直接当成这个场景的飞行入口。** 当前没有修改旧控制器、默认世界或模型输入。双机只读启动成功，但结束时两机 `pre_flight_checks_pass=false`；`flight_ready=false`、`flight_admitted=false`。尚未验证机体净空、楼内飞行、跨层路线和真实模型协作闭环。没有任何新模型成绩。

## 一条命令打开（现有 WSL）

在 WSL 终端运行，仓库是 `/mnt/d/AeroWAM`，不是旧的桌面副本：

```bash
cd /mnt/d/AeroWAM
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_urban preview \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --aircraft --px4-root /home/ryx/PX4-Autopilot \
  --output-dir "/mnt/d/AeroWAM-data/subt_urban_view_$(date +%Y%m%d_%H%M%S)"
```

可看到完整建筑和两架未接飞控的实体无人机。此命令不运行 PX4、不调用模型、不解锁；无人机不飞是预期行为。关闭窗口或 Ctrl+C 退出。日志在本次输出目录的 `gazebo_gui.log`；每次输出目录必须新建。只看环境时可去掉 `--aircraft --px4-root ...`。

已导出的动态双机场景：`/mnt/d/AeroWAM-data/subt_urban_ready_v1/world.sdf`，配套 `scene_contract.private.json`。world 内含本机绝对资源路径，不能直接复制到其他机器使用；换机器后需在当地重新 prepare/export。

## 静态渲染与遮挡复验

```bash
cd /mnt/d/AeroWAM
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_urban_probe \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --output-dir "/mnt/d/AeroWAM-data/subt_urban_check_$(date +%Y%m%d_%H%M%S)"
```

依次运行 `overview / hidden / revealed / upper` 四组双视角。原始 1280 × 960 RGB/labels_map 严格同时间戳配对，另检验 256 × 256 输入尺寸；至少三对不同时间戳。实例标签、包围框和目标坐标仅供私有检查，不输入策略。

验收额外检查：Gazebo 日志无资源加载错误、建筑标签确实出现、可见目标的 RGB 有对应颜色支持、四组场景只改相机位姿。隐藏视角目标仍在前方视场内，建筑碰撞三角形截断中心视线；墙后中心视线不再被截断，目标有有效框。它们是独立静态视角，不是运动轨迹或记忆实验。

## 可替换、可移动的工业标靶

场景中的旧彩色条纹球/方块已替换为哑光工业目标：独立支架承载可替换的球、方块或圆柱组件，目标材质为高粗糙度、非金属，不带条纹、文字、AprilTag 或自发光。默认组合为红球和蓝方块。支持颜色 `red / blue / green / yellow`，形状 `sphere / cube / cylinder`；外观与私有实例标签相互独立，标签只用于环境检查和裁判，不送入策略。

在预览命令后添加 `--target-a COLOR:SHAPE` 或 `--target-b COLOR:SHAPE` 可替换组件，例如：

```bash
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.subt_urban preview \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --target-a green:cylinder --target-b yellow:sphere \
  --output-dir "/mnt/d/AeroWAM-data/subt_custom_targets_$(date +%Y%m%d_%H%M%S)"
```

`--move-targets` 启用两个目标沿各自有界往返路线移动。它使用 Gazebo 仿真时间和位姿服务驱动的运动学小车，不是物理轮驱动，也不发送 PX4 命令。首次检查建议在 WSL 中跑 18 秒无界面预览：

```bash
cd /mnt/d/AeroWAM
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.subt_urban preview \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --output-dir "/mnt/d/AeroWAM-data/subt_industrial_preview_$(date +%Y%m%d_%H%M%S)" \
  --case revealed --move-targets --headless --preview-seconds 18 --target-speed 0.2
```

预览会在输出目录生成 `preview_summary.json`；目标运动诊断写入 `motion/motion.private.json`，前后渲染拼图为 `motion/rendered_motion.png`。诊断检查位姿变化、相机 RGB 与实例标签的精确时间戳配对、可见目标像素/质心变化、路线边界以及本次启动进程的清理状态。打开 GUI 观察时去掉 `--headless`，保留 `--preview-seconds 18` 限制运行时间。

当前证据证明的是场景目标位姿和双相机渲染随仿真时间变化，**不证明**双无人机策略协作、队友行为变化下的组合迁移、模型成绩或飞行放行。目标接触故障注入还未验证；运动预览也不是接触物理或可飞性验收。

只采集一组可用：

```bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_urban capture \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 --case upper \
  --output-dir "/mnt/d/AeroWAM-data/subt_upper_$(date +%Y%m%d_%H%M%S)"
```

## 双机只读启动

确保没有其他任务占用 PX4 实例 20/21；不并发运行同一仿真。程序会拒绝已占用的实例，不做全局 kill。

```bash
cd /mnt/d/AeroWAM
source /opt/ros/humble/setup.bash
source /home/ryx/px4_ros2_ws/install/setup.bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_urban startup \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --px4-root /home/ryx/PX4-Autopilot \
  --agent /mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent \
  --ros-domain-id 83 --timeout-s 120 \
  --output-dir "/mnt/d/AeroWAM-data/subt_startup_$(date +%Y%m%d_%H%M%S)"
```

可加 `--gui`。独立 Gazebo 分区、新 ROS 域、临时 Agent 端口、新 PX4 工作目录；不发布起飞、解锁或轨迹指令。只清理本次进程。启动成功只说明相机和遥测就绪，不说明预检或飞行通过。没有改 WSL 校时链路，也没有放宽 10 ms 飞行时钟门槛或 400 ms 推理预算。

## 已保存的实测证据（2026-10-20）

| 检查 | 结果 | 本机数据目录 |
| --- | --- | --- |
| Win Python / WSL Python 3.10 回归 | 新场景各 13 项通过；连同旧 warehouse 共 25 项通过 | `subt_urban_self_test.py` / `warehouse_self_test.py` |
| 四组双相机静态验收 | 通过，每路 4–8 对精确时间戳帧 | `/mnt/d/AeroWAM-data/subt_urban_admission_v2` |
| 墙体遮挡 / 重新可见 | 红蓝目标在两路中分别隐藏 / 有效可见；117,517 个碰撞三角形参与私有射线检查 | 同上 `inspection.json` |
| GUI + 双实体无人机 | 两次 18 秒有界启动正常，GUI 插件和两机系统加载；未接飞控，v2 保留 GUI 日志 | `/mnt/d/AeroWAM-data/subt_urban_gui_smoke_v2` |
| 独立 PX4 双机只读启动 | 通过，两机均未解锁，运动指令 0 | `/mnt/d/AeroWAM-data/subt_urban_disarmed_v1` |

最后一项两路 RGB/实例图各 8 对；status 各 2 条，odometry 为 61/60 条；无错误且本次进程全部退出。两机预检未通过，**不可由此推导飞行放行**。

真实渲染拼图：`/mnt/d/AeroWAM-data/subt_urban_admission_v2/subt_urban_views.png`。

失败证据保留：`subt_urban_static_v1` 暴露了旧 COLLADA 文件被 XML 序列化为带前缀根标签时的加载问题，代码已保留默认 COLLADA 命名空间修复；`subt_urban_admission_v1/revealed` 是一次 WSL NVIDIA/D3D12 驱动段错误。未忽略失败、未修改系统驱动；隔离重跑 `v2` 四组全部通过。另有 `emission texture not supported`、`gz_frame_id` 等兼容警告，不等于资源缺失或闭环成功。

## 外部资源、固定版本与许可

来源为 [Open Robotics Fuel](https://app.gazebosim.org/OpenRobotics/fuel)。四个官方包均由 Fuel 元数据声明为 CC BY 4.0，不是把 SubT 仓库的 Apache 2.0 自动套用到模型上。

| 名称 | 固定版本 | 用途 |
| --- | --- | --- |
| Urban 2 Story | 14 | 当前实际放入世界的完整双层建筑 |
| Urban Service Room Straight | 6 | 已准备的扩展资源，当前不放置 |
| Urban Stairwell Platform Centered | 14 | 已准备的扩展资源，当前不放置 |
| Urban Platform | 3 | 约 205 MB 的共享纹理包，不放置其平台几何 |

没有把三个不经检查的模块随意拼在一起宣称连通。当前单个完整 2 Story 模块已有楼梯和两层空间；额外模块的接口连通性尚未验收。

源压缩包/元数据：`/mnt/d/AeroWAM-data/external_scenes/subt_urban_source_v1`。完整 ZIP SHA256 固定在 `subt_urban_scene.ASSETS`；解压前校验、拒绝路径穿越和符号链接、输出不覆盖。中断的 `urban_platform.partial.zip` 保留用于诊断，准备流程不用它。

资源包：`/mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2`。`manifest.json`、`ATTRIBUTION.txt`、原始 `model.config`/`metadata.pbtxt` 和 `fuel_metadata.json` 均保留。原作者 Cole Biesemeyer / Ian Chen 等署名以各包原文件为准。

只修改本地副本的纹理引用：SDF PBR 和 COLLADA 内嵌的移动 `tip` URL 均转为固定本地纹理；生成 SDF 去掉旧材质脚本和比赛专用 PerformerDetector，保留原始几何、厘米单位、节点变换、submesh、碰撞及链接位姿。使用显式声明的中性补光，不声称复现 SubT 黑暗条件。资源文件清单另有固定摘要，不能改内容后自行更新 manifest 来绕过检查。Fuel 下载计数等动态元数据不参与固定清单摘要，但其当前副本仍有逐文件摘要。

在源包已下载的其他目录重新准备：

```bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_urban prepare \
  --source /mnt/d/AeroWAM-data/external_scenes/subt_urban_source_v1 \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_new_package
```

下载原始资源时使用版本 ZIP，例如 `https://fuel.gazebosim.org/1.0/OpenRobotics/models/Urban%202%20Story/14/Urban%202%20Story.zip`，不要使用 `tip` 或错误的 `model.zip`。其余 URL 按上表固定模型名和版本构造。大资产和运行结果留在数据盘，不加入 Git。

## 交给闭环执行对话的接口

入口 `subt_urban_scene.scene(package, models, aircraft=True)` 返回 `(ElementTree, private_contract)`，世界名为 `aerowam_subt_urban`，动态模型仍是 `annotation_uav_0/1`；契约包含两路实际相机话题、资源摘要和 ENU 初始位置。不要把这里导出的 world 直接替换旧 benchmark 的默认 world。

执行对话需要显式适配场景入口、室内坐标/高度/碰撞安全和评测对象绑定，然后再申请/验证飞行。只读启动中的预检失败必须先解释，不可关掉预检来宣称场景已可飞。起飞前需核查两机体积净空与接触监测，原 lawn/warehouse 控制器的航路和地理高度假设不能照搬进楼内。

`role_assignment=None`，不能把相机编号当作永久搜索/确认角色。世界/资源契约、目标位置、实例图、射线和私有验收结果只进仿真/安全/裁判侧，策略只接允许的公开观测及队友通信。之后的模型闭环、队友行为变化与新任务组合迁移仍需真实测评；本次静态对照不提供那些结论。

## 楼梯跨层 handoff 专家示范（2026-10-21）

此开发示范服务于本文开头的研究主线，不把场景接入或教师成功冒充模型成绩。目标沿原建筑折返楼梯由一层到二层，经过中间平台，再从二层出口进入走廊；路线约 34 m，目标中心垂直跨度约 10 m。楼梯踏面附近的支撑高度由固定碰撞网格求得，目标包围盒沿全路线做保守扫掠检查，不是穿过楼板的直线插值。相机代理另做体积净空和相互间距检查；这些检查不等于真实无人机机体、旋翼或接触动力学验收。

渲染可见性顺序为“仅初始观察者可见 → 双方可见的交接窗口 → 仅接手者可见”。因果事件教师依次给出 `found → handoff_request → handoff_accept → confirm`，确认要求原观察者已经失视且接手者稳定可见，并遵守声明的消息延迟。当前双方同时可见窗口较长，尚未验证双方同时失视下的预测恢复能力。

两条已保存示范分别采用红球 `uav_0 → uav_1`、蓝方块 `uav_1 → uav_0`。这说明采集器支持身份置换，不表示颜色决定角色，也不证明策略已经学会角色不变性。角色由可见性事件临时形成，不作为永久 UAV 编号分工；后续应将每种目标与两种身份置换交叉采集，避免颜色与身份混杂。

### 采集与独立审计

在 WSL 的 `/mnt/d/AeroWAM` 下运行；每次必须使用不存在的新输出目录。此入口只启动 Gazebo 和运动学相机代理，不启动 PX4、不解锁、不发送飞控命令：

```bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_handoff_collect \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --target red:sphere \
  --output-dir "/mnt/d/AeroWAM-data/subt_stair_red_$(date +%Y%m%d_%H%M%S)"

/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_handoff_collect \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --target blue:cube --swapped \
  --output-dir "/mnt/d/AeroWAM-data/subt_stair_blue_$(date +%Y%m%d_%H%M%S)"
```

`--speed` 默认 0.3 m/s、上限 0.4 m/s；`--teammate delayed` 改变接手者到达轨迹，并将消息延迟从 0.5 s 改为 2 s，属于联合变化条件，不能据此分别归因轨迹或通信的影响。`--reverse` 仅是候选反向路线参数，不表示反向采集已通过。所有候选条件都必须重新通过几何、渲染和因果审计。

独立审计默认写出 `inspection.json` 和拼图；重复检查现有采集目录时加 `--read-only`，保证不改写任何采集文件。旧红色数据缺少条件元数据，审计需显式指定目标和消息延迟：

```bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_handoff_inspect \
  --episode /mnt/d/AeroWAM-data/subt_stair_expert_red_v1 \
  --target red:sphere --message-delay-s 0.5 --read-only

/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_handoff_inspect \
  --episode /mnt/d/AeroWAM-data/subt_stair_expert_blue_swapped_v1 --read-only
```

两组现有数据分别为 236 个双视角采样、472 张 RGB；均有约 10 m 垂直跨度，且已有独立审计报告。旧红色数据没有原始源代码摘要或位姿请求日志，审计明确保留 `source_capture_legacy=true`，不补写伪造日志。蓝色数据可重放核验位姿请求，但请求成功不等于曝光时刻动作已精确执行。

### 学习数据接口与使用边界

先完成默认审计生成报告，再导出到新目录。导出器会重新审计并核对原报告文件摘要，不只相信成功标志：

```bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_handoff_data \
  --episodes /mnt/d/AeroWAM-data/subt_stair_expert_red_v1 \
             /mnt/d/AeroWAM-data/subt_stair_expert_blue_swapped_v1 \
  --output-dir /mnt/d/AeroWAM-data/subt_stair_handoff_training_v1
```

本机此目录已生成，包含 472 个联合样本、944 张图、470 个下一观察对，不应重复覆盖。`inputs.jsonl` 只含 `instruction / rgb / proprioception / messages / dt_s`；公开 RGB 单独复制到 `images/`。`event_supervision.jsonl` 和 `future_observation_supervision.jsonl` 是训练监督，不是策略当前输入。目标位置、实例掩码、私有可见性、路线进度、未来帧和当前教师标签均不进入公开输入；`design.json` 仅作离线溯源，包含私有文件的摘要，不是模型特征。

当前数据只支持特权教师下的联合事件模仿与下一观察预测开发：相机朝向依赖隐藏目标位置，双视角输入不是已验证的分散式策略；相机姿态来自理想代理而非 PX4 遥测。期望下一观察位姿不是已执行飞控动作，因此导出不提供运动控制监督，也不能宣称建立动作条件世界模型。

所有样本均标记 `development_training_only`，同属一个场景族。连续帧、颜色变化和身份置换不能当作独立测试重复；没有创建随机帧级测试集，没有模型成绩、组合迁移结论或飞行放行。下一阶段应围绕主线加入真正的失视恢复、独立队友变化条件及独立路线/场景留出，再评价学习策略及世界模型的贡献，而非继续堆积同一路线相邻帧。
