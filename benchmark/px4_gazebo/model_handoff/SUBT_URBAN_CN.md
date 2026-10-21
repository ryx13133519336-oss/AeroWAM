# SubT Urban 双层建筑接入与使用

主线：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。

当前阶段暂不研究通信影响：新计划固定消息延迟 0.5 s，开发采集只交叉目标外观、身份置换和队友轨迹，共 8 格。旧 16 格清单及通信 2 s 数据保留作历史证据，不计入当前覆盖。长时采集由用户运行，不自动启动。

本次把官方 SubT Urban 2 Story 接入独立 Gazebo 场景，保留楼梯、竖井、隔墙、梁柱、工业设备和原始碰撞网格。约 40 × 40 m、两层、总高约 20 m。它是有纹理的工业建筑，不是摄影测量城市，也没有新增室外复杂地形。

## 当前能直接用到什么程度

可直接打开 GUI、自由观察、导出双机场景、采集两路 RGB/私有实例图，以及执行不解锁的 PX4 双机遥测检查。已验证墙前不可见、墙后可见，二层也能真实渲染。

**还不能把原 `search_pilot --execute-flight` 命令直接当成这个场景的飞行入口。** 当前没有修改旧控制器、默认世界或模型输入。双机只读启动成功，但结束时两机 `pre_flight_checks_pass=false`；`flight_ready=false`、`flight_admitted=false`。尚未验证机体净空、楼内飞行、跨层路线和真实模型协作闭环。没有任何新模型成绩。

2026-10-21 的冻结事件头 checkpoint 已通过一次跨 Windows/WSL 的 5 秒 Gazebo 代理 smoke（`D:/AeroWAM-data/subt_proxy_checkpoint_20261021_192240`）：11 个模型请求和 11 个动作服务请求均被独立重建，进程清理无误。该适配器只输出 hold 动作并拒绝无 bbox 的 grounded event，因此 `proxy_event_success=false`、`kinematic_proxy=true`、`px4_started=false`、`flight_admitted=false`；这只是实时 Core 接口证据，不是飞行或 handoff 成绩。下一次终端操作应先运行本文件后面的不解锁 startup 检查，确认两机 PX4 遥测和控制端点，再讨论真实动作适配。

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

若要在下一轮启动检查中验证 PX4 控制写端，再加 `--control-link-check`：程序会等待六个逐机 writer 各匹配一个 PX4 reader，并发送每机一条 `command=400` 的未解锁 disarm 请求，要求收到新鲜 `result=0` ACK。它不会发送 offboard、轨迹、解锁或起飞命令；即使通过，`flight_admitted` 仍为 `false`。

```bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_urban startup \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --px4-root /home/ryx/PX4-Autopilot \
  --agent /mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent \
  --ros-domain-id 83 --timeout-s 120 --control-link-check \
  --output-dir "/mnt/d/AeroWAM-data/subt_startup_control_$(date +%Y%m%d_%H%M%S)"
```

准入检查应同时看到 `startup_passed=true`、`control_link_verified=true`、两条 `result=0` disarm ACK、`motion_commands_sent=0` 和 `owned_processes_stopped=true`。若失败，保留输出目录和 `control_link_diagnostics`，不要通过重试或放宽门限把它改成通过。

## 已保存的实测证据（截至 2026-10-21）

| 检查 | 结果 | 本机数据目录 |
| --- | --- | --- |
| Win Python / WSL Python 3.10 回归 | 新场景各 13 项通过；连同旧 warehouse 共 25 项通过 | `subt_urban_self_test.py` / `warehouse_self_test.py` |
| 四组双相机静态验收 | 通过，每路 4–8 对精确时间戳帧 | `/mnt/d/AeroWAM-data/subt_urban_admission_v2` |
| 墙体遮挡 / 重新可见 | 红蓝目标在两路中分别隐藏 / 有效可见；117,517 个碰撞三角形参与私有射线检查 | 同上 `inspection.json` |
| GUI + 双实体无人机 | 两次 18 秒有界启动正常，GUI 插件和两机系统加载；未接飞控，v2 保留 GUI 日志 | `/mnt/d/AeroWAM-data/subt_urban_gui_smoke_v2` |
| 独立 PX4 双机只读启动 | 通过，两机均未解锁，运动指令 0 | `/mnt/d/AeroWAM-data/subt_urban_disarmed_v1` |
| SubT 楼梯资源上的 PX4 双机只读启动复核 | `startup_passed=true`，`flight_ready=false`，运动指令 0；`uav_0/uav_1` 各 8 个相机配对，odometry 75/74 条，status 各 2 条，清理正常 | `/mnt/d/AeroWAM-data/subt_startup_20261021_193713` |

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

#### 楼梯 GUI 预览

不要使用 `subt_urban preview --move-targets` 查看 handoff 路线；那个入口展示的是一楼工业标靶的水平往返小车。楼梯 handoff 使用独立入口，目标、载体和两个相机代理沿已审计的折返楼梯路线更新：

```bash
cd /mnt/d/AeroWAM
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.subt_handoff_preview \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --target blue:cube --swapped \
  --speed 0.4 --preview-seconds 100 \
  --output-dir "/mnt/d/AeroWAM-data/subt_stair_gui_$(date +%Y%m%d_%H%M%S)"
```

`--preview-seconds 100` 覆盖完整路线，结束摘要中的 `reached_upper_floor` 应为 `true`。这是 Gazebo GUI 中的运动学代理预览，不启动 PX4、不解锁、不发送飞控命令，也不代表真实飞行成绩。较短的 20 秒只显示起始楼层和上楼前段；目标不再沿一楼左右往返。

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

`--speed` 默认 0.3 m/s、上限 0.4 m/s。队友轨迹变化用 `--teammate delayed` 表示；通信延迟现在有独立的 `--message-delay-s 0.5|2` 参数，可以在任意轨迹条件下交叉设置。例如，仅改轨迹而保持通信：

```bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_handoff_collect \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --target blue:cube --swapped --teammate delayed --message-delay-s 0.5 \
  --output-dir /mnt/d/AeroWAM-data/subt_stair_blue_motion_only_v1
```

显式给出 `--message-delay-s` 时，队友轨迹和通信分别受控；省略该参数仍兼容旧行为（normal=0.5 s，delayed=2 s），报告会标记它是旧式联合默认值。固定种子、按整 episode 随机化的 2×2×2 固定通信发展采集计划可用 `subt_handoff_design --output ...json` 生成；清单本身不会启动 Gazebo。计划单元是整条 episode，不是连续帧；每格一条只用于覆盖条件，不构成独立重复或显著性证据。目标颜色/形状也仍然绑定。所有候选条件都必须重新通过几何、渲染和因果审计。

当前计划器另列出 8 条 `development_training_only` 上行条件，以及 8 条候选 `held_out_route_test` 下行条件；全部固定通信 0.5 s。下行条件需同时传 `--reverse --split held_out_route_test`；采集器和独立检查器会核对 `route_id=stair_descent_v1`、方向及分区。任何留出测试 episode 都会被训练导出器拒绝。该留出只改变同一楼梯折返路线的行进方向，属于同场景族的任务方向留出，不是独立场景或新几何；需要独立场景泛化证据时，仍须接入另一场景并重新审计。

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

2026-10-21 的两条新蓝色身份置换示范分别为 `subt_stair_blue_swapped_motion_only_v1`（轨迹 delayed，消息 0.5 s）与 `subt_stair_blue_swapped_comm_only_v1`（轨迹 normal，消息 2 s）。均为 236 个双视角样本、472 张 RGB、10 m 跨层跨度；独立审计重放事件成功，位姿请求分别核验 2358 / 2361 条。这是采集条件和审计器的可用性证据，不是模型效果。旧红色示范仍属无条件元数据的 legacy；已审计蓝色基线是 normal/0.5 s，仍未组成平衡完整矩阵。新审计器允许有效检测框边界处只有 1–3 个目标色素的遮挡掩码按原 15% 色彩比例通过；可见性仍由原严格实例框门槛控制。

当前数据只支持特权教师下的联合事件模仿与下一观察预测开发：相机朝向依赖隐藏目标位置，双视角输入不是已验证的分散式策略；相机姿态来自理想代理而非 PX4 遥测。期望下一观察位姿不是已执行飞控动作，因此导出不提供运动控制监督，也不能宣称建立动作条件世界模型。

开发导出仍只接收 `development_training_only` 上行 episode。方向留出下行 episode 使用独立分区并由导出器拒绝，不能混入训练。连续帧、颜色变化和身份置换不能当作独立测试重复；同场景反向路线也不等同于独立场景。尚无模型成绩、组合迁移结论或飞行放行。下一阶段先完成 8 格固定通信开发条件逐条采集与独立审计，扩展方向留出路线的审计样本，再在 episode/路线分组固定后训练和评测事件策略与世界模型；仍需另建独立场景才能讨论场景泛化。

已加入 `subt_handoff_learn.py` 作为可复现的探索性基线：它使用公开双视角 RGB、公开代理位姿和已送达消息训练集中式事件分类器，并预测下一时刻 8×8 RGB/公开代理位姿。一次运行使用两条上行现代 episode 训练、红球下行留出 episode 评估，结果写入数据盘并经 `audit` 独立重算；事件宏 F1 0.1939，稀有事件在留出 episode 未检出，RGB RMSE 0.0587，公开代理位姿混合 MAE 1.2174。该结果是单 episode、同场景族的探索性诊断，明确 `model_score_eligible=false`，不代表分散式策略、动作条件世界模型、PX4 飞行或组合迁移。

当前采集命令使用 `subt_stair_fixedcomm_design_20261021_v3.json`，旧 v2 清单不再使用。队友轨迹的 `delayed` 是运动到达较晚，消息延迟仍为 0.5 s。8 格覆盖、单场景方向反转和独立模块离线指标都不能独自证明组合迁移；正式实验仍需新任务留出及协作技能/世界模型预测组合的对照。历史轻量基线混用了通信条件且未形成组合决策闭环，只保留作接口诊断。
