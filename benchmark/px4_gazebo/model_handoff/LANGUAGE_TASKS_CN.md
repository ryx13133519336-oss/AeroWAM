# 语言目标绑定与独立多任务验收（draft）

2026-10-18 更新：新主线不固定 UAV 编号职责。新增 `role_free_task` 的任务 /2-draft
与角色自由离线验收，任一架可先发现/发起交接，另一架完成后续独立证据。
配对任务显式使用 `role_mode="unassigned"`。下文 /1 默认角色和已有飞行诊断保留
用于兼容，不代表新任务要求；尚未迁移实际飞行流程。详见 ROLE_FREE_MODELS_CN.md。

2026-10-14 创建，2026-10-15 更新。离线验收/静态真实像素及不解锁启动分步验证；新增搜索runner草案，飞行闭环待验证，
不表示已完成新 benchmark，也不改变旧 Handoff 的成功率或专家数据。

## 公开模型接口与私有答案

`language_tasks.py` 定义 `aerowam-language-task/1-draft`：公开原文指令、
任务类型、source/receiver 角色及验收规则；结构化外观描述、场景物体 ID、
真值位置和巡检视角标签只留在 evaluator。任务可由模板生成、作者绑定
唯一外观描述，不是任意自然语言解析器，改写指令仍需人工审查语义一致性。
用户已要求实际指令手动输入：search_pilot默认交互接收原文，评分目标另行
人工选择、不传模型。新runner与旧模型辅助报告器的边界见SEARCH_PILOT_CN.md。

`paired_target_tasks` 在同一场景里为不同物体生成不同指令和不同正确答案。
至少需要两个外观可区分物体；外观描述指向零个或多个物体时拒绝绑定。
同场景所有指令条件、候选及重复应作为一个场景组留在同一数据划分。
场景组是配对实验单位，不能把图像帧数当作独立重复；场景生成、分组
划分调度、跨因素组合留出与分块随机运行次序尚未实现。

`aerowam-target-report/1-draft` 的模型输出包含 agent_id、实际观察到的
image_timestamp_ns、归一化 xyxy 图像框、confidence 和 event。模型不得
输出目标真值 ID 或巡检真值视角。事件可为 observe/found/handoff_request/
handoff_accept/confirm/inspect；confidence 只记录，不作为可绕过验收的条件。

`grounded_interface.py` 提供独立 `aerowam-grounded-action/1-draft` 信封：
旧 ModelJointAction 加每机至多一个 TargetReport。2026-10-15 已连接可选
GroundedAdapter 与 PolicyClient.infer_grounded，通过本地实际TCP回环验证。
策略函数收到经过校验的公开ModelObservation，返回GroundedAction；不传入
私有绑定或标注。双方核对报告必须引用当前请求中对应相机的实际帧。
该时间戳校验只证明引用输入帧，不证明目标身份/可见性或完成任务。
尚未连接候选选择器或实时任务执行器，也没有因此注册“已送达”私有证据。
原动作协议 `/1` 与候选决策协议保持不变。

## 三种任务的不同验收

`language_evaluator.py` 是私有离线状态机，不生成仿真标注，也不向模型
返回匹配到的真值身份。传入的盒子必须匹配指令选定物体的可见图像框，
不能只验证路线几何或 active_tracker 字段。

| 任务 | draft 证据顺序 |
|---|---|
| search | source 在自己的图像中 found 正确目标，然后 receiver 用事件后自己的图像独立 confirm 同一目标 |
| handoff | source 连续跟踪 → request → receiver accept → receiver 连续跟踪 → receiver confirm 同一目标 |
| inspection | 两机分别以私有几何计算的互补 required views 完成连续巡检 dwell，覆盖所有要求视角，然后两机各自 confirm 同一目标 |

所有任务结束前的安全违规可覆盖已完成的任务证据；success 只有 finish()
后才是最终结果。基础设施故障通过 invalidate(simulator_error, reason)
标记；缺失/冲突/未知身份标注为 evaluator_error，不冒充模型能力失败。
错误目标、模糊图像框、未观察或过期帧、无效事件顺序、证据不足则是
model_failure。报告框的歧义是模型定位问题；标注本身不可信时，harness
必须明确 invalidate(evaluator_error, reason)，不能让不可靠标注成为评分依据。

每个相机帧只支持一个事件且只累计一次 dwell；同一帧重复 confirm 不会
补出新事件。不接受已被新帧替代的旧帧。看不见目标的已送达帧和过大
证据间隔会中断连续 dwell。accept 之前捕获的帧不能增加接管后时长。

## 时间、标注和 draft 阈值

ObservedFrame 必须由可信 harness 在图像真正送达策略之后注册，包含
全部私有场景 ID；不可见物体标为 visible_bbox=None，不能用遮挡后完整
轮廓的 amodal box 伪装可见证据。inspection_view 来自私有几何计算，
不是模型输出。图像框质量、遮挡下可辨识条件和视角几何规则需另做 pilot。

image_timestamp_ns 用于查找帧；sim_time_ns 是私有、已对齐的仿真捕获时刻。
report(now_sim_time_ns=...) 使用同一仿真时钟判断新鲜度与事件顺序，不将
host UNIX、monotonic 和相机时间直接相减。实时标注与时间对齐没有实现，
现有间歇性 host 时钟异常仍需解决，此模块不会自动修复它。

当前 TaskRules 默认值：dwell 0.5 s、最大证据间隔 0.2 s、最大报告年龄
0.2 s、IoU 0.5、最佳匹配与第二匹配分差 0.1、巡检视角 front/side。
这些是可配置开发阈值，不是冻结的正式实验标准；需 pilot 后提前固定。
不替换原协议的 100 ms 双相机同步与默认 400 ms 推理期限。

## 已验证与下一步

### 私有像素标注入口（2026-10-14 补充）

`pixel_annotations.py` 已实现独立实例标签图 → 可见框/私有视角 → ObservedFrame
入口。它不生成实例图，飞行生产链路仍未接入 Gazebo 实例图。实际检查现用
mono_cam 只有 RGB 传感器（1280×960，horizontal_fov=1.74），当前旧采集
配置不能直接得到实例图。下述独立静态场景不修改 PX4 模型或旧采集命令。

可信渲染接入端必须提供同次捕获的 RGB 与实例图，两者 CaptureKey 的相机、
图像时间、仿真捕获时间、render_id 和尺寸完全一致；映射覆盖全部场景物体，
每个出现的标签都明确归属，0 为背景，已知环境标签映射 None。主体和条纹
可以有多个标签但必须映射到同一个物体。未知标签/缺失身份/错帧是标注错误，
由 harness 转为 evaluator_error，不能作为模型失败。

框只包围实际可见实例像素：遮挡处不补全。draft 默认至少16像素、框宽高
至少3像素、框内像素比例至少0.15，并拒绝贴边裁切。参数需 pilot 校准。
标签图必须与模型看到的 RGB 采用同一 raster 变换；标签缩放必须最近邻，
不能将原始1280×960标签直接对应256×256 JPEG，也不能线性插值标签 ID。
匹配元数据不自动证明传感器同步或渲染标定正确，后续还需真实画面对照。

实例像素支持不等于颜色/形状/条纹均足以辨识：当前未验证外观属性可见性，
也没有可靠的完整轮廓遮挡比例。正式语言身份评分前必须审查这些条件，
不把少量实例像素当作已证明的语言理解证据。

CaptureGeometry 用同次捕获的 ENU 相机/物体位置及物体局部到 ENU 的正交
旋转计算视角。局部+X定义 front，±Y定义 side；draft 距离0.5–6m、水平扇区
半角25°、最大仰俯角30°。不支持任意视角名称，也不推断球体语义上的
“正面”；局部坐标基准需固定并记录。几何永远不能使无实例像素的物体变可见。

有效几何但角度/距离不合格时 inspection_view=None 且
inspection_geometry_available=True，只中断巡检 dwell，不报标注缺失。
中间已送达的无效/其他视角帧即使没有模型报告，也会打断原视角连续时长。
真正缺少巡检几何仍为 evaluator_error。9项合成像素测试通过，不代表真实
Gazebo 像素标注已通过。

### 真实 Gazebo 静态标注检查（2026-10-14）

`annotation_probe.py` / `annotation_probe.sdf` 提供独立 Gazebo transport
分区的无 PX4 静态场景；包含普通 RGB 相机与同位置/参数的 instance 相机，
256×256、10 Hz、horizontal_fov=1.74。用系统 Python 的 gz.transport13 /
gz.msgs10 接收，不依赖用户训练 venv 安装 Gazebo Python 包。保存原始
protobuf、日志、实际场景及 SHA256；只停止自己启动的 Gazebo 进程组。
少于三对完全同时间戳的 RGB/labels_map 时检查失败，不近邻复用帧。

实测 Gazebo 8.15.0：分割相机发布 `<topic>/colored_map` 与
`<topic>/labels_map`，不提供普通 RGB。两者虽然都标记 RGB_INT8，
colored_map 是随机标签预览，不能当作普通 RGB 或解码出目标身份。
`gazebo_rasters.py` 严格使用实际 labels_map：前三字节中前两字节为
不透明实例计数，末字节为场景作者分配的 label；不猜实例字节序或实体 ID。
当前场景以唯一非零 label 标识物体，0 背景、33墙、11红球、22蓝方块，
红球主体与黑条纹共同映射到同一物体。此方式限制 label 在8位范围，
不等于通用语义类别标注。未知 format/packing、错时间戳、错尺寸、坏
buffer/stride、误用 colored_map 均拒绝。支持有行填充的RGB缓冲区。

`annotation_probe_inspect.py` 用真实像素接入 pixel_annotations，生成
RGB、原始标签.npy/PNG、叠框图及 inspection.json；还进行静态外观像素
核对，此处颜色只用于检查，不用于生成验收真值。

| 独立静态条件 | 同时间戳 RGB/标签对 | 红球可见标签像素 | 蓝方块像素 | 标注结果 |
|---|---:|---:|---:|---|
| 无遮挡 | 8 | 592 | 642 | 两目标可见 |
| 部分遮挡 | 8 | 500 | 642 | 框只来自可见像素 |
| 完全遮挡 | 8 | 0 | 642 | 红球不可见，干扰物仍可见 |

三个条件都通过线格式/标签身份/静态标注检查，RGB及叠框图已人工式视觉
核对；红球可见条件下黑条纹标签像素32。原始证据目录为
`/mnt/d/AeroWAM-data/annotation_probe_static_rgb_v2`（部分遮挡）、
`annotation_probe_static_clear_v3` 和 `annotation_probe_static_full_v3`。
每次检查中的连续静态帧是技术重复，不能当作8个独立实验场景。
Windows 与用户 WSL venv 均通过28项相关回归（解码5、像素9、语言14）。

这些证据仅验证静态渲染接入；不证明移动双机传感器捕获/姿态同步、
1280×960→256×256 JPEG 变换对齐或复杂遮挡下的语言属性可辨识。
`capture_pair_verified` / `moving_capture_calibration_verified` 和
`live_multitask_ready` 仍为 false。实时 runner、公开技能和外部模型接线
仍待完成，未据此启动新专家采集或训练。

如需复现，使用新输出目录（工具拒绝覆盖已有诊断）：

```bash
cd /mnt/d/AeroWAM
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.annotation_probe \
  --scene-case full --timeout-s 25 \
  --output-dir /mnt/d/AeroWAM-data/annotation_probe_full_recheck
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.annotation_probe_inspect \
  --probe-dir /mnt/d/AeroWAM-data/annotation_probe_full_recheck
```

无飞行测试验证换指令换正确目标、错误目标拒绝、严格消息格式、重复
帧不增时长、过期/未来/旧帧拒绝、交接顺序、遮挡中断、双机互补视角与
确认、晚期碰撞覆盖，以及标注/仿真错误分类；不验证实际像素识别能力。

```bash
cd /mnt/d/AeroWAM
source ~/venvs/aerowam-model/bin/activate
python3 -B -m benchmark.px4_gazebo.model_handoff.language_self_test
python3 -B -m benchmark.px4_gazebo.model_handoff.pixel_annotation_self_test
```

### 双视角原生相机配置验证（2026-10-15）

`annotation_models.py` 从已安装mono_cam/x500_mono_cam生成独立命名的可选
模型，源文件不变。保留RGB传感器、机体及相机安装位姿，在同一camera_link
添加同pose/视场/分辨率/频率的私有instance传感器；源及生成文件均有SHA256。
含固定显式topic、畸变或噪声的未知相机配置会拒绝，避免静默错配或两机topic冲突。
生成资源使用本机绝对引用，不可搬移后直接沿用。

独立无PX4探针 `annotation_probe_dual_native_v1` 使用两台上述相机，保持
原生1280×960、30Hz，实际发现两侧各8对严格同时间戳RGB/labels_map。
两侧原生红球像素12658/15138、蓝方块15924/14400，主体及条纹映射通过。
`resize_pair` 按现有图像路径全图拉伸到256×256，RGB双线性、标签最近邻：
两侧缩放红球672/812、蓝方块852/768。原生/缩放像素验收均通过，两个
缩放叠框图已视觉核对。未更改旧collect、Px4GazeboSession或用户安装SDF。
Windows及用户WSL venv均通过33项无飞行回归（相机生成4、解码/变换6、
像素标注9、语言验收14），不代表PX4闭环或组合迁移验证通过。

不能仅导出PX4_GZ_MODELS选择这些模型，因为PX4启动脚本会重置该变量；
后续独立runner需显式生成/加载机体并用PX4_GZ_MODEL_NAME连接。
生成机体的双PX4无解锁连接现已验证，详见下节；飞行仍未测试。
实时策略送达、移动曝光/姿态对齐、复杂外观下JPEG可辨识性和闭环仍未验证。
`flight_ready`、`moving_capture_calibration_verified`、`live_multitask_ready`仍为false。
通信受限时禁止用这个集中式双视角诊断路径绕过队友信息限制。

下一步将私有实例图接到独立双机runner并验证移动捕获对齐，审查可辨识属性，完成公开观测驱动的技能执行器，
再把可选接口连接独立 runner。之后少量仿真核对任务可完成性、误验收/
漏验收与时钟质量，再冻结新版本及分组划分。此前无需重采旧 280 episodes
或重跑 50k 步训练，也不能把合成测试成功包装成多任务闭环成功。

### 双机无解锁连接与报告传输（2026-10-15）

`annotation_session_smoke.py` 在独立世界/transport分区中显式加载两台生成
机体，用PX4_GZ_MODEL_NAME连接PX4实例20/21；独立工作目录、ROS命名空间、
domain和临时DDS Agent端口。只接收图像、状态及里程计，不发送解锁、
Offboard或运动指令。退出仅清理本次拥有的进程组，不清理用户其他仿真。

默认继承DDS传输的v2启动通过，但v4在两侧图像各8对的情况下，所有ROS
状态/里程计计数为0并超时；原因尚未定位，不能据此认定SHM是根因。
`--udp-only`仅给本次诊断进程配置FastDDS UDPv4，不更改系统配置或清理SHM。
两个独立运行 `annotation_session_noarm_udp_v5`、`annotation_session_noarm_udp_v6`
均通过：每侧8对RGB/labels_map，状态各2条，里程计分别各65/26条，
均未解锁、无failsafe、位姿/速度/四元数有限，运动指令数0；自有进程全部退出。
这是启动重复验证，不是长时间稳定性或旧采集故障的修复证明。

`annotation_session_inspect.py` 核对实际世界和捕获文件的SHA256、时间戳、
原生分辨率及变换后可见像素。v5/v6的两侧256×256红球像素均为711/1156，
蓝方块912/1124。v5真实JPEG及叠框已视觉核对；v6离线校验通过。
`public_images`仅写真实RGB生成的JPEG；标签、实例身份及叠框放在
`private_annotations`。这是代码/API及目录边界，不是操作系统访问隔离。
此处最新里程计没有与曝光对齐，不构造ModelObservation或伪造策略送达记录；
`capture_aligned_telemetry`、`policy_delivery_verified`均为false。

可选 `GroundedAdapter` / `PolicyClient.infer_grounded` 已通过真实TCP回环：
策略只接收既有公开ModelObservation，返回GroundedAction；报告必须引用
当前输入对应相机的时间戳。该约束不证明目标身份、可见性或任务成功，
仍由私有evaluator验收。示例只hover，不是已接入的参评模型。
现有集中式双视角输入仍仅供诊断，不能用于通信受限结论。

Windows与用户WSL venv均通过40项相关回归（相机生成4、解码/变换6、
像素9、语言14、隔离启动配置3、报告传输4）；旧接口自检也通过。
`flight_ready`、`moving_capture_calibration_verified`、`live_multitask_ready`
仍为false。后续重点是捕获/状态对齐、真实多任务runner和观测驱动技能执行，
随后落实受限队友信息、行为变化和组合留出；不扩大旧任务训练代替这些验收。

如需复现无解锁诊断，使用新输出目录：

```bash
cd /mnt/d/AeroWAM
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.annotation_session_smoke \
  --px4-root /home/ryx/PX4-Autopilot \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --timeout-s 100 --udp-only \
  --output-dir /mnt/d/AeroWAM-data/annotation_session_noarm_manual_01
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.annotation_session_inspect \
  --probe-dir /mnt/d/AeroWAM-data/annotation_session_noarm_manual_01
```
