# 手动指令搜索 pilot（更新：2026-10-18）

先连接手动指令、真实观测、动作和独立验收，再比较三个方法。此处只有一组静态
双目标场景；通信不受限，新增队友响应扰动尚未飞行验证，没有组合留出或候选预测
比较，不是正式结果。

2026-10-18 已用相同目标资产完成独立静态遮挡/侧面可见对照，48对真实RGB/实例图
通过；命令、像素证据和限制见 [SEARCH_OCCLUSION_CN.md](SEARCH_OCCLUSION_CN.md)。
该诊断不启动PX4，不改变这里的默认飞行场景，也不代表连续重新观察或模型记忆已验证。
随后同一服务器内的分步相机序列也通过：每阶段5组联合帧，依次可见、遮挡、重新可见。
同一文档记录实时位姿流验收和失败修复；仍未验证连续飞行、曝光位姿或模型历史依赖。
随后历史信息对照也通过：两布局遮挡RGB逐像素一致、过去目标位置互换、正确左右判断
相反。见 [SEARCH_HISTORY_CN.md](SEARCH_HISTORY_CN.md)；只验收信息必要性，没有模型
推理或迁移结论，不改变本页的飞行接口。

## 手动输入

默认在启动仿真前提示：

1. 用户输入原始自然语言指令，原文传给模型，不自动改写或轮换。
2. 正确目标：1=红色黑条纹球，2=蓝色白条纹方块。此项是单独的私有评分标注，
   不作为模型输入。用户需核对指令与正确目标语义一致。

示例：“寻找红色的球，让另一架无人机确认同一个目标”。当前规则固定为UAV0
报告发现、UAV1用之后实际观察帧确认。同一episode不支持中途换指令；换指令用
新目录重新启动，策略历史重置。`--instruction`与`--expected-target`可非交互传入；
默认交互。`--paired-check`只供开发检查两个作者条件，默认关闭。

## 组件与限制

`search_reporter`是三种方法共用的显式辅助RGB报告器：按文字里单一红/蓝颜色
（中英文）定位JPEG，不读取标签、真值ID、位置或评分反馈。不识别形状/条纹，
依赖每种颜色仅一个物体；不支持通用关系、否定或任意任务语言。无颜色、多颜色、
部分显式否定会拒绝，其余复杂表达仍不能保证正确理解。固定confidence=1不能
用作置信校准。颜色框可能与私有可见框IoU不足，失败保留，不改验收迎合报告器。

- `rgb_scan`：缺少目标时转动视角，可见时保持；只是接线阳性对照。
- `checkpoint`：旧模型输出动作/意图，另附上述共享报告器，不注入专家路线，
  不替换模型动作。应称“旧模型+pilot RGB reporter”，不是纯BC或端到端语言模型。
  旧检查点只训练过交接，新搜索有分布变化，失败不能直接归因于架构。
- `checkpoint_search`：显式新增的移动证据入口，保留检查点动作/意图，辅助报告器
  使用公开RGB和本机里程计，不生成转向、前进或悬停动作。与`rgb_search`共用
  `search_approach_acceptance/1`私有移动验收，不与旧`checkpoint`扫描结果混算。

新飞行控制使用独立PX4实例20/21、ROS domain78、临时DDS端口、独立世界。
两目标初始位于UAV0视野之外，需重新观察。增量除以0.5秒转速度，沿用
FLU→NED转换及速度/加速度限制；发布不等于已确认精确执行。遥测、Offboard、
局部范围、估计双机间距、请求看门狗和降落/未解锁有检查，两目标有接触传感器。
未验证完整接触覆盖，不能声称安全已全面验证。最新本机遥测不是曝光对齐状态，
`capture_aligned_telemetry=false`，不产生世界模型训练数据。
只在推理确实成功后注册本次输入帧；私有标注不传模型。目录/API分离不是系统沙箱。

## 队友响应变化接线（2026-10-18，仅非飞行验证）

主线仍为：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型
预测能力能否组合迁移到新任务。本次先把队友运动响应的变化落实到控制请求边界，
使“计划请求”和“给控制器的请求”可区分；不是独立队友策略或完整迁移评测。

`teammate_response.py`使用显式版本`aerowam-teammate-response/1-draft`。
可选`--teammate-response`，默认`nominal`。非默认条件只允许搭配`rgb_search`或
`checkpoint_search`，旧扫描入口不能启用扰动。

| 条件 | UAV1运动请求处理 |
|---|---|
| nominal | 原动作对象直通，默认行为保持原样 |
| cautious | 四维增量均乘0.5，不改变UAV0或intent |
| delayed | 延后0.75仿真秒；每次新决策到达时，选已到期的最新UAV1请求，首次到期前为零 |

延后从推理返回、交控制器前读取的仿真clock计起；仅在决策边界释放，不是精确
连续延迟线路，实际额外等待可能超过0.75仿真秒。不延迟图像、报告、消息、起飞、
普通降落或安全检查，不延长RUN/看门狗预算。各episode新建队列；回退/重复时戳、
超出队列上限拒绝。报告仍指当前观察帧，不能延后重放旧found/confirm冒充新证据。

策略原始动作留在`policy/requests.jsonl`的`response.action`；实际交控制器前的
请求另记为`controller_requested_action`，下一观测`action_history`使用后者。
这些仍是低层限幅/加速度限制之前的请求，不是PX4已确认执行；已有
`published_controls.jsonl`继续记录实际发布。条件身份、参数、源请求step与私有
事件只保存在evaluator日志/summary中，不加入策略输入；公开历史中的响应后动作
属于允许的自身控制历史，模型可从历史与观测推断响应变化。

`evaluator/teammate_response.jsonl`逐步记录原请求、响应后请求、仿真时戳及来源。
独立`audit_response_trace`检查顺序/因果、UAV0/intent不变、缩放/到期选择正确性，
并交叉核对runner控制请求记录。空记录或全零不证明扰动实际起作用；审计通过
不证明物理响应或任务成功。原私有目标/移动/框验收不受扰动条件放宽。

Windows和WSL/ROS各102项相关回归通过，候选/trace接口自检通过；控制器桩验证了
到期前UAV1为零、到期后进入实际控制转换，队列仍有待发请求时断链仍立即请求
两机降落。已存红93/蓝115条规则请求，各三条件的离线变换/审计通过：nominal
改变0条，其他条件各改变93/115条。本回放以录像捕获戳代替控制边界戳，只验算
请求变换，未运行控制器、未重算轨迹/任务结果，不是扰动后的闭环成绩。
已增加已知私有队友条件/移动标定/真实候选收益字段的递归拒绝与adapter反例；
关键字检查不是任意语义泄露检测或系统访问隔离。

未启动本版飞行或训练。当前仍为集中式双机策略，未实现独立队友决策、真实遮挡
与历史依赖验收、多任务候选执行或组合留出。后续应把这些缺口落实到场景/独立
验收；本扰动组件不能直接当作“新队友策略泛化”证据。

## 当前证据

14项新回归通过，覆盖版本化PX4回执话题、手动原文、独立答案、换目标/错目标拒绝、RGB报告、跨episode
重置、未送达/重复帧、异常分类、局部坐标、看门狗/间距、控制转换及降落命令。
旧语言、报告传输及接口测试继续通过。WSL新场景无解锁启动通过：
`/mnt/d/AeroWAM-data/search_manual_startup_v1/condition_00/evaluator/summary.json`，
`startup_passed=true`、`execute_flight=false`、自有进程退出、无清理错误。
此检查使用开发者测试文字，用户的实际指令由本人输入。

2026-10-16 首次带GUI的真实SITL搜索闭环通过：
`/mnt/d/AeroWAM-data/search_rgb_scan_gui_diagnostic_v1/condition_00/evaluator/summary.json`。
复用用户已输入原句“寻找红色的球，让另一架无人机确认同一个目标”；两机初始
均不可见目标，18次策略请求后得到发现/后续图像确认，最终未解锁，两机PX4日志
均记录`Disarmed by landing`。自有进程退出、无清理错误，RUN最大发布间隔约53 ms。
这是规则扫描正对照，不是学习模型成绩；JointBC新任务推理尚未验证。
`live_multitask_ready=false`仍保留，不立即批跑三个方法。

本次修正/诊断：命令回执按`MESSAGE_VERSION=1`订阅`vehicle_command_ack_v1`；
控制发布采用显式PX4 QoS；只在新建隔离SITL参数中设置`NAV_DLL_ACT=0`，适配无
GCS运行，不关闭估计器/解锁检查，不强制解锁或解锁后强制停桨。增加状态、回执、
逐路启动诊断及同partition GUI。一次成功不能区分各项修改的独立因果贡献。
启动曾出现图像持续正常但ROS迟到/超时，根因尚未确认；默认等待150秒，执行器
定时唤醒，不放宽运行中的图像新鲜度/遥测门槛。50项相关非飞行回归通过。
定时唤醒后的独立无解锁复查（无额外ROS查询进程）通过：
`/mnt/d/AeroWAM-data/search_control_link_v5/condition_00/evaluator/summary.json`，
约35秒启动、两机命令回执接受、输入各匹配1个PX4订阅端，进程清理无错误。
这只证明该轮独立启动/命令往返，不证明ROS启动波动的根因已消除。

### 2026-10-16 起飞前链路门槛与Agent版本

用户`search_manual_rgb_scan_02`复现：104次模式命令发布、零命令回执、始终未解锁/
未进入Offboard，不能算搜索模型失败。新增强制门槛：六个实际控制发布端提前创建，
每端只允许1个订阅端；两机健康未解锁、状态新鲜，且都收到本轮发送普通未解锁
命令后的接受回执，才创建飞行定时器。起飞复用同一组发布端；检查失败不发送模式/
解锁/运动设定值。运行中连接丢失请求降落，不保证断链时降落命令能够送达。
`evaluator/control_link.json`保存端点GID、类型、QoS、匹配历史与预检命令。
匹配与命令回执不等于运动设定值已执行；飞行仍要求实际Offboard/解锁/高度证据。

系统`/usr/local/bin/MicroXRCEAgent`实际依赖`libmicroxrcedds_agent.so.3.0`，构建缓存
为3.0.2；虽现有源码标签已为2.4.2，运行二进制并未随之更新。PX4内置客户端为2.4.x。
已用本机已有ROS Humble FastDDS2.6/FastCDR1库在独立目录编译Agent2.4.2：
`/mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent`。
没有下载依赖或系统安装，不覆盖系统Agent、用户旧构建、旧权重和数据。
runner启动前检查实际动态库主版本；拒绝3.x或无法核对的运行库，并提示`--agent`。
该检查是当前环境的保守门槛，不能据此声称所有3.x失败都已归因或所有2.x组合都兼容。

新Agent首次无解锁预检通过（本轮包含一次额外只读ROS查询）：
`/mnt/d/AeroWAM-data/search_control_gate_agent242_v1/condition_00/evaluator/summary.json`。
独立复查（无额外ROS查询）也通过：`search_control_gate_agent242_v2`同级summary，
启动约106.9秒、六端点各匹配1个订阅端、只发2条普通未解锁命令、两机接受回执。
自有进程均退出、无清理错误；启动迟到仍明显，不能宣称长期稳定或根因全部消除。
此前旧3.x Agent在`search_control_gate_v1`（所有网卡）和
`search_control_gate_loopback_v1`（回环）均图像正常/ROS计数0而启动超时；回环不是已证实
修复，默认保留`udp_all`，可选`--dds-transport udp_loopback`仅供诊断。
60项非飞行回归通过；新增9项覆盖缺端点、旧/错/拒绝回执、状态异常、不可绕过门槛等。
更新后的飞行闭环仍需验证，不以旧版本首次成功作为新版本飞行复验。

### 2026-10-16 GUI恢复后的控制下行复查

用户执行WSL重启后，普通测试窗口和Gazebo场景均恢复显示；此前WSLg共享内存
报`Input/output error`，普通非3D窗口也不可见，不是搜索任务本身未运行。
`search_manual_rgb_scan_03`已有一次15步发现/确认及安全降落成功，但不能代替下面
控制发布配置变更后的飞行验证。

`search_gui_check_20261016_191409`在GUI恢复后仍未起飞：六输入端匹配，但50次
预检发布无回执；两机ULog都没有记录到预检command=400。此轮是起飞前基础设施
失败，不计模型失败。单独改ROS回环profile未修复；PX4自带的participant XML会
覆盖Agent默认网络配置。一次性同时设置`ROS_LOCALHOST_ONLY=1`和新SITL参数
`UXRCE_DDS_PTCFG=1`后，Agent端确实只绑定localhost，但仍发生ACK超时。因此
不把回环当作已验证的修复，也不改变默认`udp_all`或全局网络设置。

搜索pilot的六个控制写端改用`RELIABLE/VOLATILE/KEEP_LAST/depth=10`（与
`ros2_px4_adapter`默认深度10写端一致），遥测订阅QoS不变，不缓存控制命令供
迟到读端回放。默认传输下两次独立不解锁检查通过：

- `search_control_reliable_post_restart_20261016_v3`：一次性诊断覆盖，启动约37秒，
  门槛约0.30秒；两条预检请求、两机接受回执。
- `search_control_reliable_post_restart_20261016_v4`：实际修改后CLI，启动约24.5秒，
  门槛约9.40秒；20次预检发布后，两机ULog各记录到command=400及接受回执。

两轮都保持`arming_state=1`，不发布模式切换/解锁/运动设定值，进程清理无错误。
25项门槛/搜索非飞行回归通过。此结果支持当前环境的写端配置修正，但不分离
可靠性、持久性、队列深度三项各自贡献，不宣称全部间歇故障或启动迟到均已消除。
起飞仍必须通过原六端点与新鲜双ACK门槛；该配置下完整飞行尚待用户复验。

## 新增可选移动演示：rgb_search（2026-10-16）

用户完整飞行 `search_gui_qos_fix_20261016_193009` 的 `ff01ebcc` 已完成旧正对照，
但其所有平移请求/发布速度均为0：用户观察到主要是起飞、转向和降落是正确的。
旧 `rgb_scan` 保留为接线正对照，不把已有成功重命名为自主移动搜索。

新增 `--policy rgb_search`，当前版本 `rgb_search_approach/2`。它仍是规则基线，不加载
三种学习模型，不是世界模型规划。沿用同一个开阔场景，红球/蓝方块仍是静态物体，
没有搬运、物体运动、避障或遮挡探索；只旋转仍可能看到目标，平移是新增任务要求，
不能据此宣称已解决需要移动才能发现的未知空间搜索。

新流程：UAV0扫描公开RGB → 对准目标颜色框后前移 → 实测靠近约1.1m、图像框
增大后稳定悬停 → 发出found → UAV1从自己的相机独立扫描并靠近 → 稳定确认。
不把两机各自局部坐标当共享坐标，不向策略传入目标位置、私有标签或专家航线。
队友接收的是集中式策略内部的发现事件，不是已实现通信限制下的消息协议。
策略以局部里程计闭环悬停并维持约2m高度；最大局部净位移2.5m，超出即失败降落。
前进请求约0.44m/s；这不是精确距离测量，颜色框也仍不检查形状/条纹。

独立私有验收 `search_approach_acceptance/1` 在原身份/框/时戳验收之外，要求：

- UAV0报告found前，实测水平净位移至少0.8m，估计目标水平距离缩短至少0.7m。
- UAV1在found之后重新累计同样的位移/距离缩短，不能用之前的运动抵扣。
- UAV1至少3个新鲜、中心可见且水平速度不大于0.22m/s的观察跨越至少1仿真秒；
  缺失观察或相邻间隔超过0.8仿真秒重置稳定观察。

距离由私有场景出生位置、初始航向和接收时局部里程计校准估计，仅裁判使用，
不是曝光时刻精确真值。指标写入 `evaluator/summary.json` 的 `motion_metrics`，
校准参考写入同一私有文件的 `motion_calibration`（下述v2回归后新增日志字段）；
逐步公开阶段/XY/动作写入终端和 `policy/requests.jsonl`。相同场景哈希不代表验收
相同，新旧模式的成功率不能直接混比。普通ACK门槛、运行看门狗、间距限制和
正常降落全部保留；无真实硬件支持。

手动输入指令的可视化命令（使用全新输出目录）：

```bash
cd /mnt/d/AeroWAM
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash

/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.search_pilot \
  --px4-root /home/ryx/PX4-Autopilot \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --agent /mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent \
  --policy rgb_search --execute-flight --gui --duration 90 \
  --startup-timeout 150 \
  --output-dir "/mnt/d/AeroWAM-data/search_rgb_search_$(date +%Y%m%d_%H%M%S)"
```

仍按终端提示原文输入，例如“寻找红色的球，让另一架无人机确认同一个目标”，
再单独输入裁判答案1。蓝色方块选2；该有限颜色解析器不支持任意物体/否定指令。

新增反例与简化渲染运动闭环见 `search_motion_self_test.py`。包括原地转向、只有
动作请求没有位移、向远离目标方向移动、队友提前移动、未稳定观察/断帧均不能
通过，以及公开像素决定转向/前移、坐标轴转换、跨回合复位。简化运动学验证不是
PX4动力学/实飞证据，实际SITL状态另行记录。

首次新模式回归 `search_motion_smoke_red_v1_20261016` 在扫描阶段因
`px4_failsafe` 结束，已正常降落、无接触、自有进程清理完成。ULog显示持续收到
Offboard消息时，其时间戳由接近HRT变成落后约1.02s，触发
`offboard_control_signal_lost`，而非策略完成或电池低电量。
新模式因此显式选择 `latest_px4_odometry_roundtrip_no_extrapolation`：各机命令/
设定点复用自己最新PX4遥测已经同步过的时间戳，不用主机墙钟推算当前仿真HRT，
也不把旧数据的时戳凭空推进。接收时新鲜度检查与PX4信号过期检查均保留；此项
只在 `rgb_search` 启用，不偷偷改变旧模式。它不使状态成为曝光对齐状态，也不
保证所有DDS/时间同步故障已解决。失败输出保留，不计成功。

蓝色启动回合 `search_rgb_blue_clean_20261016_200639` 还出现过相机/时钟持续更新、
但四个 ROS PX4 状态/里程计计数保持0，最终不解锁并报
`Isolated startup streams not ready`。PX4 日志显示输出写端已创建；搜索端原先对
这些 live 输出请求 `TRANSIENT_LOCAL`，与易失遥测端点可能不匹配。现已改为
`BEST_EFFORT + VOLATILE + depth10`；控制写端的已验证 QoS 和安全门槛不变，旧
`rgb_scan` 语义不变。此前蓝色/红色重叠运行导致的 ACK 超时不作为识别证据。

修正后完整红球SITL回归 `search_motion_smoke_red_v2_20261016/condition_00`
（`717e477f18b94c9fab77c4f636c51b6b`）通过新验收：96次请求，step54 found，
step96 confirm；UAV0净位移1.123m/目标距离缩短1.122m，UAV1在found后净位移
1.098m/距离缩短1.094m，最后稳定观察2.736仿真秒。正常降落已验证，无目标接触，
无清理错误，自有进程全部停止；最大20Hz发布间隔0.053s。该次未开GUI，不声称
已核对新模式GUI观感；蓝目标仅有离线图像/简化运动学闭环，尚无新模式完整SITL
验证。55项搜索、安全门槛、语言验收、传输回归在Windows和ROS/WSL系统Python
均通过。一个成功回合不构成稳定性统计或正式组合迁移结论。

### v2：蓝色边缘候选过早报告（2026-10-16）

用户回合 `search_rgb_blue_qos_fix_20261016_201623` 已成功启动并飞到搜索阶段，
但step20的蓝方块仍被右边缘裁切。蓝色像素框止于x=252/256，白色条纹仍延伸到
边缘；旧检测器仅检查颜色像素是否触边，立即发出observe，被独立私有验收拒绝为
`ungrounded_target_report`，随后普通降落。不是完成任务，也不是这一次DDS断链。

v2只修改移动策略：候选颜色框继续驱动CENTER转向，但只有水平中心误差≤0.08，
且四边余量≥max(8像素, 颜色框最大边长的25%)才允许报告和前进。诊断新增
`report_ready`，区别于仅表示有颜色候选的`visible`。余量是公开RGB保守筛选，
不是完整可见性证明；原裁判/框IoU/新鲜度/实际靠近要求/安全降落不变。旧rgb_scan
及共享颜色检测器未改；结果按新策略版本单独记录。

Windows及ROS/WSL的73项回归通过（包含像素标注/栅格测试），新增白条纹触边但
颜色不触边的四边反例、居中后恢复报告/
前进，以及强行上报边缘框仍被原裁判拒绝。真实20帧离线回放：第20步继续转向、
不再发报告、无裁判失败，动作与原录像一致。红球93帧回放仍在52/93步found/confirm，
包括原移动验收；另一96帧红球回放在54/96步完成原图像验收。回放使用已录状态，
不是新的闭环飞行；蓝色v2完整SITL与GUI仍待用户单独复测。未另启飞行进程。

### GUI慢仿真与运行预算（2026-10-16）

下一用户回合 `search_rgb_blue_qos_fix_20261016_202903` 共105步，step62已发出并
通过found；队友发现后净位移1.301m、目标距离缩短1.294m，最后已进入STABILIZE。
但该回合没有任何confirm报告：最后3个稳定新帧仅跨越0.496仿真秒，尚不足策略
要求的1.2秒及私有验收要求的1秒。实际运行60秒先到期，于是普通降落，最终为
`incomplete_task_evidence`。首尾相机时戳仅推进23.232仿真秒，不能把GUI墙钟
时间当成稳定观察证据，也不把“队友靠近”视为已经确认。

`--duration`明确为RUN阶段的实际单调时钟预算，不含启动/起飞/降落，不是仿真秒。
移动演示rgb_search默认改为90秒，允许5..120秒；显式`--duration 60`仍严格在
60秒停止，不自动延长。其他模式保持默认30秒、5..60秒范围。建议本次单独蓝色
GUI复验使用上方90秒命令；预算耗尽仍不能记成功，旧失败输出不改动。不同预算
应分开记录，不能直接混比成功率。20Hz控制、新鲜度/失联/间距/围栏/电池保护，
以及实际移动与稳定确认要求均不改变。

私有summary新增run_budget、run_stop_reason、RUN实际/仿真耗时和evidence_progress；
终端降落前打印具体原因（如run_wall_timeout），策略公开诊断新增速度、稳定
新帧数与仿真停留时间。私有验收进度只写日志，绝不回传给策略。Windows及ROS/WSL
76项回归通过，包含慢仿真必须累计完整新帧时长、预算上下界和不伪造完成状态。
本轮只修改预算及诊断，不改策略动作或confirm阈值；更长预算下能否完成蓝色完整
飞行仍需单独SITL复验，不用已录制尾帧伪造未来成功。

### 90秒预算下蓝色完整SITL复验通过（2026-10-16）

用户明确授权独占复验、并暂停其他运行后，检查无残留搜索/PX4/Agent/Gazebo进程，
单独运行同一条蓝色指令，使用`--gui --duration 90`。新输出
`search_rgb_blue_verify90_20261016_204334/condition_00`，episode
`1fad91bf36714e77b7d3cc72917e5787`，最终`success (task_evidence_complete)`。
step63 UAV0 found，step115 UAV1 confirm；RUN耗时65.319实际秒、25.956仿真秒，
在任务证据完成时主动正常降落，并未用满90秒或自动延长。

私有验收：UAV0净位移1.119m、距离缩短1.113m；队友found后净位移1.163m、距离
缩短1.160m，稳定证据跨度2.904仿真秒。策略自身更严格低速段累计7帧/1.388仿真秒
才发confirm。两机正常落地/未解锁状态已验证，contacts和cleanup_errors均为空，
owned_processes_stopped=true；结束后只读进程检查无残留，最大发布间隔0.05452秒。
本次仅证明该静态蓝目标移动演示可完整通过，不是多次稳定性统计；GUI客户端已启动，
未另外核对窗口的视觉观感。既有60秒失败输出保留，验收标准未改。

## 命令

### JointBC接入第一步：真实录像离线TCP推理（2026-10-16）

规则红/蓝闭环通过后，新增`checkpoint_search_probe`，只读既有检查点及公开
requests.jsonl中的observation字段，以实际GroundedAdapter/PolicyClient TCP协议
进行400ms限时推理。逐帧响应、耗时与方法/权重/输入SHA256写入新目录；不读私有
标注，不启动PX4/Gazebo，不训练，不把录像状态当模型自己飞出来的状态。
超时/错误即停止，不重试同一模型历史、不继续并发调用；旧输出与检查点不覆盖。

`search_jointbc_probe_20261016_205027` 使用
`joint_bc_confirm_fix_v1_seed_0.pt`，SHA256
`d65ba56323d2a1e426e2c810bbaccf5e33f9a0a7c36201b1d37b87ab197f93af`。
红录像93帧、蓝录像115帧均完整推理通过；TCP耗时中位数20.68/20.00ms，最大
32.42/35.37ms。权重前后哈希一致。新增4项探针反例/隔离测试，Windows总计80项
相关回归通过，WSL新增4项通过。只证明本轮加载/推理/传输可用，不是闭环性能。

独立事后只读核对（不回传模型）：这些帧上共享辅助报告框未出现IoU<0.5或不可见
报告；但旧扫描报告器在红step16/蓝step23便发found，对应录像中的实际净位移仅
0.03094m/0.04059m，不能通过移动搜索至少0.8m的独立要求。confirm也是辅助报告器
产生，JointBC自身208帧意图全为none。不能据此宣称模型学习了搜索/确认，也不能
把旧checkpoint入口的扫描成功与rgb_search移动成功直接比较。

上述下一接入点现已实现，见下节。原扫描报告器及旧结果保留；不自动启动训练或
全量评测，不把辅助报告事件解释为学得的发现/确认能力。

### 独立移动证据入口与离线验证（2026-10-16）

`search_approach_reporter.ApproachReporter`版本`pilot_rgb_approach_reporter/1`仅发报告，
只用原文单一颜色、公开JPEG、本机位置/速度及图像时间戳。目标框需居中且有边缘
余量；首次合格图像后净移动至少1.1m、框宽增大至少1.12倍，随后低速、至少3个
连续新帧跨1.2仿真秒才发found。队友的位移基线和图像增长证据从found后重新累计。
它不能保证移动朝向目标，私有验收仍独立检查真实目标距离缩短、身份/IoU及稳定帧。
不调用规则导航替模型调整动作，旧模型若不会转向/靠近，就保留其未完成结果。

新`--policy checkpoint_search`选择上述报告器、相同私有移动验收及最新PX4遥测
时间戳往返；RUN默认90墙钟秒、允许5..120秒，不自动续时。旧`checkpoint`入口和
默认预算不变。模型服务的`--grounded-search-approach`与旧报告器选项互斥。

真实离线TCP输出：`search_jointbc_approach_probe_20261016_210158`。
红93帧、蓝115帧完整推理；耗时中位数20.64/20.22ms、最大31.62/25.47ms。
208帧动作与旧扫描报告器探针逐项完全一致，检查点SHA256前后不变。
辅助found/confirm分别在红step52/93、蓝step63/115，不再在几厘米位移时提前found。
事后独立读取已保存私有标注回放同一移动验收：红源位移1.126m、队友之后1.273m，
蓝1.119m/1.163m，均通过框/身份及移动检查。回放使用捕获时刻作now，不能核实
当时真实响应到达的新鲜度；录像轨迹由规则策略产生，不是JointBC闭环成功。
Windows与WSL/ROS各87项相关回归通过，包含报告器反例、真实TCP动作保留和私有
验收接线。未训练、未改权重、未覆盖既有结果。

### 冻结JointBC单轮SITL诊断（2026-10-16，未完成搜索）

用户明确授权独占一轮后，运行`search_jointbc_blue_closedloop_20261016_210353`，
使用其原句“找到蓝色方块，并让另一架无人机确认”，私有答案blue；旧权重只读，
`checkpoint_search`、GUI、90秒RUN上限。六控制端匹配和两机新鲜ACK通过，两机
正常进入Offboard并起飞。首个策略请求超过400ms，RUN约0.53墙钟秒后触发普通
降落，成功动作请求0、found/confirm均无。保存结果为
`model_failure/policy_inference_error`，`policy_error=PolicyTimeoutError:policy inference exceeded 400 ms`。
这是推理预算失败，尚不能评价该检查点的闭环搜索能力，更不能推断架构优劣。
离线推理延迟小不代表与Gazebo同跑时满足预算；根因未定位，不猜测为读取日志导致。
原有服务已经执行CUDA预热，不能未经验证直接归因于“缺少预热”。

两机最终未解锁已验证，`contacts=[]`、`cleanup_errors=[]`、
`owned_processes_stopped=true`，事后只读进程检查无残留；未自动重试、未放宽400ms、
未训练。检查点SHA256与离线探针一致。后续仍以部分可观测、队友行为变化下的
技能与预测组合迁移为主线，本静态场景仅是接线诊断。

复现（训练venv，只用本地缓存；输出目录须新建）：

```bash
cd /mnt/d/AeroWAM
export HF_HOME=/mnt/d/AeroWAM-data/cache/huggingface
export HF_HUB_OFFLINE=1
/home/ryx/venvs/aerowam-model/bin/python3 -B -u \
  -m benchmark.px4_gazebo.model_handoff.checkpoint_search_probe \
  --checkpoint /mnt/d/AeroWAM-data/checkpoints/joint_bc_confirm_fix_v1_seed_0.pt \
  --method joint_bc --device cuda --limit 120 \
  --requests \
    /mnt/d/AeroWAM-data/search_rgb_search_20261016_195247/condition_00/policy/requests.jsonl \
    /mnt/d/AeroWAM-data/search_rgb_blue_verify90_20261016_204334/condition_00/policy/requests.jsonl \
  --output-dir "/mnt/d/AeroWAM-data/search_jointbc_probe_$(date +%Y%m%d_%H%M%S)"
```

先交互输入并检查启动；此命令不会起飞。输出目录必须全新。

```bash
cd /mnt/d/AeroWAM
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash

/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.search_pilot \
  --px4-root /home/ryx/PX4-Autopilot \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --agent /mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent \
  --output-dir /mnt/d/AeroWAM-data/search_manual_user_startup_01
```

可选先检查命令往返：在上述启动命令中加入`--control-link-check`。它只对已经
未解锁的两机发送普通未解锁命令并核对回执，不起飞，不能与`--execute-flight`并用。
飞行命令本身已强制执行相同预检，无需同时加此选项。
`--control-link-timeout`默认25秒、允许5..60秒；不要用无限重试掩盖断链。

小规模飞行正对照命令如下。运行会让隔离SITL起飞/搜索/降落，仍由用户手动输入
指令。`--gui`与服务器使用同一partition；关闭GUI不会中止服务器，结束时清理本次
GUI进程。它展示的是新的搜索场景，并非原来的绕墙handoff场景。

```bash
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.search_pilot \
  --px4-root /home/ryx/PX4-Autopilot \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --agent /mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent \
  --policy rgb_scan --execute-flight --gui \
  --startup-timeout 150 \
  --output-dir /mnt/d/AeroWAM-data/search_manual_rgb_scan_03
```

正对照闭环通过后，可单独运行JointBC移动诊断；runner自动在独立端口启动/停止
模型服务。模型用训练venv，ROS/Gazebo用系统Python。旧权重只读，不与其他终端
并发启动仿真。默认仍手动输入原文和单独的私有答案。

```bash
export HF_HOME=/mnt/d/AeroWAM-data/cache/huggingface
export HF_HUB_OFFLINE=1

/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.search_pilot \
  --px4-root /home/ryx/PX4-Autopilot \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --agent /mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent \
  --policy checkpoint_search --method joint_bc \
  --checkpoint /mnt/d/AeroWAM-data/checkpoints/joint_bc_confirm_fix_v1_seed_0.pt \
  --policy-python /home/ryx/venvs/aerowam-model/bin/python3 \
  --device cuda --execute-flight --gui --duration 90 --startup-timeout 150 \
  --output-dir "/mnt/d/AeroWAM-data/search_manual_joint_bc_approach_$(date +%Y%m%d_%H%M%S)"
```

如需复现旧扫描报告器基线，显式选`--policy checkpoint`，其默认RUN仍为30秒。

实验按experimental-design的配对/分组原则处理：三方法保留相同人工指令、答案
与场景，并记录/平衡方法顺序。同场景条件是一组，不拆到训练/测试两边；指令数、
帧数不能当独立场景数。一个场景正对照成功只证明接线/可完成性，不证明组合迁移。
