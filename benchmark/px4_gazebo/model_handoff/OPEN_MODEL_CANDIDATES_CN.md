# 开源模型与算法候选池：双无人机协作与世界模型榜单

核查日期：2026-10-18。本文是选型建议，不是已接入或已测出成绩的模型清单。

主线始终是：**在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。**
任务不按机号预分配发现、确认、规划、执行或高低空角色。方法原有的固定分工可作为其属性披露，不能变成 benchmark 的评分特权。

## 如何使用这份候选池

下面列出 **42 个模型/算法候选条目**，其中包含模型变体、一个航空基线族及需要重训练的算法；不等于42个已具备无人机权重的可部署系统。不同规模、种子不应被包装为不同方法增加榜单数量。

“权重入口”仅表示官方说明或模型仓库提供了入口；本轮不下载权重、不验证二进制完整性、不确认下载许可、不运行这些外部模型。代码许可证与权重许可证分开核对；“公开可访问”也不自动等于符合严格开源定义。

W：官方提供权重入口；G：需要申请/访问控制；C：代码已见，完整任务权重尚未确认；T：算法实现，需在本任务另行训练。所有条目均尚未获得当前双无人机闭环准入。

优先级是本项目接入顺序建议，不是对模型能力的预判：P1先做适配可行性检查，P2扩充覆盖，P3作为专项或较重的复现方向。高相关性不等于低适配成本。

## 1. 无人机原生方法：4项

| # | 候选与官方来源 | 开放证据 | 与本项目的关系及主要差距 | 建议 |
|---|---|---|---|---|
| 1 | [OpenVLA-UAV / UAV-Flow](https://github.com/buaa-colalab/UAV-Flow)，[权重](https://huggingface.co/wangxiangyu0814/OpenVLA-UAV) | W，另有Flask推理服务 | 已有航空动作训练，比直接使用机械臂VLA贴近；仍需核对动作坐标、尺度、历史与双机协作 | P1 |
| 2 | [LLaMA-UAV / TravelUAV](https://github.com/buaa-colalab/TravelUAV)，[模型](https://huggingface.co/wangxiangyu0814/llama-uav-7b)、[轨迹补全模型](https://huggingface.co/wangxiangyu0814/traveluav-traj-model) | W | 航空语言导航/轨迹参考；完整方法含轨迹补全和原AirSim接口，不能只加载语言模型就声称复现 | P2 |
| 3 | [AerialVLN 的 CMA 基线](https://github.com/liukejia121/AirVLN) | C，README提供视觉编码器权重，任务完整权重需另核 | 经典航空VLN对照；是平台中的参评基线，不把平台名当一个通用预训练模型；输入及任务策略需迁移 | P2 |
| 4 | [AeroDuo / PilotLLM](https://github.com/buaa-colalab/AeroDuo)，[任务权重](https://huggingface.co/salome1023/AeroDuo-PilotLLM) | W | 双机航空直接相关，但原方法有高低空固定分工、BEV/深度/点云/规划依赖；当前RGB轨不能无损接入，已有源码审计见ROLE_FREE_MODELS_CN.md | P2，边界审计优先 |

**重点：UAV-Flow是项目/数据与基线仓库，实际榜单行应登记OpenVLA-UAV；不能把它再当一个不同模型重复计数。该仓库的Pi-0-UAV仍写Coming Soon，本轮不计入已开放模型。**

## 2. VLN、视觉导航与导航VLA：11项

| # | 候选与官方来源 | 开放证据 | 原生接口/用途及适配障碍 | 建议 |
|---|---|---|---|---|
| 5 | [NaVILA](https://github.com/AnjieCheng/NaVILA)，[8帧权重](https://huggingface.co/a8cheng/navila-llama3-8b-8f) | W | 历史视觉+语言到导航动作；腿足运动执行层需替换成公开声明的航空执行层，双机协作另接 | P1 |
| 6 | [InternVLA-N1](https://github.com/InternRobotics/InternNav) | C，本轮确认框架与模型文档，具体配置/权重准入另核 | 语言导航与连续轨迹双系统；需选择具体模型配置，审查深度等输入、两级时序和执行器 | P2 |
| 7 | [NaviLLM](https://github.com/zd11024/NaviLLM)，[权重](https://huggingface.co/zd11024/NaviLLM) | W | 泛化语言导航参考；原生候选视点/导航任务接口与当前有限前视RGB有差异 | P2 |
| 8 | [NavGPT-2](https://github.com/GengzeZhou/NavGPT-2)，[XL权重](https://huggingface.co/ZGZzz/NavGPT2-FlanT5-XL) | W，README在master分支 | 显式导航推理+策略；原Matterport视点候选和特征不能直接当PX4连续动作 | P2 |
| 9 | [ETPNav](https://github.com/MarSaKi/ETPNav)，[权重](https://modelscope.cn/models/admagic/ETPNav/files) | W | 在线拓扑规划+连续导航；原全景/深度和航点预测器需审查，同RGB轨不能额外送深度真值 | P2 |
| 10 | [DUET](https://github.com/cshizhe/VLN-DUET) | W，官方Dropbox入口 | 全局/局部拓扑决策；原生图节点与全景候选接口不等同本任务的局部机体动作 | P2 |
| 11 | [HAMT](https://github.com/cshizhe/VLN-HAMT) | W，官方Dropbox入口 | 长历史导航对照；原生全景历史，需适配有限视野采样和新动作空间 | P2 |
| 12 | [GNM](https://github.com/robodhruv/visualnav-transformer) | W，共享官方模型目录 | 图像目标导航；原生不是语言策略，也不是双机协作；可作技能对照，需合法目标来源 | P2 |
| 13 | [ViNT](https://github.com/robodhruv/visualnav-transformer) | W | 视觉导航航点；目标图像/拓扑记忆须来自允许的观测，不能由裁判给出目标答案 | P2 |
| 14 | [NoMaD](https://github.com/robodhruv/visualnav-transformer) | W | 目标遮蔽扩散探索；适合探索技能参考，不原生完成语言找物和双机确认 | P1技能对照 |
| 15 | [NavDP](https://github.com/InternRobotics/NavDP) | G，README链接表单 | 视觉扩散导航；最新模型下载需申请，部分评测路径为RGB-D，必须核对实际配置；代码标注CC-BY-NC-SA-4.0 | P2 |

单架导航器可复制为两实例作为“独立策略”对照，但不因此自动拥有协作能力。若外接协调器、语言选目标模块或路径规划器，完整系统名称与开销必须包括它们。

## 3. 通用VLA与预测-动作联合模型：13项

| # | 候选与官方来源 | 开放证据 | 为什么保留/需要什么适配 | 建议 |
|---|---|---|---|---|
| 16 | [OpenVLA](https://github.com/openvla/openvla)，[7B权重](https://huggingface.co/openvla/openvla-7b) | W | 代表性自回归VLA；机械臂7维动作及归一化统计不能截取4维直接飞 | P2 |
| 17 | [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | W | 连续动作块、多视图/状态适配参考；需航空动作头及动作块执行时序 | P2 |
| 18 | [Octo](https://github.com/octo-models/octo)，[Base 1.5](https://huggingface.co/rail-berkeley/octo-base-1.5) | W | 较轻、可适配新输入/动作空间；适合做同数据预算迁移对照，仍需航空训练 | P1 |
| 19 | [π0](https://github.com/Physical-Intelligence/openpi) | W，官方GCS基础权重 | 流匹配动作模型；动作语义、机体状态和两机输出头都需适配 | P2 |
| 20 | [π0-FAST](https://github.com/Physical-Intelligence/openpi) | W | 自回归动作压缩，与π0形成不同动作表示对照；不能按名字推定实机推理已满足400ms | P2 |
| 21 | [π0.5](https://github.com/Physical-Intelligence/openpi) | W | 开放环境泛化参考；当前公开实现主要支持其流匹配头，仍是操纵域预训练 | P2 |
| 22 | [SmolVLA](https://github.com/huggingface/lerobot)，[基础权重](https://huggingface.co/lerobot/smolvla_base) | W，模型卡Apache-2.0 | 较轻量，适合有限资源适配；不能把消费级部署说明当本机双机时延保证 | P1 |
| 23 | [GR00T N1.5](https://huggingface.co/nvidia/GR00T-N1.5-3B)，[代码族](https://github.com/NVIDIA/Isaac-GR00T) | W | 跨机体VLA参考；两架UAV是新机体配置。当前main已演进到N1.7，评测N1.5必须匹配旧代码版本 | P2 |
| 24 | [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer)，[权重](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) | W | 统一状态/动作空间、动作块扩散；需航空量纲映射和训练，多视图不等于多智能体 | P2 |
| 25 | [CogACT](https://github.com/microsoft/CogACT)，[权重](https://huggingface.co/CogACT) | W | 语言认知+扩散动作；原生操纵7DoF，不能原样当无人机位移 | P2 |
| 26 | [SpatialVLA](https://github.com/spatialvla/SpatialVLA) | W | 空间动作表示对照；原生末端空间与航空动作差别需适配，公开RGB处理与额外几何输入须区分 | P2 |
| 27 | [RoboVLMs 的固定检查点](https://github.com/Robot-VLAs/RoboVLMs)，[模型库](https://huggingface.co/robovlms/RoboVLMs) | W | 适合比较VLM骨干/动作头；它也是框架，正式行须指定一个实际检查点，不能只登记框架名 | P2 |
| 28 | [GR-1](https://github.com/bytedance/GR-1) | W，官方CALVIN权重 | 视频预测与动作联合建模，适合WAM相关对照；不等于已提供任意候选动作条件世界模拟器 | P2 |

这些模型原生多为机械臂/人形机器人。双臂、双视图或两个动作头不等于双智能体协作。若要适配，应明确使用统一合法训练数据和允许的预算，分开记录冻结迁移与任务适配结果；不将新动作头训练后的系统写成原模型零样本。

## 4. WM、动作条件预测与模型式规划：10项

| # | 候选与官方来源 | 开放证据 | 适合哪类比较/关键限制 | 建议 |
|---|---|---|---|---|
| 29 | [NWM / Navigation World Models](https://github.com/facebookresearch/nwm)，[权重](https://huggingface.co/facebook/nwm) | G，模型卡CC-BY-4.0，访问为manual gated | 导航动作条件未来预测及轨迹规划，语义最贴近导航WM；原生地面动作和单视图域需迁移 | P1，先核访问 |
| 30 | [DINO-WM](https://github.com/gaoyuezhou/dino_wm) | W，PointMaze/PushT/Wall检查点 | 预训练视觉特征上的动作条件动力学+规划，适合作为latent WM基线；不是通用语言或双机预训练控制器 | P1 |
| 31 | [V-JEPA 2-AC](https://github.com/facebookresearch/vjepa2) | W，README另列动作条件检查点 | latent动作条件预测+图像目标规划；必须选AC而非仅V-JEPA视频编码器，机械臂动作需航空适配 | P1后续 |
| 32 | [iVideoGPT](https://github.com/thuml/iVideoGPT)，[模型库](https://huggingface.co/collections/thuml/ivideogpt-674c59cae32231024d82d6c5) | W | 视频token WM；OXE公开权重为动作无关/目标条件，候选动作预测需选BAIR/RoboNet/VP2动作条件版本并适配 | P2 |
| 33 | [DreamerV3](https://github.com/danijar/dreamerv3) | T | 在潜空间想象中学习策略，适合任务训练型WM基线；不是下载即用的双机预训练策略 | P2 |
| 34 | [TD-MPC2](https://github.com/nicklashansen/tdmpc2) | W/T，原任务检查点公开 | 潜动力学与MPC；现有状态域/像素域权重不可混用，需要本任务动作、观测与任务条件适配 | P2 |
| 35 | [AVDC](https://github.com/flow-diffusion/AVDC) | W，Meta-World/iTHOR/Bridge入口 | 从视频预测经稠密对应求动作；非原生候选动作条件WM，转换和执行器作为完整方法披露 | P3 |
| 36 | [COMBO](https://github.com/UMass-Embodied-AGI/COMBO) | C，master代码可读，完整权重与外部模型依赖待核 | 部分可观测、多智能体、组合世界模型与树搜索，与主线高度相关；TDW任务到UAV迁移成本高 | P2研究对照 |
| 37 | [IRIS](https://github.com/eloialonso/iris)，[权重](https://huggingface.co/eloialonso/iris) | W | 自回归视觉token WM；原Atari离散动作/奖励域，航空迁移需重训练 | P3方法对照 |
| 38 | [DIAMOND](https://github.com/eloialonso/diamond)，[权重](https://huggingface.co/eloialonso/diamond) | W | 扩散像素WM；原Atari权重不是无人机物理模型，重训练及规划成本较高 | P3方法对照 |

WM没有原生语言输入时，可以参加声明范围内的预测组件比较；不能隐式从私有目标真值构造goal latent。完整任务系统需要合法的语言/目标/协作与选择组件。WM图像预测好看，不直接代表候选排序正确或协作成功。

## 5. 必要的多智能体学习对照：4项

| # | 算法与官方来源 | 开放证据 | 在本项目的作用/限制 | 建议 |
|---|---|---|---|---|
| 39 | [MAPPO](https://github.com/marlbenchmark/on-policy) | T | 无显式世界模型的协作学习基线；需增加同权限RGB/历史/语言编码，核对训练critic是否用特权信息 | P1后续训练基线 |
| 40 | [IPPO](https://github.com/marlbenchmark/on-policy) | T，代码中核验具体配置后登记 | 独立学习对照，检查集中训练的收益；不能复用MAPPO名字而不改训练配置 | P2 |
| 41 | [HAPPO](https://github.com/PKU-MARL/HARL) | T | 非共享参数等异构多智能体对照；不同参数不要求永久固定机号角色 | P2 |
| 42 | [QMIX](https://github.com/oxwhirl/pymarl) | T | 离散协作技能选择对照；适合共用离散候选技能集，不宜强行对比原生连续航点控制 | P3候选决策对照 |

这些是算法实现，不是已有本任务权重，不能与开箱冻结迁移混成一个训练设定。训练期特权critic、任务奖励、训练交互量应明确披露；推理期不得读私有目标/未来真值。

## 建议先推进的名单

先保留3个已有内置方法：JointBC、DualVLN-Hier、DualWAM-MPC。它们是仓库自定义实现，不能冒充同名外部论文官方模型。

首批外部候选建议从 **OpenVLA-UAV、NaVILA、SmolVLA、Octo、NWM、DINO-WM** 六项开始做准入审计。前三类覆盖航空动作、语言导航和轻量VLA，后两项覆盖导航/特征世界模型；Octo作为可换动作空间的较轻迁移基线。该名单不是六项都能立即开飞，NWM还受权重访问约束。

随后增加 **LLaMA-UAV、V-JEPA 2-AC、MAPPO**，形成3个内置+9个外部共12个初期方法候选。MAPPO需要另行训练，不能当一个下载后即测的冻结模型。COMBO和AeroDuo同时做研究/信息边界对照，待核完依赖后再决定具体实施范围。

不建议先用IRIS/DIAMOND撑大榜单：它们有价值，但航空域重训练成本与主线首轮收益不匹配。GNM/ViNT/NoMaD更适合作为导航技能参考，不单独声称完成语言协作。

## 榜单组织建议

保留三个互相关联的结果入口：

1. **协作闭环**：完整系统完成search/handoff/inspection及新组合，报告成功、安全、耗时、资源与双方实际贡献。不预分配角色，不以两个模型输出头代替角色适应证据。
2. **候选条件预测**：相同公开历史和候选动作/时长，比较可见性、成功/风险预测及校准；latent/视频误差作补充。只有能对不同候选输出有意义不同预测的方法才参加该项。
3. **候选决策**：相同候选与执行器比较ranking/regret和选后闭环收益。没有预测头的策略可有闭环成绩，预测列填N/A；编码器也不能因有特征输出就算完整WM。

VLN/VLA/WAM/WM是方法标签，不是互斥赛道。GR-1等可同时含预测和动作，但要根据实际开放接口核验能力。

主RGB轨统一当前有限视野、局部状态、历史和语言权限。RGB-D/BEV/全景/特权地图等需求须单列输入轨；推断深度模块也要列入方法、算力和时延，而非悄悄当作免费观测。图像目标只可从合法数据/历史得到，不能由私有评分目标合成答案。

区分冻结迁移与使用统一训练划分的任务适配，登记适配数据、参数、步数、交互量、骨干预训练与外部模块；测试失败不能回流训练或调提示。固定角色是参评方法自身属性，不是benchmark答案。

当前400ms闭环预算与安全门不变。大模型先测实际端到端时延，未通过就不进入该实时榜。未来若另设慢速高层规划轨，需公开固定决策节奏、执行层和预算，并让同轨全部方法遵守；不得临时给某个模型放宽时限。

榜单行应是“模型+适配器+规划/协调器+训练设定+版本”的完整参评系统。正式排位只填真实测得结果；尚未接入、权重待申请、准入失败、运行异常、任务失败要分开记录，不能把待测填成0%或筛掉真实模型失败。

## 开放状态中需要单独处理的事实

- OpenVLA-UAV代码仓库标Apache-2.0，但2026-10-18权重API的`cardData.license`为空；需核模型文件/基座条款，不能将代码许可自动套给权重。
- NWM权重API：`gated=manual`，模型卡`cc-by-4.0`；需要用户按发布方流程取得访问，不自动代接受条款。
- SmolVLA模型卡：`apache-2.0`，`gated=false`。NaVILA权重API无license字段，不据此认定没有许可证或默认为MIT。
- GR00T代码与模型权重条款分开，且main随版本演进；本目录中的N1.5必须与固定版本代码匹配。
- iVideoGPT需要指定动作条件检查点；V-JEPA需要指定2-AC；不能仅凭“world model”标题判断已能评价候选动作。
- COMBO代码README已读，但尚未闭合完整权重下载路径；OpenFly本轮检出的`Eziotic/OpenFly`仅为占位README，未核出可复现完整模型，暂列待核而非上述42项。它的论文[2502.18041](https://arxiv.org/abs/2502.18041)存在，不代表此占位仓库就是正式实现。
- RT-2、Genie 2、UniSim等不能仅因论文/演示公开就记为已具备可下载完整权重；本轮不列作可立即接入的开源模型。

## 来源与可重复核查

每行官方仓库/模型链接是本轮选型依据；共读取38个候选来源仓库README（包含暂缓的OpenFly占位仓库），另外核对TravelUAV子模型说明、NaVILA/SmolVLA等模型卡与API。没有进行权重下载或运行验证。

论文检索使用arXiv官方API，查询：
`ti:AerialVLN OR ti:OpenFly OR ti:UAV-Flow OR ti:TravelUAV OR ti:UniNaV OR (ti:COMBO AND ti:world)`，`start=0,max_results=15`。
它是定向检索而非穷尽综述；标题搜索有无关UAV flow结果，已人工剔除。COMBO核为[2404.10775](https://arxiv.org/abs/2404.10775)，AerialVLN核为[2308.06735](https://arxiv.org/abs/2308.06735)，TravelUAV论文由官方README定位[2410.07087](https://arxiv.org/abs/2410.07087)。GitHub搜索API一度限流，后续依据官方链接直接核README，不将搜索缺项解释为项目不存在。

本机原始README与读取URL/日期位于：
`C:/Users/R/.codex/visualizations/2026/10/18/01a14f57-a476-7bb3-a96a-0c52d6eea5f6/model_catalog_sources/`。
该目录不进入模型输入，也未随代码上传。版本固定、权重许可证/哈希、完整依赖、模型实际消费字段、动作尺度与预算验证，应在选定具体条目后逐项完成；这些工作不能以本目录替代。
