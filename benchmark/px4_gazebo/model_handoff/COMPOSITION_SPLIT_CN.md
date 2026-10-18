# 组合留出清单审计（离线草案）

主线：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。

`composition_split.py` 在采集/训练前检查声明的划分和已有训练暴露。它不生成新任务、
不调用模型或仿真、不自动分组/随机分配，也不把旧固定交接数据重新命名成组合测试。
当前只实现元数据一致性审计；正式场景、预算、分支匹配和测试划分尚未冻结。

## 检查什么

同一 scene_family_id、parent_episode_id、decision_id 的所有候选和重复分支必须在同一
train/validation/test 分区。决策点的父episode、场景族和条件须一致；一个父episode不能
被登记到不同场景族。重复的 decision/candidate/repeat 拒绝，缺少样本清单的条件不能
提供训练因素覆盖。计数分别列场景族、父episode、决策点和候选重复分支，独立样本数
始终为null：分支、帧或数值重复不能自动当作独立实验重复。

因素固定为 task（search/handoff/inspection）、target_motion（static/dynamic）、
observability（finite_fov/occluded_history）、teammate_behavior（显式登记的条件ID）。
`composition_axes` 必须在检查前明确，至少两个不同因素；该字段定义“哪个组合未见”。
例如留出 task×target_motion 中的 handoff+dynamic，就不能因训练时使用不同遮挡或
另一队友条件，仍把它声称为未见；如果主张的是四因素组合，须显式登记四个轴，
不能把通过四因素留出审计写成二因素组合也未见。更改组合轴属于新设计版本，须在
新采集及模型测试前声明；不能为消除既有泄漏而事后改轴并覆盖原声明。

所有训练前暴露、微调及验证调参条件均要登记到exposures；其中预训练/微调计入训练
支持及暴露，validation_tuning禁止看到留出的测试组合。训练因素支持不能只来自验证集。
`training_inventory_complete=false` 会拒绝清洁留出声明；设为true只是作者声明，
工具不认证上游通用预训练、检查点训练历史或来源清单完整性。未知来源须如实保留不完整，
不能把空exposures当作未暴露的证据。

| 测试claim | 组合要求 | structure_id要求 |
| --- | --- | --- |
| seen_combination | 完整四因素组合在训练中出现 | 训练中出现 |
| heldout_combination | 声明轴上的组合未见于训练和验证调参，所有单项因素已在训练出现 | 训练中出现 |
| new_structure | 完整四因素组合在训练中出现 | 训练及验证调参均未见 |
| heldout_combination_new_structure | 留出组合的要求 | 训练及验证调参均未见 |

这样可分别报告组合新颖性和遮挡结构变化，不把二者一次变化的结果直接归因为组合迁移。
每个条件不必占用不同structure_id；同一结构下的独立场景族仍须使用真实、稳定的族ID。
结构ID表示遮挡拓扑/生成结构身份，场景族ID表示同源场景/变体族；改随机seed、相机编号、
物体颜色或文件名不能自动创造新的结构/场景族。工具只比较ID，不识别这些语义或内容别名。

队友条件须登记kind和source_ref。被样本/暴露使用的command_transform或unknown条件不通过本审计，
避免把已有请求减速/延后代理标为独立队友策略变化。independent_policy也只是声明，
仍需审计实际策略实现、配置与运行轨迹。清单不指定哪架固定负责发现、确认或某个高度，
role_mode必须为unassigned；这不证明模型实际形成了动态分工。

## 输入和运行

清单schema为 `aerowam-composition-split/1-draft`，包含以下精确字段；完整可执行合成
例子见 `composition_split_self_test.fixture()`，不能将其当作正式分配方案或真实数据。

- design_id、source_kind（synthetic_fixture/declared_inventory）、role_mode、
  training_inventory_complete、composition_axes。
- teammate_conditions：behavior_id、kind（independent_policy/command_transform/unknown）、source_ref。
- cells：cell_id、split、claim、四因素factors、structure_id。训练/验证claim均为development。
- samples：sample_id、cell_id、scene_family_id、parent_episode_id、decision_id、candidate_id、repeat_id。
- exposures：exposure_id、usage（pretraining/finetuning/validation_tuning）、factors、structure_id、source_ref。

每条sample代表计划/登记的一份候选重复分支，不是视频帧。分支是否跑齐、是否达到匹配起点
及其标签是否有效由独立采集和candidate_scoring继续检查，不由此处认证。

```bash
python3 -B -m benchmark.px4_gazebo.model_handoff.composition_split \
  --inventory /absolute/private/inventory.json \
  --output /absolute/private/new_split_audit.json

# 可选：仅生成明确标注的合成正确/泄漏示例，新目录必须不存在。
python3 -B -m benchmark.px4_gazebo.model_handoff.composition_split_self_test \
  --write-synthetic-demo /absolute/private/new_synthetic_demo
```

一致时退出0，报告status=declared_split_consistent；声明泄漏/不完整时退出2并保存全部
issues；格式、重复JSON键或未知字段错误不生成报告。输出独占创建，不覆盖既有报告。
报告绑定清单原始字节和规范JSON的SHA256，前者包含换行差异；不加载source_ref文件，
不哈希实际训练数据/权重。清单含私有划分、条件和训练来源，不进入模型输入，不自动发布。

## 验证范围及后续门槛

新增25项合成回归，加上适配器运行、清单、候选评分及角色自由/grounded传输回归，
Windows/WSL各117项通过。覆盖父组跨分区、旧动态交接预训练泄漏、验证调参暴露、
未见单项因素、结构新颖性、队友代理、缺失清单、重复键、CLI拒绝和不可覆盖。
合成例子按场景族与父episode分组，避免伪重复；不据不完全交叉条件估计因素因果作用。

另将同一份正确/训练泄漏合成清单送入两端正式CLI，报告分别为declared_split_consistent
和rejected（两个留出条件被旧预训练暴露）；两端报告逐字节一致。私有复核目录：
`D:/AeroWAM-data/composition_split_synthetic_20261018_v1/`。报告SHA256分别为
`f163d904c38daec77d94fbe33bfe02d8c2ae624981d9e670a3f6c3688071c168`、
`ecfca651d0ec184c8bdc637c7565f7750245724cee12505953f63ca566f9b47e`。
上述均为合成元数据，未读取真实训练数据、调用模型、生成分支真值或修改旧实验。

审计通过不代表真实部分可观测/历史依赖、队友策略变化、来源完整、候选起点匹配或迁移
能力通过；报告相关verified字段及formal_split_frozen/flight_ready均为false。
当前没有通过此工具认证任何真实训练/测试清单，也没有获得新模型成绩。

正式执行前还须完成真实因素与角色自由技能实现、匹配分支pilot和收益可区分检查，
之后才冻结条件、候选/执行器、统一预算与分组划分。候选和方法运行顺序应在配对区组内
预先随机化并保存seed/顺序；本工具尚不生成或验证运行调度。独立重复数要在pilot后根据
分支复现与方差确定，不能凭合成例子的2次重复声称统计功效。保持现有10ms时钟与400ms
客户端预算，不绕过当前真实环境拒绝，不扩大采集、训练或飞行。
