# WSL 校时隔离与已验证范围

主线仍是部分可观测、队友行为变化下的双无人机技能和世界模型组合迁移。
本入口仅解决当前 WSL 实验会话的计时干扰，不代表模型任务成功。

2026-10-18 已直接追踪到 VM 初始 PID 命名空间中的 `/usr/sbin/chronyd`
调用 `clock_adjtime(ADJ_FREQUENCY|ADJ_TICK)`，tick 曾降到9167。
仅停 Ubuntu 的 systemd-timesyncd 不够；该服务已在失败后恢复。
没有发现可验证的 `.wslconfig` 关闭此根层同步设置；时区选项不等于关闭校时。

采用会话隔离：核对初始命名空间、chronyd 可执行路径和 PID/starttime，暂时
SIGSTOP 根层PPID=1的进程；临时停止 Ubuntu 校时，清掉残留 offset/frequency/tick 修正。
不设置 UTC epoch 或同步状态位，不修改 Windows 服务或持久 WSL 配置。
会话结束恢复 Ubuntu 服务及根层进程，另有600秒独立恢复保护。
正常 WSL 使用继续校时；直接执行旧仿真命令仍可能复发，应从隔离入口启动。
不要同时运行其他需要精确 UTC 校时的 WSL 任务，或并发运行同一组 SITL。

## 实测证据

- `AeroWAM-data/wsl_dual_clock_repair_20261018_v3`：固定60秒、1197个样本，
  原10ms门限通过，frequency始终0、tick始终10000；两层恢复确认。
- `AeroWAM-data/wsl_clock_repaired_jointbc_20261018_v2`：20秒预检通过；
  模型初始化至降落及清理连续219.174940秒、4367个样本，全程时钟通过，
  最大采样间隔55.394ms，RAW净偏移区间[-0.000300,0.000452]ms。
  这只是相对时钟质量，不证明相机曝光对齐或与Windows绝对UTC一致。
- 冻结JointBC、`--role-mode unassigned`、蓝色目标实际双机起飞、完成22次决策。
  随后 `RGB search local travel budget exceeded`，归为model_failure，正常降落，
  landing_verified及进程清理通过，权重SHA256未变。没有完成found/confirm，
  不计任务成功或组合迁移成绩。没有放宽10ms/400ms门限，没有训练。
- 中间环境加载失败和隔离脚本失败记录均保留，未用成功窗口覆盖失败。
- 新入口另经 `AeroWAM-data/wsl_clock_entry_validation_20261018_v1` 完整执行：
  自动提权、60秒时钟门控、双层恢复均通过，退出0，没有起飞。两种模式生成的
  7个脚本已核对实测版本；新增根层PPID检查及未触碰服务时不启动服务的保护。

## Windows 入口

在当前Windows部署的仓库根目录运行（Python可用现有Anaconda）。默认只生成
可检查脚本，不更改系统；添加 `--execute` 才请求UAC并执行一次。每次输出目录必须新建。

```powershell
python -m benchmark.px4_gazebo.model_handoff.wsl_clock_session --output-dir D:/AeroWAM-data/wsl_clock_check_NEW --mode probe --execute
```

probe只测60秒，不起飞。确需一轮冻结模型诊断时将mode改为pilot，另换输出目录。
`--method`可选joint_bc（默认）、dual_vln_hier或dual_wam_mpc；只允许已核验的
三个confirm_fix_v1_seed_0权重，准备前和运行前均核对冻结SHA256。
`--policy-seed 0`默认显式固定torch/规划随机种子，不改变权重；旧直接运行入口仍
保持未指定种子时的兼容行为。不同seed或模型不合并为同一重复实验。
pilot使用已核验的当前本机路径、Ubuntu-22.04、ryx、ROS Humble及px4_ros2_ws；
不用于别的部署。角色自由接线已在75fdf76提交；旧仓库缺少role-mode时
在生成脚本前拒绝，probe不依赖模型权重。
90秒RUN预算、420秒主监督上限及普通降落清理逻辑保留。
该入口没有开启Gazebo GUI，不预先分配发现/确认角色；事件仍来自公开辅助报告器，
不能宣称模型已学会动态角色分工。旧固定UAV1队友扰动未迁移到此模式。

生成脚本来自与实测一致的模板，路径有字符检查、Python语法检查和文件摘要。
`prepared.json`只表示已生成；`launch.json`表示请求启动；最终以
`admin_result.json`、`ubuntu_result.json`及pilot任务summary分别判断恢复、时钟、任务。
退出0表示probe通过或pilot任务成功；2表示隔离/恢复/时钟失败；3表示时钟通过但
pilot任务未成功。若进程意外退出，先核对恢复日志，不能把没有结果当作通过。

这是一项已实测的临时运行办法，不是永久修复微软WSL内置校时，也没有验证休眠、
宿主大幅改时或其他负载组合。无需迁移原生Linux才能继续当前诊断；原生Linux仍可
作为后续独立平台验证选项。

## 2026-10-19 新边界

DualVLN角色自由蓝色诊断已实际飞行14次决策；120.620秒会话全程10ms门通过，
搜索距离耗尽而失败，正常落地恢复。下一轮DualWAM在Ubuntu清理校时修正后
读回offset=1908us、frequency=0、tick=10000、status=0，被原有零残留前置检查拒绝；
没有开始预检或模型，不算模型任务失败，也不能说这次已测出10ms门失败。
两层恢复确认。未自动重试、未改变拒绝条件或系统状态位；报告改为保留具体读数，
并区分“未评估”与“失败”。临时隔离办法仍有此未解决边界，不能当作永久稳定修复。
完整原始目录、像素/轨迹核对和协议见[FROZEN_SEARCH_DIAGNOSTIC_CN.md](FROZEN_SEARCH_DIAGNOSTIC_CN.md)。
