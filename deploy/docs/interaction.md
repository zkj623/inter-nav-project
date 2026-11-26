# 真机推箱交互

现场可在一次普通点导航中查看认箱、推移和到站结果。[三站连续验收](../README.md#三站连续验收)

deploy 复用仿真的完整流程：在线探索/导航 → 比较绕行与推移 → 接近并对齐 → 推开箱体 → 退出接触 → 用新扫描确认通道 → 继续原目标并停稳。

已接入的非语义路径使用登记尺寸和实时 LiDAR 位姿，不需要 DINO、SAM、Qwen 或 RGB-D。真机通过 SportClient 执行机身速度；仿真的 loco 权重不在真机中加载。普通配置运行点导航。30 × 80 × 30 cm 箱体的首次短推使用标称机头尺寸，接触速度上限为 6 cm/s。完整推箱需要本机实测参数。

## 模块与数据流

| 模块 | 输入 | 输出 / 调用对象 |
|---|---|---|
| `robot/sensors.py`、`ExperimentFrame` | ROS 点云、odom | 本轮 experiment 坐标下的 Snapshot |
| `perception/box_fitting.py` | 当前扫描、登记长宽高 | 实测箱面、位姿、残差、边界证据 |
| `perception/registered_boxes.py` | Snapshot、登记区域 | 新鲜 BoxState、登记来源权限、截止时间 |
| `interaction/config.py` | `configs/interaction.yaml` | 校验后的 provider、SDK profile 和 writer 上限 |
| `app/tasks.py` | 同一 Snapshot 与 provider 输出 | 调用 `NavigationEngine`，管理开始/暂停/继续 |
| `app/navigation.py` | 实测地图、箱体、目标 | 原 `OnlineGoalSelector / SidePushPlanner / PushAsideSkill` 的决策和速度 |
| 共享 `push_aside_skill.py` / `push_aside_backend.py` | 规划、观测与显式注入的执行后端 | 唯一推箱状态机；默认后端保持仿真控制方式 |
| `interaction/profile.py` | 完整实测参数 | 仅供 deploy 使用的 SDK 参数；没有隐式默认标定值 |
| `interaction/skill_backend.py` | SDK 参数、登记许可、真实观测 | `create_push_skill()` 为共享技能注入 `SdkPushBackend` |
| `interaction/motion.py` | 实测运动响应与当前碰撞地图 | SDK 对齐、转向、退让原语及导航恢复适配；只返回速度请求 |
| `interaction/box_motion.py` | 登记尺寸、实测位姿、独立源扫描 | 箱体平移/旋转停稳依据，供通道证书核验 |
| `interaction/collision.py` | 未被箱体图层屏蔽的扫描证据 | 接触时的非目标障碍、未知区及箱体停止空间检查 |
| `interaction/execution.py` | 技能阶段、箱体有效期 | 不可续期的技能/接触截止时间及执行代次 |
| `app/controller.py` → `robot/motion.py` | 速度和交互约束 | 单个独立 writer 调用 SportClient，超时/接管锁止 |
| Session → view / record | 同一观测和阶段信息 | 地图、箱体诊断、请求/确认、停止与通道证据 |

在线由 `app/observations.py` 的独立线程持续采集、转换坐标并更新箱体跟踪；导航主线程只读取固定的观测/证据对。登记跟踪在观测线程内同步完成，回放使用同一拟合器，不再依赖导航轮询来推进身份关联。拟合本身过慢、真实断流或遮挡仍会触发原有过期与身份失效检查。

地图原始射线证据只计算一次，接触检查复用这些证据，不修改导航图来制造通路。耗时的准备位置和通道规划仍走停稳后规划流程；计算期间感知与记录继续运行，地图和技能始终由导航主线程管理。恢复前以新观测重新检查，不能直接发送阻塞计算前生成的速度。

共享模块不导入 deploy，也不判断 `is_sdk`。仿真直接构造 `PushAsideSkill` 时使用原有默认后端；deploy 在自己的工厂中选择 SDK 后端并绑定实测几何。后端不直接发送机器人命令，所有请求仍经过 controller 和独立 writer。接口边界与调用约定见 [interaction 模块说明](../interaction/README.md)。

## 1. 登记箱体并只观察

编辑 `configs/interaction.yaml`：

```yaml
source: registered_box_lidar
tracking_enabled: true
geometry_verified: false
registry_reference: "本轮尺寸测量记录的位置"
boxes:
  - id: box-1
    dimensions_m: [null, null, null]
    search_region: [null, null, null, null]
    push_allowed: true
tracking:
  max_age_s: 0.25
max_skill_seconds: 180.0
contact_profile: null
```

把所有 `null` 尺寸/区域换成实测数字后再启动；这个片段是填写模板，不能原样运行。

- `dimensions_m`：箱体完整长、宽、高，米；不是可见点云的 AABB 尺寸。当前支持每维 0.15–2 m 的直立刚性箱体。
- `search_region`：`[xmin,xmax,ymin,ymax]`，本轮 experiment 坐标，用于首次关联，不是预设箱体坐标或推移目的地。区域应覆盖实验中的重新关联位置，宽度不超过 10 m，且位于导航地图内。
- `push_allowed`：允许规划器选择这个已登记目标；不跳过实时几何或运动检查。
- `geometry_verified: false` 时只输出跟踪诊断，不向导航提供可交互箱体。先在这一状态核对尺寸、位姿、时间和遮挡。
- 开启 `--box-window` 时，用两段独立的点云确认箱位。MID-360 在机器人站稳时每段最多积累 2 秒。对 30 cm 高的箱子，认箱单独保留到离地 46 cm，可以用箱顶和侧面核对尺寸。区域内出现多个候选时，等待下一次能分清的观测。导航地图仍只用 28 cm 及以下的点。

Orin 终端 1：

```bash
source ~/unitree_ros2/setup.sh
cd ~/inter-nav-project/deploy
source ../.venv-deploy/bin/activate
python run.py --observe-only
```

终端 2：

```bash
cd ~/inter-nav-project/deploy
source ../.venv-deploy/bin/activate
python record.py --output ~/go2-interaction-records
```

view 的启动与 SSH 转发沿用 [README](../README.md)。它会显示箱体跟踪状态、残差、观测年龄和接触检查原因；`record` 保存完整诊断与原观测。未核验几何不会被画成可推障碍，但诊断栏仍可核对拟合结果。

## 2. 接近正面时如何继续跟踪

首次确认箱位后，机器人可能只看到箱体正面。此时允许连续单面观测，但必须同时满足：

- 本帧重新测得完整面宽、高度、法向和距离，符合登记尺寸；不是根据上一帧预测位置。
- 两条横向边缘外各有独立的当前射线，证明射线经过边缘外空域并到达箱体后方的背景或地面；不能以 ROI 截边、图像缺失或无返回当作自由空间。
- 位姿变化与仍新鲜的已确认身份连续，只有一个可行关联。

箱体厚度来自登记尺寸。首次确认后，地图保存同一个箱号和箱位。暂停、换目标和暂时遮挡都会保留箱位。接近时用记住的位置寻找箱面，接触指令要核对新扫描的箱面证据。

接触控制使用 `tracking.max_age_s`，默认从采集时刻起算 0.25 秒。需要在接触距离验证能否持续看到所需边界。

`--box-window` 的地图观察使用 `window_max_age_s`，省略时沿用 `tracking.max_age_s`。30 × 80 × 30 厘米箱体的观察配置为 0.35 秒。两个有效期分别按实测点的时间计算。地图上的框用于核对箱位，接触控制仍需当前扫描给出的几何证据。

## 3. 填写 SDK 接触参数

首次实推可直接使用 `configs/box_30x80x30.yaml` 和 `--box-window`。等地图出现 `box-1` 后按 `n` 输入目标。Go2 会沿选定的箱面接近，转正后用新的扫描核对箱面和推路。首次短推测到沿推向移动约 10 cm 就请求停车。停稳后再扫描，确认沿推向移动至少 2 cm，才对原目标重新找路。

首次短推的参数保存在 `go2_first_push.yaml`。机头尺寸来自宇树模型，速度、预留距离和等待时间是试推设置。记录中的 `first_push_result` 保存停止原因、箱体位移，以及能测到的停车后位移。完整推箱按下面的步骤填写本机实测参数。[宇树 Go2 机身模型](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf)

在 deploy 目录创建填写用配置：

```bash
cp -n configs/contact_profile.example.yaml configs/contact_profile.yaml
```

填完全部字段后，在 `interaction.yaml` 中设：

```yaml
contact_profile: contact_profile.yaml
```

路径相对 `interaction.yaml` 所在目录解析。模板没有可执行的默认数值；缺字段、`null`、非有限值、倒置的包络或超过导航/SDK 限速都会被拒绝。

| 字段组 | 含义 |
|---|---|
| `backend / calibration_reference` | 固定 `sdk`；记录本机测量/标定数据位置 |
| `nose_offset_m`、`foot_*`、`head_*` | base 坐标中的接触部位、足部和头部完整碰撞包络，米 |
| `max_*_mps / max_yaw_rps` | 前、后、侧向和转向上限 |
| `forward_mps / reverse_mps / lateral_mps` | 对齐与退出速度；`min_*` 是实测有效控制下限 |
| `push_start_mps / push_max_mps / push_acceleration_mps2` | 接触推移速度和加速度 |
| `linear_acceleration_mps2 / yaw_acceleration_rps2` | 全技能加速度约束，不超过已有导航与 writer 限额 |
| `*_stop_seconds / *_margin_*` | 实测停止响应及保守余量；推移与普通退让分别配置 |
| `settle_seconds / stall_seconds` | 实测停稳等待与无进展截止 |
| `box_position_tolerance_m / box_settle_seconds / box_max_settled_speed_mps` | 箱体观测抖动界限、停稳窗口及窗口内速度界限；需用静止和缓慢移动记录分别验证 |
| `max_contact_seconds / max_tilt_degrees` | 首次接触开始后的总预算及倾角上限；重试和规划暂停不续期 |

接触包络要包括实际装配件，测距基准必须一致。之前墙距偏差尚未解释，不能直接减一个固定常数后认定几何已正确。SDK 回执只说明请求收发情况，不是动作执行或箱体移动证明。

箱体停稳使用多个独立源扫描的连续窗口，检查完整箱体角点的偏移和速度，包含原地旋转；不要求带噪测量逐帧完全相同。三个 `box_*` 参数必须由静止抖动与缓慢移动实验确定，不能把分辨率以下的变化当成已测得静止。通道确认同时检查地图实际融合扫描的采集时间，不用计算完成时间冒充移动后的新扫描。

空载响应与接触标定的方法、最终完整验收口径见 [interaction_plan.md](interaction_plan.md)。

## 4. 启用完整交互导航

推箱实验使用以下配置。

| 文件 | 设置 |
|---|---|
| `configs/interaction.yaml` | `source: registered_box_lidar`、`tracking_enabled: true`、`geometry_verified: true`，登记箱体及完整 `contact_profile` |
| `configs/navigation.yaml` | `interaction_enabled: true` |

在 deploy 目录运行：

```bash
python run.py
```

地图与跟踪就绪后，按 `n` 输入导航坐标。按 `p` 暂停、`r` 继续、空格取消、`q` 退出。

规划器自动决定绕行或推移，并自行计算待推位置、方向和侧停目标。推移结束后要求箱体实测位移、箱体最终移动后的新扫描和整机通道证书；独立 writer 确认停稳后才恢复普通导航。运动中丢失目标、观测过期、接触无进展、预算到期或遥控接管均停止，不能靠重新发送旧命令自动恢复。

## 5. 回放和记录

对实际录制目录运行同一跟踪器，不连接 SDK：

```bash
python run.py --replay /实际记录目录 --output ../results/deploy/interaction-replay
```

带导航决策的回放可附加 `--task navigate --goal X Y`。回放只能分析记录中的观测与决策，不会模拟新指令引起的箱体运动，不能当作物理成功证据。回放使用当前配置；复现实验时应使用该次保存的登记尺寸、区域和 profile。

新记录的 `observations.jsonl` 来自独立采集通道，目标频率默认 10 Hz；`frames/` 是最多约 5 Hz 的显示证据，二者不要求一一对应。`run_summary.json` 的 `missing_observations` 和 `last_observation` 分别记录原始观测缺失及最终游标。`observation_stream_complete` 检查已发布的观测是否收齐，`recording_complete` 还检查显示与事件是否完整；这些字段不证明 ROS 上游从未丢帧。旧版本中因显示降频未保存的扫描无法补回，不能据此认定实时箱体身份曾经丢失。

推移完成后的回放也会等待停稳：记录必须包含完成后的新扫描，以及连续、新鲜的完整机身线速度和角速度。证据不足时以 `interaction_stop_handoff_required` 结束，不推进后续导航；通过时记录 `recorded_stop_handoff`，不会伪造 SDK 停稳回执。

| 记录字段 / 事件 | 用途 |
|---|---|
| `perception_result` | 身份、实测位姿、可见面、边缘射线、误差、登记来源及源时间 |
| `decision`、技能状态转换 | 为什么绕行/推移、准备与实际执行阶段 |
| `interaction_command` | 目标几何截止、固定技能预算、执行代次和速度限制 |
| `command_requested / command_accepted` | 请求与 writer 接收；均不替代实测运动 |
| `planning_hold_* / motion_state` | 停稳后重规划，以及技能完成后恢复导航的交接 |
| `recorded_stop_handoff` | 仅用于回放的录制停稳证据，未向机器人发送命令 |
| `task_result` 中的 `skill_reports` | 实测推移量、通道证书、profile 和成功/失败原因 |

原始 LiDAR/odom/IMU 问题仍按 [sensor_check.md](sensor_check.md) 录制 ROS bag。RGB-D 视觉模型接口保留为诊断/候选来源；其可见点 AABB 没有完整朝向几何，本实现不会把它伪装成可执行接触模型。
