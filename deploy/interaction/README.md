# 推箱执行适配

这个目录负责首次短推、完整推箱的 SDK 适配，以及接触和停稳检查。首次短推测量箱子移动了多远。完整推箱还要确认机器人能从腾出的通道通过。

## 调用关系

```text
LiveObservations：采集 Snapshot → RegisteredBoxPerception → 固定观测/证据对
  ↓
Tasks → NavigationEngine
          ├─ RegisteredFacePlanner → FirstPushSkill（首次短推）
          └─ OnlineGoalSelector / SidePushPlanner → PushAsideSkill（完整推箱）

create_push_skill() 按接触配置选择技能，两者都使用 SdkPushBackend。

SkillCommand → NavigationEngine → NavigationController
             → MotionProcess / ContactGuard → SportClient.Move
```

后端返回动作意图和测量依据，不连接 SDK、不发送命令、不修改机器人或箱体位姿。独立 `MotionProcess` 仍是唯一运动写入者。

## 模块职责

| 文件 | 职责与调用边界 |
|---|---|
| `config.py` | 配置检查、provider 工厂、writer 限额；live/replay 共用 |
| `first_push.py` | 根据登记箱体朝向选择可达的箱面，检查短推和刹停空间，停后复测位移 |
| `profile.py` | 全字段显式填写的 SDK 参数；统一平移/转向停止响应计算，无仿真参数继承或默认标定 |
| `skill_backend.py` | 工厂组装共享技能，SDK 后端提供权限校验、运动原语、执行检查和通道观测证据 |
| `motion.py` | SDK 对齐/转向/导航恢复原语；`bind(planner)` 仅设置该 planner 的运动后端、足部几何和接触余量 |
| `box_motion.py` | 基于源扫描的箱体角点运动与停稳窗口，不推进技能阶段 |
| `collision.py` | 未经目标图层擦除的实测点云检查；不向地图伪造自由空间 |
| `execution.py` | 技能执行身份、分段交接、绝对预算和观测期限 |
| `replay.py` | 以录制停稳反馈完成技能→导航交接；没有 SDK 回执 |

## 共享层边界

`FirstPushSkill` 继承 `PushAsideSkill`，复用对齐和接触动作，另外处理短推的停止和测量。真机技能工厂为两者注入 `SdkPushBackend`。仿真直接使用共享技能时，使用默认的 `PushAsideBackend`。真机各模块及箱位记忆的关系见[项目结构](../../ARCHITECTURE.md#真机从观测到运动)。

后端接口按职责分为：

- **控制原语**：`yaw_servo`、`heading_trim`、`new_turn/update_turn`、`translate`；共享技能决定何时调用。
- **运动约束**：`before_step/after_step`、停止响应、退让参数和无进展处理；后端只能限制当前执行，不能跳过通道阶段。
- **观测证据**：`observe_box`、`passage_evidence`、`verify_passage`；SDK 使用真实源时间及停稳窗口，并调用共享通道几何证明。
- **诊断**：`feedback/report`；SDK 参数和观测依据由 deploy 添加，默认仿真报告保持原字段。

通用导航查询允许 planner 显式绑定 `body_geometry` 和 `motion_backend`。前者是纯足部几何，后者提供接触命令检查、缓冲区恢复、局部动作限制和拐角动作选择。未绑定时原有仿真分支照常执行。`go2_heading_trim.py`、`layered_navigation_map.py` 保持原实现；在线地图默认自身过滤仍为 `0.44 m`，不会因 deploy 几何或导航半宽配置改变。

所有适配均绑定到实例；不修改模块全局变量，不替换全局函数，不复制技能状态机。机器人尺寸、现场接受开关、录制输入与命令有效期均继续由 deploy 管理。

SDK 后端显式实现全部执行接口，新增共享默认行为必须单独完成真机适配。隔离回归会检查这一约定，以及仿真独立导入和不同 planner 实例之间的几何隔离。

操作与现场标定见 [真机推箱交互](../docs/interaction.md)。
