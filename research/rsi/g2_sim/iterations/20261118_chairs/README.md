# 2026-11-18：剩余两把椅子的针对性迭代

本轮保留全部碰撞、姿态、接触表面和抬升验收门槛。没有为获得成功而更改物体碰撞模型或添加附着。

## 改动

- 座面托底从中部扩展到前后角：x=±12/18厘米，分别检查窄开度和全开；逐个检查真实手指和腕部几何。
- 站位增加前后距离与左右偏移，并对餐椅两组通过手部检查的候选增加0/±15/±30/±45度接近角度。
- 改抓结构件：餐椅金属腿搜索高度扩展到总高度25%–50%；办公椅轮脚区域10%–32%。保留手臂与物体的精确几何检查，用全身几何检查替代保守的躯干X平面启发式。
- 区分接近/抬升阶段逆解失败，避免把几何候选、轨迹规划和物理抓取混为一个成功指标。
- 餐椅角部方案216次尝试都在向外退9厘米的预接近姿态失败；随后改为向外2厘米、向下5厘米进手，并单独测试允许躯干配合预接近的版本，保持20度躯干包络。
- 躯干配合版本出现身体碰椅身和自碰撞后，又加入躯干前向退让平面作为优化引导，并将站位间隔增加到10/18/24厘米；最终仍要求精确全身几何检查，不以启发式代替验收。

## 规划结果

| 物体 | 方案 | 最终可执行计划 | IK调用 | 淘汰计数 |
| --- | --- | --- | --- | --- |
| chair_dining | strategy15_chaircoordinated_validated_plans | 0 | 644 | `{"object_contact_body_link3": 12, "visible_self_collision": 23, "object_contact_body_link2": 10, "object_contact_body_link4": 10, "arm_ik_with_fixed_torso": 18, "preapproach_ik_failed": 10, "object_contact_base_link": 3}` |
| chair_dining | strategy15_chaircorners_validated_plans | 0 | 864 | `{"arm_ik_with_fixed_torso": 216, "preapproach_ik_failed": 216}` |
| chair_wheeled | strategy15_chaircorners_validated_plans | 0 | 0 | `{}` |
| chair_dining | strategy15_chairentry_validated_plans | 0 | 530 | `{"arm_ik_with_fixed_torso": 136, "preapproach_ik_failed": 122, "search_time_budget": 1}` |
| chair_dining | strategy15_chairguarded_validated_plans | 0 | 426 | `{"arm_ik_with_fixed_torso": 142}` |
| chair_dining | strategy15_chairlegs_validated_plans | 0 | 1009 | `{"arm_ik_with_fixed_torso": 129, "object_body_link5": 4, "visible_self_collision": 39, "search_time_budget": 4, "object_arm_r_link5": 7}` |
| chair_wheeled | strategy15_chairlegs_validated_plans | 0 | 566 | `{"arm_ik_with_fixed_torso": 151, "visible_self_collision": 10, "search_time_budget": 1}` |
| chair_dining | strategy15_chairyaw_validated_plans | 0 | 723 | `{"arm_ik_with_fixed_torso": 185, "preapproach_ik_failed": 168}` |

本轮完成物理试验 0 次，通过 0 次。规划无合格轨迹时不会运行穿模动作，也不会生成冒充成功的抬升视频。

两把椅子此前存在的物理失败样本另见20261118归档；其中餐椅曾抬起约7.8厘米但触发接触点偏离可视表面检查。新几何候选未解决完整轨迹时，不能计作这项物理失败已修复。有限搜索失败不证明所有可能策略都不可行。

## 复现

先用 ../20261118_support/replay.py prepare 创建新工作区，然后复制本目录source/*.py到工作区（均为本轮新文件）。使用Isaac环境Python执行 `run_chaircorners_plan.py chair_dining`、`run_chaircorners_plan.py chair_wheeled`、`run_chairlegs_plan.py chair_dining`、`run_chairlegs_plan.py chair_wheeled`。餐椅角部搜索完成生成corners_geometry_chair_dining.json后，再运行 `run_chairyaw_plan.py chair_dining`、`run_chairentry_plan.py chair_dining`、`run_chaircoordinated_plan.py chair_dining`、`run_chairguarded_plan.py chair_dining`。各runner内部限制BLAS/OMP为单线程。

plans保存生成和审计后的计划及淘汰信息，logs保留逐朝向手部候选计数，planning_summary.json区分不同方案。保持1千克刚体、理想躯干驱动等原仿真假设，未进行实机验证。
