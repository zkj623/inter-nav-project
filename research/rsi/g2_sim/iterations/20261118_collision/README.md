# 2026-11-18：椅子避碰逆解与共享躯干姿态

本轮把避碰引导加入逆解，尝试左右夹爪不同开度、中央支柱夹取、预抓取与抬升终点共享躯干的联合逆解。所有最终轨迹仍需通过原可视、自碰撞、原生碰撞与全身/物体几何审计；没有通过关闭检查或添加附着获得成功。

## 优化引导校准

最初的外包球近似过于保守，在已验证的木凳姿态上给出35.88厘米的代理自重叠。它只是失败的优化近似，不代表真实模型碰撞，相关三组 broad/avoid 规划结果不可当成物体本身不可抓取的证据。

修正后的较小内部球只用于引导搜索；在空载与木凳参考姿态上均无代理自重叠。它不是碰撞表面，也不能替代最终真实网格/凸包审计。校准记录见 diagnostics/guidance_validation.json。

共享躯干求解同时优化预抓取和抬升20厘米终点的手臂姿态，用同一组躯干关节值保证两端的可达余量。已验证木凳的逆解回归结果见 diagnostics/shared_ik_validation.json；这仅是端点运动学与姿态包络检查，不是新一轮物理抬升验证。shared方案的ik_calls只统计后续逐点IK，不包含共享端点优化调用，不能与普通方案直接比较计算次数。

办公椅中央支柱方案为单手夹取，另一臂可自由寻找避让姿态；其控制器跟踪非抓取臂的规划关节值。该策略与双手方案单独记录。不同开度的托底方案对两侧手指分别建模，不强制左右使用相同开度。

## 完成的规划结果

| 物体 | 方案 | 最终可执行计划 | 淘汰计数 |
| --- | --- | --- | --- |
| chair_dining | chairasym | 0 | `{"arm_ik_with_fixed_torso": 5, "visible_self_collision": 8, "lift_ik_failed": 2, "preapproach_ik_failed": 2, "search_time_budget": 1}` |
| chair_wheeled | chairasym | 0 | `{}` |
| chair_dining | chairavoid | 0 | `{"arm_ik_with_fixed_torso": 90}` |
| chair_wheeled | chaircolumncore | 0 | `{"arm_ik_with_fixed_torso": 59}` |
| chair_wheeled | chaircolumnshared | 0 | `{"shared_body_endpoint_ik_failed": 30}` |
| chair_dining | chaircore | 0 | `{"arm_ik_with_fixed_torso": 32, "preapproach_ik_failed": 18, "visible_self_collision": 7}` |
| chair_dining | chairlegavoid | 0 | `{"arm_ik_with_fixed_torso": 21}` |
| chair_wheeled | chairlegavoid | 0 | `{"arm_ik_with_fixed_torso": 31}` |
| chair_dining | chairlegcore | 0 | `{"arm_ik_with_fixed_torso": 15, "preapproach_ik_failed": 17, "visible_self_collision": 5, "search_time_budget": 2}` |
| chair_wheeled | chairlegcore | 0 | `{"arm_ik_with_fixed_torso": 32}` |
| chair_dining | chairshared | 0 | `{"shared_body_endpoint_ik_failed": 52}` |

本轮完成物理试验 0 次，通过 0 次。几何候选或端点逆解成功不计为稳定抓取；没有合格轨迹时不执行物理抬升。有限时间、有限初值搜索失败不证明不存在可行抓取。

## 复现

先使用 ../20261118_support/replay.py prepare 创建新工作区；复制 ../20261118_chairs/source/*.py，并复制该归档plans/corners_geometry_*.json到工作区根目录；再复制本目录source/*.py。运行各run_VARIANT_plan.py ASSET，例如 `run_chairshared_plan.py chair_dining` 和 `run_chaircolumnshared_plan.py chair_wheeled`。运行器调用同一Isaac Python，限制BLAS/OMP为单线程。可用原run_VARIANT_trials.py在审计完成后启动对应物理实验。

原始资产保持外部只读，继承1千克刚体与理想躯干驱动假设。源码、规划结果与诊断按SHA256归档，不修改此前通过记录。
