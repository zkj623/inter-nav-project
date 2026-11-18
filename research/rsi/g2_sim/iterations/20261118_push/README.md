# 2026-11-18：两把大椅子的推移验证

把无法稳定抬起的餐椅、办公椅改为沿地面推移。此目录独立保存新方案及全部失败，不更改此前抓取通过记录。完成结果见下表和 baselines.json；只有同一源码、同一初态至少三次通过独立复核的方案才进入基线。

- chair_wheeled：同初态3次通过；前移13.99–13.99厘米，最大倾斜0.54度，最大偏航0.35度。
- chair_dining：同初态3次通过；前移12.24–12.24厘米，最大倾斜4.74度，最大偏航0.73度。

## 范围与标准

从已规划的低姿态初始化，底盘不主动行驶且不锁定。手部接近刚性结构，水平推进13–15厘米（各计划push_offsets_m记录确切指令行程），后撤4厘米，松手保持3秒。判据：物体前进至少10厘米，侧移不超过3厘米，倾斜和偏航均不超过10度，末1秒速度不超过2厘米/秒且手部接触力低于0.5牛。必须有真实接触见证；餐椅双手方案另要求至少5个推移采样点双手同时有力。详细判据按10 Hz记录独立复算，不能把采样峰值当成120 Hz全步最大值。

保留手部可视距离1.5毫米、接触点表面误差2.5毫米检查；保留可视、自碰撞及原生碰撞审计。clear、reach和center方案额外给非手部留15毫米规划间隙。没有物体附着、物体跟随、直接向物体施力或基座锁定。

复核另检查全部非手部接触记录，任何非手部冲量超过0.001牛秒的试验均不收为通过，不依赖是否穿透；避免上臂只是贴住椅子、但已参与推动而被忽略。

力的独立复核要求采样峰值不超过35牛、连续超过20牛的记录不长于1.05秒。反馈控制用18牛以上减小前压量替代固定轨迹的暂停逻辑；35牛运行中停止检查仍保留，持续力限制由统一复核再次检查。

两把椅子都只有一个刚体、没有关节；办公椅轮子不转动。本轮验证的是刚体在地面滑动，不能作为真实脚轮滚动验证。沿用1千克统一质量与理想躯干驱动，手臂及夹爪原驱动上限；质量中心和惯量由仿真自动计算，未测量运行时质心。尚未验证站立接近、下降到推姿、长距离移动、真实质量或实机鲁棒性。

## 根据失败调整

- 初版先解推进终点，许多候选回不到预接近位置；改为两端共享躯干的联合逆解。
- 低位餐椅路径容易使前臂穿过底盘；改为按手腕与底盘的实际间距选站位，并保留最终真实几何审计。
- 发现初版控制器在 world.reset 后才旋转物体。原 push 的两次试验受启动碰撞干扰，作为初始化诊断保留，不用于能力结论。init/start控制器在reset前写入朝向和默认位姿。
- 餐椅一次初始化校验将reset期间2.1毫米落地位移当成错误；start控制器改为先严格校验reset前位姿，再分别核对reset后水平偏移、落地位移和朝向。原软件错误保留，未改变接触或成功判据。
- 办公椅侧向单手推动约13.4厘米，但偏航13–16度未过；随后转向正前/正后方向，在几何对称线的中央支柱上推，减少偏转。
- 餐椅26厘米双手推点实际推进约5.2厘米后翘起；下调推点并以已通过碰撞审计的躯干姿态为参考，在其附近小范围求解，避免为够到低点而过度折叠身体。
- 22厘米推点的15厘米指令行程仍缺少手臂余量；short方案把指令改为13厘米，实际前进至少10厘米的通过阈值不变。
- 进一步核对发现原餐椅两个推点的外法向都约为(-0.707,-0.707,0.008)，同向侧向分力导致擦过椅腿。front方案对每个细椅腿横截面用0.5毫米步长细化射线，选择最靠近机器人的前缘；保留真实碰撞和接触审计，不直接改动物体位置。
- front方案倾斜降至1度以内，但仅推进约8.8厘米，撤手阶段触发79牛接触峰值。release方案在24厘米推点使用15厘米行程，先将双夹爪张开至-0.18弧度，再用更慢的退手动作；接触力上限和成功标准不变。
- release方案三次均能撤手停稳，但最终前移约8.3厘米，未达10厘米。记录显示固定手部轨迹下接触会逐渐丢失；feedback方案使用实测物体位姿更新手部接触目标，额外前压量限制在4–20毫米，并根据接触力调整。只控制机器人关节，不修改交互中的物体位姿、速度或直接施加物体外力。目标实际前移11.2厘米，为撤手后的少量回退保留余量，原10厘米验收线不变。
- feedback三次前移约12.3厘米、撤手后停稳，但侧移3.21厘米超过3厘米阈值。centerline方案对手部目标增加0.2倍横向误差反馈，修正量最多2毫米，保持原验收阈值；这仍是机器人关节控制，不是修正物体的位置。

## 完成的试验

表中位移与角度来自实际采样；软件错误没有实测值，不计入能力结论。

| 物体 | 方案 / 候选与重复 | 类别 | 复核通过 | 失败 | 前移m | 最大偏航deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chair_wheeled | push / c0 r1 | superseded initialization order; not a valid capability conclusion | False | NONHAND_OBJECT_CONTACT | -5.662441253662109e-06 | 0.5394375552123497 |
| chair_wheeled | push / c1 r1 | superseded initialization order; not a valid capability conclusion | False | NONHAND_OBJECT_CONTACT | 5.960464477539063e-08 | 0.0008258636725553626 |
| chair_wheeled | pushcenterinit / c0 r1 | physical trial | True | None | 0.1398819088935852 | 0.35070282827996835 |
| chair_wheeled | pushcenterinit / c0 r2 | physical trial | True | None | 0.1398819088935852 | 0.35070282827996835 |
| chair_wheeled | pushcenterinit / c0 r3 | physical trial | True | None | 0.1398819088935852 | 0.35070282827996835 |
| chair_dining | pushcenterlinestart / c0 r1 | physical trial | True | None | 0.12244653701782227 | 0.7307863289240815 |
| chair_dining | pushcenterlinestart / c0 r2 | physical trial | True | None | 0.12244653701782227 | 0.7307863289240815 |
| chair_dining | pushcenterlinestart / c0 r3 | physical trial | True | None | 0.12244653701782227 | 0.7307863289240815 |
| chair_wheeled | pushclearinit / c0 r1 | physical trial | False | PUSH_STOP_CRITERIA_NOT_MET | 0.13437551259994507 | 15.695235165992647 |
| chair_wheeled | pushclearinit / c1 r1 | physical trial | False | PUSH_STOP_CRITERIA_NOT_MET | 0.0939018726348877 | 15.966582751418214 |
| chair_dining | pushfeedbackstart / c0 r1 | physical trial | False | PUSH_STOP_CRITERIA_NOT_MET | 0.12282896041870117 | 1.2854748785337433 |
| chair_dining | pushfeedbackstart / c0 r2 | physical trial | False | PUSH_STOP_CRITERIA_NOT_MET | 0.12282896041870117 | 1.2854748785337433 |
| chair_dining | pushfeedbackstart / c0 r3 | physical trial | False | PUSH_STOP_CRITERIA_NOT_MET | 0.12282896041870117 | 1.2854748785337433 |
| chair_dining | pushfrontstart / c0 r1 | physical trial | False | PUSH_FORCE_LIMIT | 0.08846127986907959 | 1.4195070330959594 |
| chair_wheeled | pushinit / c0 r1 | physical trial | False | PUSH_STOP_CRITERIA_NOT_MET | 0.13437843322753906 | 12.960131200418013 |
| chair_dining | pushreachinit / c0 r1 | software error | False | HARNESS_ERROR | None | None |
| chair_dining | pushreachstart / c0 r1 | physical trial | False | PUSH_UNINTENDED_LIFT | 0.05243980884552002 | 3.9298847188108126 |
| chair_dining | pushreleasestart / c0 r1 | physical trial | False | PUSH_STOP_CRITERIA_NOT_MET | 0.08297932147979736 | 7.474373892412457 |
| chair_dining | pushreleasestart / c0 r2 | physical trial | False | PUSH_STOP_CRITERIA_NOT_MET | 0.08297932147979736 | 7.474373892412457 |
| chair_dining | pushreleasestart / c0 r3 | physical trial | False | PUSH_STOP_CRITERIA_NOT_MET | 0.08297932147979736 | 7.474373892412457 |
| chair_dining | pushsafestart / c0 r1 | physical trial | False | PUSH_STOP_CRITERIA_NOT_MET | 0.05796802043914795 | 6.625939788988019 |
| chair_dining | pushshortstart / c0 r1 | physical trial | False | PUSH_STOP_CRITERIA_NOT_MET | 0.039827704429626465 | 8.214246031010758 |

## 复现与证据

`python3 replay.py verify` 校验归档字节与源码语法。`replay.py prepare` 依次准备此前support、chairs、collision及本轮源码，参数沿用support/replay.py（新工作区、原资产、collision-deps）。`replay.py run --workspace ... --python ... --asset ... --gpu ... --repeat ...` 只运行baselines.json记录的已重复通过方案，并拒绝覆盖已有试验。

完整视频、samples和场景保留服务器路径及哈希；Git内保存执行源码、规划、协议、结果、接触诊断及独立复核。有限搜索未通过不证明不存在可行交互；三次同初态重复不代表随机扰动成功率。

联合求解方案的ik_calls只统计后续逐点逆解，shared_endpoint_attempts另计联合端点求解次数；不能把ik_calls=0解释为没有尝试规划。

规划中的q_lift只是供旧审计器复用的字段别名，与水平q_push相同，不代表本轮抬升了物体。正式运行入口以baselines.json和replay.py为准，其他源码保留实验过程。

feedback控制器复用release方案的已审计参考计划，实际手部轨迹按物体观测在线修正；不能把参考计划的静态审计声称为所有在线修正姿态都已预先审计。实际运行保持自碰撞物理、接触表面见证、非手部接触复核及姿态/力/跟踪误差停止检查，采样与源码均归档。

物体反馈来自仿真真值位姿，尚未接入视觉位姿估计；这是几何与控制策略验证，不是训练后模型或视觉鲁棒性验证。
