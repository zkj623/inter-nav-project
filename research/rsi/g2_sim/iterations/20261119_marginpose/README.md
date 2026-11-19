# G2 地面抬升姿态：增加躯干限位余量

替换旧的 [-50,150,-85,0,0] 度躯干配置，改为 [-40,142,-87,0,0] 度。依据未锁定前的 G2 URDF 原始关节限位，五关节最近限位余量约为 [22.006,10.006,22.263,25.004,174.500] 度。第二关节余量由约 2 度提高到约 10 度。10 度是本轮筛选约束，不是厂家实机认证标准。

肩高约 0.823 m，整体躯干前倾 15 度。固定底盘、腰、头和轮组；只允许左右各 7 个手臂关节和两只夹爪主关节运动，共 16 维。双臂仍按物体预抓取姿态初始化，未验证统一双臂 reset 路径。

## 筛选

比较 10 组躯干配置，每组都要求所有躯干关节距原 URDF 上下限至少 10 度。复用既有接触区域，重新求解预抓取、抓取与抬升全路径，允许前后站位偏移至多 12 cm。保留原关节限位、静态负载检查、可视/原生自碰撞和全身物体碰撞审计。最终候选 7 的 11 个物体通过完整几何路径审计。

覆盖发生变化：边桌 table_side 在新姿态下有合格路径；脚踏桶 bin_pedal 在本轮有限搜索中没有合格路径，暂不进入该姿态训练集。两把椅子和软包圆凳沿用此前排除原因。几何通过不等于物理抬升通过。

## 物理验证

| 物体 | 独立复核通过 | 失败原因 | 最低保持抬高 m | 最大保持倾斜 deg |
| --- | --- | --- | --- | --- |
| box_reinforced | False | NO_SETTLED_FOUR_FINGER_GRASP | None | None |
| case_handheld | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.18401074409484863 | 26.377353760858057 |
| box_reinforced | True | None | 0.16902665793895721 | 12.920181632020928 |
| stool_rectangular | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.19836711883544922 | 0.585307224665361 |
| stool_rectangular | True | None | 0.19832982122898102 | 0.5348927669351157 |

成功判据保持原样：抬高至少 15 cm、真实保持 3 s、倾斜不超过 20 度、相对漂移不超过 2 cm、四接触覆盖率至少 90%、实际接触表面误差和自碰撞检查。独立复核按实际采样关节和物体位姿检查非手部几何穿透不超过 1 mm，并验证锁定关节误差小于 0.005 rad、根部位置漂移小于 1 mm。几何穿透复核不等价于非手部冲量测量。

首轮加固箱在抬升后的四指稳定阶段失败，steady 版本对三个采样点的指力取平均，并把最多 30 次抬后稳定调节增加到 60 次；通过判据不变。木凳首轮抬高约 19.8 cm，但保持阶段一个指面承力不足；supportsettle 版本增加抬后四分支稳定门槛，通过后再开始 3 秒保持。硬箱首轮保持倾斜约 26.38 度，超过 20 度，仍判失败；收紧末端逆解误差至 0.5 mm、细化到 1 cm 站位搜索后暂未找到合格新路径，不计入当前验证通过集。

仍使用统一 1 kg 物体、理想躯干驱动、原手臂和夹爪力矩上限，固定根部是本阶段训练简化。结果不证明真实腰部承载能力，不包括起身、行驶或随机扰动鲁棒性。全部失败试验保留。

## 训练入口与复现

当前入口配置为 ../../ground_lift_training_config.json，原配置与原结果保留在 ../20261119_fixedpose/，不覆盖历史。

先按 ../20261119_fixedpose/README.md 准备新的工作区，再覆盖本目录 source/ 和 fixtures/ 到工作区根目录。勿运行资产生成器，资产仍为共享只读。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /path/to/isaaclab/bin/python marginpose_deadband_trial.py --asset box_reinforced --plan-dir marginpose_selected7_validated_plans --gpu 0 --repeat 2
```

首轮手提硬箱同用 deadband，木凳用 marginpose_support_trial.py；后续版本为 marginpose_steady_trial.py、marginpose_supportsettle_trial.py。实际通过入口见当前配置的 validated_simulation_baselines，不要把首轮失败入口当成通过基线。完整视频和采样数据的服务器路径及 SHA256 见 trial_index.json。已执行计划保留原来源的历史 envelope/descent 元数据；执行依据是重解后的 q_pre/q_grasp/q_close/q_lift。姿态余量应查看新配置和实际 q 值，不能读取历史 envelope 推断当前姿态。尚未开始训练。
