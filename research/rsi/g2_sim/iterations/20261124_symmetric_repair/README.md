# 剩余4类交互修复：2026-11-24

4类均通过独立验收，合并上一轮8类，当前对称低位初姿共12/12类名义初态成功。不是随机扰动成功率，也未重新采集240条。初姿、身体−59°/149°/−75°、头部27°、物体原初始位置不变；固定底盘与身体，原接触、速度、碰撞和保持标准不变。

|物体|最低抬升cm|最大倾斜°|最大表面偏差mm|实际速度/模型上限|
|---|---|---|---|---|
|basket_rectangular|18.3|8.58|1.53|0.462|
|box_white|16.7|2.70|0.90|0.458|
|box_reinforced|17.6|13.19|1.26|0.580|
|table_side|19.7|0.22|1.00|0.260|

视频审阅：[飞书文档第16节](https://ocnprlp82ms4.feishu.cn/docx/WPm9djZqoo9mgfx3GDLcBF9OnYf#doxcnnK5JnhsdQzcHQ9pDmTtlRb)。每个成功入口的同版本复跑通过次数见根目录ground_lift_training_config.json的latest_baseline_verified_repeat_counts；所有尝试保留在results/attempts.json。

具体成功入口与失败尝试见results。加固箱调整接触带并改善顺应性；白箱调整腕部角度、接触高度和抬升间隙，采用从早期轨迹中提取的平衡关节目标作为候选，再通过新的完整仿真验证，最终保持阶段使用固定目标，避免反馈来回调节破坏接触；边桌降低夹爪刚度、增加阻尼抑制夹爪超速；长方篮改为偏航0°、两侧居中篮壁夹持，按真实曲面重新计算夹持点并加深，原中心位置不变。不要把未通过的候选替换为最终入口。部分早期v1试验额外延长保持，实际时长见controller_manifest；最终都按原至少3秒及全程接触/速度标准验收。

本轮不同物体使用不同夹爪增益，准确参数见controller_manifest.json和归档源码；部分继承的feedback描述字符串仍为旧说明，执行常数以源码为准。统一ACT扩量前应统一低层控制，或显式记录/建模控制配置，不能把这些参数差异隐藏在数据里。新录制器遇到落后的渲染帧会最多重渲染3次，断言不推进物理时间；原时间戳容差未放宽。

创建独立工作区：`python iterations/20261124_symmetric_repair/source/repair_workspace.py --workspace /path/to/work --package /path/to/g2_sim`。
四类执行：`python realpose_<asset>_trial.py --asset <asset> --plan-dir repair_final_plans --gpu <id>`，然后`python verify_realpose.py --trial-dir realpose_trials/repair_final_plans/<asset>/candidate0_repeat1`。
其余8类使用上一轮入口。不要运行旧build_realpose_pilots.py覆盖本轮逐物体控制器；重建应使用repair_workspace.py，其覆盖源码来自已验证试验的SHA256归档。

原始物理记录与视频：`/data20t/embodied/zkj/inter-nav-results/g2-symmetric-repair-20261124/realpose_trials`。物体未被绑定或直接写位姿，仍采用既定1kg刚体和理想身体驱动设置。完整站姿准备仅手提箱在上一轮物理复验；本轮4类从统一低位开始。
