# 四类物体全程姿态优化

两个篮子、四脚木凳和边桌的控制器新增围绕规划姿态的有限幅度IK反馈，防止七自由度冗余解漂移；抓取及抬升规划采用镜像关节与连续逆解。木凳/边桌重新选点并优化肘部外展，必要时采用高位接近路点和笛卡尔接近路径。编织篮具体选择及各资产参数见执行plan和controller_manifest，失败候选保留在attempts。

|物体|保持最低抬升cm|最大倾斜°|全程实际镜像关节偏差最大°|
|---|---|---|---|
|basket_rectangular|19.5|4.93|0.49|
|basket_woven|19.7|1.58|0.50|
|stool_wood_four_leg|20.3|0.32|0.46|
|table_side|19.9|0.20|0.44|

四类均通过原物理验收，以及新增全采样姿态检查：镜像关节偏差≤15°、交互阶段肘部不高于肩部10cm、相邻抬升路点关节变化≤30°，并人工检查实际状态正面/俯视重建及视频关键帧。数值只是本次筛选规则，不代表真机人体工学认证。原碰撞、可视表面、关节速度、四指接触、抬升与保持门槛未降低。

从当前统一低位初姿开始，身体−59/149/−75°、头部27°、固定底盘/身体、1kg刚体与理想身体驱动设定不变。其他8类控制器及旧240条数据不变。12/12仍指各自名义初始摆放的成功案例，不代表完成随机扰动采集。不同物体夹爪参数仍有差异，ACT扩量前须统一控制或显式记录。

重建：`python iterations/20261124_natural_four/source/natural_workspace.py --workspace /path/to/work --package /path/to/g2_sim`。
验收：`python run_natural_validation.py --asset <asset> --plan-dir natural_final_plans --gpu 2`。采集入口须同时检查verification.json与posture_verification.json，不能仅凭episode.hdf5存在就接收。实验生成器仅为溯源，复现采用已验证fixtures和哈希归档控制器。

原始轨迹/视频：`/data20t/embodied/zkj/inter-nav-results/g2-natural-four-20261124/realpose_trials`；实际状态重建：`/data20t/embodied/zkj/inter-nav-results/g2-natural-four-20261124/natural_posture_preview`。仅手提箱保留之前完整站姿准备复验，四类此次均从共同低位开始。


[四类优化前后对比及完整视频（飞书第19节）](https://ocnprlp82ms4.feishu.cn/docx/WPm9djZqoo9mgfx3GDLcBF9OnYf#doxcnsb504cmqMap9z5PCt8xRJd)。`results/review_artifacts.json` 记录原始视频路径和哈希，`results/bootstrap_verification.txt` 记录独立工作目录重建检查。
