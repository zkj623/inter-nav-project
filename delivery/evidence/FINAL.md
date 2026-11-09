# Go2 原五任务材质重录与五条新交互任务：最终验收

原T01–T05保持原场景、起终点和流畅度要求，完成保留材质的重新录制；新T06–T10完成实际推箱、新扫描确认通路、整机穿越与自动停稳。共50个完整25fps视频。所有任务曲折度>1.3，初始杂乱度38.195603%，终点误差≤0.5m。

场景原贴图像素没有缩放或修改，颜色与UV保留；通过合并重复材质和相同图像资源、停用未使用材质、兼容转换导出PBR着色器，满足原24GiB cgroup约束。高级折射、位移与贴图式specular采用兼容近似，不能声称原MDL光照效果完全相同。

|任务|类型/版本|曲折度|真实交互|停稳误差cm|完整时长s|综合视频|
|---|---|---:|---|---:|---:|---|
|T01|原任务材质重录 / qv13|1.582989|无|0.59|31.80|[播放](E:/inter-nav-results/go2-textured-interaction-20261103/existing-five/runs/qualified-qv13-t01-try1/T01/video/combined.mp4)|
|T02|原任务材质重录 / qv13|1.483066|无|9.00|47.76|[播放](E:/inter-nav-results/go2-textured-interaction-20261103/existing-five/runs/qualified-qv13-t02-try1/T02/video/combined.mp4)|
|T03|原任务材质重录 / qv13|1.488507|无|14.06|40.76|[播放](E:/inter-nav-results/go2-textured-interaction-20261103/existing-five/runs/qualified-qv13-t03-try1/T03/video/combined.mp4)|
|T04|原任务材质重录 / qv13|1.417087|无|4.03|34.32|[播放](E:/inter-nav-results/go2-textured-interaction-20261103/existing-five/runs/qualified-qv13-t04-try1/T04/video/combined.mp4)|
|T05|原任务材质重录 / qv13|1.405217|无|7.57|34.44|[播放](E:/inter-nav-results/go2-textured-interaction-20261103/existing-five/runs/qualified-qv13-t05-try1/T05/video/combined.mp4)|
|T06|新交互任务 / qv22|1.325453|B06 / 0.817m|13.58|98.64|[播放](E:/inter-nav-results/go2-textured-interaction-20261103/interaction-five/runs/qualified-qv22-t06-try1/T06/video/combined.mp4)|
|T07|新交互任务 / qv22|1.575777|B06 / 0.830m|19.65|89.84|[播放](E:/inter-nav-results/go2-textured-interaction-20261103/interaction-five/runs/qualified-qv22-t07-try1/T07/video/combined.mp4)|
|T08|新交互任务 / qv22|1.542677|B07 / 0.844m|13.92|93.76|[播放](E:/inter-nav-results/go2-textured-interaction-20261103/interaction-five/runs/qualified-qv22-t08-try1/T08/video/combined.mp4)|
|T09|新交互任务 / qv22|1.359387|B07 / 0.843m|11.69|130.80|[播放](E:/inter-nav-results/go2-textured-interaction-20261103/interaction-five/runs/qualified-qv22-t09-try1/T09/video/combined.mp4)|
|T10|新交互任务 / qv23|1.317166|B06 / 0.845m|16.31|108.40|[播放](E:/inter-nav-results/go2-textured-interaction-20261103/interaction-five/runs/qualified-qv23-t10-try1/T10/video/combined.mp4)|

每个video目录还有third_person、robot_rgb、map_topdown、pointcloud_map四路。完整帧解码、原始物理步、源/模型/输入hash及真实关键帧已分别验证；未宣称人工观看每一帧。

当前交互视频中，普通及通路导航的橙色虚线是从实际姿态、当前在线全局图和当前子目标重新搜索的A*参考，约1s刷新；蓝色实线是实际跟随路线，细绿线是实走轨迹。推箱期间的历史对比明确标为Before push (historical)，进入通路后保留独立Before push对比，同时显示新的当前Reference及年龄。旧交互试录的标注更正见REFERENCE-DISPLAY-ERRATUM.md，旧录像不拼入本轮交付。

|新任务|起点|目标|要求交互的箱体|
|---|---|---|---|
|T06|[14.6625, 1.6875]|[10.3, 1.75]|B06|
|T07|[8.3875, 3.8375]|[10.3, 1.4]|B06|
|T08|[4.6625, 4.1125]|[3.3, 1.4]|B07|
|T09|[7.6125, 4.8125]|[3.3, 1.4]|B07|
|T10|[4.6875, 2.9125]|[10.3, 1.75]|B06|

T07早期试录的固定转角区域，停留从28.50s降至15.00s，多余转向从421.76°降至186.60°。这些区域仅用于离线评价，不传给导航器。
本轮导航修复包含：在原占据/未知区掩码上检查短的初始骨架连接，避免先接到身后的支路；完成首次有界转向后留出一次最多2s的起步机会，且不重置真实无进展时钟；探索候选评分计入当前朝向的初始转向代价，避免为很小的路径收益反复转身。实际任务目标、碰撞/未知区/制动检查和policy均保留。
推箱对齐还增加了4cm的侧移启动余量，避免在20cm鼻尖离箱距离边界反复停止、纠偏；停止侧移和实际接触判定仍保留原阈值。纵向起步停滞后，经实测停稳再执行限时、限位移且经过完整路径检查的平移输入，避免速度爬升尚未送达就反复制动。
推箱后的明显后退/侧移组合改用机身单轴分段，每段仍检查原机身、未知区、完整步幅及制动路径；来回摆动不再重置进展时钟。新交互录像同时检查重复对齐循环和通路连接中的低进展，不因最终到达就忽略长期停滞。

曲折度沿用2.5cm真实mesh占据表示上的连续几何最短路径/欧氏距离，不用执行绕行或原地转圈抬高。新任务几何路径有限；0.30m余量通路交互前被箱体阻断，侧推后连通。离线地图、参考路径和历史停放见证均不输入在线导航，机器人从空图使用真实LiDAR。

保留原尺寸、.01m辅助地面、物理200Hz/Actor50Hz、全局/局部1/10Hz、机身/未知区/制动检查与原policy。交互资格仍使用已授权的真值箱子名单和LiDAR可见性，未训练视觉识别模型或新的步态policy。

新任务仍会有必要的接触对齐、停稳和恢复。T06功能诊断曾在名义推距还差约6cm时触发推进停滞恢复，安全后退后通过实际新扫描证明通路已形成；这不是放宽几何安全条件。最终以完整录像中的实际结果为准。

|组别|运行版本|运行源码|回归通过数|
|---|---|---|---:|
|existing-five (T01, T02, T03, T04, T05)|qv13|2888eb8813da095172c39319c71e4d6c44086521|974|
|interaction-five (T06, T07, T08, T09)|qv22|d718f88179ddb8c544da310173df6ac395f39385|1006|
|interaction-five (T10)|qv23|514cd51dc7a4400a3e7db431b1f18a7aba7e3f50|1012|

T10的QV22尝试真实移动B06约0.845m，但在后退和停稳余量检查之间循环84次并超时，未获通路证书，失败录像与日志保留。QV23只在实际停稳后余量不足时要求下一次结束后退前先取得8cm净后退；保留原4cm机身余量、未知空间、完整运动/制动检查及恢复时限。新录像用于T10，已验收的QV22 T06–T09仍按原版本如实列出，本轮没有重新训练policy。

每次回归均有1项既有G1相机配置检查未选择。原QV7五项及历史v218原始T04/T05交互验收保持有效，后者是不同起终点，不能与本轮T04/T05混用。

|尝试|类型|结果|原因|最终录像|
|---|---|---|---|---|
|existing-five/qualified-qv10-t01-try1|qualified|未成功|resource_limit_before_task|否|
|existing-five/qualified-qv13-t01-try1|qualified|成功|goal_reached|是|
|existing-five/qualified-qv13-t02-try1|qualified|成功|goal_reached|是|
|existing-five/qualified-qv13-t03-try1|qualified|成功|goal_reached|是|
|existing-five/qualified-qv13-t04-try1|qualified|成功|goal_reached|是|
|existing-five/qualified-qv13-t05-try1|qualified|成功|goal_reached|是|
|existing-five/qualified-qv8-t01-try1|qualified|未成功|resource_limit_before_task|否|
|interaction-five/diagnostic-qv13-t06-try1|diagnostic|成功|goal_reached|否|
|interaction-five/diagnostic-qv14-t07-try1|diagnostic|未成功|stop_requested|否|
|interaction-five/qualified-qv13-t06-try1|qualified|成功|goal_reached|否|
|interaction-five/qualified-qv13-t07-try1|qualified|未成功|stop_requested|否|
|interaction-five/qualified-qv15-t07-try1|qualified|未成功|stop_requested|否|
|interaction-five/qualified-qv16-t07-try1|qualified|未成功|stop_requested|否|
|interaction-five/qualified-qv17-t07-try1|qualified|未成功|stop_requested|否|
|interaction-five/qualified-qv18-t07-try1|qualified|未成功|exception_before_task_result|否|
|interaction-five/qualified-qv19-t06-try1|qualified|成功|goal_reached_but_fluency_failed|否|
|interaction-five/qualified-qv19-t07-try1|qualified|成功|goal_reached|否|
|interaction-five/qualified-qv21-t06-try1|qualified|成功|goal_reached|否|
|interaction-five/qualified-qv21-t07-try1|qualified|成功|goal_reached|否|
|interaction-five/qualified-qv21-t08-try1|qualified|成功|goal_reached_but_fluency_failed|否|
|interaction-five/qualified-qv22-t06-try1|qualified|成功|goal_reached|是|
|interaction-five/qualified-qv22-t07-try1|qualified|成功|goal_reached|是|
|interaction-five/qualified-qv22-t08-try1|qualified|成功|goal_reached|是|
|interaction-five/qualified-qv22-t09-try1|qualified|成功|goal_reached|是|
|interaction-five/qualified-qv22-t10-try1|qualified|未成功|push_aside:skill_timeout|否|
|interaction-five/qualified-qv23-t10-try1|qualified|成功|goal_reached|是|

QV8和QV10的首次材质加载受24GiB上限终止，均未进入行走，失败记录保留；无录像功能诊断不替代最终录像。

完整验收汇总：[final-textured-acceptance.json](E:/inter-nav-results/go2-textured-interaction-20261103/final-textured-acceptance.json)。
