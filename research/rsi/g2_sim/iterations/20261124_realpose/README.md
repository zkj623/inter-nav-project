# G2 真机参考低位默认姿态与头部视野

当前默认身体角：**脚踝J1 −59°、膝J2 149°、腰J3 −75°**。这三个数来自用户真机可达性反馈；假定真机控制界面零位和正方向与当前URDF一致。头部第3关节采用 **27°**，依据本轮仿真视野检查选择，尚非真机相机标定或头部姿态验证。

模型中的J1/J2角度余量约3.01°，头部上限29.97°、余量约2.97°。原10°身体余量规则与新姿态不相容，故仅该明确指定profile将独立验收的身体余量门槛改为2.9°；原关节硬限位、根漂移、关节冻结、碰撞、接触及抬升保持判据保留。不能把2.9°称为真机安全余量。

## 已完成的检查

- 12类重新求双臂IK，保持原接触策略。7类保持原摆位，5类向机器人方向移动2–4cm；精确平移见配置 `object_shift_x_m`。全接近/抓取/抬升路径按<=0.06rad间隔检查非手部物体穿透，独立检查可视自相交。
- 12/12初态的完整可视模型顶点位于名义头部相机视锥内，且语义渲染检测到目标像素。头部25°时矩形凳底部约3px超出画面，27°时该底部距画面下沿约11px。全模型在视锥内不等于无遮挡；这里不是对每条动作全程遮挡的保证。
- 手提箱、提手编织篮、脚踏桶3类，在**准确的27°头部默认配置**下完成真实物理抬升并独立验收，同时记录20Hz、640×400头部RGB，与16维qpos/qvel/action和时间戳逐项对齐。
- 其余9类仅通过几何/路径和初态视野检查，尚未在新姿态下完成物理抬升复验。待机到新低位的61个插值点通过可视自碰撞检查，完整物理过渡仍未复验。

抬升要求仍为>=15cm、保持>=3秒、倾斜<=20°、四点承力、接触面和非手部碰撞检查等。物体仍按1kg、原理想腰腿驱动1e7、固定底座和工作时冻结腰腿，未验证真机力矩上限。

## 第一视角与数据

名义相机位于 `head_link3`，局部平移 `[0.10237,0.02375,0.10256]`m，USD四元数wxyz `[-0.5945104,-0.3868316,0.3787291,0.5945443]`，分辨率640×400，fx/fy约305.64/305.60、cx/cy约318.80/202.49，源于本地官方仿真配置。地面采用家居木地板视觉材质，原碰撞面与摩擦参数未改。

新物理验证目录：`/data20t/embodied/zkj/inter-nav-results/g2-realpose-20261124/realpose_trials/realpose_plans/<asset>/candidate0_repeat2`。`head_pilot.hdf5`保存真正的头部图像、16维状态与执行目标；`episode.hdf5`是同次试验外部观测流。`head_pilot_verification.json`检查两路状态、动作、时间逐项一致及相机时间偏差；通过记录偏差为0。

重复编号1是较早的头部25°诊断，独立保留在 `head25_archive` 配置对应范围；本轮正式复测使用重复编号2。源码和实际执行的头部录制器版本按SHA归档。二进制HDF5、MP4和全量samples在服务器，Git仅包含索引、计划、报告和关键图片。

**原240条冻结数据仍属于旧身体姿态 −40°/142°/−87°，且为外部相机输入，未转换成新姿态或第一视角。** 不可不区分profile直接混训。若以新默认和头部相机训练，应重新采集并独立冻结新数据。

## 复现入口

创建独立工作区，沿用服务器现有资产，只读共享模型：

```bash
python research/rsi/g2_sim/iterations/20261124_realpose/source/realpose_workspace.py \
  --workspace /data20t/embodied/zkj/inter-nav-results/g2-realpose-new \
  --package /absolute/path/to/research/rsi/g2_sim
```

此后用Isaac Python运行，例如：

```bash
python realpose_case_handheld_trial.py --asset case_handheld --plan-dir realpose_plans --gpu GPU
python verify_realpose.py --trial-dir realpose_trials/realpose_plans/case_handheld/candidate0_repeat1
```

`GPU`替换为可用设备编号。不要对已有试验目录反复计数。其他9类入口也已提供 `realpose_<asset>_trial.py`，必须实际执行与验收后再计入物理通过名单。此启动方式直接初始化低位，不证明站姿到低位的完整过渡已验证。

交互查看器： https://g2-pose-lab.lofty-flame-4116.chatgpt.site 。其扭矩仍是无接触、无载荷的模型自重估算；可保留旧姿态预设作对照。
