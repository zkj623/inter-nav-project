# 长方篮：自然镜像夹持

原方案左右末端使用相同朝向，而非镜像朝向，且两臂分别求逆解，造成左肩/腕绕转明显。新方案保留篮子中心和偏航0°、双侧篮壁上方夹持，用镜像夹爪朝向，并从共同低位初姿连续求解；规划关节按[-1,+1,-1,+1,-1,+1,-1]镜像，实际载荷反馈仍允许两侧微调。

选择mirror_right：沿用已验证右臂逆解并镜像到左臂，两腕同形态、双肘对称。更收肘的mirror_home_seed候选在保持末段触发LOADED_VISIBLE_SURFACE_GAP，未采用，失败记录保留。静态图片是规划姿态预览，实际成功由完整物理轨迹独立复核。固定初姿、身体、头部、底盘、物体、质量、材料和夹爪控制器均沿用上一轮；只调整篮子双臂目标与路径。

独立验收：保持最低抬升19.29cm，最大倾斜0.75°，四指接触比例1.000。该版本通过次数1，不代表随机摆放鲁棒性。当前12/12类仍有成功名义初态案例。

重建：`python iterations/20261124_basket_mirror/source/basket_mirror_workspace.py --workspace /path/to/work --package /path/to/g2_sim`。
测试：`python realpose_basket_rectangular_trial.py --asset basket_rectangular --plan-dir basket_mirror_plans --gpu 5`，再运行`python verify_realpose.py --trial-dir realpose_trials/basket_mirror_plans/basket_rectangular/candidate0_repeat1`。

原始视频/记录：`/data20t/embodied/zkj/inter-nav-results/g2-basket-posture-20261124/realpose_trials/mirror_right/basket_rectangular/candidate0_repeat1`。对比图片：`/data20t/embodied/zkj/inter-nav-results/g2-basket-posture-20261124/posture_preview`。可执行姿态以plan中的q_*为准，candidate内点位为历史特征估计。未改变其他11类，也未修改旧240条数据。
