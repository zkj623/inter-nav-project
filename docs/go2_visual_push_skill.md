# Go2 视觉识别与 PushAsideSkill

使用原尺寸 Go2、原尺寸 MV7、1 cm 辅助地面和原始任务起终点。占据图从空地图开始，经 LiDAR 在线构建，全局 1 Hz、局部 10 Hz。

## 识别与交互决策

1. LiDAR 命中后，几何跟踪器提供待识别对象的位置和尺寸；类别为 `unknown`，不可交互。
2. 第一视角 RGB 经 GroundingDINO 提出候选框，MobileSAM 得到掩膜；掩膜内 RGB-D 点与几何跟踪对象关联。
3. 保留 SAM 前景的原始像素、弱化背景干扰后，将候选裁剪交给 Qwen3-VL 分类；同时保存未弱化背景的裁剪供复核。DINO 查询词、仿真对象名和布局标签不能授予交互权限。
4. 至少两个不同渲染帧得到一致类别，检测分数不低于 0.30，才查询白名单。目前仅 `box` 支持 `push_aside`。
5. 目标方向上出现附近的未识别障碍时，即使目标区域尚未观测，也先到约 1.5–2.8 m 的可达观察点，朝向物体取得识别证据。未确认仍视为障碍。
6. 比较绕行代价与侧推代价：`接近路径长度 + 2.5 × 推动距离 + 推后剩余路径长度`。地图显示恢复连通或预计路径缩短的收益。

识别请求异步执行，物理控制持续运行。证据按任务隔离；重复帧不增加支持数，明确类别冲突会重新计数，证据默认在 60 s 仿真时间内有效。

**仿真接口边界：** 类别来自实际相机模型推理；实例 ID、几何尺寸、定位和移动后的位姿跟踪仍借助仿真接口，并受 LiDAR 可见性门控。这不是完全摆脱仿真真值的实机感知系统。

## 技能执行契约

`interaction_skill.py` 定义技能注册、上下文和速度/导航命令接口。`push_aside_skill.py` 实现 `push_aside/1.0`：

- 前置条件：目标 ID 有至少两帧视觉证据，类别及技能在白名单中，存在侧向停放方案。
- 准备：到允许完整机身转向的位置，停稳后转向，再分轴对准前后距离和横向位置。
- 接触与推动：先低速接触，再逐渐增加速度请求；反馈包括实际速度、航向、横向误差和箱子位移。
- 恢复：对准偏差、姿态异常或箱子停滞时制动、退开、重新准备；有次数和总时限约束。
- 效果验证：完成规划的侧向停放位移后退开，以新扫描确认全局路线和局部完整机身通路；根据新路线修正通路方向，必要时前往通路入口重新观察；不能直接将旧占据格擦成自由空间。
- 任务结束：技能成功后经过通路入口到原始目标。必须完整机身穿过通路、停稳误差小于 0.25 m，全程不触发 30° 硬健康阈值。

技能只输出导航子目标或速度，不设置机器人或物体位姿。位姿重置只用于任务初始化。

## 运行 T04、T05

需要已启动的 DINO / SAM / Qwen 服务（端口 12181 / 12183 / 12185）。在仓库根目录运行；GPU 按可用资源选择。完整材质场景预留至少 92 GiB 内存及 14 GiB 显存。

```bash
source /data20t/embodied/share/inter-nav/env.sh
export MALLOC_ARENA_MAX=2
export PYTHONPATH="$PWD:$PYTHONPATH"
go2_inputs=/data20t/embodied/zkj/inter-nav-results/go2-voronoi-heading-up-20261025/collision-corrected-inputs
go2_out="/data20t/embodied/zkj/inter-nav-results/go2-visual-skill-$(date +%Y%m%d-%H%M%S)"
"$ISAAC_PYTHON" -u grutopia/demo/go2_online_interactive_navigation.py \
  --benchmark-dir "$go2_inputs" --output "$go2_out" --task-ids T04,T05 --gpu 3 --max-steps 40000 \
  --floor-top .01 --map-source online --actuator-mode explicit_dc --path-controller continuous \
  --navigation-attitude-recovery --route-planner voronoi --navigation-heading path \
  --roadmap-shape circle --route-clearance .30 --navigation-half-width .22 --voronoi-clearance-weight .30 \
  --global-map-hz 1 --local-map-hz 10 --local-map-radius 3 \
  --interaction-catalog grutopia/demo/profiles/interactive_object_catalog.json \
  --push-max-speed .95 --turn-speed 1.0 --contact-centering-speed .25 --contact-lateral-speed .35 \
  --contact-yaw-tolerance .10 --contact-front-gap .35 \
  --record-video --material-mode preview \
  --third-person-height 1.35 --third-person-forward .25 --third-person-fov 100
```

每组输出：

- `recognition/`：原始 RGB、分类裁剪、逐帧模型输出、`recognition_events.jsonl`、`recognized_objects.json`。
- `events.jsonl`：观察、识别、技能开始/阶段切换/结束及任务事件。
- `result.json`：识别来源、技能报告、实际位移、通路证书、最终误差和全程健康。
- `video/combined.mp4`：完整组合视频；同目录还有第一视角、第三视角和两种地图视频，25 fps。

地图中未识别对象为灰色，已识别且允许交互的箱子为金色。面板显示类别、技能阶段、重试次数和路径收益。

## 关闭纹理做开发测试

先在完整原尺寸 MV7 场景中测试控制、交互和在线地图，可使用上方命令并将 `--material-mode preview` 改为 `--material-mode simple`，去掉 `--record-video`。
第一视角 RGB-D 相机和视觉识别继续运行；墙体、家具、箱子的碰撞几何不变。纯色材质模式保留物理材质绑定，箱子的质量、摩擦和纯色外观保持原配置。

```bash
# 无纹理测试：在上方完整命令中使用这组开关，并去掉 --record-video
--material-mode simple

# 测试通过后，正式录像重新使用
--material-mode preview --record-video
```

无纹理开发测试不能替代开启纹理后的视觉识别和完整视频验收。内存峰值需要实际运行测量；不能把启动时的空闲内存门槛当作进程实际占用。

## 扩展类别

编辑 `grutopia/demo/profiles/interactive_object_catalog.json` 中的 `recognition_labels`、`detection_terms`、`enabled` 和 `skills`。标签会传给视觉分类器。通用 `box` 与细分类共同参与识别，避免无法判断材质时拒绝明确的箱体形状。纸箱、塑料收纳箱、木箱归一为 `box`；柜子不归为箱子。

新增类别还必须提供支持该类别的技能；仅加白名单不会让 `PushAsideSkill` 自动支持未知物体。

## 验收状态（2026-10-28）

新流程的 444 项回归检查通过。在独立的小型 Isaac Sim 场景中，5 个真实渲染帧经过 DINO、SAM、RGB-D 关联和 Qwen3-VL 后被识别为 `box`；系统在第二个独立帧确认后才授予权限。未知类别未被默认放行。此检查仅验证识别接口，不是原始 T04/T05 的导航或推箱验收。

完整原尺寸 MV7 的无纹理录像已完成一次（`recognition-skill-v42-notexture-video-try1`）：采样进程 RSS 峰值18.88 GiB，T04/T05 的五路视频均25 fps并通过逐帧解码及时序检查，但两组均在观察子导航/转向阶段失稳，尚未启动推箱技能。T04最大倾角35.03°、T05为46.82°，均未通过任务验收。

已补齐观察子导航的姿态恢复保护，并用实测线速度、角速度、位置和倾角稳定窗口控制到点/转向切换。识别现在使用前景强调的裁剪；主动观察必须取得新视点的两帧证据，未知结果会换视点，非白名单判断需复核两个主动视点。回归检查444项通过。

下轮继续保存无纹理完整录像，使用固定200 Hz的既有DC执行器接口及原50 Hz Go2 policy，检查原PD接口中出现的转向失稳问题。policy权重、30°硬健康阈值、0.25 m最终到点误差、场景尺寸和原任务坐标不变。此复测尚未验收通过；旧成功视频不能替代新流程的T04/T05验收。
