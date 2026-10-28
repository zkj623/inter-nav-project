# Go2 视觉识别与 PushAsideSkill

使用原尺寸 Go2、原尺寸 MV7、1 cm 辅助地面和原始任务起终点。占据图从空地图开始，经 LiDAR 在线构建，全局 1 Hz、局部 10 Hz。

## 识别与交互决策

1. LiDAR 命中后，几何跟踪器提供待识别对象的位置和尺寸；类别为 `unknown`，不可交互。
2. 第一视角 RGB 经 GroundingDINO 提出候选框，MobileSAM 得到掩膜；掩膜内 RGB-D 点与几何跟踪对象关联。
3. 候选裁剪交给 Qwen3-VL 分类。DINO 查询词、仿真对象名和布局标签不能授予交互权限。
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
  --floor-top .01 --map-source online --actuator-mode implicit_pd --path-controller continuous \
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

## 扩展类别

编辑 `grutopia/demo/profiles/interactive_object_catalog.json` 中的 `recognition_labels`、`detection_terms`、`enabled` 和 `skills`。标签会传给视觉分类器。通用 `box` 与细分类共同参与识别，避免无法判断材质时拒绝明确的箱体形状。纸箱、塑料收纳箱、木箱归一为 `box`；柜子不归为箱子。

新增类别还必须提供支持该类别的技能；仅加白名单不会让 `PushAsideSkill` 自动支持未知物体。

## 验收状态

新流程的 434 项回归检查通过。T04/T05 的含视觉识别物理验收尚未完成；旧的无识别或已知地图成功视频不能作为本版本的验收证据。
