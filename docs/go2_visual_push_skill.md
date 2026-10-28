# Go2 交互资格来源与 PushAsideSkill

使用原尺寸 Go2、原尺寸 MV7、1 cm 辅助地面和原始任务起终点。占据图从空地图开始，经 LiDAR 在线构建，全局 1 Hz、局部 10 Hz。

## 当前开发模式：交互资格使用真值

按当前测试要求，默认 `--interaction-source ground_truth`：从场景 `layout.json` 读取已知箱子的交互能力，只有声明 `interactions` 包含 `push`、未被禁用且类别在白名单中的对象才能调用 `push_aside`。当前原始场景 B01–B07 都声明支持推。

真值模式跳过视觉分类和主动观察绕行，不需要 DINO/SAM/Qwen 服务。箱子的类别与交互资格已知，空间地图仍从空地图在线构建；规划只使用已被 LiDAR 观测到的几何对象。初始化不会把整屋地图或所有箱子位置预填给规划器。

运行输出 `interaction_ground_truth.json` 和每组 `interaction/known_objects.json`，记录真值来源、布局哈希和允许的技能。技能许可明确标记 `ground_truth_layout`，视觉帧与分类器记录为空，不伪造视觉识别证据。

需要恢复视觉识别时使用 `--interaction-source vision`，流程如下。

## 视觉模式：识别与交互决策

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

- 前置条件：目标 ID 有明确许可来源（真值清单，或至少两帧视觉证据），类别及技能在白名单中，存在侧向停放方案。
- 准备：到允许完整机身转向的位置，停稳后转向，再以不超过 0.05 rad 的航向误差完成精调并停稳，随后分轴对准。前后方向接受鼻端的安全距离区间，横向依据实际鼻端误差，不追固定身体中心。
- 接触与推动：先低速接触，再逐渐增加速度请求；反馈包括实际速度、航向、横向误差和箱子位移。
- 恢复：对准偏差、姿态异常或箱子停滞时制动、退开、重新准备；有次数和总时限约束。
- 效果验证：完成规划的侧向停放位移后退开，以新扫描确认全局路线和局部完整机身通路；根据新路线修正通路方向，必要时前往通路入口重新观察；不能直接将旧占据格擦成自由空间。
- 任务结束：技能成功后经过通路入口到原始目标。必须完整机身穿过通路、停稳误差小于 0.25 m，全程不触发 30° 硬健康阈值。

技能只输出导航子目标或速度，不设置机器人或物体位姿。位姿重置只用于任务初始化。

## 运行 T04、T05

真值模式不依赖视觉模型服务；视觉模式需要 DINO / SAM / Qwen 服务（端口 12181 / 12183 / 12185）。在仓库根目录运行；GPU 按可用资源选择。完整材质场景预留至少 92 GiB 内存及 14 GiB 显存。

```bash
source /data20t/embodied/share/inter-nav/env.sh
export MALLOC_ARENA_MAX=2
export PYTHONPATH="$PWD:$PYTHONPATH"
go2_inputs=/data20t/embodied/zkj/inter-nav-results/go2-voronoi-heading-up-20261025/collision-corrected-inputs
go2_out="/data20t/embodied/zkj/inter-nav-results/go2-visual-skill-$(date +%Y%m%d-%H%M%S)"
"$ISAAC_PYTHON" -u grutopia/demo/go2_online_interactive_navigation.py \
  --benchmark-dir "$go2_inputs" --output "$go2_out" --task-ids T04,T05 --gpu 3 --max-steps 40000 \
  --floor-top .01 --map-source online --interaction-source ground_truth --actuator-mode explicit_dc --path-controller continuous \
  --navigation-attitude-recovery --route-planner voronoi --navigation-heading path \
  --roadmap-shape circle --route-clearance .30 --navigation-half-width .22 --voronoi-clearance-weight .30 \
  --global-map-hz 1 --local-map-hz 10 --local-map-radius 3 \
  --interaction-catalog grutopia/demo/profiles/interactive_object_catalog.json \
  --push-max-speed .95 --turn-speed 1.0 --contact-centering-speed .25 --contact-lateral-speed .35 \
  --contact-yaw-tolerance .10 --contact-front-gap .35 \
  --record-video --material-mode simple \
  --third-person-height 1.35 --third-person-forward .25 --third-person-fov 100
```

每组输出：

- 视觉模式的 `recognition/`：原始 RGB、分类裁剪、逐帧模型输出、`recognition_events.jsonl`、`recognized_objects.json`。
- `events.jsonl`：观察、识别、技能开始/阶段切换/结束及任务事件；`skill_feedback` 记录鼻端间距、鼻端横向误差、航向误差、机身实测速度、倾角、准备位置、技能输出和实际下发命令。
- `result.json`：识别来源、技能报告、实际位移、通路证书、最终误差和全程健康。
- `video/combined.mp4`：完整组合视频；同目录还有第一视角、第三视角和两种地图视频，25 fps。

地图中可交互箱子为金色，其余对象为灰色。面板注明交互资格来源（GT 或 RGB）、类别、技能阶段、重试次数和路径收益。

## 关闭纹理做开发测试

上方命令默认在完整原尺寸 MV7 中使用 `--material-mode simple --record-video`，录制无纹理的完整测试视频。需要节省录像开销时可去掉 `--record-video`；正式贴图录像改用 `--material-mode preview --record-video`。
视觉模式仍运行第一视角 RGB-D 识别；真值模式跳过识别，录像相机仅用于展示。墙体、家具、箱子的碰撞几何不变。纯色材质模式保留物理材质绑定，箱子的质量、摩擦和纯色外观保持原配置。

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

最新控制修订的自动回归检查为 **464 passed、1 deselected**；未运行的 G1 相机测试与本分支的 Go2 任务无关。物理任务仍未验收通过，单元测试不代表 T04/T05 成功。

持续侧移的 v45/v46 记录中，平均指令约 0.348 m/s、实测约 0.306–0.307 m/s，片段峰值倾角 8.623°。这只表明持续侧移可执行，不能证明低速、短距离纠偏或负载接触同样稳定。v47 的 T04 在第二次接触准备后失稳：短前移指令从 0.25 m/s 突然归零时，实测仍向前约 0.16 m/s，随后持续前倾；原目标未到达。T05 仍出现接触前纵向对准反复启停。

据此加入以下修复，准备在 v48 重新验证：技能常规停下请求通过原有速度变化限制器和机身/惯性碰撞检查；提前制动一旦触发，必须完整停稳再请求下一次步伐，停止距离还计入命令降速所需的行程；停稳必须同时满足命令、实测速率、位置、高度与倾角稳定窗口；短准备子目标减速半径缩为 0.30 m，避免在到达阈值外降入无效低速；大转向后单独精调航向，安全接触距离内不再追固定前后位置。健康失败或倾角达到 18° 的保护停止仍立即归零，30° 硬健康阈值不变。这些修改尚未通过完整物理复跑，不能称导航失稳已解决。

### 先前试验记录

最新安排：暂用真值交互资格，保留原 policy、DC 控制、无纹理完整录像，重点验证导航稳定性与推箱执行。导航失稳问题已做代码修复，但原始 T04/T05 尚未验收通过。

新流程的 448 项回归检查通过。在独立的小型 Isaac Sim 场景中，5 个真实渲染帧经过 DINO、SAM、RGB-D 关联和 Qwen3-VL 后被识别为 `box`；系统在第二个独立帧确认后才授予权限。未知类别未被默认放行。此检查仅验证识别接口，不是原始 T04/T05 的导航或推箱验收。

完整原尺寸 MV7 的无纹理录像已完成一次（`recognition-skill-v42-notexture-video-try1`）：采样进程 RSS 峰值18.88 GiB，T04/T05 的五路视频均25 fps并通过逐帧解码及时序检查，但两组均在观察子导航/转向阶段失稳，尚未启动推箱技能。T04最大倾角35.03°、T05为46.82°，均未通过任务验收。

已补齐观察子导航的姿态恢复保护，并用实测线速度、角速度、位置和倾角稳定窗口控制到点/转向切换。识别现在使用前景强调的裁剪；主动观察必须取得新视点的两帧证据，未知结果会换视点，非白名单判断需复核两个主动视点。回归检查448项通过。

下轮继续保存无纹理完整录像，使用固定200 Hz的既有DC执行器接口及原50 Hz Go2 policy，检查原PD接口中出现的转向失稳问题。policy权重、30°硬健康阈值、0.25 m最终到点误差、场景尺寸和原任务坐标不变。此复测尚未验收通过；旧成功视频不能替代新流程的T04/T05验收。
