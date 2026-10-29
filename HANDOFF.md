# HANDOFF

更新时间：2026-10-29（Asia/Shanghai）
当前实验分支：`codex/go2-online-smooth-20261027`
服务器源码工作树：`/data20t/embodied/zkj/inter-nav-visual-skill-20261028`；完整仿真使用独立、不可变源码工作树。
上一版发布分支：`codex/go2-interactive-navigation-20261026`（提交 `a2a07ab`）

## 最新检查点：原始 T04/T05 仍未通过（2026-10-29）

保留原尺寸 Go2 / MV7、1 cm 地面、空地图在线 LiDAR、全局 1 Hz / 局部 10 Hz、原 policy 50 Hz / DC 200 Hz、23.5 Nm 限矩；交互资格用 B01–B07 真值白名单，箱体几何受可见性门控。原始坐标：T04 `[2.11,-0.78] → [10.3,1.75]`，T05 `[13.6,-0.4] → [3.3,1.4]`。全程倾角 <30°，最终实测停稳误差 <0.25 m，五路完整 25 fps 视频。

最新 T04 v85（3dbc909）也未通过：15277 步，B06 侧移 0.620681 m，`no_safe_observed_retreat`，目标误差 1.326674 m，最大倾角 22.47435°；五路视频均为 3111 帧 / 25 fps / 124.44 秒，下载到本机同名目录后 SHA256 全部一致。原任务没有完成穿越或到点。

v85 在 118.135 秒探测到完整通路，随后停稳复核失败并返回撤离。125.710 秒保存地图显示实际机身包含 3 个已知占据采样点，未知格和箱体重叠均为 0。原后退方向不安全，现有 `ForwardExplorationPathTracker` 的避障恢复能找到约 `[10.625,0.025]` 的安全退让位置；原侧移请求约 `[-0.0068,0.35,0]`，完整机身平移检查通过，地图未改动。详见结果根目录 `v88-release-recovery-probe.json`。

候选新增 `recover_release_pose`：仅当已真实推箱、新观测曾证明通路、退出箱体接触并实测停稳后，才允许在姿态漂移导致机身占据重叠时调用原避障恢复。保留原未知格、箱体与机身约束及加速度限制；当前位置和实测停止扫掠都安全后，仍要返回停稳状态并重新检查新地图。恢复单次限 12 秒，次数有界，不能从一次几何证书直接跳到成功。新增实际在线地图回归夹具仅供测试使用，运行导航不读取它。

`v88-release-recovery-tests.log`：522 passed，1 个既有 G1 相机测试 deselected。可选独立物理诊断 `tools/diagnose_go2_release_recovery.py --gpu 5 --fixture tests/fixtures/t04_release_body_overlap_v85.npz --output /absolute/new/diagnostic-directory` 会把局部已知静态格转换为 0.5 m 高矩形，使用固定箱体和原 Go2；它没有原 MV7 的完整几何、接触状态和在线传感器，不能替代任务验收。第一次冷启动诊断的站立漂移导致初始位置不满足恢复入口条件，未执行恢复；增加 `[0.10,0.09]` 初始偏移的第二次简化场景诊断进入了恢复，但 12 秒超时，最大倾角 10.046°，最终机身仍未通过地图检查。两次都不是恢复成功证据。这个格子重建场景不含原接触与关节状态；不再根据它调整原场景几何或安全阈值。当前候选只有代码与实际保存地图上的几何验证，真实场景复跑仍是最终验证。

此前结束的完整测试同样失败，不能作为任务成功：

- T04 v79（c2620b7）：B06 实际侧推约 0.825 m，撤离时技能超时；目标误差 1.569 m，最大倾角 22.474°，6556 帧 / 262.24 秒。
- T05 v84（b0a137f）：接触准备重试耗尽；目标误差 2.017 m，最大倾角 29.746°，4439 帧 / 177.56 秒。

两组五路 MP4 已保存到本机 `E:/inter-nav-results/go2-online-smooth-20261027/` 下同名运行目录，逐文件 SHA256 与服务器一致；视频格式、源码、空地图检查通过，任务、交互效果和通路穿越检查未通过。实时运行句柄见产物根目录 `VISUAL_SKILL_CURRENT_STATE.md`；本机视频索引 `ONLINE_VIDEO_STATUS.md`。

当前候选修复三处衔接：持续前进停止反馈保留到下一真实运动意图，准备恢复不会把它误当成接触恢复关闭；撤离时可按新观测确认完整通路已被推开，先停稳再复核；粗转向已有的实测停稳结果可以用于满足前后间距的侧向对齐衔接，省去重复静止等待。推箱完成仍须由机器人实际穿过通路、到达原目标，不能把几何证书当作完成。

T05 v84 最后三次粗转向在 167.825 / 173.400 / 178.510 秒完成，鼻端间距约 0.472–0.475 m、侧向误差约 0.415 m。之后重复等待期间仍有约 0.10–0.14 rad/s 的朝向漂移，几乎没有侧向动作就触发 0.18 rad 对齐上限。新衔接保留原转向停稳、前后间距和完整机身碰撞检查，后续侧移仍使用原加速度限制、航向保持、实测动量检查、接触精度和倾角上限。

`v87-turn-handoff-tests.log`：513 passed、1 个既有 G1 相机测试 deselected。`preparation-handoff-v87-flat` 独立物理对照中，两种方式都完成接触准备：原方式 6.05 秒 / 最大 9.77°，复用停稳结果方式 5.11 秒 / 最大 10.35°；新方式侧向误差 0.0033 m、航向误差 0.00027 rad。该平地试验没有复现旧场景的全部漂移，不能据此宣布原任务故障已解决。保存在线地图上的三个原姿态均能发出通过原几何检查的侧移指令，但该快照并非每个时刻的地图，也不能替代物理复跑。

独立接触准备对照命令（已知平地诊断，只到接触前，不推箱，也不算 MV7 任务）：

```bash
source /data20t/embodied/share/inter-nav/env.sh
export PYTHONPATH="$PWD:$PYTHONPATH"
"$ISAAC_PYTHON" tools/diagnose_go2_preparation.py --gpu 5 --output /absolute/new/diagnostic-directory
```

完整场景运行命令见 [README 在线交互导航部分](README.md#go2-在线建图规划与交互导航)。下一轮仍需同版本原始 T04/T05 全流程、源码、五路视频与实际穿越验收。下面为历史检查点，旧进程状态已经过时。

## 历史检查点：未知地图与连续运动（2026-10-27）

入口改为 `grutopia/demo/go2_online_interactive_navigation.py`，具体运行命令在
[README 的在线交互导航部分](README.md#go2-在线建图规划与交互导航)，设计与验证记录见
[docs/go2_online_navigation.md](docs/go2_online_navigation.md)。

旧交互版本的全局碰撞网格确为预先已知。当前分支改为初始空地图、LiDAR 在线融合、
Voronoi 边界探索、1 Hz 全局层与 10 Hz 局部层。保留原 Go2 policy，恢复连续探索速度律，
并修复遗漏空量程/地面自由射线及不可执行目标假设引起的反复改路。
箱子识别仍使用激光可见性门控的仿真候选接口，定位仍是仿真位姿。

实验产物根目录：服务器 `/data20t/embodied/zkj/inter-nav-results/go2-online-smooth-20261027`；
本机 `E:\inter-nav-results\go2-online-smooth-20261027`。
新版本端到端复跑状态以在线导航说明为准。下方历史成功视频使用预生成地图，不能替代在线版本验证。

当前代码检查点 `f8ce43a`：每 5 ms 重新计算 DC 电机力矩，策略仍为原模型的 50 Hz 推理。
已加入急转弯前制动、恢复模式切换停稳、箱前航向反馈及小角度制动，412 项测试通过。
旧隐式驱动的 `full-online-v19` 只有 T01/T02/T03 通过录像验收，不能当作新版 DC 结果。
`full-online-v22` 最终只有 T01 通过新版录像验收（727 帧 / 25 fps / 29.08 秒），T03 失稳。
`online-v22-T02` 到点但有 4 次姿态恢复；T04 箱前对准超时。
`online-v23-T05` 侧推 0.648 m 并由新扫描确认了完整通路，随后因恢复控制在转向/后退间频繁切换超时，
这一状态切换问题已在 `cb91e07` 修复，等待端到端验证。
`full-online-v25` 的 T02 完整录像通过，但有 3 次恢复、峰值倾角 29.70°；T03 停稳恢复超时。
该批 T05 已在开始时停止，等待修正低速侧移控制后复跑。
独立物理试验发现 0.25 m/s 侧移未有效迈步；0.35 m/s 配合 0.65 s 制动预测和航向反馈完成了对准，
最终横向误差 2.69 cm、航向误差 0.70°、最大倾角 8.46°。它是控制器试验，不能计为导航任务成功。
`full-online-v29-contact` 正在以 `f8ce43a` 录制原始 T04/T05；尚待到点、全程健康及录像验证。
新版已验收视频入口为本机 `E:\inter-nav-results\go2-online-smooth-20261027\ONLINE_VIDEO_STATUS.md`。
索引当前选入 T01 的 DC 结果，以及较平稳的 T02/T03 原 PD 在线结果，逐行记录源码和控制方式。
这些是三个个体已验收任务，不代表单一配置统一通过五组；预先已知地图的录像不计入该索引。

## 历史记录（已知地图分支）

工作树 `/data20t/embodied/zkj/inter-nav-dynamic-go2-20261026`，分支 `codex/go2-dynamic-interaction-20261026`。

<a id="go2-interactive-navigation"></a>

## Go2 原尺寸 MV7：可交互箱子点到点导航（2026-10-26）

入口为 `grutopia/demo/go2_side_clearance_navigation.py`。场景和 Go2 均为原尺寸，辅助碰撞地面高 0.01 m；程序判断是否需要交互，需要时将挡路箱子侧推到通路一边，再穿过通路到达终点。本轮加入分层地图、交互收益显示和全机身转向检查；正式实跑结果见下，校准试验不能作为原始长距离任务成功记录。

<a id="go2-layered-results"></a>

### 本轮正式结果

五组均已选出个体审计通过的完整录像，采用两套已记录的源码/配置，**不是同一配置五组统一通过**。总索引见 [五组已验证录像](docs/go2_verified_runs.md)，本机入口为 `E:\inter-nav-results\go2-dynamic-interaction-20261026\VERIFIED_VIDEOS.md`。

**T04 和 T05 已使用同一版 v19 代码完成原始起终点任务，全部审计通过。** 两组都保持原尺寸场景、原 Go2 policy、1 cm 地面和完整材质；完整机身通过门口，停稳后误差小于 0.25 m，所有环境 tick 的健康检查通过。

| 任务 / 版本 | 控制步数 | 停稳后误差 | 每路录像 | 最终箱子侧移 |
|---|---:|---:|---:|---:|
| T01 / v19 | 7036 | 0.166725 m | 1459 帧，25 fps | 无推箱 |
| T02 / v23 | 5400 | 0.167428 m | 1132 帧，25 fps | 无推箱 |
| T03 / v19 | 3206 | 0.077250 m | 693 帧，25 fps | 无推箱 |
| T04 / v19 | 18484 | 0.124120 m | 3820 帧，25 fps | B06：0.653359 m |
| T05 / v19 | 10456 | 0.085938 m | 2215 帧，25 fps | B07：0.641608 m |

本机完整视频：

- T04：`E:\inter-nav-results\go2-dynamic-interaction-20261026\full-layered-v19-T04\T04\video\combined.mp4`
- T05：`E:\inter-nav-results\go2-dynamic-interaction-20261026\full-layered-v19-T05\T05\video\combined.mp4`

服务器目录为 `/data20t/embodied/zkj/inter-nav-results/go2-dynamic-interaction-20261026/full-layered-v19-T04` 和同一父目录下的 `full-layered-v19-T05`。各目录有 `validation.json`，任务子目录有结果、事件、轨迹、最终分层地图和五路完整 MP4。全体视频均已完整解码，清路、转向及终点画面已抽查。

T04 在第 15540 步通过实际通路检查，退出接触后再次确认通路可用，第 17005 步用更新地图重规划，第 18483 步停稳确认到达。B06 最终侧移 0.653359 m，沿通路仅偏移 0.014516 m；地图清除 758 个旧箱体体素，固定底图保持不变。记录轨迹长 24.51 m，发生 4 次接触重新对齐。

T05 第 7180 步通过清路检查，第 8803 步重规划，第 10455 步停稳确认到达。B07 最终侧移 0.641608 m，沿通路仅偏移 0.003095 m；清除 665 个旧箱体体素，固定底图保持不变。记录轨迹长 19.29 m，发生 1 次接触重新对齐。

两组交互的收益是恢复原本被箱子切断的连通性。初次决策预测的清障后路线分别为 T04 10.31 m、T05 11.99 m；实际轨迹包括接触准备、退让、转向及局部修正，不能直接把它们的差值当作节省距离。录像中的决策距离以当次决策时的机器人位置为起点，S/G 则始终标记任务原始起终点。当前交互执行仍有多次对齐，效率有改进空间。

旧 T05 v18 的清路检查越过原终点约 0.116 m，误碰终点后方固定障碍，因此主动停止。v19 将检查限定到原终点，并保留末端连接段、完整机身和 4 cm 余量检查，随后完成上述正式任务。几何诊断和运行机制见 [分层交互导航说明](docs/go2_layered_navigation.md)。

选定的 T01/T03/T04/T05 源码快照逐文件匹配 `e028847d8036e328e9867f39fda4aaf55f22af6c`，使用 `--path-controller pure`。T02 匹配 `f7e33cf606123e85399c1ff537e509041b2458a8`，使用 `--path-controller guarded --navigation-attitude-recovery`，视频位于 `full-layered-v23-all/T02/video/`。T02 的5659个运行 tick 全程健康，最高倾角29.929°、3次恢复，共450个保持 tick；五路视频及逐步姿态重算、地图、源码检查全部通过。

同一 v23 配置的 T01 于3867步失稳，峰值34.650°；T02完成后停止该统一试验，后面三组为 `stop_requested`。因此 `full-layered-v23-all/validation.json` 整批为 false，但 T02 个体检查及全部五路视频通过；其余四组使用表中的 v19 成功记录。新配置尚未证明普遍稳定，不能将单次通过解释为重复运行保证。旧失败、校准、快速无录像验证和主动停止视频均不计入选定结果。

### 连接服务器并运行一组推箱任务

在本机终端连接：

```bash
ssh zkj@162.105.195.166
```

随后在服务器的 Bash 中执行以下命令。示例进入已准备好的发布工作树；若自行 clone，请先检出上述分支，并把 `cd` 路径改为自己的仓库目录。示例运行原始起终点的 **T05**，使用 GPU 1；可按空闲 GPU 情况修改 `--gpu`。

```bash
cd /data20t/embodied/zkj/inter-nav-dynamic-go2-20261026
git switch --detach e028847
source /data20t/embodied/share/inter-nav/env.sh
export MALLOC_ARENA_MAX=2

go2_inputs=/data20t/embodied/zkj/inter-nav-results/go2-voronoi-heading-up-20261025/collision-corrected-inputs
go2_out="/data20t/embodied/zkj/inter-nav-results/go2-interactive-nav-$(date +%Y%m%d-%H%M%S)"

set -o pipefail
"$ISAAC_PYTHON" -u grutopia/demo/go2_side_clearance_navigation.py \
  --benchmark-dir "$go2_inputs" \
  --task-ids T05 \
  --gpu 1 \
  --floor-top 0.01 \
  --max-steps 30000 \
  --record-video --material-mode preview \
  --route-planner voronoi --navigation-heading path \
  --path-controller pure --roadmap-shape circle --route-clearance 0.30 \
  --voronoi-clearance-weight 0.30 --navigation-half-width 0.22 \
  --global-map-hz 1 --local-map-hz 10 --local-map-radius 3 \
  --third-person-height 1.35 --third-person-forward 0.25 --third-person-fov 100 \
  --push-max-speed 0.95 --turn-speed 1.0 \
  --contact-centering-speed 0.25 --contact-yaw-tolerance 0.10 --contact-front-gap 0.40 \
  --output "$go2_out" 2>&1 | tee "${go2_out}.log"
```

使用共享环境的 `$ISAAC_PYTHON`，GPU 通过 `--gpu` 选择，不设置 `CUDA_VISIBLE_DEVICES`。每次使用新的输出目录。上述命令前台运行，日志同步保存为 `${go2_out}.log`。

完整场景录制建议启动前至少有 14000 MiB 空闲显存。可用 `nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv` 选择显存余量充足的卡。本轮队列还监控渲染日志；出现 `ERROR_OUT_OF_DEVICE_MEMORY` 会请求停止并将录制标为无效，即使物理仿真仍能继续运行也不作为完整可视化结果。

输入目录已包含箱子布局、门口通道、修正后的碰撞栅格及任务：`scene_profile.json`、`layout.json`、`passages.json`、`navigation_grid.npz`、`original_metadata.json`、`tasks_preflight.json`，另有地面配置和碰撞修正审计。场景配置引用服务器资产 `/data20t/embodied/zkj/inter-nav-results/rectangular-boxes-five-scenes-xy100-20261021-v2/mv7/after_scene.usda`；复现时需要保留这些外部数据和共享 Go2 模型/policy。

### 任务选择和主要参数

这些是同一个 MV7 场景中的五组点到点任务；坐标单位为米，起点/终点在地图上显示为 S/G。下表交互类型为当前初始布局下的决策，运行时会根据实际位姿重新判断。

| 任务 | 起点 `(x, y)` | 终点 `(x, y)` | 初始决策 |
|---|---|---|---|
| T01 | `(2.11, -0.78)` | `(12.0, -0.5)` | 普通导航 |
| T02 | `(13.6, -0.4)` | `(2.11, -0.78)` | 普通导航 |
| T03 | `(7.74, -4.7)` | `(8.0, 4.2)` | 普通导航 |
| T04 | `(2.11, -0.78)` | `(10.3, 1.75)` | 侧推 B06 |
| T05 | `(13.6, -0.4)` | `(3.3, 1.4)` | 侧推 B07 |

- 只运行两组推箱任务：将命令中的 `--task-ids T05` 改为 `--task-ids T04,T05`。
- 五组复现按上表选择源码/配置。T02 将示例的源码版本改为 `f7e33cf`、任务改为 `T02`、控制改为 `guarded --navigation-attitude-recovery`；T01/T03/T04/T05 保持 `e028847` + `pure`。列表参数支持批量运行，但本轮没有同一配置五组统一通过的结论。`--max-steps 30000` 是每组任务的控制循环上限，录制索引还包含初始化后的停步/对齐帧，二者不能直接等同。
- 无录像的物理调试：去掉 `--record-video`，将 `--material-mode preview` 改为 `--material-mode simple`。
- 普通导航复用纯探索的 Voronoi 骨架规划和原路径跟随控制，机身朝向随路径变化。规划图使用已有碰撞先验和仿真箱子位姿，LiDAR 在线建图同时运行；图不连通时的栅格回退记录在结果中。
- `--contact-centering-speed 0.25` 是接触对齐的纵向速度请求；当前控制器横向纠偏为 0.35 m/s、后退至少 0.35 m/s。`--contact-front-gap 0.40` 为鼻端预接触目标间距，转向容差 `0.10` 的单位是 rad。接触平移预测制动窗口为 0.40 s，由入口显式传入并记录在录像配置中；原位姿容差、步态 policy 与场景尺寸保持不变。各组已有表中通过记录；尚未证明同一配置五组稳定复现。
- 录像为完整 25 fps，第三视角机身上方 1.35 m、前移 0.25 m、俯角 90°、水平 FOV 100°，狗前方朝画面上方并随实际 yaw 转动。降低相机用于减少灯具遮挡；没有改变场景物体的可见性或碰撞属性。
- 同一时间只启动一个完整场景进程；通过 `--task-ids` 列表可在进程内顺序运行多组，每组复位机器人、七个箱子、地图、相机和记录器。本轮 T04/T05 单独进程，后续 T01/T02/T03 使用列表逐组复位运行。

### 分层地图与交互解释

详细数据流和限制见 [Go2 分层交互导航](docs/go2_layered_navigation.md)。

- 固定碰撞先验只读保存；1 Hz 全局层更新箱子布局和持续观测到的稳定障碍，供 Voronoi 全局规划使用。
- 10 Hz 局部层维护机器人周围 6 m × 6 m 窗口，融合 LiDAR 占据/自由射线，供局部避障与全机身运动检查使用。临时证据过期不影响墙体或已跟踪箱子。
- 箱子旧位置、新位置分别维护；腾空区去除旧箱子证据，保留重叠物体。侧推结束触发即时全局刷新，机器人从真实当前位置规划穿门路线。
- 每次决策记录不交互的路线、侧停后的预测路线、接近距离、推动距离和代价。原本不连通时显示“恢复连通”，不输出无穷大的距离收益。
- 地图显示 S/G、箱子编号、推向侧边的箭头、当前路线、预测路线、局部碰撞图与地图版本号。这里的箱子身份/尺寸/位姿由仿真提供，尚未实现现实场景中的自主箱子识别与跟踪。

### 输出与结果判断

运行输出保存于 `$go2_out`：

```text
summary.json                       # 各任务 success、reason 和完成情况
recording_config.json               # 本次实际录像/控制配置
source/、source_manifest.json        # 启动源码快照及哈希
T05/result.json                     # 本组结果、侧推/穿门状态、规划统计
T05/events.jsonl、trajectory.json    # 事件和轨迹
T05/layered_map.npz、layered_map.json # 最终分层地图及更新统计
T05/map_events.jsonl                # 物体移动、全局刷新等地图事件
T05/video/combined.mp4              # 四画面完整录像
T05/video/third_person.mp4          # 头朝上第三视角
T05/video/robot_rgb.mp4             # 第一视角
T05/video/map_topdown.mp4           # 带 S/G 的俯视地图
T05/video/pointcloud_map.mp4        # LiDAR 点云
```

多任务会分别生成 `T01/` 至 `T05/` 子目录。`summary.json` 的 `complete` 表示本批任务已经结束；判断导航成功需看各任务的 `success`、`reason`，推箱任务还应核对箱子侧停、清路和 `full_body_passage_crossed`。退出码 0 或视频正常保存不代表成功。

上一版 `full-voronoi-final-remaining-v19` 的结果：T01 的独立正式录像已成功；箱前局部 T05 校准成功（4531 步、终点误差约 0.031 m），但该校准修改了起点且未录像。该旧流程的原始起终点批次中，T04 侧推 B06 约 0.808 m 并清路，随后在 `cross_passage` 阶段失稳；T05 侧推 B07 约 0.655 m，尚未清路，在 `push_center` 阶段失稳。该批次 T02/T03 也未成功，不能将这批旧结果描述为五组全部成功。

正式 v19 的完整诊断录像和结果：

- 服务器：`/data20t/embodied/zkj/inter-nav-results/go2-voronoi-heading-up-20261025/full-voronoi-final-remaining-v19/{T02,T03,T04,T05}/video/`。
- 本机：`E:\inter-nav-results\go2-voronoi-heading-up-20261025\full-voronoi-final-remaining-v19\{T02,T03,T04,T05}\video\`。
- T01 已验证录像：本机 `E:\inter-nav-results\go2-voronoi-heading-up-20261025\T01-current-verified\T01\video\`。

### 场景准备与代码范围

`grutopia/demo/prepare_interactive_box_scenes.py` 和 `grutopia_extension/interactive_navigation/interactive_box_scenarios.py` 保存五场矩形箱子布局及 USD 叠加层的生成逻辑；`grutopia/demo/render_interactive_box_previews.py` 提供前后对比图与物理检查。这些准备工具使用服务器保留的 20261019 五场 profile、原始资产和架构栅格。上面的导航命令直接复用已生成的 MV7 输入目录。

本次推送包含 Go2 导航、箱子交互、可配置辅助地面、录像及相关测试。机器人/场景资产、模型权重、生成的输入数据和视频继续保存在上述服务器及 E 盘目录，不随 Git 推送。仅 clone 仓库还需要安装共享环境并取得这些外部数据。

`grutopia/demo/go2_walk_demo.py` 提供无路径规划的恒速步态诊断，可用于区分 locomotion 与导航控制问题。

### 上一版发布验证（a2a07ab）

当前 v23 代码自动测试为 326 passed、1 deselected；T02 正式完整录像个体审计通过，v23 的 T01 回归失败，五组交付按上表选取两套配置的成功录像。T04/T05 v19 均已完成原始起终点导航并通过全部审计。下面保留上一版发布的验证信息，便于区分代码检查与物理导航结果。

在基于 `origin/main` 的独立 Go2 发布工作树中，265 项测试通过。主分支已有的 `test_g1_third_person_is_close_and_elevated` 依赖未包含在该分支的 G1 配置文件，本次未运行该项；其余测试均通过。测试日志和命令预检记录位于服务器结果目录的 `publish-20261026/`。

上一版发布时，T05 单组与全部五组的文档命令均通过 `--preflight-only` 检查，确认参数、输入数据及所选原始起终点可加载；该预检不替代本轮正式实跑结果。录像编码依赖 `imageio-ffmpeg`，已补入 `requirements/runtime.txt`。
