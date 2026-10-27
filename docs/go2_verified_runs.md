# Go2 五组已验证录像与复现版本

五组原始起终点任务各有一份完整录像通过个体审计；采用两套源码和控制配置，不能解释为同一配置五组统一通过。

| 任务 | 源码提交 | 控制配置 | 停稳误差 | 控制步数 | 录像目录（相对结果根目录） |
|---|---|---|---:|---:|---|
| T01 | `e028847` | pure | 0.167 m | 7036 | `full-layered-v19-rest/T01/video/` |
| T02 | `f7e33cf` | guarded + navigation-attitude-recovery | 0.167 m | 5400 | `full-layered-v23-all/T02/video/` |
| T03 | `e028847` | pure | 0.077 m | 3206 | `full-layered-v19-rest/T03/video/` |
| T04 | `e028847` | pure | 0.124 m | 18484 | `full-layered-v19-T04/T04/video/` |
| T05 | `e028847` | pure | 0.086 m | 10456 | `full-layered-v19-T05/T05/video/` |

结果根目录：

- 本机：`E:\inter-nav-results\go2-dynamic-interaction-20261026`
- 服务器：`/data20t/embodied/zkj/inter-nav-results/go2-dynamic-interaction-20261026`

各目录的 `combined.mp4` 是完整四画面，另有 `third_person.mp4`、`robot_rgb.mp4`、`map_topdown.mp4`、`pointcloud_map.mp4`，均为25 fps。本机 `VERIFIED_VIDEOS.md` 提供可点击入口，`verified_video_manifest.json` 保存25段视频的SHA256、帧数及单组审计状态。

T04/T05 侧推B06/B07后通过原门口，固定地图保留，旧箱体占据及点云被清理。两例交互收益为恢复原本断开的路径连通性。第三视角为机身上方1.35m、前移0.25m、90°俯视、FOV100°，随狗实际朝向转动。

v23 统一试验的T01失稳，T02完成后其余三组主动停止，整批审计为false。选定的T02个体及五路视频审计全部通过，最高倾角29.929°，接近30°失稳阈值；新配置尚未证明普遍稳定。四组继续使用v19成功记录。以上是单次通过证据，不是重复运行成功率或现实机器人部署结论。

源码快照逐文件匹配 `e028847d8036e328e9867f39fda4aaf55f22af6c` 和 `f7e33cf606123e85399c1ff537e509041b2458a8`；两者均在当前实验分支历史中。每组保留 `recording_config.json`、`source/`、`source_manifest.json`、结果、事件及地图。

复现命令见 [HANDOFF](../HANDOFF.md#go2-interactive-navigation)，地图与交互方案见 [设计说明](go2_layered_navigation.md)。箱子身份、尺寸和位姿来自仿真。
