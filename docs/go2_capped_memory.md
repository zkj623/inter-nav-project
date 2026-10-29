# Go2 完整场景的受限内存试跑

这是资源调度模式，导航代码、原 Go2/MV7、1 cm 地面、原始坐标、空图在线 LiDAR、原策略/DC 和五路25fps录像保持一致。

不设硬上限的串行模式仍按48GiB可用内存启动。受限模式使用用户自己的 systemd scope，将单次仿真及其编码子进程限制为24GiB，禁用交换分区；主机40GiB可用内存稳定60秒后才启动，保留16GiB余量。队列仍保持单个完整场景，主机持续低于16GiB时停止本任务。

探针已确认内核实际设置为25769803776字节、交换上限为0，并确认exec保持PID及独立进程组、退出信号可正确传递。没有做内存压力测试。证据见 [scope 探针](data/go2_capped_scope_probe_20261029.json)。24GiB能否覆盖完整场景和录像导出仍由真实运行验证；超限不能记作任务成功，不会自动降低场景规模或减少录像。

本轮v94保持93ae182导航代码。T04启动前其他任务释放了内存，实际可用约140.62GiB，因此这次启动不是40GiB低余量条件下的实证。运行目录的scope_identity.json记录真实PID及内存上限，resource_telemetry.jsonl同时记录主进程和整个scope的占用、峰值及OOM事件。受限资源失败后只保留一次尝试，等待分析所需资源。

以下是在当前实验室服务器运行一组T04的完整命令。它使用独立输出目录，所有控制和录像参数与当前原任务复跑一致。启动前应确认达到上述可用内存和显存条件；同一时间只运行一组。T05使用新进程和新输出目录，将任务ID改为T05。

```bash
source /data20t/embodied/share/inter-nav/env.sh
export PYTHONPATH="$PWD:$PYTHONPATH"
export MALLOC_ARENA_MAX=2
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
go2_inputs=/data20t/embodied/zkj/inter-nav-results/go2-voronoi-heading-up-20261025/collision-corrected-inputs
go2_out="/data20t/embodied/zkj/inter-nav-results/go2-capped-T04-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$go2_out"
systemd-run --user --scope --quiet --collect \
  --unit="codex-go2-capped-T04-$(date +%s)" \
  --property=MemoryMax=24G --property=MemorySwapMax=0 -- \
  "$ISAAC_PYTHON" tools/scoped_go2_exec.py "$go2_out" \
  "$ISAAC_PYTHON" -u grutopia/demo/go2_online_interactive_navigation.py \
  --benchmark-dir "$go2_inputs" --output "$go2_out" --task-ids T04 --gpu 2 --max-steps 40000 \
  --floor-top .01 --map-source online --interaction-source ground_truth \
  --actuator-mode explicit_dc --path-controller continuous --forward-stop-feedback \
  --navigation-attitude-recovery --route-planner voronoi --navigation-heading path \
  --roadmap-shape circle --route-clearance .30 --navigation-half-width .22 --voronoi-clearance-weight .30 \
  --global-map-hz 1 --local-map-hz 10 --local-map-radius 3 \
  --push-start-speed .60 --push-max-speed .95 --turn-speed 1.0 \
  --contact-centering-speed .25 --contact-lateral-speed .35 --contact-yaw-tolerance .10 --contact-front-gap .35 \
  --record-video --material-mode simple \
  --third-person-height 1.35 --third-person-forward .25 --third-person-fov 100
```

辅助脚本在加载模拟器之前检查实际scope限制，再原样exec导航命令；若不在指定24GiB/零交换的scope中，会拒绝启动。完整任务仍须通过实际推箱、通路穿越、最终实测停稳、全程健康及所有视频/地图/源码检查。
