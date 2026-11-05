# go2-simulation-验收版本 独立交付

交付分支：go2-simulation-验收版本。分支基点 da6acda；实测运行版本 514cd51dc7a4400a3e7db431b1f18a7aba7e3f50。交付仅新增文档、输入和证据，不修改导航算法。源码位于仓库根目录。

## 入口与结构
- grutopia/demo/go2_online_interactive_navigation.py：在线导航入口。
- grutopia/demo/go2_side_clearance_navigation.py：仿真、参数解析、任务循环。
- grutopia_extension/interactive_navigation/：在线建图、路径选择、执行安全、推箱技能与停稳反馈。
- grutopia_extension/controllers/isaac_go2_ctrl.py：Go2控制器。
- tests/：现有自动回归测试；delivery/regression-case-index.csv 为可定位索引。
- tools/：任务资格和运行审计、资源保护及诊断工具。
- delivery/inputs/T01–T10：每项历史运行的原始输入快照；评估图只能供审计，不能输入导航器。
- delivery/*.py：原服务器go2-simulation-验收版本启动与功能审计脚本，含固定路径和运行存在性保护。历史脚本是复现依据，不可直接覆盖既有运行。

## 环境与启动
使用原Linux服务器上的Isaac Sim/IsaacLab环境；现有初始化脚本 /data20t/embodied/share/inter-nav/env.sh。先 source 该文件，进入源码目录，export PYTHONPATH="$PWD:$PYTHONPATH"。requirements/ 下保存依赖，Isaac环境不应由普通Windows Python替代。
入口参数："$ISAAC_PYTHON" -m grutopia.demo.go2_online_interactive_navigation --help。
回归命令："$ISAAC_PYTHON" -m pytest -q tests/（完整历史命令还包含环境依赖和一项 deselect，见原测试日志；不要声称此命令必然复现相同数量）。
本次复核：python -m pytest -q tests/test_online_navigation.py tests/test_final_push_release.py tests/test_passage_certificate.py tests/test_release_reserve_cycle.py；原环境43 passed。

## 外部资产交接清单
本包是源码、配置、模型权重和验证证据交付，不含大型USD场景/纹理及Isaac安装。完整仿真仍需以下既有依赖：
- 场景：/data20t/embodied/zkj/inter-nav-results/rectangular-boxes-five-scenes-xy100-20261021-v2/mv7/after_scene.usda 及其引用纹理/网格。
- 机器人：/data20t/embodied/zkj/inter-nav-results/go2-online-smooth-20261027/official-policy-review-20261030/nvidia_isaaclab_4_5/go2_training_flattened.usdc 及引用资产。
- grutopia/assets 目录：原环境为资产符号链接，源码压缩包不包含此链接目标。
- policy SHA256：8862682da34ebb090a2e56f39254af430d5645f20cb5e3e104c72a4720671db8；文件见 delivery/model/actor_checkpoint.pt。
- 启动脚本引用 go2-online-smooth-20261027 下 capture_go2_policy_state_v193.py、contact_report_launch.py、launch_online_checked.py、supervise_motion_v218_t04.py 等外部历史支持文件；随包另存 runtime-support，部署时须显式映射其路径。
跨机器部署前将scene_profile.json、启动器、机器人与支持脚本中的原服务器路径映射到新环境，保持资产比例、policy及物理契约不变。路径变更后须重新验收，不得沿用旧试验结论冒充新运行。

## 复现实验要求
保留原运行目录，使用新的输出目录与运行标识；原启动器禁止覆盖或自动重试。先校验policy与源码哈希，再检查单机共享锁、24GiB/零swap资源限制、空闲内存56GiB持续60秒与GPU空闲14GiB。自身仿真串行。原脚本与契约见交付证据。审计调用格式参考 audit_uniform_qv23_functional.py 的 argparse；启动参数以 launch_uniform_qv23_diagnostic.py 为完整冻结依据。
本交付不启动新训练或仿真。

## 验收结论与限制
固定十项每项一次任务完成验收10/10，耗时不设门槛；T02–T05耗时增加，仅作参考。历史完整测试1012 passed, 1 deselected；本次定向43 passed。T01–T09这轮不录视频，T10沿用同提交已有录像。交互资格使用真值箱体名单加LiDAR可见性，不是新视觉识别模型。不能据此推断任意场景成功率。

## 文件校验
在仓库根目录运行 python delivery/verify_delivery.py。manifest.json记录除自身以外的delivery文件SHA256。源码通过Git提交和bundle标识；evidence/source-hash-check.json记录十项源码清单与交付树比对结果。


当前正式文档为-v2.docx。验收以任务成功完成为准，无耗时门槛，十项均通过；原始证据耗时字段仅供参考。


## 正式交付文档
- 01-交付说明与测试用例.docx / -V4.pdf：合并后的交付说明、依赖、运行方式和测试方法。
- 02-测试报告与可视化材料.docx / -V4.pdf：go2-simulation-验收版本功能结果及20张历史录像截图。
两份文档统一微软雅黑、A4版式、标题、表格、页眉和页码。截图来源与版本见figures/frame-sources.json，耗时仅作参考。
文档检查：合并说明9页，测试报告13页，报告内20张截图。已导出PDF并检查分页、图注和代表页面。

## 路径与场景复杂度
同一MV7场景中的十组起终点任务，曲折度均大于1.3，初始杂乱度38.195603%大于20%。曲折度=最短几何路径长度/起终点直线距离；不是实际执行路径长度。逐项起终点、完整参考路点、距离、比值与原始资格证据见task-paths-and-complexity.json、.csv和evidence/qualification/。资格路线仅供离线评估，不输入在线导航器。两份Word及PDF已补充指标、定义和达标说明。
