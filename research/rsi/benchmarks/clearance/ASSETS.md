# 仿真资产与复现边界

- 机器人：Galaxea R1 Lite 相关 URDF/USD/mesh，来源于实验室现有的 `GalaxeaManipSim` 与 `galaxea_urdf` 资产副本。需要自行取得合法可用的原始资产并遵守其许可；本功能分支不转载第三方 mesh、权重或机器人 USD。
- 实验室已导入资产位于 `/data20t/embodied/zkj/third_party/r1lite_isaac_v2/r1lite_navigation.usd` 及配套同名 URDF。配置环境变量 `R1LITE_USD_PATH` 可以覆盖默认位置；URDF 引用的 mesh 也必须可访问。
- round-06 使用独立 `robot_snapshot/` 和 `robot_asset_manifest.json`，位于服务器结果目录，不属于 Git 源码。仅用未匹配资产的运行结果不能冒充原实验复现。
- `grutopia/demo/prepare_r1lite_asset.py` 是实验室资产导入工具，带有当前本机原始资产路径。不要在其他机器直接运行并假设能下载资产；先配置来源并核对导入后的关节、质量与碰撞形状。
- 箱子为代码程序化构造的无把手动态刚体，尺寸和质量在各任务配置明确给出；墙/静态障碍也由场景配置创建。各候选评测之间不能悄悄改质量、摩擦、碰撞代理或成功门槛。
- 本地地面复制 Isaac 4.2 默认环境的无限 Z 碰撞平面，无需下载远程场景 USD。实际安装 SDK 原地面没有绑定那个被创建的 0.8 材质；实现保留其有效未绑定状态，而不是声称有效摩擦一定为 0.8。原/本地地面的跌落与滑动对照各 1000 样本、最大轨迹差为 0，此对照不是搬运成功证明。
