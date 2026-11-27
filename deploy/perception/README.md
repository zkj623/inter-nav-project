# MID-360S 点云认箱

点导航、认箱和推箱使用同一路 Swarm-LIO 配准点云。箱体尺寸、初始搜索区域和是否允许推移来自登记配置。前置 RGB 画面用于现场预览。

| 输入 | 处理 | 用途 |
|---|---|---|
| 多帧点云 | `WindowProcess` 接收导航观测线程的扫描，`WindowBoxObserver` 用多个扫描窗口拟合箱位和朝向。有效结果写入 `BoxMapMemory` | 地图显示、规划前往已知箱位的路线 |
| 当前扫描 | `RegisteredBoxPerception` 用登记尺寸复核箱面，返回与扫描时间及箱体身份绑定的证据 | 接触许可、短推位移测量和停后复测 |

默认配置只运行点导航。查看登记箱时，现场配置为 `configs/box_30x80x30.yaml`，`--box-window` 开启多帧跟踪。只需观察时加 `--observe-only`。[现场步骤](../README.md#点云认箱与查看)
