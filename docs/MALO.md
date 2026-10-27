# Zeno Malo 本地工作区

从已有的 Zeno Malo EDU 描述包导入真实网格和关节，保存为本地设计，再从查看器添加到场景。
底盘使用独立转向控制器驱动 MuJoCo 执行器；双臂、躯干、头部和夹爪可使用关节控制。
导入与控制均在本地完成，不需要启动 ROS 或 Isaac。

## 创建工作区

在 robotforge 源码目录运行，参数指向已经展开的 `zeno_malo_edu.generated.urdf`：

```bash
uv run --frozen python scripts/setup_malo.py /path/to/zeno_malo_description/zeno_malo_edu.generated.urdf
```

保留描述包的目录结构及 `meshes/`，让 URDF 中的 `package://zeno_malo_description/` 引用能够解析。
脚本也接受已有的 xacro 入口；已展开的 URDF 可直接复用，避免重新展开宏。
首次导入会转换和缓存需要处理的网格；源 URDF、网格或导入参数变化后，缓存会重新检查。

默认创建以下内容：

| 名称 | 内容 |
| --- | --- |
| `zeno_malo_workshop` | Malo、工作台、零件盒和橙色标记方块 |
| `zeno_go2_warehouse` | Go2、仓库和模拟激光雷达 |
| `zeno_panda_workbench` | Panda、工作台和橙色标记方块 |
| `zeno_malo` | 可重复添加的 Malo 本地设计，包含导入参数和模拟传感器 |

Go2 与 Panda 从现有机器人库按需下载并缓存。
只需要 Malo 时添加 `--malo-only`。
用 `--prefix another_name` 创建另一组工作区；已有同名世界或设计时脚本会拒绝覆盖。
创建中途失败可能留下已经创建的世界，请先检查脚本输出和本地目录，再选择新的前缀。

默认世界位于 `~/robotforge/worlds/`，设计位于 `~/robotforge/designs/`。
可用 `--home /path/to/local-data` 指定独立数据目录；之后打开查看器时也要设置同一个 `ROBOTFORGE_HOME`。

```bash
uv run --frozen robotforge open ~/robotforge/worlds/zeno_malo_workshop
```

独立目录示例：

```bash
uv run --frozen python scripts/setup_malo.py /path/to/zeno_malo_description/zeno_malo_edu.generated.urdf --home /path/to/local-data --malo-only
ROBOTFORGE_HOME=/path/to/local-data uv run --frozen robotforge open /path/to/local-data/worlds/zeno_malo_workshop
```

## 查看器操作

在“场景 · 搭建”的“真实模型与本地设计”中选择已保存的 `zeno_malo`，再选择空闲落点。
同一区域提供 Go2 与 Panda 卡片，显示本地缓存或首次下载状态；加载期间显示下载、准备、编译和发布阶段。
目录列表本身不会下载或编译模型。
手动保存的其他设计也复用这个入口，程序调用仍可使用 `add_robot(name, design="zeno_malo")`。

选中 Malo 后，用播放按钮开始仿真，W A S D、手柄或屏幕方向按钮控制前进、后退和转向。
转向轮先对齐再滚动，轮速和转向角受模型限制；停止指令清零轮速目标，实际速度仍受物理惯性影响。
暂停后可检查关节，切到手动模式使用关节滑条。
28 个独立关节执行器包括 4 个轮速执行器，其余为位置伺服；两侧夹爪的从动手指通过 URDF mimic 关系跟随主动手指。

R 重置到当前保存世界的起始状态。
在“地图 → 保存的布局”中保存布置，之后恢复同一布局可再次试玩。
布局保存机器人配置、初始位置、道具和资源引用，不保存实时姿态、速度、仿真时间或控制器状态。
保存设计与布局都不复制外部网格，移动或删除原始描述包后需要修复引用。
更多交互细节见[创造玩法](CREATIVE_PLAY.md)。

在“机器人 → 显示 → 光照”中，可选择默认的“摄影棚（Studio）”或“日光（Daylight）”。
选择保存在当前浏览器，适用于整个查看器场景；两种预设复用已有灯光和环境，不新增纹理或渲染通道。
它们调整观看光照，保留源 URDF 的白色、灰色材质，并不保证所有场景都更好看或帧率更高。
传感器预览固定使用 Studio 光照，避免查看偏好改变传感器图像；RGB、双目、鱼眼和事件相机的相关五项像素检查已通过。

## 模型与控制边界

导入设置明确使用 `respect_effort_limits=True`，保留源 URDF 的关节力矩或推力上限，不自动放宽弱电机限制。
执行器增益、碰撞处理及其他导入修复仍由导入体检报告说明；这些仿真参数不代表厂商控制器标定。
源文件的白色和灰色材质保留，不为 Malo 单独重涂。

独立转向底盘由通用控制器支持，运动通过轮速和转向执行器产生。
`drive_to` 实验任务使用左右两轮的二维动作接口，因此明确拒绝独立转向底盘；底盘可以驾驶不代表这个任务已适配。
现阶段没有为 Malo 定义经过验证的 TCP、IK 或高层夹爪端点，不能把双臂和夹爪关节可动等同于 `move_arm`、抓取或放置任务可用。
查看器和工具以实际能力元数据决定开放哪些控制。

工作区附加的是理想化模拟传感器：

- `head_rgb`：在 `stereo_camera_link` 上的单路 RGB，160 × 120、垂直 FOV 70°、5 Hz。
- `floor_scan`：在 `lidar_link` 上的 36 射线激光雷达，5 Hz。

它们用于观察场景，不是实机双目、镜头畸变或雷达的标定复现。

## 已记录的物理检查

2026-10-27 的 `drive-probe-v2.json` 记录了本轮冻结源码在真实 MuJoCo 步进中的四段各 3 秒检查：

| 输入 | 实测结果 |
| --- | --- |
| 前进 0.15 m/s | 位移约 0.421 m |
| 后退 −0.15 m/s | 位移约 0.422 m |
| 原地转向 0.45 rad/s | 朝向变化约 1.203 rad，平移约 0.003 m |
| 前进 0.10 m/s、转向 0.25 rad/s | 位移约 0.251 m，朝向变化约 0.647 rad |

四段均记录有限状态、接触和接近竖直的底盘，停止后的四个轮速控制目标均为零。
记录中的轮执行器力矩范围为 ±2 N·m。
该检查支持上述短时平地运动结果，不代表长时间稳定、载荷、复杂地形或实机性能认证。
记录保存在仓库外的 `robotforge-evidence/2026-10-27-malo-workspace/drive-probe-v2.json`，对应源码 SHA-256 `b867b4be92b970f4f23e7c29fa7697c5d2355c6c24f4a17c8fe5a9dd38dfa104`。
初次检查记录另行保留；后续代码改动需对应新的验证记录。

同轮源码与安装包冒烟、60 个 MCP 工具、子进程脚本、导航及 `drive_to` 按预期拒绝独立转向模型的检查已通过。
源码查看器的 v3 实录为 20.8333 秒，已核对源码哈希、Reset 运行代次、初始关节位置和控制目标的精确恢复、零 WebGL 错误及进程清理。
完整常规/browser 汇总为 3951 通过、127 跳过、1 失败；另行 5 项传感器检查通过。唯一失败在 main 原版也复现，因此保留草稿 PR 和合并阻塞，详见[验证记录](bench/2026-10-27-malo-workspaces.md)。

## 资源与许可

用户自行提供已有的 Malo 描述包，robotforge 不下载或打包它的 CAD 网格。
描述包的 `package.xml` 将 Apache-2.0 标记为软件许可，并注明导入 CAD 资产仍受原所有者条款约束。
不要从软件许可推断网格也获准重新分发。
本地设计保存资源引用，分享世界或导出模型前应分别核对所引用资产的许可。
