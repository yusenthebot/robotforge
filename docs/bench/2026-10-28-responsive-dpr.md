# 显示密度事件与画布分辨率（2026-10-28）

查看器收到实际分辨率媒体事件或窗口 resize 时，重新读取显示密度并同步主画布，保留原有 DPR2 上限。
密度单独变化时保持 CSS 坐标、相机投影和放置锚点；相同尺寸与倍率无需重设缓冲，并修复默认 300×150 画布的初始相机比例边界。
没有新增每帧检测、贴图、shader、渲染 pass、请求或依赖，小功能成功率未调整。

## 真实路径与边界

原始 Malo、Go2、Panda 世界保持暂停，实际 Chrome/Metal 中使用 CDP 改变设备指标。
两套相同捕获流程共 54 图，每会话 485 个模型／qpos／qvel／ctrl 字段保持，跨会话编译模型／网格数组精确相同，仅路径元数据可不同；网格标签相同或绝对路径资源哈希相同，相同的相对文件标签不冒称已单独解析原文件。
12 组固定页面控制逐像素一致，包含重复返回采样；实际主画布分辨率为 DPR1／2，设备 DPR3 封顶为2。

| 路径 | 实际结果 |
|---|---|
| 媒体通知 1→2 | 原版仍1280×800；候选2560×1600，等于fresh2画布 |
| 媒体通知 2→1.25 | 候选1600×1000；验证通知、buffer和相机，无fresh1.25像素参考 |
| 媒体通知 1.25→3 | 候选封顶2560×1600，等于设备3的fresh控制 |
| 3→1 加真实CSS resize | 候选1280×800，等于fresh1；原版高密度启动后仍2560×1600 |
| 初始300×150、重复相同指标、密度后实际点击放置 | 比例与尺寸正确，无额外缓冲重设，落点与角度正确 |

纯 CDP 3→1 在本环境中没有向产品或固定媒体查询哨兵发送新通知，直到真正改变窗口尺寸才恢复；此自动同步路径仍未通过。
不能据此证明是与产品真实使用无关的 Chromium 缺陷；未验证实体跨显示器或原生浏览器缩放。
独立 headed Chrome 仅完成初始建立和清理，Mac锁屏导致未操作快捷键，不能作为原生缩放实测。
相机投影、目标、灯光和fit保持精确一致，eye使用绝对1e-10容差并记录实际最大误差。

## 验证与成本

- 最终源码 `88dfa18d0afff74c525b4bf261944f5021bd2273bcfd3cbb9b07bf3519099d90`，冻结760项输入。
- 完整非慢速浏览器290通过/1可选集合跳过，资产／传感器GPU9通过，相关后端579通过；共878个不同pytest节点。
- 原7案例框架基线通过，使用与原记录一致的真实Python3.12.13；其他检查Python3.12.16。
- 源码及新离线锁定安装包的真实CLI／查看器／60工具MCP冒烟通过，原安装／查看器预算11/11通过。
- wheel 3941294B，较已验证PR46产物+239B；静态JS原文+778B，413源码／wheel／安装文件一致，仅app.js产品字节变化。
- 运行包／依赖不增加，原工作目录13项WIP哈希保持；完整常规／其他慢速套件未重跑，缺失type／security工具不计通过。
- 原预算实测：冷首帧0.5299s，Go2／组合资产60.12／60.24FPS，绘制1.90／3.00ms；安装163.6372MB／42包，服务器峰值134.0MB／idleCPU4.4%。
- ABBA各世界／cohort两次浏览器冷导航和真实重载；Malo冷均值差+5.00ms、Go2-1.50ms。本地模型／系统盘缓存已暖、服务器已编译，不能声称普遍提速。
- 固定截图控制中的资源路径集合、纹理数、程序数及绘制量一致；请求送达顺序和renderer帧计数不要求一致。
- DPR1→2正确增加四倍画布像素；DPR2→1减少75%，这是缓冲像素数量，不是实测VRAM或峰值内存。浏览器RSS仅为快照。

加载就绪指主画面已绘制加2RAF与gl.finish，不等待所有传感器预览编译；加载中的纹理／程序计数是瞬时值，不用于稳态资源比较。

## 视频与记录

六段录像保留完整时间戳、全量原帧，完整时间戳作为编码输入并30fps重采样，不保证每张原JPEG进入输出视频；随后全部解码，root逐段查看六张解码样本，共36张，并非逐帧认证。
视频是暂停世界中的设备指标切换与显式fresh控制，包含重载和加载画面；不是驾驶、实体跨屏或原生快捷键视频。

- malo: [before 17.17s](../../../robotforge-evidence/2026-10-28-responsive-dpr/screen-before-v4/malo/workflow.mp4), [candidate 17.60s](../../../robotforge-evidence/2026-10-28-responsive-dpr/screen-candidate-v4/malo/workflow.mp4)
- go2: [before 15.60s](../../../robotforge-evidence/2026-10-28-responsive-dpr/screen-before-v4/go2/workflow.mp4), [candidate 15.90s](../../../robotforge-evidence/2026-10-28-responsive-dpr/screen-candidate-v4/go2/workflow.mp4)
- panda: [before 17.90s](../../../robotforge-evidence/2026-10-28-responsive-dpr/screen-before-v4/panda/workflow.mp4), [candidate 18.60s](../../../robotforge-evidence/2026-10-28-responsive-dpr/screen-candidate-v4/panda/workflow.mp4)

原始证据保存在同级 `../robotforge-evidence/2026-10-28-responsive-dpr/`，封闭索引和外部资产哈希另经独立重验。
此前两报告族四份raw report中64个名为DPR1的样本实际均为renderer DPR2，本次更正fine-heading和environment-scale覆盖说明，旧原图与哈希不改，不追认为DPR1。
初始等待time>0、严格eye浮点相等、媒体通知等待和离线比较误要求请求顺序／frame计数相等、误把相对网格标签当作绝对文件的失败均保留；最终通过不能覆盖这些失败。
同期阴影恢复筛查未达到两个真实场景的普通视距门槛，没有实现或合并阴影候选；QA根位姿往返不是实际驾驶，其原始记录保存在 `../robotforge-evidence/2026-10-28-shadow-recovery/`。
整体材质质量、轻量加载和自由搭建目标继续。
