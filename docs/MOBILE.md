# 移动操作：底盘和机械臂一起干活

带臂的机器人（`go2_arm`、`mobile_panda`、`lift_picker`、Spot 背 Z1、ANYmal 背 UR5e）用 `pick_and_place`、`operate` 和 `move` 干整件事，不用自己拆成"先走过去再伸手"。

## 底盘和臂怎么一起规划

- 找站位：`fetch.spots_for` 在目标周围找一圈能站的地方，臂在那里能够得着，机器人的身子放得下，周围有转身的空间。
  每个候选站位先在世界的副本里把整个动作排练一遍，排练不成的跳过。
- 站位的排序：离机器人现在的位置近、侧向偏得少的排前面。
  偏在侧面的站位要先转、再走、再转回来，轮式底盘在抽屉旁边转不开，狗转身会扫到抽屉。
- 到了之后重新规划：导航停在离站位 8 cm 以内、12 度以内，臂在实际停下的地方重新算一遍。
  够不着就在原地蹭进站位（轮式），或者从现在的位置重新找一个站位（最多两次）。
- 蹭进站位时不再来回晃：偏在侧面时先走到站位的点，再转到站位的朝向；离站位很近（12 cm 内）时直接转向再沿朝向蹭，侧向留 6 cm 以内。
- 臂干活时底盘不动：轮子靠反馈把机器人拉在原地（臂一甩，轮子底盘会被带偏），狗站定。
  需要边走边干的活，传送带上取箱子，臂在原地等、跟着箱子走一段，底盘不动。
- 拿着东西走：臂折到身上，让东西在机器人上方，路径按加宽后的身子规划，门、过道不够宽就说"过不去，要 0.88 m 宽"。
- 夹爪：抓的时候夹爪捏紧（`fetch.squeeze`），整个伺服一起放大。
  Robotiq 的夹爪在库里是反过来用的（0 张开 255 闭合变成 255 张开），带一个偏置；之前只放大了增益和刚度，没放大偏置，指头停在离闭合还差 0.08 rad 的地方，ANYmal 背的 UR5e 抓不住任何东西。

## 干的活

| 活 | 怎么说 | 做什么 |
|---|---|---|
| 过门 | `move(robot, to="kitchen")` | 推开摆门，穿过去，门自己回去 |
| 取东西 | `pick_and_place(arm, "apple", "living_table")` | 去厨房，拿起苹果，穿过门，放在客厅的桌上 |
| 开抽屉取东西 | `operate(arm, "chest", "open")`，再 `pick_and_place(arm, "block", "kitchen_table")` | 站到抽屉旁，拉开，再站到能够到抽屉里的位置，拿出来 |
| 按按钮 | `operate(arm, "button", "press")` | 走到按钮旁，臂从上方按下去，折回来 |
| 传送带上拿 | `pick_and_place(arm, "box_6", "workbench")` | 站在传送带旁，臂跟着转动的箱子走一段，夹住，不停传送带，再带到工作台 |

## 实测

`uv run python scripts/mobile_bench.py --seeds 5` 在两居室的家（客厅、厨房、中间一扇摆门）和工厂里，把每台机器人放在不同位置、不同朝向，跑每件活。
种子改变桌上东西的摆放和机器人的起点；真实的 Menagerie 模型，无头仿真。
每格是成功次数（八个种子），括号里是失败的原因和次数。

| 机器人 | 过门 | 取苹果 | 抽屉取东西 | 按按钮 | 传送带 |
|---|---|---|---|---|---|
| go2_arm | 8/8 | 8/8 | 7/8 (take:stuck x1) | 7/8 (not_pressed x1) | 0/8 (refused:cannot_catch x8) |
| mobile_panda | 8/8 | 8/8 | 5/8 (take:out_of_reach x3) | 8/8 | 8/8 |
| lift_picker | 8/8 | 8/8 | 0/8 (refused:too_high_or_low x8) | 8/8 | 1/8 (refused:cannot_catch x4, not_there x2, missed x1) |
| spot_z1 | 8/8 | 0/8 (refused:cannot_pick x8) | 0/8 (refused:nowhere_to_stand x8) | 6/8 (too_narrow x2) | 0/8 (refused:cannot_pick x8) |
| anymal_ur5e | 8/8 | 5/8 (knocked x2, fell x1) | 0/8 (refused:nowhere_to_stand x8) | 8/8 | 8/8 |

失败的原因：

- `refused:cannot_catch`：传送带 0.1 m/s，臂在带旁够不到箱子经过的路线（Go2 的臂只比带高一点，lift_picker 的臂伸出去不够远），说的是该怎么办（停带、挪机器人）。
- `refused:too_high_or_low`：lift_picker 的夹爪朝下只在 0.20 到 0.80 m 高处，抽屉里的方块在 0.04 m。
- `refused:nowhere_to_stand`：抽屉把手在 0.23 m 高，要拉出 0.33 m，臂在背上很高的位置（ANYmal 背 UR5e）或臂短（Spot 背 Z1，朝下 0.53 m）站不出一个整个过程都够得着的位置。
- `refused:cannot_pick`：Z1 的单爪夹子在臂摆动时会松手（库里写着，十次里三次），所以 Spot 背 Z1 不拿东西，只开门和按按钮。
- `take:stuck`、`take:out_of_reach`：抽屉拉开后机器人站在它旁边，转向站位时车头或狗身扫到抽屉前板，把抽屉推回去 2 到 3 cm，东西跟着动，站位刚好够得着的臂就差这一点。
  轮式底盘的三次是这样；狗的一次是转身 170 度时扫到。还没解决：转向要先躲开贴着的东西，或者先倒车。
- `knocked`、`fell`：ANYmal 背着臂在传送带旁或拿苹果时被臂的反作用推开超过 8 cm 三次，或者摔倒一次。
- `too_narrow`：背着东西的身子比过道宽（Spot 的臂折起来要 0.65 m，客厅的桌子和沙发之间只有 0.50 m）。
- `not_pressed`：Go2 的臂按键时按键没按到底，一次。

## 录像

`scripts/demos.py` 里的 `mobile_*`：每段是真实时间里一件完整的活，镜头跟着机器人。
`mobile_fetch_go2` 是"去厨房把苹果拿到客厅的桌上"，Go2 带臂推开摆门、拿起苹果、再推开门回来。
