# robotforge 0.0.1

robotforge is a robot playground you drive with words.
You tell Claude or GPT what you want; it builds the robot, puts it in a world, makes it move, and shows it live in a 3D viewer in your browser, on MuJoCo physics and over MCP.

This is the first version number.
It is a tagged snapshot of `main`, not a published release: nothing is on PyPI and there is no `.mcpb` download yet (see "Not in 0.0.1").

## What 0.0.1 can do

### Create robots

- **Parts.** Build a robot from servos, links, wheels and sensors with `add_robot(build=...)`, `add_part`, `update_part` and `remove_part`.
  Checks refuse a design that cannot work and say how to fix it (overlapping parts, a servo too weak for its arm stretched out, a body that would tip) ([builder guide](BUILDER.md)).
- **Sentences.** `add_robot(describe="a tall wheeled courier with a lidar")` assembles a robot from a sentence, in English or Chinese; `update_robot` changes it by another sentence ("six legs instead of four", "make it red").
- **Asset engine.** Every generated part is made by a parametric generator in a style you choose (cobot, lab, hobby, rugged): filleted bases with bolt rings, joint housings, shells with parting lines, tread, brushed or cast surfaces, with collision shapes and inertia worked out from the geometry ([assets](ASSETS.md)).
- **Mount.** `mount` puts any robot on another: a PiPER on a Go2, a UR5e with a Robotiq gripper on a wheeled base, a lidar on a wrist.
  It checks mass, balance margin and reach, and refuses what would tip over.
- **Open robots by name.** 22 tuned robots (Go2, Spot, ANYmal C, Panda, UR5e, PiPER, G1, H1, T1 and others) and 175 more from the open robot catalogue, about 200 in all, added with `add_robot(library=...)` and filtered by kind with `list_library` ([catalogue](CATALOGUE.md)).
- **Import and export.** URDF, xacro, MJCF and mesh files are imported through a health check that repairs what it can and reports the rest; worlds and single robots export to MJCF or URDF with meshes ([extending](EXTENDING.md)).
- **Plugins.** A Python file in `~/.robotforge/plugins` adds a part, a sensor, a controller or a behavior that works like a built-in one; syntax and runtime errors are reported with file and line.

### Worlds and tasks

- **Scenes from a sentence.** `load_scene(description=...)` lays out a flat, a kitchen, a warehouse, a park or a hillside; terrain and floors are set separately ([scenes](SCENES.md)).
- **Tasks from a sentence.** `add_task("bring the red cup from the kitchen to the bedroom desk")` makes a task with a success check, in English or Chinese; sentences that name a table or desk in a room that has none get one.
- **Objects.** 24 generated objects (mugs, bottles, bowls, tools, fruit, labelled boxes, cans) with grasp hints, also from a description ("a tall blue mug with a big handle") ([objects](OBJECTS.md)).
- **Task suites.** `scripts/task_suite.py` generates easy and hard suites (bring, put, open, go to, stack, insert, pour, hand over) and checks that each case can be done before it is kept.

### Doing things

- **Legged robots on terrain.** Library quadrupeds (Go2, A1, ANYmal C, Barkour, Go1) cross flat ground, a 12 degree slope and stairs; Spot climbs the slope and refuses stairs; humanoids (G1, H1, T1, OP3) stand and take a push ([legged](LEGGED.md), [humanoid](HUMANOID.md)).
- **Mobile manipulation.** A mobile arm plans its drive and its reach together and fetches things with `pick_and_place`; arms ride on dogs and carts ([mobile](MOBILE.md)).
- **Mechanisms.** `operate` opens drawers and cabinets and presses buttons, switches, levers and valves; a factory conveyor feeds a sorting arm.
- **Force control and grasp planning.** `plan_grasp` scores grasps (a mug is taken by its handle); `operate(..., force=N)` limits the force of a turn or a press; peg in hole is in [manipulation](MANIPULATION.md).
- **Behavior trees and state machines** for task logic in scripts ([flexible](FLEXIBLE.md)).
- **Gamepad.** A gamepad drives anything that walks or rolls and jogs an arm; W A S D does the same from the keyboard.
- **Record and replay.** `demonstration` records what you or an agent did and replays it on another robot of the same kind.
- **Water and air.** A boat, an underwater robot and a quadcopter, each with a task.

### Sensing

- **Sensors on any robot.** Colour, depth, segmentation, stereo, fisheye and event cameras; 2D and 3D lidar, sonar, IMU, encoders, wheel odometry, force, touch and skin, temperature and gas.
  `add_sensor` and `read_sensors` work on templates, catalogue robots, imported and built ones, and the viewer draws what they sense ([sensors](SENSORS.md)).
- **Perception and mapping.** `mapping` builds an occupancy map from lidar and drifting odometry, a particle filter keeps the pose, the robot explores unseen space and goes to a named place on its own map ([perception](PERCEPTION.md)).

### Learning

- **Any-algorithm training.** `train_policy` trains any robot on any task with ARS, CEM, PPO, SAC or an algorithm you wrote, several side by side, or a halving search over settings; the learner sees what the robot's sensors read.
  The viewer's training view draws every trial's curve and settings live and plays the best policy ([training](TRAINING.md)).
- **Learned walks.** Any legged robot, built from parts or imported, learns a rhythm in the viewer, keeps it in `walks/<robot>.json`, learns to turn on the spot, and then walks from `move`, the keyboard or the gamepad.
- **Imitation.** You demonstrate a grasp; a nearest-neighbour rule and a small neural network copy it, and DAgger corrects them.
- **Teaching mode.** 13 lessons, 4 benches, 2 face-offs and 3 challenges in the viewer; `scripts/playtest.py` plays all 22 to the end.

### Agents

- **MCP for Claude and Codex.** 60 tools over stdio, a skill that teaches an agent how to build robots and teach beginners, a Claude Code plugin and a Codex plugin ([plugins](PLUGINS.md)).
  The viewer opens inside the chat where the client supports MCP Apps.

## How it was verified

Before the version number, a review round tried to break each ability the way a beginner would, in a real viewer and through real MCP sessions, not by reading code.
Every ability got a verdict.

| Area | PASS | PARTIAL | FAIL |
| --- | --- | --- | --- |
| Real agent sessions | 6 | 2 | 0 |
| Creating robots | 5 | 3 | 0 |
| Worlds, objects, manipulation | 9 | 4 | 0 |
| Locomotion, sensing, learning | 6 | 3 | 1 |
| Total | 26 | 12 | 1 |

The review found 17 problems; all 17 were handled on `main` before this version (two by documentation and a warning only, one only partly), most with a regression test, and each batch passed lint, the suite and the smoke test.
The ones that mattered most:

- `add_task` with no robot named crashed in the kitchen preset (`KeyError: 'pos'`); both kitchen agent sessions hit it.
- The README's own example, "bring the red cup from the kitchen to the bedroom desk", failed in 4 of 6 generated flats because the bedroom had no desk; it now works as typed in 30 of 30 seeds.
- A sentence such as "a green apple on the coffee table" was ignored; the placement is now honoured.
- ANYmal C with a UR5e and a Robotiq gripper was refused by `mount`, because the adapter plates' mass counted against the payload limit.
- A lidar return far beyond its range spoiled `box_m` and the point cloud; returns beyond `range_m` are now dropped.
- The G1's default speed sat inside the walking policy's dead zone, so it did not walk; the speed is now raised with a note.
- `pick_and_place` reported failure for a mug that stood upright on the shelf, and the apple slipped out of the grip in the kitchen preset.
- Imported arms were not recognised as arms, and a wheeled URDF was classed as a humanoid.

Real agent sessions ran Codex with the plugin mounted from a git archive of the code, 8 sessions with deterministic checks of the saved world and the MCP event stream (never of what the agent said):

| Request | Result |
| --- | --- |
| zh: a four-legged dog walking on grass | PASS, 1.29 m, upright, answered in Chinese |
| en: an arm puts the red cube in the box | PASS, cube ends in the bin |
| en: a six-legged robot with two arms, make it walk | PASS, 1.51 m in 6 s |
| en and zh: a Go2 with a lidar in a warehouse, what does it see | PASS twice, the nearest obstacle was checked against the world file |
| en and zh: a kitchen, bring the apple to the table | PARTIAL twice: the outcome was right, the agent first hit the `add_task` crash above, now fixed |
| en: train an arm to reach a ball and show the training | PASS, 4 of 5 held-out episodes reached the ball, after an honest report of two weaker runs |

`claude -p` was not run: the Claude CLI on the test machine was not signed in.
The suite has 3055 tests (2992 run by default, 63 marked slow or browser, run by hand at milestones); `scripts/smoke.py` starts the viewer and an MCP session as separate processes.
The cold-install start-up of the plugins (the first dependency download) was not measured in this round.

CI: see below.

## Known limits

Honest, from the PARTIAL items of the review and from the docs.

- **Humanoids walk on flat ground only.**
  They stand and take small pushes, but a 60 px mouse drag knocks a G1 or a T1 over, and the template humanoid tips at small drags by design.
  The G1 walks only above its policy's dead zone (0.2 m/s; a lower speed is raised to 0.25 with a note).
- **Some catalogue robots fall.**
  Added by name, a catalogue robot is imported, not tuned: in the review sample the humanoids `oli_mj` and `h1_2` fell at home, a `tron2` fell, and `b1` and `anymal_d` walked a short way and fell.
  Arms reach a point; hands move their joints; some catalogue drones (the Crazyflie) are rigid bodies without motors.
- **URDF round trip loses motors.**
  A URDF carries no motors, so an exported and reloaded URDF robot gets new, weaker servos and walks differently; MJCF export keeps them.
- **Mobile pick and place is not reliable on every body.**
  Fetching from a drawer with a wheeled base and fetching an apple with an ANYmal carrying an arm both succeed only about a third of the time (measured tables in [mobile](MOBILE.md)).
  The Go2 with an arm cannot fetch from counter height.
- **Handover can fail mid-way.**
  Two arms that touch at the start are refused before moving, but arms that only collide during the handover fail with a reason and cannot be predicted in advance.
- **Task suites.**
  The hard suite passed 3 of 5 cases in the review (a pour dropped 4 of 7 balls, a wrench handover lifted nothing); the default suite run is now verified case by case and is slow (about two minutes for eight cases).
- **Generated scenes.**
  A bedroom has a desk in 28 of 30 seeds, and in one seed in the review's bring run the picker still refuses the desk ("walls or furniture close it off").
  Placement words ("on the coffee table") are understood in English only.
- **Learning.**
  The default ARS settings sit at about 16 cm from the ball; a small search helps but a single run on one seed did not always reach (the review's own run ended at 5.7 cm, the documented run at 2.9 cm).
  A kept walk turning straight out of a walk reaches about 55 percent of its turn rate.
- **Viewer.**
  The panel can cover part of the arm at 1280 px width; the viewer's selection stays on the scene's starter rover until you pick the robot; a forward drag by the back flips a never-moved dog.
- **Plain failures.**
  A drawer and a button placed close together can make `operate` fail without a note; a UR5e refuses a drawer at some positions with a distance to move it.
- **Platforms.**
  CI runs on Ubuntu and Windows by hand only (the repository is private); macOS is tested locally.
  Offscreen rendering needs OpenGL; without it `observe` returns the state and a note instead of a picture.

## Not in 0.0.1

- No PyPI package and no `.mcpb` bundle for Claude Desktop; install from git or build the bundle locally (`scripts/build_desktop.py`).
- No ROS 2 bridge.
- No LeRobot dataset format.
- No browser physics (MuJoCo in WebAssembly); physics runs in a local Python process and the browser only draws.
- No licence file chosen yet.
- Drones are parked: a quadcopter flies a square, but there is no new drone work.

## 中文摘要

robotforge 是一个用话来驱动的机器人游乐场：在 Claude 或 GPT 里说一句话，agent 就把机器人搭出来、放进世界、让它动起来，并在浏览器的 3D 查看器里实时演示。
物理是 MuJoCo，通过 MCP 对接。
0.0.1 是第一个版本号，只是 `main` 的一个标记，还没有发布到 PyPI，也没有 `.mcpb` 安装包。

0.0.1 能做的：

- 造机器人：零件拼装、一句话生成、资产引擎（有圆角、螺丝圈、分模线的真实外形）、把一台装到另一台上、按名字加载约 200 台开源机器人（22 台调好的加 175 台目录里的）、导入导出 URDF、MJCF、插件。
- 世界和任务：一句话生成场景和带成功判定的任务、24 种生成物体、任务套件。
- 做事：足式机器人上地形、移动抓取、机构（抽屉、按钮、阀门等）、力控和抓取规划、行为树、手柄、录制和回放。
- 感知：各种传感器、建图和定位。
- 学习：任意算法训练并在查看器里看训练过程、学出来的走法、模仿学习。
- 智能体：给 Claude 和 Codex 用的 MCP 工具（60 个）。

怎么验证的：发版前做了一轮评审，每项能力都在真实查看器和真实 MCP 会话里试着弄坏它。
结果 26 项通过、12 项部分通过、1 项失败，发现的 17 个问题都已在 `main` 上修好并配了测试。
8 个真实的 Codex 会话里 6 个完全通过，2 个（厨房取苹果）结果正确但先撞上一个已修好的错误。
测试共 3055 个。CI 结果见上面的 "CI: see below"，稍后补上。

已知的限制：

- 人形机器人只能在平地上走，轻轻一拉就会倒；G1 的速度要高于死区才走得动。
- 目录里的部分机器人会摔倒，导入的是原样，没有调过。
- URDF 往返会丢掉电机，MJCF 不会。
- 轮式底盘的抽屉取物和 ANYmal 背臂取苹果都只有大约三分之一的成功率；递物过程中两臂可能撞到而失败。
- 困难任务套件在评审里 5 个通过 3 个；单次训练的结果不一定达到文档里的数字。
- 放置用语（"放在茶几上"）目前只识别英文。

0.0.1 里没有：发布到 PyPI 和 `.mcpb` 安装包、ROS 2、LeRobot 数据格式、浏览器里的物理。
