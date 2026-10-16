"""The beginner learning path: nine short lessons an agent teaches with the robotforge tools.

The learner reads docs/lessons/NN-slug.md (Chinese). This module holds the agent's side of each lesson:
what the learner asks for, the ideas to teach, the tool calls that get there and how to check the result.
MCP exposes it two ways, so "teach me lesson 3" works in every client: a `lessons` tool (clients without
prompt support, such as Codex) and one prompt per lesson (Claude Desktop and Claude Code list them).
"""

from __future__ import annotations

from dataclasses import dataclass

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from robotforge import wording

REPO_URL = "https://github.com/yusenthebot/robotforge"

TEACHING = (
    "Teach in the user's language (the lesson pages are Chinese; answer in English if the user writes English). "
    "Before each step say in one or two plain sentences what you will do and the idea behind it; define every "
    "technical word the first time. Open the viewer early and invite the learner to try things by hand (sliders, "
    "dragging, W A S D). Every reply runs at least one step that makes something happen in the simulation (a "
    "robot moves, balances or falls) and shows its numbers: building the scene and explaining is not yet a lesson. "
    "After every action check the numbers the tool returns and look at an image (observe); "
    "never say it worked without checking, and if it did not, say so and fix it. Keep each message short. End "
    "with the lesson's challenge, link the lesson page, and offer the next lesson."
)


@dataclass(frozen=True)
class Lesson:
    number: int
    slug: str
    title: str
    title_zh: str
    ask: str  # what the learner can say; the page quotes it word for word
    goal: str
    ideas: tuple[str, ...]
    steps: tuple[str, ...]
    check: tuple[str, ...]
    challenge: str
    optional_tools: tuple[str, ...] = ()  # tools a step uses only when this install has them

    @property
    def page(self) -> str:
        return f"docs/lessons/{self.number:02d}-{self.slug}.md"

    @property
    def url(self) -> str:
        return f"{REPO_URL}/blob/main/{self.page}"

    @property
    def prompt_name(self) -> str:
        return f"lesson_{self.number}"

    def summary(self) -> dict[str, object]:
        return {"number": self.number, "title": self.title, "title_zh": self.title_zh, "ask": self.ask}

    def plan(self) -> dict[str, object]:
        return {
            "number": self.number,
            "title": self.title,
            "title_zh": self.title_zh,
            "learner_says": self.ask,
            "goal": self.goal,
            "ideas_to_teach": list(self.ideas),
            "steps": list(self.steps),
            "check": list(self.check),
            "challenge": self.challenge,
            "page": self.url,
        }

    def prompt_text(self) -> str:
        lines = [
            f"Teach robotforge lesson {self.number}: {self.title} ({self.title_zh}).",
            "",
            f"Goal: {self.goal}",
            f"The learner's request: {self.ask}",
            "",
            "Ideas to teach, one at a time as they come up:",
            *(f"- {idea}" for idea in self.ideas),
            "",
            "Plan (robotforge tools):",
            *(f"{i}. {step}" for i, step in enumerate(self.steps, 1)),
            "",
            "Check before you say it worked:",
            *(f"- {item}" for item in self.check),
            "",
            f"Challenge to end with: {self.challenge}",
            f"Lesson page for the learner: {self.url}",
            "",
            TEACHING,
        ]
        return "\n".join(lines)


LESSONS: tuple[Lesson, ...] = (
    Lesson(
        number=1,
        slug="joints-and-actuators",
        title="Joints and actuators: build an arm and move it joint by joint",
        title_zh="关节和舵机：搭一条机械臂，一个关节一个关节地动",
        ask="给我做一条 4 个关节的机械臂，打开查看器，教我关节和舵机是什么。",
        goal="The learner builds a small arm, moves each joint by hand and by command, and can explain joint, "
        "degree of freedom (DOF) and servo.",
        ideas=(
            "A joint lets two parts move relative to each other; a hinge turns about one axis, so each hinge adds "
            "one degree of freedom (DOF).",
            "An actuator makes a joint move. A hobby servo is a position actuator: you send a target angle in "
            "radians and it pushes toward it like a stiff spring.",
            "Joints have limits (range), and gravity pulls on the arm, so a servo holding the arm out stops a "
            "little short of its target.",
        ),
        steps=(
            '`new_world(name="lesson1")`, then `add_robot(name="arm", template="arm", params={"dof": 4})`.',
            "`open_viewer()`; invite the learner to drag the arm/yaw, arm/shoulder and arm/elbow sliders.",
            "`describe_model()`: read out the actuators and their ranges; names are `arm/<joint>`, units radians.",
            '`set_controls(controls={"arm/shoulder": 0.8})`, `simulate(seconds=1)`, then `observe()`: compare '
            "the target (controls) with the angle reached (joints).",
            "Move one joint at a time so the learner sees which part each joint turns.",
        ),
        check=(
            'observe: joints["arm/shoulder"] is within about 0.1 rad of the 0.8 target.',
            "The image shows the arm in the new pose.",
        ),
        challenge="Make it a 6-joint arm with update_robot, count the degrees of freedom again and name the part "
        "each joint turns.",
    ),
    Lesson(
        number=2,
        slug="drive-a-rover",
        title="Driving a rover: how two wheels steer",
        title_zh="开一台小车：两个轮子怎么转弯",
        ask="给我一台两轮小车，开着它转一圈，告诉我它是怎么转弯的。",
        goal="The learner drives a differential-drive rover and can explain how unequal wheel speeds turn it.",
        ideas=(
            "A differential-drive rover has two powered wheels: equal speeds drive straight, unequal speeds turn it, "
            "like a tank.",
            "Speed is in m/s and turning rate in rad/s; a positive turn is to the left (counter-clockwise seen from "
            "above), and heading is reported in degrees.",
            "Where the robot ends up is measured, not assumed: wheels slip a little, so check position and heading.",
        ),
        steps=(
            '`new_world(name="lesson2")`, `add_robot(name="rover", template="rover")`, `open_viewer()`; the learner '
            "can drive with W A S D in the viewer.",
            '`move(robot="rover", speed=0.3, seconds=3)`: report distance_m.',
            '`move(robot="rover", speed=0.2, turn=1.0, seconds=3)`: report turned_deg and explain the wheel speeds.',
            '`set_controls(controls={"rover/left_wheel": 4, "rover/right_wheel": 8})` and `simulate(seconds=2)` to '
            "command the wheels directly (rad/s), then `robots()` to see where it went.",
        ),
        check=(
            "distance_m is close to speed x seconds (about 0.9 m for 0.3 m/s over 3 s).",
            "turned_deg is positive (a left turn) but below the ideal turn x seconds (1 rad/s for 3 s is 172 "
            "degrees; about 150 is normal): speeding up and wheel slip eat the rest, a good point to discuss.",
        ),
        challenge="Drive a full circle back to the start, see how far off it ends, and work out why.",
    ),
    Lesson(
        number=3,
        slug="dog-walk-and-gait",
        title="Making a robot dog walk: what a gait is",
        title_zh="让机器狗走起来：什么是步态",
        ask="给我一只四足机器狗，放在草地上，让它往前走 10 秒。",
        goal="The learner makes a quadruped walk on grass and can explain a trot gait and stance versus swing.",
        ideas=(
            "A gait is the rhythm in which legs lift and land. In a trot the diagonal legs move together (front "
            "left with rear right), so two feet always hold the body up.",
            "Each foot follows an arc: stance pushes back along the ground, swing lifts and moves forward. Leg "
            "inverse kinematics turns each foot position into hip, thigh and knee angles.",
            "Walking needs friction and balance: grass grips better than a smooth floor, and faster steps leave "
            "less time to correct.",
        ),
        steps=(
            '`new_world(name="lesson3", floor="grass")`, `add_robot(name="dog", template="quadruped")`, '
            "`open_viewer()`.",
            '`robots()` to read max_speed, then `move(robot="dog", speed=0.25, seconds=10)`.',
            "`observe()`: the four touch sensors (dog/front_left_touch ...) show which feet are on the ground.",
            "Let the learner steer with W A S D in the viewer.",
        ),
        check=(
            "move reports distance_m above 1.0 and to.upright true.",
            "The image shows the dog standing on the grass, ahead of where it started.",
        ),
        challenge="Walk an L: 3 s straight, then 3 s turning left, and describe how the gait changes in the turn.",
    ),
    Lesson(
        number=4,
        slug="ik-pick-and-place",
        title="Inverse kinematics and pick-and-place",
        title_zh="逆运动学：让机械臂把方块放进盒子",
        ask="做一条机械臂，把红色方块放进蓝色盒子里，再告诉我它是怎么算出每个关节角度的。",
        goal="The learner sees an arm reach a point and move a cube into a bin, and can explain forward versus "
        "inverse kinematics and reachability.",
        ideas=(
            "Forward kinematics goes from joint angles to where the hand is. Inverse kinematics (IK) goes the other "
            "way, from where the hand should be to joint angles; robotforge solves it numerically in small steps "
            "using the Jacobian.",
            "Not every point is reachable: link lengths and joint limits bound the workspace, so the reach is "
            "checked before moving.",
            "Pick-and-place is a sequence: open, go above, descend, close, lift, carry at a safe height, lower, open.",
        ),
        steps=(
            '`new_world(name="lesson4")`, `add_robot(name="arm", template="arm")`.',
            '`add_object(name="cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, -0.12], color="red")` and '
            '`add_object(name="tray", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="blue")`, '
            "then `open_viewer()`.",
            '`move_arm(robot="arm", position=[0.25, -0.12, 0.15])` to show IK reaching above the cube; read '
            "tool_position.",
            '`pick_and_place(robot="arm", object="cube", target="tray")`.',
            '`observe(view="top")` to look at the result.',
        ),
        check=(
            "move_arm's tool_position is within about 3 cm of the requested point (servos sag a little under gravity).",
            "pick_and_place returns success true, and the top view shows the red cube inside the blue tray.",
        ),
        challenge="Add a green cube and put both cubes in the tray; discuss why the second one is harder.",
    ),
    Lesson(
        number=5,
        slug="sensors-and-camera",
        title="Sensors and the robot's camera",
        title_zh="传感器和相机：机器人眼里的世界",
        ask="让我看看小车摄像头里的画面，再告诉我它的传感器读数是什么意思。",
        goal="The learner sees the world through the robot's camera and reads its IMU while it turns.",
        ideas=(
            "A camera on the robot shows the world from its point of view. The default view is for us; the robot "
            "only has its own camera.",
            "An IMU measures acceleration (accel, m/s^2), turning rate (gyro, rad/s) and orientation (a quaternion), "
            "so a robot knows when it tilts or turns.",
            "Touch sensors on feet and position sensors on tools tell a controller what is happening; every reading "
            "is a number you can log and compare.",
        ),
        steps=(
            '`new_world(name="lesson5")`, `add_robot(name="rover", template="rover")`, '
            '`add_object(name="ball", shape="sphere", size=[0.1], pos=[0.8, 0], color="red")`, '
            '`add_object(name="box", shape="box", size=[0.1, 0.1, 0.1], pos=[0.6, 0.4], color="blue")`.',
            '`describe_model()` lists sensors and cameras; `observe(view="rover/front_cam")` shows what the rover '
            'sees; compare with `observe(view="top")`.',
            '`move(robot="rover", speed=0, turn=1.0, seconds=null)`, `simulate(seconds=1)`, '
            '`observe(include_image=false)`: read rover/gyro while it turns, then `stop(robot="rover")`.',
        ),
        check=(
            "The front_cam image shows the red ball ahead.",
            "While turning, the third rover/gyro value (rotation about the vertical axis) is clearly positive, "
            "about 0.6 rad/s for the 1.0 commanded: measuring beats assuming. After stop it is near 0.",
        ),
        challenge="Using only the camera image to choose the direction, drive the rover up to the red ball and "
        "stop; explain how you decided which way to turn.",
    ),
    Lesson(
        number=6,
        slug="pd-lqr-mpc",
        title="Control: PD vs LQR vs MPC",
        title_zh="控制算法：PD、LQR 和 MPC 比一比",
        ask="做一个倒立摆，分别用 PD、LQR 和 MPC 让它保持平衡，中途推它一下，比较哪个最好。",
        goal="The learner balances a cart-pole with three controllers, survives a push, and reads a comparison table.",
        ideas=(
            "Feedback control measures the error and pushes against it. PD pushes in proportion to the error (P) and "
            "brakes by how fast it changes (D).",
            "LQR linearizes the physics around upright and computes the best gains from what matters (Q) and how "
            "costly effort is (R): gains are calculated, not hand-tuned.",
            "MPC simulates many possible futures a fraction of a second ahead and acts on the best one: general, but "
            "it costs a lot of computation. Doing nothing (none) shows the pole falls.",
        ),
        steps=(
            '`new_world(name="lesson6")`, `add_robot(name="pole", template="cartpole")`, `open_viewer()`.',
            "`list_algorithms()`; explain the balance methods in one line each.",
            '`run_controller(robot="pole", task="balance", method="lqr", push_at=2)` live, then `stop(robot="pole")`.',
            '`compare_controllers(robot="pole", task="balance", methods=["none", "pd", "lqr", "mpc"], push_at=3)` '
            "and walk through the table.",
        ),
        check=(
            "run_controller with lqr returns success true after the push.",
            "In the table none fails (failed_at set) and the others succeed; explain reward, effort and wall time "
            "from the actual numbers.",
        ),
        challenge="Make the pole longer (update_robot with pole_length) and compare again: which method gives up "
        "first?",
    ),
    Lesson(
        number=7,
        slug="your-own-script",
        title="Writing your own robot script",
        title_zh="写自己的脚本：让小车走一个正方形",
        ask="帮我写一个 Python 脚本，让小车走一个边长 0.5 米的正方形，代码要让我看得懂。",
        goal="The learner gets a readable, closed-loop Python script that drives a square, runs it and changes it.",
        ideas=(
            "A script reads the world and sends commands in a loop. robotforge runs it in lockstep, so time only "
            "passes when the script calls step().",
            "Open loop means drive two seconds and hope; closed loop means keep measuring position and heading and "
            "correct, which copes with slip.",
            "A script is a file in the world's behaviors folder: the learner can read it, edit it and run it again.",
        ),
        steps=(
            '`new_world(name="lesson7")`, `add_robot(name="rover", template="rover")`, `open_viewer()`.',
            '`write_script(name="square", code=...)` with commented code using sim.position, sim.heading, sim.move, '
            "sim.step and sim.stop; drive each side by distance and turn by heading feedback.",
            '`run_script(name="square")`; read its output (print each corner).',
            '`read_script(name="square")` when the learner wants to look at the code again.',
        ),
        check=(
            "run_script returns ok true with no traceback.",
            "The printed corners are within about 0.1 m of a 0.5 m square and the rover ends near its start.",
        ),
        challenge="Rewrite the script to drive a five-pointed star (turn 144 degrees at each point) and print the "
        "heading at every point.",
    ),
    Lesson(
        number=8,
        slug="train-a-policy",
        title="Training a policy: learning by trial and error",
        title_zh="训练一个策略：让机器人自己学会平衡",
        ask="不要写控制规则，让倒立摆自己学会保持平衡，再和 LQR 比一比。",
        goal="The learner watches a policy learned by trial and error balance a pole and compares it with LQR.",
        ideas=(
            "Reinforcement learning (RL): the robot tries actions, gets a reward (here staying upright and centered) "
            "and keeps the changes that earn more. The result is a policy, a function from what it senses to what "
            "it does.",
            "ars (augmented random search) learns a simple linear policy with numpy in seconds. Neural-network "
            "methods (PPO, SAC) learn harder skills such as walking, need the optional rl extra and take minutes.",
            "A learned policy is only as good as its reward and its practice; test it with a push it never saw.",
        ),
        steps=(
            '`new_world(name="lesson8")`, `add_robot(name="pole", template="cartpole")`, `open_viewer()`.',
            '`run_controller(robot="pole", task="balance", method="ars", push_at=2)`: it trains for a few seconds, '
            "then runs live; report the training time.",
            '`compare_controllers(robot="pole", task="balance", methods=["ars", "lqr"], push_at=3)` and compare the '
            "learned policy with the designed one.",
            'Only if the tool list has `train_policy(task="balance", robot="pole")`: follow it with '
            '`training_status(name=...)`, then `run_controller(method="policy:<name>")`.',
        ),
        check=("ars returns success true and its reward is close to lqr's; explain any gap honestly.",),
        challenge='With the rl extra installed, teach the dog to walk: train_policy(task="walk", robot="dog").',
        optional_tools=("train_policy", "training_status"),
    ),
    Lesson(
        number=9,
        slug="import-a-urdf",
        title="Bring your own robot: importing a URDF and reading the doctor's report",
        title_zh="带上你自己的机器人：导入 URDF，读懂体检报告",
        ask="我有一个 URDF 机器人文件，帮我导入进来，告诉我体检报告说了什么，再让它动一动。",
        goal="The learner imports a URDF (their own, or a tiny one written together), understands each line of "
        "the import doctor's report and drives the imported joints.",
        ideas=(
            "URDF is the common file format for robots: links (rigid parts with shape and mass) joined by joints "
            "(with axis and limits). Meshes are separate files it points to, often with package:// paths.",
            "Real URDFs are often incomplete for simulation: missing mass or inertia, missing motors, mesh paths "
            "that only work on the author's computer. The import doctor repairs what it can and reports the rest.",
            "A robot is either bolted down (fixed base, like an arm on a table) or free (floating base, like a dog); "
            "the doctor guesses, and base=fixed or base=floating overrides it.",
        ),
        steps=(
            '`new_world(name="lesson9")` and `open_viewer()`.',
            "Get a URDF path: the learner's own file or folder (URDF, xacro, or a mesh such as .stl), or, if they "
            "have none and you can write files, write a tiny two-joint arm URDF together (one link without "
            "<inertial>, no motors), explaining each tag.",
            '`add_robot(name="myarm", file="<path to .urdf>")`: read the result\'s doctor field line by line '
            "(fixed, NOT fixed, notes) in plain words.",
            "`describe_model()` for the new actuators (`myarm/<joint>`), then "
            '`set_controls(controls={"myarm/<joint>": 0.6})`, `simulate(seconds=1.5)` and `observe()`.',
            'If it falls over or should be free, re-import with params={"base": "fixed"} or {"base": "floating"}; '
            "if the file's torque limits are too weak to hold the robot, the doctor relaxes them unless "
            'params={"respect_effort_limits": true}.',
        ),
        check=(
            "add_robot succeeded and doctor.unfixed is 0, or every unfixed item is explained to the learner.",
            "The joints reach the commanded angles within about 0.1 rad and the image shows the robot moving.",
        ),
        challenge="Import a real open-source robot URDF (for example an open-source arm with its mesh folder) and "
        "try every joint with the viewer sliders.",
    ),
)

START = (
    "Welcome the user to robotforge in their language, in at most five short lines: it is a robot playground on "
    "the MuJoCo physics engine; they describe a robot in plain words and you build, move and explain it. Offer "
    "two ways in: build something right away (ask what robot they want, or suggest a robot dog that walks on "
    "grass), or follow the learning path. Call `lessons()` and show the lessons as a numbered list with "
    "their Chinese or English titles to match the user. Once a robot exists, open the viewer so they can watch."
)


def get(number: int) -> Lesson:
    for lesson in LESSONS:
        if lesson.number == number:
            return lesson
    raise ToolError(f"there is no lesson {number}; lessons go from 1 to {len(LESSONS)}")


def register(server: MCPServer) -> None:
    """Add the `lessons` tool and one prompt per lesson (plus a `start` prompt) to the MCP server."""

    @server.tool()
    def lessons(number: int | None = None) -> str:
        """The beginner learning path: 9 short lessons (joints, driving, walking gaits, inverse kinematics and
        pick-and-place, sensors and cameras, PD vs LQR vs MPC, scripts, training a policy, importing a URDF).
        Without a number it lists them; with a number it returns that lesson's teaching plan (ideas, tool steps,
        checks). When the user asks for a lesson ("teach me lesson 3", "第三课"), call this and follow the plan."""
        if number is None:
            return _json(
                {
                    "lessons": [lesson.summary() for lesson in LESSONS],
                    "how": "Call lessons(number) for a lesson's teaching plan.",
                    "pages": f"{REPO_URL}/tree/main/docs/lessons",
                }
            )
        return _json({"lesson": get(number).plan(), "how_to_teach": TEACHING})

    server.prompt(name="start", title="Start here: robotforge tour and learning path")(_start_prompt)
    for lesson in LESSONS:
        server.prompt(
            name=lesson.prompt_name,
            title=f"Lesson {lesson.number}: {lesson.title}",
            description=f"{lesson.title_zh}. {lesson.goal}",
        )(_lesson_prompt(lesson))


def _start_prompt() -> str:
    return START


def _lesson_prompt(lesson: Lesson):
    def prompt() -> str:
        return lesson.prompt_text()

    prompt.__name__ = lesson.prompt_name
    return prompt


def _json(data: object) -> str:
    return wording.dumps(data)
