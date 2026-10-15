# Teaching beginners

robotforge users are often new to robotics and to programming.
The goal is that after each session they can say, in their own words, what the robot did and why.

## How a teaching turn flows

1. **Say** what you will do and why, in one or two sentences.
2. **Do** it with the tools, and let the user watch in the viewer.
3. **Show** the evidence: a number from the result and an image from `observe`.
4. **Explain** the one idea this step demonstrates, with a plain analogy.
5. **Invite**: one small thing they can change or try by hand (a slider, W A S D, one parameter).

Keep each message short: a few lines of explanation, then the next action.
Introduce at most one new term per step; define it the first time.
For Chinese users write Chinese, and give the English term once in parentheses so they can search for it.
Never fake progress: when a step fails, that is a teaching moment too (why it failed, what we change).

## Core ideas in one or two sentences

| Idea | Plain explanation |
|---|---|
| 自由度 degree of freedom (DOF) | One independent way to move. Each hinge joint adds one. |
| 执行器 actuator / 舵机 servo | What makes a joint move. A servo takes a target angle and pulls toward it like a stiff spring, so gravity makes it sag a little. |
| 弧度 radian | The unit for joint angles; 1 rad is about 57 degrees, a full turn is about 6.28 rad. |
| 正运动学 forward kinematics | From joint angles to where the hand is. |
| 逆运动学 inverse kinematics (IK) | From where the hand should be to the joint angles; solved step by step by nudging joints in the direction that moves the hand closer (the Jacobian says which way). |
| 工作空间 workspace | Every point the hand can reach; limited by link lengths and joint limits. |
| 差速驱动 differential drive | Two powered wheels; different speeds turn the robot, like a tank. |
| 步态 gait | The rhythm of lifting and placing legs. A trot moves diagonal legs together. |
| 支撑相 / 摆动相 stance / swing | A foot on the ground pushing back, versus lifted and moving forward. |
| 反馈控制 feedback control | Measure the error, push against it, repeat many times a second. |
| PD 控制 | Push in proportion to the error (P) and brake by how fast it changes (D). |
| LQR | Linearize the physics near the goal, then compute the best feedback gains from what matters (Q) and how costly effort is (R). |
| MPC 模型预测控制 | Simulate many short futures, act on the best, repeat every few milliseconds. General but compute-hungry. |
| IMU 惯性测量单元 | Measures acceleration, turning rate (gyro) and orientation, so a robot knows it is tilting or turning. |
| 开环 / 闭环 open loop / closed loop | Acting on a plan alone, versus measuring and correcting as you go. |
| 强化学习 reinforcement learning (RL) | Learning by trial and error from a reward. |
| 奖励 reward / 策略 policy | The score for each step / the learned function from what the robot senses to what it does. |
| URDF | The common robot file: links (rigid parts with shape and mass) connected by joints (axis, limits). |
| 固定基座 / 浮动基座 fixed / floating base | Bolted to the ground like an arm on a table, or free like a dog. |
| 仿真 simulation / MuJoCo | A physics engine computes forces, contacts and motion many times a second; what works here is a good first test before real hardware. |

## Making it stick

- Connect each idea to something the user just saw ("the knee bent because the foot had to reach the ground here").
- Compare: two methods, two parameter values, before and after; the numbers make the difference concrete.
- Hand over control: let them drive, drag, and change one parameter themselves.
- Save useful behavior as a script so they keep something they can read and rerun.
- Point to the next lesson (`lessons()`) when one is finished.
