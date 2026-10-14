"""Algorithm adapters: one task, many ways to solve it.

    from robotforge.control import compare_methods, make_env

    env = make_env("reach")                       # Gym-style environment, no extra dependencies
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    print(compare_methods("balance", ["lqr", "pd", "mpc"], seconds=5, push_at=2.5).table)

Tasks (tasks/): reach, balance, stand, walk, drive_to. Methods (methods.py): ik, pd,
cartesian, lqr, mpc, mppi, hold, gait, none. Every method is a session Controller, so
it can also drive the live world in the viewer.
"""

from robotforge.control.compare import Comparison, RunResult, compare_methods
from robotforge.control.env import RobotEnv, make_env, task_world
from robotforge.control.methods import METHODS, make_controller
from robotforge.control.mpc import MPCConfig, SamplingMPC
from robotforge.control.spaces import Box
from robotforge.control.tasks import TASKS, Task, TaskError, make_task

__all__ = [
    "METHODS",
    "TASKS",
    "Box",
    "Comparison",
    "MPCConfig",
    "RobotEnv",
    "RunResult",
    "SamplingMPC",
    "Task",
    "TaskError",
    "compare_methods",
    "make_controller",
    "make_env",
    "make_task",
    "task_world",
]
