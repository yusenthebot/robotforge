"""Deep reinforcement learning (PPO, SAC with Stable-Baselines3) on the built-in tasks.

    from robotforge.control.rl import train

    policy = train("balance", minutes=2)     # needs the optional extra: uv sync --extra rl

Training runs in its own process (train.py, the only module here that imports PyTorch); trained
policies run with numpy alone (runtime.py), as the method "policy:<name>" of run_controller and
compare_controllers. Importing this package does not import gymnasium, SB3 or PyTorch.
"""

from robotforge.control.rl.api import load_policy, train
from robotforge.control.rl.jobs import INSTALL_HINT, JOBS, missing_rl_modules
from robotforge.control.rl.runtime import LearnedPolicyController, MlpPolicy, TrainedPolicy
from robotforge.control.rl.spec import DEFAULTS, TrainSpec, make_spec
from robotforge.control.rl.store import find_policy, list_policies

__all__ = [
    "DEFAULTS",
    "INSTALL_HINT",
    "JOBS",
    "LearnedPolicyController",
    "MlpPolicy",
    "TrainSpec",
    "TrainedPolicy",
    "find_policy",
    "list_policies",
    "load_policy",
    "make_spec",
    "missing_rl_modules",
    "train",
]
