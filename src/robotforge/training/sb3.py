"""PPO and SAC from Stable-Baselines3 as training algorithms. Imports PyTorch: only a training process loads this.

The network is trained on the trial's own environment (one simulation per trial: a search runs several trials at
once instead). Its episodes and losses are reported to the training view as they happen, and it can be continued
by a later segment (successive halving) because the model and the observation normalizer are kept in the trial's
folder. The result is exported to plain numpy arrays (control/rl/runtime.MlpPolicy), so it runs without PyTorch.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn

from robotforge.control.gym_compat import as_gymnasium
from robotforge.control.rl.train import export_policy

MODEL_FILE = "model.zip"
NORM_FILE = "vecnormalize.pkl"
LOSS_KEYS = ("loss", "value_loss", "policy_gradient_loss", "entropy_loss", "actor_loss", "critic_loss", "ent_coef")


class Reporting(gymnasium.Wrapper):
    """Tells the training context about every finished episode (its return, length and success)."""

    def __init__(self, env: gymnasium.Env, ctx: Any) -> None:
        super().__init__(env)
        self.ctx = ctx
        self.total = 0.0
        self.length = 0

    def reset(self, **kwargs: Any):
        self.total, self.length = 0.0, 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total += reward
        self.length += 1
        if terminated or truncated:
            self.ctx.note_episode(self.total, self.length, bool(info.get("is_success")))
        return obs, reward, terminated, truncated, info


class Progress(BaseCallback):
    """Stops when the context says so, and sends the losses to the training view."""

    def __init__(self, ctx: Any) -> None:
        super().__init__()
        self.ctx = ctx

    def _on_step(self) -> bool:
        return not self.ctx.expired()  # the step budget is learn()'s own total_timesteps

    def _on_rollout_end(self) -> None:
        values = self.model.logger.name_to_value  # the losses of the update that just ran
        found = {key.split("/")[-1]: float(v) for key, v in values.items() if key.split("/")[-1] in LOSS_KEYS}
        if found:
            self.ctx.log(**found)


def run(ctx: Any, algo: str):
    torch.set_num_threads(1)  # small networks: one thread is fastest, and a search runs several trials at once
    p = ctx.params
    folder = ctx.folder
    model_path, norm_path = folder / MODEL_FILE, folder / NORM_FILE
    venv = DummyVecEnv([lambda: Reporting(as_gymnasium(ctx.env), ctx)])
    width = int(p["width"])
    net = [width, width]
    settings = {k: v for k, v in p.items() if k != "width"}
    if algo == "ppo":
        # One update uses n_steps experiences: keep the minibatch a divisor of it.
        while settings["n_steps"] % settings["batch_size"]:
            settings["batch_size"] //= 2
    if model_path.exists() and norm_path.exists():
        norm = VecNormalize.load(str(norm_path), venv)
        model = (PPO if algo == "ppo" else SAC).load(str(model_path), env=norm, device="cpu")
    else:
        norm = VecNormalize(venv, norm_obs=True, norm_reward=algo == "ppo", gamma=settings.get("gamma", 0.99))
        if algo == "ppo":
            kwargs = {"net_arch": {"pi": net, "vf": net}, "activation_fn": nn.Tanh}
            if ctx.env.action_mode == "residual":
                kwargs["log_std_init"] = -1.0  # explore gently around a controller that already works
            model = PPO("MlpPolicy", norm, policy_kwargs=kwargs, seed=ctx.seed, device="cpu", **settings)
        else:
            kwargs = {"net_arch": {"pi": net, "qf": net}, "activation_fn": nn.ReLU}
            model = SAC("MlpPolicy", norm, policy_kwargs=kwargs, seed=ctx.seed, device="cpu", **settings)
    try:
        model.learn(total_timesteps=max(1, ctx.budget), callback=Progress(ctx), reset_num_timesteps=False)
        model.save(str(model_path))
        norm.save(str(norm_path))
        policy = export_policy(model, norm)
    finally:
        venv.close()
    ctx.sink.steps = ctx.steps
    return policy


__all__ = ["run"]
