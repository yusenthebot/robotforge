"""Policies as flat parameter vectors, for algorithms that search parameter space (ARS, CEM, your own).

Every policy robotforge trains is an `MlpPolicy` (control/rl/runtime.py): an observation normalizer, a few
matrices and a squash, saved as plain numpy arrays and run without PyTorch. A linear policy is a network with one
layer. `Shape` turns a flat vector into those matrices and back, so an algorithm only handles numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotforge.control.rl.runtime import MlpPolicy

NORM_EPSILON = 1e-8


@dataclass(frozen=True)
class Shape:
    """The layer sizes of a policy: obs_dim -> hidden... -> act_dim."""

    obs_dim: int
    act_dim: int
    hidden: tuple[int, ...] = ()

    @property
    def sizes(self) -> list[int]:
        return [self.obs_dim, *self.hidden, self.act_dim]

    @property
    def n_params(self) -> int:
        return sum((a + 1) * b for a, b in zip(self.sizes, self.sizes[1:], strict=False))

    def unpack(self, theta: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta.size != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters for layers {self.sizes}, got {theta.size}")
        weights, biases, at = [], [], 0
        for a, b in zip(self.sizes, self.sizes[1:], strict=False):
            weights.append(theta[at : at + a * b].reshape(b, a))
            at += a * b
            biases.append(theta[at : at + b])
            at += b
        return weights, biases

    def pack(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
        return np.concatenate([np.concatenate([w.ravel(), b]) for w, b in zip(weights, biases, strict=True)])

    def zeros(self) -> np.ndarray:
        return np.zeros(self.n_params)

    def random(self, rng: np.random.Generator, scale: float = 0.1) -> np.ndarray:
        """Small random weights (scaled by the layer's input width), zero biases."""
        weights = [
            rng.normal(0.0, scale / np.sqrt(a), (b, a)) for a, b in zip(self.sizes, self.sizes[1:], strict=False)
        ]
        return self.pack(weights, [np.zeros(b) for b in self.sizes[1:]])

    def policy(self, theta: np.ndarray, mean: np.ndarray, var: np.ndarray, clip_obs: float = 10.0) -> MlpPolicy:
        weights, biases = self.unpack(theta)
        return MlpPolicy(
            [w.copy() for w in weights],
            [b.copy() for b in biases],
            "tanh" if self.hidden else "identity",
            "clip",
            np.asarray(mean, dtype=np.float64),
            np.asarray(var, dtype=np.float64),
            clip_obs,
            NORM_EPSILON,
        )


class RunningStats:
    """Mean and variance of every observation seen (Welford), for normalizing inputs."""

    def __init__(self, n: int) -> None:
        self.count = 0
        self.mean = np.zeros(n)
        self.m2 = np.zeros(n)

    def push(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def var(self) -> np.ndarray:
        if self.count < 2:
            return np.ones_like(self.mean)
        return np.maximum(self.m2 / (self.count - 1), 1e-4)

    def snapshot(self, center: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(mean, var) to normalize with. center=False keeps the zero point and scales by the typical size
        (root mean square): early episodes are mostly failures, and shifting by their mean would make the policy
        push even at rest."""
        if self.count < 2:
            return np.zeros_like(self.mean), np.ones_like(self.mean)
        if center:
            return self.mean.copy(), self.var
        rms2 = np.maximum(self.var + self.mean**2, 1e-4)
        return np.zeros_like(self.mean), rms2

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"stats_count": np.array(self.count), "stats_mean": self.mean, "stats_m2": self.m2}

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], n: int) -> RunningStats:
        out = cls(n)
        if "stats_mean" in arrays and arrays["stats_mean"].shape == (n,):
            out.count = int(arrays["stats_count"])
            out.mean = np.array(arrays["stats_mean"], dtype=float)
            out.m2 = np.array(arrays["stats_m2"], dtype=float)
        return out
