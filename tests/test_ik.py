"""The IK solver's guard rails (behaviors/ik.py): its fast linear solve must fail like numpy's, never with NaN."""

from __future__ import annotations

import warnings

import mujoco
import numpy as np
import pytest

from robotforge.behaviors import ik

TWO_LINKS = """
<mujoco>
  <worldbody>
    <body name="upper">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-170 170"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      <body name="lower" pos="0.3 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-170 170"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_the_fast_solve_gives_numpys_numbers_on_the_sizes_ik_solves():
    rng = np.random.default_rng(3)
    for n in (3, 6):
        for _ in range(20):
            j = rng.standard_normal((n, n))
            a, b = j @ j.T + 1e-4 * np.eye(n), rng.standard_normal(n)
            assert np.array_equal(ik._solve(a, b), np.linalg.solve(a, b))


def test_a_singular_system_raises_like_numpy_instead_of_returning_nan():
    """LAPACK's solve without numpy's wrapper returned [nan nan nan] here; numpy raises."""
    singular = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the fast path's own warning, before it hands over
        with pytest.raises(np.linalg.LinAlgError, match="Singular"):
            ik._solve(singular, np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("damping", [0.0, -1e-3, float("nan")])
def test_ik_refuses_damping_that_is_not_positive(damping):
    model = mujoco.MjModel.from_xml_string(TWO_LINKS)
    chain = ik.Chain.from_joints(model, [0, 1], body=2, local_point=[0.3, 0.0, 0.0])
    solver = ik.IKSolver(model)
    with pytest.raises(ValueError, match="damping must be positive"):
        solver.solve(chain, np.array([0.4, 0.0, 0.2]), np.zeros(2), damping=damping)


def test_ik_still_reaches_a_target_with_its_usual_damping():
    model = mujoco.MjModel.from_xml_string(TWO_LINKS)
    chain = ik.Chain.from_joints(model, [0, 1], body=2, local_point=[0.3, 0.0, 0.0])
    q, err = ik.IKSolver(model).solve(chain, np.array([0.4, 0.0, 0.2]), np.array([0.1, 0.3]), iterations=50)
    assert err < 1e-3
    assert np.all(np.isfinite(q))
