"""Write the tiny ONNX policy used by tests/test_onnx_policy.py.

The model is one affine layer, action = W @ obs + b, with W copying the joint_pos_rel slice of
a Go2-style observation (values 6..17) into the 12 actions, and b a known per-joint ramp. Tests
can then predict every output exactly, which checks the observation layout, the joint order
and the action mapping end to end through onnxruntime.

`onnx` is only needed to build the file, not to use it, so it is not a robotforge dependency:

    uv run --no-project --with onnx --with numpy python scripts/make_policy_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OBS_DIM = 49
JOINTS = 12
JOINT_POS_START = 6  # after base_ang_vel (3) and gravity (3)
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "policies" / "echo_joint_pos.onnx"


def bias() -> np.ndarray:
    return (np.arange(JOINTS, dtype=np.float32) - 5.5) * 0.01


def main() -> None:
    weights = np.zeros((JOINTS, OBS_DIM), np.float32)
    weights[np.arange(JOINTS), JOINT_POS_START + np.arange(JOINTS)] = 1.0
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["obs", "W", "b"], ["actions"], transB=1)],
        "echo_joint_pos",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, OBS_DIM])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, JOINTS])],
        [numpy_helper.from_array(weights, "W"), numpy_helper.from_array(bias(), "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="robotforge-tests")
    model.ir_version = 8  # readable by every onnxruntime >= 1.20
    onnx.checker.check_model(model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(OUT))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
