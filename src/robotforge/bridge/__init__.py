"""Bridges to learned policies trained outside robotforge.

- `spec`: the policy card (policy.json) that says exactly how a policy sees and moves its robot.
- `observations`: builds the policy's observation vector from the live MuJoCo state.
- `onnx_policy`: runs an ONNX policy (optional extra `policy` = onnxruntime) as a walking controller.
- `policies`: where policies live (shipped with robotforge or installed by the user).
- `unilab`: the UniLab training backend, run as a subprocess in UniLab's own environment.
"""
