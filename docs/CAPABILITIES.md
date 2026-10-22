# Robot capabilities

`BehaviorManager.robots()` supplies one capability description for the assembled MuJoCo model.
Templates, imported files, library models and robots assembled from parts or mounts use the same inspection path.
MCP `robots`, script `sim.robot(name).capabilities` and viewer descriptions carry the same JSON value.
Existing tool signatures and the legacy `name`, `kind`, `model`, `arms`, `max_speed`, flight and balance fields remain available.
`kind` is a body classification; it does not establish that an operation is supported.

## Contract

Each robot has a `capabilities` object with `schema_version: 1`.

| Field | Meaning |
| --- | --- |
| `source` | Primary source kind and identifier, composition flag and mounted source entries. A directly opened model uses `kind: model`. |
| `move` | `supported`, unavailable `reason`, command `axes` and, when supported, `max_speed` in m/s. |
| `move_arm` | Whether any powered arm has a TCP, and its fully qualified `endpoints`. |
| `gripper` | Whether an operable arm has a gripper actuator, and the corresponding endpoints. |
| `arms` | Per endpoint: `name`, `tcp`, `move` and `gripper` support. |
| `manual` | Raw control availability and fully qualified actuator names. |
| `sensors` | Named MuJoCo sensors belonging to the robot. Script sensor conveniences remain available separately. |
| `tasks` | Fit and unavailable reason for the viewer's existing control task choices. |
| `drive_modes` | Built-in drive/manual/learned/rhythm choices, before adding the task modes. |

An arm classification without a TCP does not invent an arm endpoint.
A bare arm can support reaching without supporting gripping.
A standalone gripper retains raw controls but cannot move itself as an arm.
An imported humanoid without a supported walk remains capable of standing and manual control, without claiming walking support.
Compositions retain the base's movement and each arm's endpoint, for example `carrier/left` and `carrier/right`.
The viewer keeps its existing task/body menu policy; this change does not expand task or algorithm support.

## Discovery and lifetime

Structural inspection is cached by the compiled model identity and robot classification.
World recompilation, adding or removing a mount and model replacement invalidate that cache.
Descriptions are serialized into fresh dictionaries, so a consumer cannot mutate the shared cached structure.
World and lesson metadata are read before acquiring the simulation lock.

Mobility discovery shares binding and speed calculations with runtime controllers, without constructing a gait, IK solver, LQR/preview controller, marine mixer or ONNX policy.
Kinematic checks use private MuJoCo data.
Existing balancing-body recognition can rehearse stability on private simulation data; discovery never advances the live clock or changes live controls.
Active mover limits take precedence where the existing interface did so, including changes to a gait period.
Marine limits use the current immersion and water drag; library and payload limits retain their previous rules.

Saved walks and policy cards are read afresh.
A saved walk must match the current driven joints before it establishes movement support.
Legacy listing fields for a mismatched walk are retained for compatibility, while `capabilities.move` reports why it cannot run.
Policy cards identify a configured provider without loading or certifying its model during discovery.
Actual commands still validate the runtime binding, optional runtime, state, target and execution ownership.
Support describes an available interface and binding, not success at a particular task or pose.

Script robot objects hold a snapshot.
Use `robot.refresh()` or call `sim.robot(name)` again after changing a saved walk.
Servers without `capabilities` retain the script facade's legacy inference behavior.
`robot.arms[i].gripper` addresses a specific arm; `robot.gripper` requires one unambiguous gripper.

The viewer broadcasts changed descriptions at most once per second, without resending meshes.
Joining clients receive a fresh scene without consuming updates for existing clients.
Updates carry the model version and are buffered during mesh loading, then applied to the matching scene.
The viewer uses capability fields for driving, gamepad profiles and available modes, with legacy fallback for old servers.
Multiple arm endpoints require explicit selection through MCP or scripts; the gamepad does not choose an arm implicitly.

## Validation

Focused regression coverage lives in `tests/test_capabilities.py`, `tests/test_mobility_description.py` and `tests/test_viewer_padmap.py`.
It covers source consistency, unsupported operations, compositions, cache isolation and invalidation, saved walk changes, legacy mode arguments, constructor-free discovery and exact runtime speed parity.
Real CLI/MCP, MuJoCo movement and Chrome verification are recorded in `bench/2026-10-21-capabilities.md`.
Controller algorithms, drawer behavior and individual manipulation success rates are outside this change.
