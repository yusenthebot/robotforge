# Common experiment records

Control comparison, training and live playback now use `robotforge.experiments.ExperimentRecord` schema version 1.
Records add references to existing results; the 60 MCP tool definitions, policy folder names and existing result fields remain compatible.
No controller formula, task reward or manipulation success-rate setting changes here.

## What is recorded

Each execution has a UUID, kind, lifecycle status and timestamps.
Its specification includes environment, registered task identity and implementation name, validated task parameters, algorithm and resolved settings, seed roles, requested budget and code/runtime/plugin versions.
Measured usage, namespaced metrics, termination reason, parent/origin references and SHA256 artifact manifests are separate fields.
The specification fingerprint excludes execution UUIDs, timestamps and known storage locations, while retaining arbitrary user parameter keys and content identities of environment, policy and resume inputs.
Different executions can share a specification fingerprint.
A fingerprint is an integrity check and input identity, not authentication or proof of deterministic behavior.

Records live under the current world's `experiments/`, or `$ROBOTFORGE_HOME/experiments` without a saved world.
Training records are outside replaceable `policies/<name>/` folders.
Each record owns an atomic `experiment.json` and relative `artifacts/<sha>.<format>` files; a shared content store avoids duplicating identical bytes when hard links are available.
Reads reject unsafe artifact paths, containment escapes, wrong size and wrong content hashes.
Terminal records cannot be updated through the store API.

## Environment and reproducibility

Snapshots contain the compiled MuJoCo model (MJB), complete `mjSTATE_INTEGRATION` state (NPY), and a world specification when available.
This captures controls, applied forces, mocap and solver warm starts in addition to positions and velocities.
Restoring requires the exact recorded MuJoCo version; it does not bundle an editable source asset tree or install recorded plugins.
Version metadata includes active installed package contents, Python, NumPy, MuJoCo, platform and already loaded plugin/helper file hashes.
Plugin hashes describe current files, not serialized Python memory or external services.

Comparison records retain the original current state and the actual reset seed: Python's default is 0, while the MCP current-state comparison retains `None`.
Each method has its own child record, identical randomized start, final integration state, metrics and terminal status.
Policy comparisons run immutable archived inputs so a concurrent checkpoint replacement cannot change which weights they execute.
`reproduce_comparison` recreates a private comparison and refuses incompatible runtime/plugin versions and unsupported callbacks or task factories.
Wall-time measurements are expected to differ.

Training records contain a run parent and one child per trial segment or halving rung.
They retain `RunSpec` or legacy `TrainSpec`, actual environment scope, algorithm settings, checkpoint inputs, training/search seeds, ranking seeds and held-out seed protocols.
The coordinator writes its parent and workers write their own children; queued cancellation, launch/submit failures and worker failures have terminal records.
Requests and actual consumed training steps are distinct because whole algorithm iterations can overshoot a step request.
Policy, checkpoint and progress artifacts survive replacement of the named policy folder.
The winning policy origin and a viewer-selected trial origin are matched against actual archived weight bytes; unmatched legacy lineage remains unknown.
The legacy PPO/SAC bridge records its existing final-evaluation seed protocol separately from the newer pipeline.
A training snapshot alone cannot reconstruct an arbitrary plugin's internal RNG, external files or optional backend runtime.

Live control and policy replay record the actual attachment state and archive the loaded network and metadata.
They preserve the original controller object and simulation scheduling.
Stopping, replacing, resetting, loading a model and shutting down close observations before discarding their state.
A bounded MCP measurement can finish while its controller remains attached, matching existing behavior.
Unbounded script/viewer attachments record a nullable requested duration and their actual lifetime.
Contact or plugin task metrics that require unperformed observations are `null`, with an explicit coverage reason.
Nonfinite measurements are `null` with their field names recorded.
Recording failures are logged, and post-attachment storage failures preserve script ownership so cancellation can still stop the controller.

`reproduce_replay` restores a closed isolated attachment in a private simulation, advances its recorded physics-step count and reports both exact equality and maximum absolute error for final qpos/qvel/ctrl, with a separate 1e-8 absolute-tolerance result.
It refuses unknown callable state, on-the-spot ARS internal policies, other controllers (including hidden wind/mechanism controllers), replacement release effects, incompatible runtime/plugins and active MuJoCo callbacks.
Interactive changes after attachment are not replayed, so the returned parity and error fields must be checked.
Controller preparation can precede attachment while a live world advances, and derived MuJoCo fields are recomputed on restore, so bitwise equality is not guaranteed for every live replay.
This is physical replay evidence, not evidence of task quality or learned-policy success.

## Existing interfaces

MCP `compare_controllers`, `run_controller`, `train_policy` and training status gain additive `experiment` references.
Training metadata also retains `policy_experiment`, and status exposes each trial's segment references.
Scripts and viewer adapters use the same record store and attachment lifecycle.
Scripts can query the current world's history and inspect a record by canonical UUID:

```python
from robotforge.script import connect

sim = connect()
print(sim.experiments(limit=20))
run = sim.run_controller("arm", "reach", "ik")
sim.step(0.2)
sim.stop("arm")
print(sim.experiment(run["experiment"]["id"]))
```

Low-level comparison recording stays opt-in for existing Python callers:

```python
from pathlib import Path
from robotforge.control.compare import compare_methods, reproduce_comparison
from robotforge.experiments.playback import reproduce_replay

comparison = compare_methods("balance", ["lqr", "pd"], seconds=0.2, record_dir=Path("experiments"))
repeated = reproduce_comparison(comparison.experiment["path"])
# For a closed isolated live attachment:
# result = reproduce_replay(record_path)
# assert result["matches_recorded_final_state"]
```

There is no new dependency, training backend, MCP tool or release in this change.
See the [verification report](bench/2026-10-22-experiments.md) for the actual executed checks and video.


## Owned executions and regression

MCP `run_controller` also returns an additive `execution` reference.
That execution record links its original control/replay attachment through origins and records cumulative preparation/measurement budgets and terminal reason.
A reset can close the attachment before the execution reports invalidation; the execution never copies a replacement world's state as its old final state.
The fixed [framework regression suite](FRAMEWORK_REGRESSION.md) validates the common records and compares versioned inputs and physical outputs across source revisions.
Source differences remain visible; incompatible runtime or incomplete evidence cannot be counted as a pass.

MCP `pose` and `gripper` use the same recorded execution lifecycle and return the same additive reference.
Their records retain the requested behavior settings, resolved endpoint, captured initial integration state, actual steps, final arrays and terminal reason.
`behavior_done` reports whether the existing controller finished its behavior; a bounded five-second gripper observation can leave earlier work queued.
Position-servo hold, mounted-carrier ownership and the humanoid's existing zero-speed stop semantics are preserved, as described in [EXECUTION.md](EXECUTION.md).
