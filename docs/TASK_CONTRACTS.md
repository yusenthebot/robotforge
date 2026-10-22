# Task and algorithm contracts

Task discovery and binding share `control.task_registry` across MCP, scripts, the viewer and training.
The plugin registry remains the owner of plugin loading, trust, reload and unload.
Existing MCP tools and their signatures are unchanged.

## Tasks

`task_definitions(world)`, `resolve_task(name, world)` and `bind_task(name, model, robot, params, world=...)` cover built-in tasks, tasks saved with a world and plugin factories, in that precedence order.
`TASKS` remains the built-in compatibility mapping; `make_task` delegates to the common binder.
Catalog inspection reads metadata without invoking plugin factories or constructing controllers.

`@task(name, params=...)` retains its existing behavior.
It also accepts `schema`, `methods` and `robot_hint`:

```python
from pydantic import BaseModel, ConfigDict, Field
from robotforge.extend import task
from robotforge.control.tasks.reach import Reach


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tolerance: float = Field(0.02, gt=0)


@task("precise_reach", schema=Options, methods=("ik", "pd"), robot_hint="an arm")
def build(model, robot, params):
    return Reach(model, robot, params)
```

An explicit schema supplies its own coercion, bounds and extra-field policy.
Legacy factories without a schema keep unknown JSON options and receive fresh copies of their defaults.
Parameters must be finite; Python NumPy scalars and arrays are normalized before validation, and tuples remain available to strict Python schemas.
Built-in schemas are selected after model adaptation, so WheelBalance keeps its existing defaults and limits.
Saved-world tasks retain their saved configuration and robot selection.

Binding checks that factories return a Task for the current model, with valid actuator indices belonging to the requested robot or its mounted descendant.
Live control and playback also reject a saved task belonging to a different requested robot.
These checks establish compatibility, not task success.

`Task.name` retains its implementation meaning for existing controller dispatch.
`registered_task_name(task)` identifies the registered task; `describe_task(task)` reports that identity and adds `implementation_task` when they differ.
Policies store the registered identity and the environment spec.
Playback prefers `env.task`, which also supports older plugin policy records whose top-level task was `custom` or an underlying built-in name.

## Control methods and training algorithms

`control.methods` owns control method names, adapters and parameter schemas.
`resolve_method`, `check`, `validate_params` and `make_controller` apply the same rules at every control boundary.
Schemas derive from existing adapter signatures and MPCConfig.
Validation does not insert controller defaults or alter control math.
Method aliases are normalized; policy names retain their case and are checked by the existing policy store.
Unknown or nonfinite settings fail before attachment.
Residual policies require a valid declared base controller; a direct policy can still train a reward task with no live control methods.

`training.algorithms` remains the common registry for built-in and plugin training algorithms.
Its `Algorithm` uses the existing `Hyper` settings validation and dependency checks.
Settings must name a selected algorithm; aliases cannot supply the same settings twice.
Workers validate settings and dependencies before creating an environment, and close the environment when later initialization fails.

Control methods and training algorithms retain distinct domains.
For example, live control ARS and training ARS have different interfaces and defaults; a catalog does not make them interchangeable.

## Entry points

`control.runtime.contract_catalog(world)` supplies `tasks`, legacy `algorithms`, additive `method_schemas` and `training_algorithms`.
MCP `list_algorithms`, script `sim.task_catalog()` and viewer `scene.task_contracts` expose the same information.
Existing learning and policy fields remain available in the MCP response.

`sim.run_controller(robot, task, method, task_params=None, method_params=None)` attaches a controller; `sim.step` advances time.
It shares the script's execution owner, cancellation handling and budgets.
Existing `sim.start_controller` still runs a plugin controller by name, and `sim.run_policy` remains compatible.
The viewer's `task_run` request uses the same binding and execution admission; its existing Training Play button uses the same saved-policy reconstruction.
The existing curated mode menus remain unchanged.

Stored policy task settings are restored before explicit overrides across these entry points.
Comparisons use the policy's saved goal for all compared controllers and reject conflicting saved goals from multiple policies unless explicit settings reconcile them.
Contact comparison duration remains derived from its initialized task.

Viewer catalog caches include task and algorithm registration identities and saved task definitions.
Reloads and removals refresh existing viewers without rebuilding meshes or consuming another viewer's update.

## Verification

Contract regression tests are in `test_task_registry.py`, `test_method_contracts.py`, `test_task_contracts.py`, `test_script_viewer_contracts.py` and `test_training_contracts.py` under `tests/`.
Real installed CLI/MCP, separate training workers, script processes and Chrome checks are recorded in [the verification report](bench/2026-10-22-task-contracts.md).
Algorithm tuning and manipulation success rates are outside this change.
