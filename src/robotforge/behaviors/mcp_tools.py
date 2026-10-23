"""MCP tools for robots working the things around them: operate drawers, doors, buttons, switches, levers and valves.

Registered by robotforge.mcp_server.build_server via `register(...)`, one line there.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from robotforge.behaviors import operate as operating
from robotforge.behaviors import visit as visiting
from robotforge.behaviors.execution import ArmExecution
from robotforge.experiments.execution import RecordedExecution
from robotforge.playground import Playground
from robotforge.sim import mechanisms

OPERATE_SECONDS = 90.0  # sim seconds an operation may take: a valve turned a full turn takes about 50


def register(
    server: MCPServer,
    playground: Playground,
    run_until: Callable[..., Awaitable[dict[str, Any]]],
    *,
    watched: Callable[[], bool] = lambda: False,
    wall_limit: Callable[[], float] = lambda: 60.0,
) -> None:
    from robotforge.mcp_server import _json, _tool_errors  # here: mcp_server imports this module

    execution = RecordedExecution(playground, watched, wall_limit, owner="operate", sim_seconds=150.3, phase_count=2)

    @server.tool()
    @execution.owned
    async def operate(
        robot: str,
        target: str,
        action: Literal["open", "close", "press", "turn"],
        amount: float | None = None,
        go: bool = True,
        force: float | None = None,
    ) -> str:
        """Have an arm work a mechanism: it takes hold of the handle and moves it along the joint's own line or arc.
        drawer: open / close, amount in m (default: all the way). cabinet: open / close the door, amount in degrees
        (default 90 open, all the way closed). button: press (it springs back). switch, lever: press flips it to the
        other side; turn tips it by amount degrees (+ toward -y). valve: open (counter-clockwise) / close / turn by
        amount degrees (default 90; past 90 it lets go and takes hold again until the wheel is there). Works with
        fingers or a suction cup; a magnet holds only the steel bars of drawers and doors (and pushes buttons). The
        result: moved (m or degrees), the state now, success (judged against how far the thing can go), and notes
        when the arm could not follow the whole way or did not take hold; a handle out of reach is refused with its
        distance. An arm on a robot that drives or walks ("go2/arm") goes over by itself when the handle is out of its
        reach or it would not follow it the whole way: the robot walks or drives to a spot from which the arm does the
        whole job, clear of what moves (a drawer pulled out, a door swung open), stands still while the arm works and
        folds the arm back; the result then also says "went", "stood_at" ([x, y, heading_deg]) and "visit_s".
        go=false keeps the robot where it stands. force (newtons) makes the hand push the handle with at most that
        force along its way instead of dragging it with its servos: a handle that takes more stops the hand (the
        result's "force" says the limit, how hard it pushed and felt, and whether it stalled; for an arm that works
        the mechanism from where it stands). Put things in an open drawer with pick_and_place(robot, object, drawer)."""
        async with ArmExecution(playground, execution.call.run) as family:
            execution.call.metrics["phases"] = family.phases
            with _tool_errors(preserve_execution=True):
                behaviors = playground.behaviors
                # Standing-place planning runs in a tracked worker, under the same execution deadline.
                arm = behaviors.resolve(robot, arm=True)
                family.reserve(arm)
                execution.call.record.update(
                    task={"domain": "behavior", "robot": arm, "name": "operate"},
                    algorithm={
                        "domain": "behavior",
                        "name": "operate",
                        "params": {"target": target, "action": action, "amount": amount, "go": go, "force": force},
                    },
                    seeds={"mode": "current_state"},
                )
                op, found, shape = await family.plan(
                    operating.start, behaviors, robot, target, action, amount, go, force
                )
                trip = op if isinstance(op, visiting.Visit) else None
                family.budget(execution, [OPERATE_SECONDS if trip is None else trip.time_limit, 0.3])
                if trip is None:
                    run = await family.advance(lambda: op.done, OPERATE_SECONDS, "operation")
                else:  # a reset or a world edit ends it; one no longer attached is waited for no more
                    run = await family.advance(
                        lambda: op.done or behaviors.errand(trip.robot) is not trip, trip.time_limit, "visit"
                    )
                    if not trip.done:  # out of time: it stops where it is
                        with family.command(check_deadline=False):
                            behaviors.stop(trip.robot)
                        trip.interrupt(
                            "it ran out of time; call operate again to go on", float(playground.session.data.time)
                        )
                        trip.why = "out_of_time"
                settle = await family.advance(lambda: False, 0.3, "settle", legacy_elapsed=True)
                family.check()
                execution.call.metrics["behavior_done"] = bool(op.done)
                session = playground.session
                with session.lock:
                    family.check()
                    state = mechanisms.read(
                        session.model, session.data, {found: shape}, session.controllers.get(mechanisms.KEY)
                    ).get(found, {})
                    if trip is None:
                        report = operating.result(op, session.model, session.data, state)
                    else:
                        report = trip.result(session.model, session.data, state)
                seconds = round(run["sim_seconds"] + settle["sim_seconds"], 2)
                return _json(
                    {
                        **report,
                        "sim_seconds": seconds,
                        "shown_live": run["shown_live"],
                        "execution": execution.reference(),
                    }
                )
