"""Behavior trees and state machines for robot scripts, and the viewer shows which node is running.

    from robotforge.flow import Action, Condition, Fallback, Retry, Sequence, Tree
    from robotforge.script import connect

    sim = connect()
    robot = sim.robot("rover")
    tree = Tree(
        Sequence(
            Condition(lambda c: c.robot.sensors["lidar"]["ahead"]["distance_m"] > 0.5, name="path clear"),
            Retry(Action(lambda c: c.robot.arm.pick_and_place("cube", "tray"), name="fetch the cube"), times=3),
        ),
        name="errand",
    )
    print(tree.run(sim, robot))      # success, failure; the viewer's tree panel follows it node by node

A tree is ticked every `tick` simulated seconds until it stops running. Every node answers SUCCESS, FAILURE or RUNNING:

- Leaves: `Action(fn)` calls fn(ctx) (True or None is success, False failure, a Status is itself; a call that blocks,
  such as arm.move_to, simply blocks inside the tick), `Condition(fn)` tests something (a sensor, say), `Wait(seconds)`,
  `WaitFor(test, timeout)`.
- Composites: `Sequence` (all, in order), `Fallback` (the first that works), `Parallel` (all at once). The first two
  carry on from the child that was running; `reactive=True` re-checks the earlier children every tick.
- Decorators: `Retry(child, times)`, `Repeat(child, times)`, `Inverter`, `Timeout(child, seconds)`.
- `StateMachine` with `State`s and transitions is a node too, so the two nest.

The context `ctx` (what fn gets) has ctx.sim, ctx.robot, ctx.time, ctx.bb (a dict nodes share) and ctx.log(text).
Needs only what scripts have (a Sim), so it imports nothing heavy; tests use any object with time and step(seconds).
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable, Iterator
from collections.abc import Sequence as Seq
from typing import Any

from robotforge.script import ScriptError, ScriptTimeLimit

__all__ = [
    "Action",
    "Condition",
    "Context",
    "Fallback",
    "FlowError",
    "Inverter",
    "Node",
    "Parallel",
    "Repeat",
    "Retry",
    "Sequence",
    "State",
    "StateMachine",
    "Status",
    "Timeout",
    "Tree",
    "Wait",
    "WaitFor",
    "sensor_is",
]

TICK = 0.1  # simulated seconds between ticks of a running tree
MAX_SECONDS = 120.0  # simulated seconds a tree may run before Tree.run gives up
NOTE_LIMIT = 80  # characters of a node's note (why it failed) the viewer shows


class FlowError(ValueError):
    """A tree or state machine that is put together wrongly (an unknown state, nothing to run)."""


class Status(enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"

    def __str__(self) -> str:
        return self.value


SUCCESS, FAILURE, RUNNING = Status.SUCCESS, Status.FAILURE, Status.RUNNING

_LETTER = {None: ".", Status.RUNNING: "r", Status.SUCCESS: "s", Status.FAILURE: "f"}


def _answer(value: Any) -> Status:
    """What a leaf's function returned, as a Status: True or None is success, False failure, a Status itself."""
    if isinstance(value, Status):
        return value
    return FAILURE if value is False else SUCCESS


def _name_of(fn: Callable[..., Any], fallback: str) -> str:
    name = getattr(fn, "__name__", "")
    return name.replace("_", " ") if name and name != "<lambda>" else fallback


class Context:
    """What a node's function gets: the sim and robot of the script, a blackboard to share values, the time."""

    def __init__(self, sim: Any = None, robot: Any = None, bb: dict[str, Any] | None = None) -> None:
        self.sim = sim
        self.robot = robot
        self.bb: dict[str, Any] = {} if bb is None else bb
        self.lines: list[str] = []  # the trace: what nodes ended and said, in order
        self._touch: Callable[[], None] = lambda: None

    @property
    def time(self) -> float:
        return float(self.sim.time) if self.sim is not None else 0.0

    def log(self, text: str) -> None:
        """Say something in the tree's printed trace (Tree.trace)."""
        self.lines.append(f"{self.time:6.2f} s  {text}")

    def touch(self) -> None:
        """Tell the viewer where the tree is now (done before a node starts a call that blocks)."""
        self._touch()


class Node:
    """A node of a tree. Make your own by overriding `run(ctx)`, which returns a Status."""

    kind = "node"

    def __init__(self, name: str | None = None, children: Seq[Node] = ()) -> None:
        self.name = name or self.kind
        self.children: list[Node] = list(children)
        self.status: Status | None = None
        self.note = ""
        self.entered: float | None = None  # when this node began its current run (simulated seconds)

    # ---- to override ----
    def run(self, ctx: Context) -> Status:
        raise NotImplementedError

    def halt(self) -> None:
        """Forget a run in progress (the node is not going to carry on): the children, and what it counted."""
        self.status = None
        self.note = ""
        self.entered = None
        for child in self.children:
            child.halt()

    # ---- the tick ----
    def tick(self, ctx: Context) -> Status:
        if self.entered is None:
            self.entered = ctx.time
        status = self.run(ctx)
        self.status = status
        if status is not RUNNING:
            self.entered = None
        return status

    def walk(self) -> Iterator[Node]:
        """This node and all below it, parents before children."""
        yield self
        for child in self.children:
            yield from child.walk()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


# ---- leaves ----------------------------------------------------------------------------------------------------


class Action(Node):
    """Do something: fn(ctx). True or None is success, False failure, or return a Status (RUNNING keeps it going: fn is
    called again next tick). A ScriptError (the playground refused: out of reach, no such robot) is a failure with its
    message as the node's note, so a Retry or Fallback can deal with it; running out of time is not caught."""

    kind = "action"

    def __init__(self, fn: Callable[[Context], Any], name: str | None = None) -> None:
        super().__init__(name or _name_of(fn, "action"))
        self.fn = fn

    def run(self, ctx: Context) -> Status:
        self.status = RUNNING
        ctx.touch()  # a call that blocks shows as running while it does
        try:
            return _answer(self.fn(ctx))
        except ScriptTimeLimit:
            raise
        except ScriptError as exc:
            self.note = str(exc)[:NOTE_LIMIT]
            ctx.log(f"{self.name}: {exc}")
            return FAILURE


class Condition(Node):
    """Test something: fn(ctx) true is success, false failure. Never runs for more than a tick."""

    kind = "condition"

    def __init__(self, fn: Callable[[Context], Any], name: str | None = None) -> None:
        super().__init__(name or _name_of(fn, "condition"))
        self.fn = fn

    def run(self, ctx: Context) -> Status:
        return SUCCESS if self.fn(ctx) else FAILURE


_COMPARE: dict[str, Callable[[Any, Any], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def sensor_is(sensor: str, path: str, op: str, value: float, name: str | None = None) -> Condition:
    """A Condition on a sensor of the tree's robot: sensor_is("lidar", "ahead.distance_m", ">", 0.5). `path` goes down
    the reading by keys (dots) and list indexes; a reading that is not there yet (or a sensor the robot lacks) is a
    failed condition, not an error."""
    if op not in _COMPARE:
        raise FlowError(f"sensor_is compares with {', '.join(_COMPARE)}, not {op!r}")

    def test(ctx: Context) -> bool:
        reading: Any = ctx.robot.sensors.get(sensor)
        for key in path.split(".") if path else []:
            try:
                reading = reading[int(key)] if isinstance(reading, list) else reading[key]
            except (KeyError, IndexError, TypeError, ValueError):
                return False
        try:
            return bool(_COMPARE[op](reading, value))
        except TypeError:
            return False

    return Condition(test, name or f"{sensor} {path} {op} {value:g}")


class Wait(Node):
    """Running for `seconds` of simulated time, then success."""

    kind = "wait"

    def __init__(self, seconds: float, name: str | None = None) -> None:
        super().__init__(name or f"wait {seconds:g} s")
        self.seconds = float(seconds)

    def run(self, ctx: Context) -> Status:
        assert self.entered is not None
        return SUCCESS if ctx.time - self.entered >= self.seconds - 1e-9 else RUNNING


class WaitFor(Node):
    """Running until test(ctx) is true (success); failure after `timeout` simulated seconds, if given."""

    kind = "wait"

    def __init__(self, test: Callable[[Context], Any], timeout: float | None = None, name: str | None = None) -> None:
        super().__init__(name or _name_of(test, "wait for"))
        self.test = test
        self.timeout = timeout

    def run(self, ctx: Context) -> Status:
        if self.test(ctx):
            return SUCCESS
        assert self.entered is not None
        if self.timeout is not None and ctx.time - self.entered >= self.timeout - 1e-9:
            self.note = f"not within {self.timeout:g} s"
            return FAILURE
        return RUNNING


# ---- composites ------------------------------------------------------------------------------------------------


class Sequence(Node):
    """Run the children in order; fail at the first that fails, succeed when all have. It carries on from the child
    that was running; reactive=True starts again from the first every tick, so a condition up front is re-checked."""

    kind = "sequence"

    def __init__(self, *children: Node, name: str | None = None, reactive: bool = False) -> None:
        super().__init__(name, children)
        self.reactive = reactive
        self._at = 0

    def halt(self) -> None:
        self._at = 0
        super().halt()

    def run(self, ctx: Context) -> Status:
        start = 0 if self.reactive else self._at
        for i in range(start, len(self.children)):
            status = self.children[i].tick(ctx)
            if status is RUNNING:
                if self.reactive:
                    for later in self.children[i + 1 :]:
                        later.halt()
                self._at = i
                return RUNNING
            if status is FAILURE:
                self._finish()
                return FAILURE
        self._finish()
        return SUCCESS

    def _finish(self) -> None:
        self._at = 0
        self.note = ""


class Fallback(Node):
    """Try the children in order until one succeeds; fail when all have failed. (Also called a selector.)"""

    kind = "fallback"

    def __init__(self, *children: Node, name: str | None = None, reactive: bool = False) -> None:
        super().__init__(name, children)
        self.reactive = reactive
        self._at = 0

    def halt(self) -> None:
        self._at = 0
        super().halt()

    def run(self, ctx: Context) -> Status:
        start = 0 if self.reactive else self._at
        for i in range(start, len(self.children)):
            status = self.children[i].tick(ctx)
            if status is RUNNING:
                if self.reactive:
                    for later in self.children[i + 1 :]:
                        later.halt()
                self._at = i
                return RUNNING
            if status is SUCCESS:
                self._at = 0
                return SUCCESS
        self._at = 0
        return FAILURE


class Parallel(Node):
    """Run all the children at once. success="all" (default), "any" or a number of children that must succeed; it fails
    as soon as that can no longer happen. Children still running when it ends are stopped."""

    kind = "parallel"

    def __init__(self, *children: Node, success: str | int = "all", name: str | None = None) -> None:
        super().__init__(name, children)
        if success == "all":
            self.needed = len(children)
        elif success == "any":
            self.needed = 1
        elif isinstance(success, int) and 1 <= success <= len(children):
            self.needed = success
        else:
            raise FlowError(f'Parallel success is "all", "any" or a number from 1 to {len(children)}, not {success!r}')
        self._done: dict[int, Status] = {}

    def halt(self) -> None:
        self._done.clear()
        super().halt()

    def run(self, ctx: Context) -> Status:
        for i, child in enumerate(self.children):
            if i not in self._done:
                status = child.tick(ctx)
                if status is not RUNNING:
                    self._done[i] = status
        wins = sum(1 for s in self._done.values() if s is SUCCESS)
        losses = sum(1 for s in self._done.values() if s is FAILURE)
        verdict: Status | None = None
        if wins >= self.needed:
            verdict = SUCCESS
        elif losses > len(self.children) - self.needed:
            verdict = FAILURE
        if verdict is None:
            return RUNNING
        for i, child in enumerate(self.children):
            if i not in self._done:
                child.halt()
        self._done.clear()
        return verdict


# ---- decorators ------------------------------------------------------------------------------------------------


class _Decorator(Node):
    def __init__(self, child: Node, name: str | None = None) -> None:
        super().__init__(name, [child])

    @property
    def child(self) -> Node:
        return self.children[0]


class Retry(_Decorator):
    """Run the child again when it fails, up to `times` attempts in all (then it fails), waiting `wait` simulated
    seconds between them. The viewer shows the attempt it is on."""

    kind = "retry"

    def __init__(self, child: Node, times: int = 3, wait: float = 0.0, name: str | None = None) -> None:
        if times < 1:
            raise FlowError("Retry needs times of 1 or more")
        super().__init__(child, name or f"retry {times}x")
        self.times = int(times)
        self.wait = float(wait)
        self.attempt = 1
        self._resume_at: float | None = None

    def halt(self) -> None:
        self.attempt, self._resume_at = 1, None
        super().halt()

    def run(self, ctx: Context) -> Status:
        if self._resume_at is not None:
            if ctx.time < self._resume_at - 1e-9:
                return RUNNING
            self._resume_at = None
        status = self.child.tick(ctx)
        if status is FAILURE and self.attempt < self.times:
            ctx.log(f"{self.name}: attempt {self.attempt} of {self.times} failed")
            self.attempt += 1
            self.child.halt()
            self.child.status = FAILURE  # shown as failed until the next attempt starts
            self._resume_at = ctx.time + self.wait
            status = RUNNING
        self.note = f"attempt {self.attempt} of {self.times}" if self.attempt > 1 or status is RUNNING else ""
        if status is not RUNNING:
            self.attempt = 1
        return status


class Repeat(_Decorator):
    """Run the child again and again: `times` times in a row (then success), or with times=None for ever. Fails if the
    child does."""

    kind = "repeat"

    def __init__(self, child: Node, times: int | None = None, name: str | None = None) -> None:
        super().__init__(child, name or ("repeat" if times is None else f"repeat {times}x"))
        self.times = times
        self.count = 0

    def halt(self) -> None:
        self.count = 0
        super().halt()

    def run(self, ctx: Context) -> Status:
        status = self.child.tick(ctx)
        if status is FAILURE:
            self.count = 0
            return FAILURE
        if status is SUCCESS:
            self.count += 1
            if self.times is not None and self.count >= self.times:
                self.count = 0
                return SUCCESS
            self.child.halt()
        self.note = f"round {self.count + 1}" + ("" if self.times is None else f" of {self.times}")
        return RUNNING


class Inverter(_Decorator):
    """Success when the child fails and failure when it succeeds."""

    kind = "inverter"

    def __init__(self, child: Node, name: str | None = None) -> None:
        super().__init__(child, name or "not")

    def run(self, ctx: Context) -> Status:
        status = self.child.tick(ctx)
        return {SUCCESS: FAILURE, FAILURE: SUCCESS}.get(status, RUNNING)


class Timeout(_Decorator):
    """Fail, and stop the child, if it is still running after `seconds` of simulated time."""

    kind = "timeout"

    def __init__(self, child: Node, seconds: float, name: str | None = None) -> None:
        super().__init__(child, name or f"within {seconds:g} s")
        self.seconds = float(seconds)

    def run(self, ctx: Context) -> Status:
        assert self.entered is not None
        status = self.child.tick(ctx)
        if status is RUNNING and ctx.time - self.entered >= self.seconds - 1e-9:
            self.child.halt()
            self.note = f"over {self.seconds:g} s"
            return FAILURE
        return status


# ---- state machines --------------------------------------------------------------------------------------------


class State(Node):
    """A state of a StateMachine. `do` is run every tick while the machine is here (a function of ctx, or a node);
    `enter` and `exit` once on the way in and out. A state with `result=SUCCESS` or `FAILURE` ends the machine."""

    kind = "state"

    def __init__(
        self,
        name: str,
        do: Callable[[Context], Any] | Node | None = None,
        enter: Callable[[Context], Any] | None = None,
        exit: Callable[[Context], Any] | None = None,
        result: Status | None = None,
    ) -> None:
        super().__init__(name, [do] if isinstance(do, Node) else [])
        self.do = do
        self.enter = enter
        self.exit = exit
        self.result = result
        self.outcome: Status | None = None  # how the node in `do` last ended

    def run(self, ctx: Context) -> Status:  # the machine runs states; a bare state tick is its work
        if isinstance(self.do, Node):
            self.outcome = self.do.tick(ctx)
        elif self.do is not None:
            self.do(ctx)
        return RUNNING


class StateMachine(Node):
    """States with transitions between them. Each tick it runs the current state's `do`, then takes the first
    transition from it whose `when` holds. `when` is a function of ctx, or SUCCESS or FAILURE: the state's node
    finished that way. A state with a result ends the machine with it.

        fsm = StateMachine("patrol", start="go")
        fsm.add(State("go", do=lambda c: c.robot.move(0.2)))
        fsm.add(State("turn", do=lambda c: c.robot.move(0.0, 1.0)))
        fsm.add(State("done", result=SUCCESS))
        fsm.transition("go", "turn", when=lambda c: c.bb.get("blocked"))
        fsm.transition("turn", "go", when=lambda c: not c.bb.get("blocked"))
    """

    kind = "machine"

    def __init__(self, name: str, start: str, states: Seq[State] = ()) -> None:
        super().__init__(name)
        self.start = start
        self.current: State | None = None
        self._transitions: dict[str, list[tuple[Callable[[Context], Any] | Status, str]]] = {}
        for state in states:
            self.add(state)

    @property
    def states(self) -> dict[str, State]:
        return {c.name: c for c in self.children if isinstance(c, State)}

    def add(self, state: State) -> State:
        if state.name in self.states:
            raise FlowError(f"{self.name} already has a state called {state.name!r}")
        self.children.append(state)
        return state

    def transition(self, source: str, target: str, when: Callable[[Context], Any] | Status) -> None:
        for end in (source, target):
            if end not in self.states:
                raise FlowError(f"{self.name} has no state {end!r}; its states are {', '.join(self.states) or 'none'}")
        if self.states[source].result is not None:
            raise FlowError(f"{source!r} ends the machine, so nothing leaves it")
        self._transitions.setdefault(source, []).append((when, target))

    def halt(self) -> None:
        self.current = None
        super().halt()

    def _go(self, name: str, ctx: Context) -> State:
        if self.current is not None:
            if self.current.exit is not None:
                self.current.exit(ctx)
            self.current.halt()
        state = self.states[name]
        ctx.log(f"{self.name}: {name}")
        self.current = state
        state.status = RUNNING
        state.outcome = None
        if state.enter is not None:
            ctx.touch()
            state.enter(ctx)
        return state

    def run(self, ctx: Context) -> Status:
        if self.start not in self.states:
            raise FlowError(f"{self.name} starts in {self.start!r}, which is not one of its states")
        if self.current is None:
            self._go(self.start, ctx)
        state = self.current
        assert state is not None
        if state.result is None:
            state.tick(ctx)
            for when, target in self._transitions.get(state.name, []):
                if when == state.outcome if isinstance(when, Status) else when(ctx):
                    state = self._go(target, ctx)
                    break
        self.note = state.name
        if state.result is not None:
            state.status = state.result
            return state.result
        return RUNNING


# ---- the tree --------------------------------------------------------------------------------------------------


def _short(text: str, limit: int = 60) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "..."


class Tree:
    """A root node, ticked until it stops running. `run(sim, robot)` ticks every `tick` simulated seconds, steps the
    clock between ticks and shows the tree in the viewer; `tick_once(ctx)` is the single tick, for tests."""

    def __init__(self, root: Node, name: str = "tree") -> None:
        if not isinstance(root, Node):
            raise FlowError(f"Tree needs a node (Sequence, Action, ...), not {type(root).__name__}")
        self.root = root
        self.name = name
        self.nodes: list[Node] = list(root.walk())
        if len({id(n) for n in self.nodes}) != len(self.nodes):
            raise FlowError("a node appears twice in the tree; make a second one for the second place")
        self.trace: list[str] = []
        self.ticks = 0

    # ---- what the viewer gets ----
    def structure(self) -> list[dict[str, Any]]:
        """The nodes in order: {"id", "parent" (an id or -1), "kind", "name"}."""
        index = {id(n): i for i, n in enumerate(self.nodes)}
        out = []
        for i, node in enumerate(self.nodes):
            parent = next((index[id(p)] for p in self.nodes if node in p.children), -1)
            out.append({"id": i, "parent": parent, "kind": node.kind, "name": _short(node.name)})
        return out

    def statuses(self) -> str:
        """One letter per node in order: . not run, r running, s succeeded, f failed."""
        return "".join(_LETTER[n.status] for n in self.nodes)

    def notes(self) -> dict[str, str]:
        return {str(i): _short(n.note, NOTE_LIMIT) for i, n in enumerate(self.nodes) if n.note}

    def reset(self) -> None:
        self.root.halt()
        self.ticks = 0

    # ---- running ----
    def tick_once(self, ctx: Context) -> Status:
        self.ticks += 1
        return self.root.tick(ctx)

    def run(
        self,
        sim: Any,
        robot: Any = None,
        tick: float = TICK,
        max_seconds: float = MAX_SECONDS,
        blackboard: dict[str, Any] | None = None,
        show: bool = True,
    ) -> Status:
        """Tick until the root succeeds or fails (or `max_seconds` of simulated time pass: failure). `robot` is a
        robotforge.robot.Robot (ctx.robot); `blackboard` seeds ctx.bb. The viewer shows the tree while it runs unless
        show=False."""
        if tick <= 0:
            raise FlowError("tick is a number of seconds above 0")
        ctx = Context(sim, robot, blackboard)
        ctx.lines = self.trace
        publish = getattr(sim, "show_flow", None) if show else None
        sent = {"statuses": None, "structure": False}

        def push() -> None:
            if publish is None:
                return
            now = (self.statuses(), self.notes())
            if now == sent["statuses"] and sent["structure"]:
                return
            publish(
                name=self.name,
                nodes=None if sent["structure"] else self.structure(),
                status=now[0],
                notes=now[1],
            )
            sent["statuses"], sent["structure"] = now, True

        ctx._touch = push
        self.reset()
        self.trace.clear()  # (ctx.lines is this list)
        start = ctx.time
        previous = self.statuses()
        push()
        status = RUNNING
        while status is RUNNING:
            if ctx.time - start >= max_seconds - 1e-9:
                self.root.halt()
                ctx.log(f"stopped after {max_seconds:g} s")
                status = FAILURE
                break
            status = self.tick_once(ctx)
            self._trace_changes(ctx, previous)
            previous = self.statuses()
            push()
            if status is RUNNING:
                sim.step(tick)
        return status

    def _trace_changes(self, ctx: Context, previous: str) -> None:
        for node, before, now in zip(self.nodes, previous, self.statuses(), strict=True):
            if before != now and now in "sf":
                ctx.log(f"{node.name}: {'success' if now == 's' else 'failure'}")

    def summary(self) -> str:
        """The trace as text: when each node ended and how, and what nodes said with ctx.log."""
        return "\n".join(self.trace)
