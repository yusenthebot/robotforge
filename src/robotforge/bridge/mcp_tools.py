"""MCP tools for learned policies and the UniLab training backend.

Registered by robotforge.mcp_server.build_server via `register(server)`. Training runs in a
separate process with UniLab's own environment; these tools only start it and read its status
file, so nothing from the trainer ever reaches the MCP stdout.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from robotforge.bridge import policies, unilab
from robotforge.bridge.spec import SpecError


def register(server: MCPServer) -> None:
    @server.tool()
    def list_walkers() -> str:
        """List the learned walking policies robotforge can play (neural networks trained with reinforcement
        learning, e.g. in UniLab), which library robot each fits and the speeds it learned. Use one with
        move(robot, speed, turn, walker="rl"); walker="gait" goes back to the built-in trot."""
        from robotforge.bridge.onnx_policy import INSTALL_HINT, onnxruntime_available

        with _tool_errors():
            found = [p.describe() for p in policies.all_policies().values()]
            out: dict[str, Any] = {"walkers": found, "user_folder": str(policies.user_dir())}
            if not onnxruntime_available():
                out["note"] = f"playing policies needs onnxruntime: {INSTALL_HINT}"
            return _json(out)

    @server.tool()
    def unilab_check() -> str:
        """Check whether the UniLab training backend is set up ($ROBOTFORGE_UNILAB_DIR, its own Python
        environment) and which tasks robotforge can train with it. UniLab is an external reinforcement
        learning framework: it simulates about a thousand robots at once to learn a walking policy in minutes."""
        with _tool_errors():
            return _json(unilab.check())

    @server.tool()
    def unilab_train(task: str = "go2_joystick_flat", iterations: int = 1000, name: str = "go2_walk_trained") -> str:
        """Start training a walking policy with UniLab in the background and return at once with a job id.
        Each iteration simulates 1024 robots for about half a second; 1000 iterations take about 13 minutes on
        a recent Mac and learn a good walk with turning, 150 learn to walk forward, 20 prove the pipeline.
        When it finishes the policy is exported to ONNX and installed under `name`; follow it with unilab_job,
        then try it with move(robot, speed, turn, walker=name)."""
        with _tool_errors():
            started = unilab.start_training(task, iterations, name)
            started["next"] = f"unilab_job(job={started['job']!r}) shows progress"
            return _json(started)

    @server.tool()
    def unilab_job(job: str | None = None, cancel: bool = False) -> str:
        """Progress of a UniLab training job (the newest when job is omitted): iteration, mean reward (higher
        is better), time left, and when done the installed policy name. cancel=true stops it."""
        with _tool_errors():
            if cancel:
                return _json(unilab.cancel(job))
            status = unilab.job_status(job)
            status.pop("pid", None)
            return _json(status)


@contextlib.contextmanager
def _tool_errors() -> Iterator[None]:
    """Anticipated failures become ToolError so the agent sees the actual message."""
    try:
        yield
    except (unilab.UniLabError, SpecError, ValueError) as exc:
        raise ToolError(str(exc)) from exc


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1, default=str)
