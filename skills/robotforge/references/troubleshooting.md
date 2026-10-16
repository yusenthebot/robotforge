# Troubleshooting

Tool errors are written for you: they say what was wrong and what to do instead.
Read them, fix, and check again; tell the user plainly what happened.

## The robotforge tools are missing

Right after the client starts, the server may still be starting: a few seconds, a minute or more on the very first start.
Wait about 20 seconds (`sleep 20` in a shell) and look for the tools again, up to three times, before telling the user anything.
If they still do not appear, the MCP server did not start:

- It runs with `uv` (`uv run ... robotforge mcp`). If `uv` is not installed, ask the user to install it (https://docs.astral.sh/uv/getting-started/installation/), then restart the client.
- The first start downloads Python packages (about 50 MB, plus Python 3.12 if missing) and can take a minute or more on a slow network, which can exceed the client's startup timeout. Ask the user to reconnect: `/mcp` in Claude Code, a new session in Codex, or disable and enable the extension in Claude Desktop. Later starts take about a second.
- Behind a firewall or in mainland China, a PyPI mirror helps: `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` in the environment, or `index-url` in `~/.config/uv/uv.toml`.
- In Codex the tools are named `mcp__robotforge__<tool>` and may be listed only once the server is up; `/mcp` in the Codex TUI shows its status. Tool calls refused with an approval message mean the tools are not approved: the plugin approves them, a hand-written `config.toml` entry needs `default_tools_approval_mode = "approve"`.

## Common mistakes

| Symptom | Cause and fix |
|---|---|
| "no world is open" | Call `new_world` (or `open_world`) before adding robots or objects. |
| "no actuator named ... Did you mean ..." | Names need the robot prefix (`arm/shoulder`); read them from `describe_model`. |
| `move` refused for a robot | `robots()` shows its kind; `move` needs a quadruped, humanoid or rover. Arms use `move_arm`; robots of kind `other` use `set_controls` or a script. |
| The robot falls (`upright` false) | Slow down below `max_speed`; for templates use a stronger `servo` or a lighter `material`; for imports read the doctor's hold test and try `base`. Reset with `sim_control(action="reset")`. |
| "cannot reach" | Move the object or target closer to the arm (0.15 to 0.35 m for a default arm) or raise the arm's `reach`. |
| A library robot times out on first use | The download (4-55 MB) continues in the background; call `add_robot` again. |
| `simulate` warns about NaN or unstable | Reset; undo the last edit; for imported meshes check `scale` or use `decompose`. |
| `observe` returns a note instead of an image | No OpenGL on this machine (servers, some VMs). The numbers are still right; the user can look at the viewer. |
| The viewer does not open | Give the user the `viewer_url` from `open_viewer` to open in a browser (`ROBOTFORGE_NO_BROWSER` may be set). |
| A pole keeps balancing or a dog keeps walking after the tool returned | `run_controller` leaves the controller attached (`keeps_running`), and `move(seconds=null)` keeps going: call `stop(robot)`. |
| "the world was changed during the run" | Editing or resetting the world removes running controllers: run it again. |
| "something else took over" | The user drove the robot with the keyboard during the run; run again or let them drive. |
| A script hangs or hits the limit | Loops must call `sim.step()`; stop on measured conditions with a fallback limit; raise `max_sim_seconds` (up to 300) only when needed. |
| A script fails with `ScriptError` | The message names the refused request (unknown name, out of reach); fix the script and rerun. |
| An edit made things worse | `undo()` reverts the last world edit; `show_world()` shows the current description. |

## When a result is disappointing

Say so with the numbers (a 0.4 m walk when 1 m was asked, a missed placement), explain the likely cause, try one fix, and check again.
Never round a failure up into a success.
