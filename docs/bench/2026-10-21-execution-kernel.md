# Execution kernel verification, 2026-10-21

This round implements the first framework optimization: a session-scoped execution kernel, with `move` and `run_script` migrated first.
It preserves the existing uncommitted mobile/runtime work and does not tune manipulation success rates or training algorithms.
The implementation and remaining migration boundary are described in [EXECUTION.md](../EXECUTION.md).

## Real execution

The installed `robotforge mcp` command was started as a separate process and driven with the official MCP client.
The before/after comparison used the same commands and template rover.
Both tools' advertised definitions matched exactly, and every previous response field remained present.
The two-second turning move measured 0.559 m before and 0.559 m after.
Destination navigation, continuous movement, explicit stop and a real child script all completed.

Actual Chrome then watched the same MCP process.

| Check | Observed result |
|---|---|
| Headless one-second move | 0.278 m, `shown_live=false` |
| Live two-second move | 0.574 m, `shown_live=true` |
| Live recording | WebP written, more than 20 frames |
| Child script without a viewer | Finished, `ok=true`, `shown_live=false` |
| Child script with Chrome watching | Finished, `ok=true`, `shown_live=true` |
| Browser closes during move | Completes exactly 3.00 simulated seconds and stops |
| Move while script owns the clock | Refused before controller mutation |
| Reset a real waiting child | Child reaped; failed run reports reset and retains its measured 0.20 s progress |
| Reset-to-child-cleanup latency | 0.016 s in this local run |
| Move after invalidation/cleanup | Completes normally |
| Browser errors | No JavaScript errors recorded in the inspected live screenshots |
| Standard CLI/MCP smoke | Passed in 2.9 s, 60 tools, HTTP/WebSocket physics, PNG and child-script movement |

Unit tests additionally compare identical physics state and step count with headless, watched and changing viewer presence, without depending on a viewer ticker.
The table's headless and live distances come from different starting states and durations; they are not a speed comparison.

The first live-verification harness attempted `control` instead of the existing `sim_control` tool name.
It was corrected and the complete live sequence rerun in a fresh evidence directory.
No product change was needed for that harness error.

## Automated checks

Wheel build and Ruff lint/format passed.
The expanded compatibility selection passed 273 tests with one optional skip; the final kernel/process selection passed 45 tests after the last ownership fix.
Code review and real adversarial probes found no remaining critical, high or medium issue in the scoped changes.
The complete default selection finished with **2920 passed, 127 skipped and zero failures** across 3047 cases in 590.2 seconds using two shards.
No selected heavy case was omitted, and both shards exited successfully without a repair rerun.
The source fingerprint was unchanged throughout the full run: `9f6416b653af67bb9b71300fc0bd2a3d90d71cc5fdfd84394a9b4fa21fac29e4`.
Machine-readable results and provenance are in [2026-10-21-execution-kernel.json](2026-10-21-execution-kernel.json).

Mypy/Pyright and Bandit/pip-audit are unavailable in the current environment and were not run.
Native Windows execution has not been run here; the existing gated Job Object path was preserved.
Slow/browser tests excluded by the repository's default pytest selection are not counted as passing.
Blocking planning/plugin/native work remains cooperative; retaining its lease can delay final completion while keeping the next run from racing it.

## Local evidence

The durable local directory is `../robotforge-evidence/2026-10-21-execution-kernel/` relative to the repository root.
It contains the pre-edit patch and selected source snapshots, before/after tool definitions and command results, live screenshots and WebP, real-child records, smoke/build logs, test logs and JUnit XML.
`live-final/` is the authoritative complete browser/MCP pass; `live/` preserves the earlier harness failure.
These results verify simulation and software behavior, not physical robot hardware.
Changes remain local and uncommitted; no paid CI, push or release was started.
