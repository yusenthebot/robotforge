# Avoid copying unchanged viewer task metadata

This scoped increment supports the visual-quality, playability and lightweight goal by removing verified repeated work in the shared viewer.
It follows the existing-prop editor in PR #18; the broad product goal remains active.
Baseline is `905cbe8fb9b03069984ec415a47caee7bd6720a2`.
Verified runtime is `ee36b9a8de315fb9debe8fbfb8b070a59bc0720fec145796cfb0d3558ffc4799`.
Raw evidence is outside Git at `../robotforge-evidence/2026-10-24-idle-cost/`.

## Change and compatibility

The viewer already caches its task catalog, but every unchanged robot-description poll deeply copied that catalog before discarding the unchanged message.
The broadcast now compares the private cached catalog and only deeply copies an actual outgoing message.
Welcome messages retain their existing detached catalog, and a failed output copy does not consume a pending broadcast.
Task, algorithm, saved-task and learned-capability invalidation remain unchanged, including updates without a model recompile and reconnects that must not consume other viewers' updates.
Public interfaces, polling frequency, world loading, simulation, controllers and rendering remain unchanged.
Only `playground.py` differs between the two runtime packages.

## Official budget passes; supplementary CPU remains mixed

The changed runtime received exactly one unchanged full install/viewer budget run: all 11 measured budgets pass, with idle CPU at 4.6% against the existing 6.0% maximum.
The benchmark is partial with respect to its other sections; import, physics and MCP performance budgets were not rerun.
No threshold, sampling method or aggregation rule changed.
This does not establish that CPU stays below 6.0% or that uninstrumented performance consistently improved.

| Metric | Current measured value |
|---|---:|
| Wheel | 3,741,757 B, +101 B from the pre-optimization editor |
| Installed runtime / packages | 163.1911 MB / 42 |
| Locked wheel download | 52.5149 MB |
| CLI to first frame | 0.5068 s |
| Standard / asset FPS at DPR 2 | 60.1069 / 60.1989 |
| Standard / asset draw | 3.3 / 3.1 ms |
| Dog-server peak RSS | 132.2 MB |
| Official idle server CPU | 4.6% |

A fixed supplementary A1/B1/A2/B2 comparison uses the same existing five-second method and one frozen saved dog world.
Baseline samples are 5.4% and 6.0%; candidate samples are 6.2% and 5.8%.
The candidate 6.2% sample still exceeds the original 6.0% limit; the supplement is mixed, not a passing upper bound or a speedup result.
The benchmark's existing formal rule evaluates its specified sample for that run, not the maximum or aggregate of supplementary runs.
No extra budget repeats were used to search for a passing sample.

Ten-second CPU-timer profiles locate the removed work: nine task-catalog calls originally make 53,487 recursive deepcopy calls, with 0.337790 s cumulative instrumented time.
The candidate's nine catalog calls make no deepcopies and record 0.000358 s cumulative instrumented time.
Robot-description cumulative instrumented time changes from 0.622585 to 0.222179 s.
Instrumentation changes cost, so these values diagnose repeated work rather than measure uninstrumented CPU savings.
Repeated YAML reads remain outside this change.

The prior editor's two official 6.2% failures and its failed record `89af1f39-7f9f-4d1d-b087-8d9233f1d6a1` remain immutable and failed.
The [prior report](2026-10-24-object-editing.md) retains its original scope and outcome.

## Functional and actual-app verification

319 focused default cases, all 107 browser cases and eight additional slow GPU/asset/sensor cases pass: 434 distinct cases on this runtime.
The unchanged seven-case framework baseline passes, including control comparison, replay, training, policy lineage, snapshot restoration and MCP control.
Actual source CLI/viewer/MCP/script smoke and a fresh offline locked-dependency wheel installation pass, including all 60 MCP tools.
The complete default suite was not rerun for this scoped metadata change; the prior 3447-pass run belongs to the previous runtime.
Ruff, formatting, package/source equality, diff and scoped secret/debug checks pass, as does independent Python review.
Optional mypy/pyright/bandit/pip-audit tools are unavailable and were not installed.

Three new regressions cover unchanged-poll copying, deep output mutation isolation and pending updates after a copy failure.
All three fail against the archived pre-optimization runtime and pass on the candidate.
The old failure log is retained as a negative control, not a candidate failure.

The same frozen gallery, kitchen, park and wood-prop fixtures reproduce 17 byte-identical full-canvas PNGs against the prior candidate.
Full scene payloads, initial/final frozen physics and GPU resources match exactly.
Four reconstructed camera arrays differ by at most 8.882e-16 despite identical prescribed inputs and PNGs.
An initial summary helper incorrectly required exact floating camera arrays; its source and failure remain retained.
Only reconstructed camera/target observations use a 1e-12 comparison tolerance; image, payload, physics and resource equality checks remain exact.
No new dependencies, frontend resources, textures or passes are added; fixed-view timings remain measurements rather than a speed claim.

Native desktop and Chinese touch-emulation replays repeat cancel, move/rotate, removal, Undo, scene picking, support/occupancy refusal and retry.
Each replay retains seven raw same-connection edit request/reply pairs; 100 desktop and 97 phone receipt checks pass.
Desktop also verifies actual native-keyboard rover/barrier contact and reset preserving the edited layout.
Phone verifies native touch editing, not touch driving or a physical phone.
Both full uncapped H.264 videos completely decode: desktop 24.0 s / 720 frames and phone 14.966667 s / 449 frames.
The coordinator inspected decoded contact sheets; this is sampled visual inspection rather than a claim of watching every frame.
The summary verifies 195 additional source, state, image, resource and media properties.

## Limits and next direction

The scoped verification is complete under the existing formal budget rule; not all supplementary performance samples pass.
CPU variability, original wood motion-stability tradeoffs, physical phones, Windows and other GPU drivers remain open.
Existing upright editing, one-use Undo, carried unrelated poses/controls and restarted velocities/time retain their previous limits.
The technical support-refusal message remains English in Chinese touch emulation.
Further work returns to shared material quality during motion and enjoyable scene editing, composition and exploration.
Individual manipulation success-rate tuning, releases and paid Actions remain deferred.
The companion JSON identifies the sealed experiment record and retained evidence.
