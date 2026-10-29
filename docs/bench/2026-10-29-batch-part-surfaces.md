# Batch robot part surfaces

Explicitly selected direct parts of one robot can share an existing surface in one guarded rebuild and one Undo.
This makes applying Original/plastic/wood/steel to several parts practical without repeating the complete model rebuild for each part.
The existing single-part dropdown remains the default.
This is opt-in creative styling with existing materials, not new shared shading or inferred real-world material identity.

## Behaviour and compatibility

“Select multiple parts” starts with the current part checked and lazily builds a scrollable list from existing server metadata.
Names and existing finishes remain visible; the group outline encloses selected geometry, including gaps between parts.
Different finishes produce a disabled “Choose a surface” prompt, requiring an explicit choice before applying; empty selection disables Apply.
One batch contains 1–64 distinct body names; the resulting saved override map also retains its existing 64-entry limit.
Original removes only the selected overrides.
Physics, base colours, source files, child-part boundaries and sensor housings retain their existing rules and eligibility limits.
Opaque untextured direct parts of eligible unmounted rigid robots are supported; transparent/native-image/mounted/flex cases remain outside this editor.

`set_robot_surface` accepts exactly one of legacy `body` or new `bodies`.
Legacy responses retain `body`; batch responses use `bodies`.
The entire selection is validated before one compile/install/carry/reset, with both existing publication guards and rollback retained.
Mixed already-matching/changed selections still produce one edit.
All-no-op selections preserve the exact runtime, saved bytes and previous Undo.
One Undo restores the complete previous mapping, including heterogeneous finishes.
No new MCP tool, world schema, shader, texture, dependency or startup request is introduced.

Selection clears on robot/world/version/generation/socket changes.
Only a successful acknowledgement matching the displayed robot, version, generation and world digest restores this operation's selection for comparison.
Saved worlds/layouts retain finishes, not checkbox state.
Applying still preserves poses and control targets, clears motion/time and pauses; it is not seamless styling during motion.

## Actual workflows

The frozen before package and candidate independently serve isolated world copies.
The native MALO URDF world styles torso_link, head_link and left_arm_link_1; the cached Go2 world styles base, FL_hip and FR_hip using narrow Chinese touch controls; a built-in rover template styles chassis and its two wheel bodies.
Each primary comparison requests the same three steel finishes.
Before requires three separate applications and three compile/load pairs; candidate requires one explicit selection and one compile/load pair.
The extra Undo/reapply demonstration is counted separately from the primary task.
The old latest-only Undo restores the final part; the new group Undo restores the whole selection, and reapply/reload retain the intended final design.

Recorded controls compare 444 nonappearance model ndarray fields per workflow, qpos/qvel/ctrl at 1e-12 tolerance, mesh payload bytes and per-geom material semantics outside the exact eligible selected geometry IDs.
A separate stored-array postcheck compares all ten mat_* fields, including mat_texrepeat and mat_texuniform, for every unselected geom and all final before/candidate geoms.
Its paired four-checkpoint main-canvas hashes are exact in all three workflows; camera vectors use absolute1e-12 tolerance, with maximum observed difference2.67e-15.
Resource/program observations are recorded separately because rebuild histories differ.
Appearance/name/path/index fields are excluded explicitly rather than mislabeled as physical equality.
Source-world bytes and the original 13 WIP files remain unchanged.
The default closed-editor comparison retains 24 main-canvas pixel-exact pairs at actual DPR1/2 under two lights on original MALO, Go2 and Panda worlds.
Whole-window screenshots and asynchronous render-frame counters are separate.

## Evidence and validation

Candidate runtime: `7ebb8412c4e941a3921b79fa11224ea57494a54e620f540f9e52a5a1a8496e00`.
Frozen before runtime: `b399c6c405fedc612c92f0e8323529f5f238b42bd01a03bd2df2fdb768d490ae`.
External captures, raw frames, decode receipts, command receipts and the machine-readable summary are retained in `../robotforge-evidence/2026-10-29-batch-part-surfaces/`.
Tests and measurements use frozen 770 source/test/script/dependency inputs.
Optional type/security tools are not installed and are not reported as passes.
Final scoped results and measured costs are recorded in the accompanying JSON.
681 viewer-default cases,322 complete non-slow browser cases and9 explicit GPU cases pass:1,012 distinct pytest passes.
Six default skips and one optional USD browser collection skip are retained; the full default suite was not rerun.
Seven canonical framework cases pass under actual Python3.12.13 with the current source first on PYTHONPATH.
Source and fresh offline installed CLI/viewer/MCP smoke pass with60 tools.
Fresh installed batch and legacy single-part apply/Undo also pass.
Ruff,format,changed JS parsing,high-confidence added-secret/debug audit and416source/wheel/installed product-file equality pass.
All770frozen inputs and13original WIP hashes remain exact.
The original11install/viewer budget thresholds all pass; other budget groups are not run.
Six final movies fully decode,with36root-inspected decoded samples; preliminary Go2v1 movies remain separate.

Initial browser verification retained three failures: a static disabled mixed placeholder changed old phone-key navigation, and a test fixture treated the rover's last part as both selected and unaffected.
The placeholder is now created only for mixed/empty batch choices, preserving legacy single-select order; the fixture leaves the third part unselected.
The corrected old/new focused browser run passes 12 cases.
Initial Ruff findings were test formatting/fixture-import annotations and were fixed before freezing the candidate.
The first portrait Go2 workflow had poor scripted framing and is preliminary; the corrected paired capture centres the actual saved robot and closes the existing drive overlay through native input, without changing the world.

Timing samples describe scripted click-to-ready, compile/load wall time and GPU-finished main-frame readiness, not human discovery/latency or all-device performance.
Observed JS heap/process-tree RSS are samples, not peak GPU memory or leak proofs.
Complete movie decode and sampled still inspection do not certify every frame, native phone hardware, Windows or manipulation/locomotion success.
The broad visual-quality/lightweight/creative-play goal remains active.

## Measured costs

| Primary three-part task | Compile ms before / candidate | Load ms before / candidate | Mesh encoded bytes before / candidate |
|---|---:|---:|---:|
| malo | 891.81 / 285.62 | 63.80 / 20.07 | 141,194,016 / 47,064,672 |
| go2 | 321.49 / 115.74 | 418.63 / 151.01 | 60,997,536 / 20,332,512 |
| rover | 67.01 / 21.94 | 22.57 / 7.14 | 1,888,704 / 629,568 |

These are one actual workflow sample per source version,excluding initial navigation and later Undo/reapply.
Each task makes3→1surface requests and3→1compile/load pairs; final authored world mappings and mesh bytes are exact.
The shared main material/shader appearance is unchanged.

Wheel3,951,703B is1,828B larger than the before package;42runtime packages,52.5149MB dependency download and163.6722MB installed footprint are recorded by the existing budget.
No additional request,texture,shader pass or dependency is added.
Fresh original worlds retain31HTTP resources; candidate cold encoded bytes rise5,320B per world,and warm reload resource bytes are retained in JSON.

| Four fresh profiles per cohort | Browser-cold main-ready mean ms before / candidate | Actual reload mean ms before / candidate |
|---|---:|---:|
| malo | 621.75 / 574.78 | 316.07 / 292.20 |
| go2 | 654.32 / 639.95 | 389.85 / 381.65 |

ABBA then BAAB ordering uses fresh package processes/profiles and explicit light at actual DPR2; main GPU-finished readiness excludes server compile and is not complete sensor/UI readiness.
Chrome process-tree sampled cold RSS means are1,765.52→1,780.28MiB for Malo and1,682.36→1,679.83MiB for Go2; these are neither peaks nor GPU-memory measurements.
The standard budget records0.9465s cold start,60.1069/60.3729FPS standard/assets,3.4/5.0ms draw,135.2MB dog-server peak and4.6%idle CPU.
The warehouse-server diagnostic is recorded separately from the dog baseline threshold.
Faster sampled loading means do not establish a general or causal speed improvement.

MALO candidate recording: `candidate-malo-v1/workflow.mp4`; Go2 narrow Chinese candidate: `candidate-go2-v2/workflow.mp4`; template candidate: `candidate-rover-v1/workflow.mp4`,relative to the external evidence directory.
