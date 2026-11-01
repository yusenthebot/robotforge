# Focus selected prop groups (2026-11-02)

The existing Focus / 查看 action now frames a selected prop group, including groups selected with Include items on top.
It uses current visible geometry, preserves the viewing direction, and keeps the complete group inside the unobstructed viewing area.
Selection, latest Undo, world, simulation and existing single-prop behavior remain intact.
Group Focus is available for one or more current members; mutation actions retain their existing paused/two-member admission rules.
There are no new assets, dependencies, rendering passes, backend commands or protocol changes.

## Source and retained corrections

The accepted baseline is the furniture-finishes tree, identical to main `dd4a114347c382bfb7db89e91b638d4c7a02ecf0`.
Native captures and the first passing test phases use the 814-input source v3, SHA `7f0295192249c7498b44b6e76f58603a711dc997ab51c3f66fef1692d55a15f5`.
Final source v4, SHA `ffdd92f6e36d6ebc63796f71ad225939f069678f242b1331569205ea46227623`, differs only in one inherited test assertion file. All 430 product payloads, scripts and dependencies are identical.
The [machine record](2026-11-02-group-focus.json) pins source manifests, methods, receipts and media under `/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-group-focus`.

The first Malo candidate failed exact pre-Focus RGBA: inserting Focus into the first toolbar row shrank the hint and added 18.1875 px of height, changing the camera projection before the action.
The same button now appears after group Move and before Arrange; single selection restores its original position before Surface. DOM movement occurs only when its sibling changes, retaining keyboard order and listeners.
A browser regression now compares actual toolbar/hint/freeRect composition with only the added button temporarily hidden, restoring it synchronously before rendering or native input.
The failed capture remains failed and retained. The revised product passed against the original A1, with unchanged helper, fixture, camera, pixel and benefit gates.

Two integration failures were also reproduced on the unchanged S78 baseline: the old group-copy test required Surface to be hidden even though furniture groups are now eligible.
The test now explicitly requires Surface visible/enabled while retaining hidden single Move/Copy/Remove checks and the full native copy/cancel/rotate/Undo journey.
An earlier pre-execution test assumption about Undo visibility was corrected: selection hides Undo; Escape exposes the preserved receipt for a real inverse click.

## Verification

66 distinct scoped Chrome cases pass in phases: 25 Focus/selection cases, 39 integration cases, then the two corrected complete copy journeys.
The original integration result remains `2 failed, 39 passed`, with its separate baseline reproduction retained; the final two passing cases are recorded under source v4.
Coverage includes existing single-prop Focus, live falling-prop bounds, manual/carried groups, 16/partial/empty/one/zero members, busy/reset guards, actual 320px Chinese input and native Undo after Focus.
Integration covers group move/copy, row arrangement, batch surface editing, saved arrangements, driving-camera behavior and inventory/dock access.

Fresh locked offline installation passes the original viewer/WebSocket/60-tool MCP smoke, carried-selection checks and 24 furniture compiler cases plus batch apply/Undo.
All 430 source, wheel and installed payloads match byte-for-byte. Ruff lint/format (671 Python files), JavaScript syntax and diff checks pass.
These are scoped local macOS results, not a full-suite, Windows or remote-CI pass. Bandit, pip-audit and type checkers were unavailable; no development dependency was installed.
The native desktop tool continued reporting locked after the unlock reply, so OS/browser zoom remains unverified.

## Original-model evidence

Malo workshop and Go2 warehouse retain their original robots/assets/sensors, with declared tabletop fixtures identical between modes.
Four passing captures produce 16 native 1280×800 diagnostic frames. All four baseline/candidate initial and pre-Focus full-RGBA pairs match exactly.
Focus uses a trusted native click; baseline records the action as unavailable and skips it honestly.
Each capture preserves all 481 model-array fingerprints, qpos/qvel/ctrl typed bytes, paused time zero, scene version/generation, authored files and model/data/backend receipt identity.
After the read-only selection query, Focus through Escape sends zero world requests. All eight selected-union corners remain inside the actual freeRect and clip depth.

Projected group bounding-rectangle area changes from 39,456.81 to 69,513.70 px² for Malo (1.762×), and 32,012.54 to 104,167.55 px² for Go2 (3.254×, above the prospective 2× gate).
Root and independent inspection support useful group inspection, with the stronger gain in Go2. This is not a material, lighting or overall rendering-quality improvement.
Diagnostic frames hide only identified selection overlays temporarily; ordinary UI samples retain the actual controls and highlights.

## Cost

The wheel is 4,152,068 B, only 185 B above baseline, with 42,236 B remaining under the unchanged 4 MiB total limit.
Locked installation is 164.0751 MB across 42 packages; listed platform wheel downloads are 52.5149 MB, with installation itself performed offline.
All four install budgets and all seven viewer budgets in each fixed A-B-B-A run pass. Unselected categories stay `not run`.

| Run | Cold start s | FPS | Frame ms | Asset FPS | Asset frame ms | Dog peak MB | Idle CPU % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 baseline | 0.9483 | 60.1209 | 1.6 | 60.2029 | 4.1 | 143.3 | 4.8 |
| B1 candidate | 0.9169 | 60.1049 | 1.6 | 60.1649 | 3.9 | 143.4 | 4.6 |
| B2 candidate | 0.9210 | 60.0930 | 1.6 | 60.1049 | 4.0 | 144.4 | 4.0 |
| A3 baseline | 0.9196 | 60.3273 | 1.6 | 60.1009 | 4.2 | 143.7 | 5.0 |

The dog-server column differs from warehouse memory, retained in JSON. These fixed benchmark results establish original-budget compliance, not a general speedup or native-model cold-load claim.
Native complete HTTP sessions retain 34 requests each and add 765 encoded bytes per candidate. WebSocket bodies are excluded; native navigation timing is diagnostic only.

## Films

Four media packages retain all 84 acquired UI JPEGs and original timestamps; their films fully decode. Root and independent review inspected 24 decoded contact panels.
Sampling is at most 5 Hz and reconstructed into 30-fps containers; it does not certify continuous-animation smoothness.

- [Malo baseline](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-group-focus/malo-A1-media-v1/workflow.mp4) / [group Focus](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-group-focus/malo-B1-media-v2/workflow.mp4).
- [Go2 baseline](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-group-focus/go2-A1-media-v1/workflow.mp4) / [group Focus](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-group-focus/go2-B1-media-v1/workflow.mp4).

Original WIP, retained accepted/rejected trees and native inputs remain protected. [Rejected mesh-cache screens](2026-11-02-mesh-buffer-screen.md) remain unpromoted; the bounded diagnosis did not justify another cache candidate.
The broader material/lightweight/play goal stays active; single-action success tuning remains deferred.
