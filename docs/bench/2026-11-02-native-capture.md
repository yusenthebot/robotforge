# Native screenshot/input exclusion at high DPR

Concurrent native surface screenshots could misdirect headless Chrome mouse input at DPR2.
The test-only `NativeCapture` helper now excludes a complete existing mouse/touch click from the complete screenshot RPC, preserving CSS coordinates and trusted target checks.
Product code, assets, materials, physical behavior, interfaces and dependencies are unchanged.
The [machine record](2026-11-02-native-capture.json) binds the diagnostic, source freezes, test receipts, raw events, actual original-world recordings and media under `/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-native-input-dpr`.

## Reproduction and source mechanism

Accepted baseline is main `bc46a7c1497c3180bbecd0d4b64f5c8cf2f21f13`.
Its independent input probe used fresh Chrome profiles/pages for DPR1/2 × four conditions × two repeats, exactly 16 slots without retakes.
All use the same actual Fit button and unchanged native-click helper.

| Condition | DPR1, two repeats | DPR2, two repeats |
| --- | --- | --- |
| Sequential screenshots.Page input | Correct target [658,754] | Correct target [658,754] |
| Sequential multiplexed Cdp input, no screenshot | Correct target [658,754] | Correct target [658,754] |
| Multiplexed Cdp, screenshot fully complete before input | Correct target [658,754] | Correct target [658,754] |
| Multiplexed Cdp, screenshot and pointer gesture overlap | Correct target [658,754] | Trusted canvas click [329,377]; delivery timeout |

The table shows quantized DOM client positions; the original fractional CSS dispatch parameters remain in the RPC logs.
All four overlap slots prove both press and release occurred inside the actual screenshot request/reply interval.
Outbound CSS parameters, RPC IDs/sessions/timing, DOM trusted coordinates/targets, viewport/layout metrics and native screenshot dimensions remain retained.
Fourteen correct clicks and the two wrong-target timeouts remain separate diagnostic outcomes; a completed diagnostic is not sixteen successful clicks.
Source, world, model/runtime and cleanup guards passed.

The [CDP input definition](https://github.com/ChromeDevTools/devtools-protocol/blob/master/pdl/domains/Input.pdl#L154-L167) requires CSS coordinates.
The exact observed Chrome `156.0.8078.74`, revision `d331de35ca8ab1358c7a17e4bd41627d55eaef51`, [temporarily modifies emulation scale for surface capture](https://github.com/chromium/chromium/blob/d331de35ca8ab1358c7a17e4bd41627d55eaef51/content/browser/devtools/protocol/page_handler.cc#L1406-L1444) and [restores it on completion](https://github.com/chromium/chromium/blob/d331de35ca8ab1358c7a17e4bd41627d55eaef51/content/browser/devtools/protocol/page_handler.cc#L1860-L1868).
The fixed controls support screenshot/input overlap as this reproduced failure mechanism; multiplying coordinates by DPR would break the successful controls.
The old E83 paths remain failed/closed, rather than being retrospectively relabeled or replaced by this new input investigation.

## Small test-support change

`tests/viewer_capture.py` provides one shared `NativeCapture(tab)` per recording tab.
Its `.click(selector=None, point=None)` delegates the complete existing click preparation, mouse/touch sequence and target delivery check.
Its `.screenshot(**params)` returns the unchanged screenshot response after the complete RPC.
Both use the same lock; unrelated RPCs remain concurrent.
Capture callers must route every pointer gesture and screenshot through that same instance; raw low-level calls bypass this guarantee.

Cancelling or failing an operation after it starts makes the gate unusable, releases the lock and rejects queued/future work without dispatch.
The owner closes the tab because cancelling a local await does not establish that Chrome restored screenshot state.
Cancelling a waiter that never acquired the gate does not invalidate the active owner.
No recovery/retry method, global RPC lock, synthetic DOM event or coordinate compensation was added.

## Verification and real original worlds

Forty distinct scoped candidate tests pass: seven lifecycle/forwarding units, four actual Chrome cases, and 29 existing resolution/dock/camera-view cases.
The four new Chrome cases cover desktop/mouse and phone/touch at DPR1/2, screenshot-first and gesture-first scheduling, selector/point input, actual request/reply exclusion, trusted Fit delivery and native JPEG dimensions.
They preserve the paused template world's model/data identity, arrays, state and authored bytes; the phone is browser emulation, not a physical device.
Two earlier unchanged-baseline direct-input cases also pass, and are already included in the 29 distinct compatibility IDs.
Repository Ruff lint/format pass; no development dependency was installed.
Bandit, pip-audit and mypy were unavailable and were not run.

Malo workshop and Go2 warehouse then each complete one independent original-world DPR2 recording check, using exact original YAML/assets/sensors and the new gate.
Each performs seven trusted clicks: Fit, Views, name field, Save, close Views, reopen Views and Restore.
The two ordinary camera positions are established with the declared public lookAt setup API; those setup moves are not claimed as native pointer input.
Save/Restore preserves the recorded bookmark and restores eye/target within the predeclared 1e-9 tolerance.
Each records twenty genuine 2560×1600 JPEGs, and directly observes a 2560×1600 GL buffer, DPR2 and GL0.
Actual RPC intervals show no screenshot/pointer overlap; all client messages are audited, with only readiness acknowledgments present and zero world/control commands.
All 481 model-array fingerprints, typed qpos/qvel/ctrl, paused time zero, version/generation, model/data and edit receipt identities, authored files, native/cache/source inputs and owned cleanup remain consistent.
These recordings qualify input capture; they are not depth or material comparisons.

## Media, cost and limits

Both media packages preserve all twenty original JPEGs and timestamps, with screenshot request starts at least 200 ms apart.
Movies fully decode without truncation or decoder errors; root and independent review inspect twelve decoded contact panels and the two original restored-final images.
The 30-fps containers resample the acquired frames and need not contain every JPEG; discrete inspection is not continuous-animation or flicker-free certification.

| World | Native size | Duration s | Decoded frames | Movie bytes |
| --- | --- | ---: | ---: | ---: |
| Malo | 2560×1600 | 6.033333 | 181 | 1,504,310 |
| Go2 | 2560×1600 | 6.066667 | 182 | 1,983,871 |

- [Malo recording](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-native-input-dpr/malo-media-v1/workflow.mp4)
- [Go2 recording](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-native-input-dpr/go2-media-v1/workflow.mp4)

All original 814 source/script/test/dependency inputs remain byte-identical; the candidate adds only three test-support files for 817 inputs.
All 430 product payload files and dependency declarations match baseline.
There is no fresh wheel/install, loading/FPS/CPU/memory budget, Windows, remote-CI, full-suite or native desktop/browser zoom qualification here.
The Browser plugin is unavailable in this session; validation uses the repository's existing native Chrome/CDP workflow without installing another browser dependency.
This phase repairs the evidence pipeline for the continuing shared visual-quality/lightweight/playability goal, without claiming a product appearance improvement or tuning single-action success.
