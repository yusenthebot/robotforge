# Near-focus depth screen: closed without a product candidate

Normal near Focus lowers the main camera's near plane and bookmark Restore retains that clipping range. This is a real camera-history mechanism, but this bounded investigation did not establish a shared ordinary-view visual defect. No renderer, material, camera, world, asset, dependency or interface change is proposed.

The accepted main baseline is `c7b8231c438f3588075385dbdedecfbf3c345d6c`, tree `b2f50108ff6addb851ed3d4b63e1456d2d568bd7`. All 814 frozen source/script/test/dependency inputs remain identical in this documentation checkout. The [machine record](2026-11-02-depth-precision-screen.json) pins the prospective plan, method, capture helper, original receipts and retained frames under `/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-depth-precision`.

## Fixed workload and actual results

Original Malo workshop and Go2 warehouse YAML bytes, native robot assets, sensors, poses and props are preserved. Each fresh path uses native Fit View, two fixed saved views, repeated native Restore, one native Focus of the existing marker/crate_5, repeated Restore, then native Fit View recovery and Restore. Only initial view setup uses the existing public lookAt API. No artificial geometry, direct clipping setter, fog normalization or private frontend hook is introduced.

Four paths ran once, serially, in the declared order. Failed paths remain failed; no retakes, image-threshold changes, alternative target or candidate followed.

| Path | Result | Diagnostic frames | Native workflow JPEGs | Limitation |
| --- | --- | ---: | ---: | --- |
| Malo DPR1 | Failed controls | 12 | 72 | Oblique fog ranges differ; A/A, B/B, C/C and A/C RGBA are not exact |
| Go2 DPR1 | Passed controls | 12 | 72 | Localized clip-dependent pixels do not establish a shared ordinary-view defect |
| Malo DPR2 | Failed input delivery | 0 | 4 | First Fit View native-click confirmation timed out |
| Go2 DPR2 | Failed input delivery | 0 | 4 | Same first-click confirmation timeout |

Actual CSS is 1280×800. DPR1 buffers are 1280×800; DPR2 UI JPEGs are 2560×1600, but no DPR2 main-buffer diagnostic completed. Thus this record contains **24 of 48 planned diagnostic frames**, with 152 acquired native JPEGs retained. It is not a four-path pass or a new regression-suite result.

All four receipts confirm unchanged frozen source, external inputs, protected checkouts and native assets. Each retains the original world bytes, all 481 model-array fingerprints, typed qpos/qvel/ctrl records and numeric zero qvel; initial runtime equals cleanup runtime. Chrome, socket, reader and server cleanup passed in every path. Complete paths additionally retain phase-by-phase state, lighting/shadow/material/geometry controls, raw pixels, full difference masks and native request/input audits.

## Interpretation

Malo front A/A, B/B, C/C and A/C are exact. All four front A/B combinations change the same 88 pixels of 1,024,000, with a maximum channel difference of 49. This supports a localized clip-dependent raster difference; ordinary full images do not establish a salient shared loss.

Malo oblique controls fail honestly. The final camera positions differ only around 1e-13, but fog near/far are latched at slightly different values. `app.js` updates fog from camera distance on OrbitControls change events; the vendored controls use a squared-motion threshold, and tween completion has no final fog update. Recorded fogNear lags the final-pose formula by approximately 0.042–0.451 mm. A/A, B/B and C/C differences have maximum channel magnitude 1, with signs consistent with their shifted fog intervals. This is a source-supported explanation, not a counterfactual proof of every fragment's cause. Waiting longer for a stationary camera does not refresh event-latched fog. The original failed control gates stand.

Go2 DPR1 repeats and A/C recovery are exact in both views, with unchanged shadow/material/geometry controls. Every corresponding A/B repeat combination changes 2,092 front pixels or 2,644 oblique pixels; maximum channel differences are 154 and 167. Near changes from 0.05 to 0.0038223513230716847 while far stays 900.8692803671978; native Fit View recovers the original clips and exact pixels. The observed differences are localized near edges/seams. Root and independent full-image inspection do not establish the required shared ordinary-view defect across both worlds. Projected ownership bounds are coverage estimates, not visible-pixel segmentation; no robot/prop/terrain-wide gain is claimed.

The DPR2 failures show an input-coordinate mismatch: the trusted click reached the canvas at [329,377], half the DPR1 button hit [658,754]. The frozen native-click helper computes a CSS DOM center, passes it unchanged to its CDP adapter and requires trusted delivery to that button. No button event occurred in either DPR2 path. The exact Chrome/CDP scaling cause remains unproven because dispatch parameters and visualViewport were not independently audited. These failures do not demonstrate a Fit View product regression, and do not establish browser/OS zoom support.

The prerequisite for a camera-clipping candidate was not met. This screen is closed without a product prototype; the broad visual quality, lightweight loading and playability goal remains active. Correct high-density capture input is a separate verification prerequisite, not permission to repeat this screen until it produces a defect. Single-action success tuning remains deferred.

## Available film and limits

Only the passed Go2 DPR1 path was exported. Its media package preserves all 72 acquired JPEGs and timestamps; the 54.267-second movie fully decodes to 1,628 frames. Root and independent review inspected the six decoded contact panels and full ordinary A/B images. This is sampled workflow evidence, not a continuous-motion or flicker-free certification. The 30-fps container resamples samples and need not contain every original JPEG.

- [Go2 native workflow](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-depth-precision/go2-dpr1-media-v1/workflow.mp4)
- [Decoded contact sheet](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-depth-precision/go2-dpr1-media-v1/decoded-contact.jpg)

The failed Malo path retains all original samples but was not passed to the pass-only exporter. The two aborted DPR2 paths retain their four samples each; there are no complete DPR2 films. All planned four-film and 48-frame deliverables were therefore not achieved, and are not claimed.

No new pytest, wheel/install, cold-load, FPS, CPU/memory, Windows, remote-CI or native desktop zoom qualification was run for this documentation-only closure. Existing accepted budgets belong to the separate group-Focus report and are not fresh results here. No change in product size or performance is claimed.
