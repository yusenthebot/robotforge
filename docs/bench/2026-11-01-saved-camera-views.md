# Saved camera views acceptance, 2026-11-01

Users can name a free main-camera view and manually restore it after exploration, refresh and reopening the same saved source at the same browser origin.
Existing Back history, camera arcs, reduced motion, follow/input ownership and interfaces remain compatible.
This improves free exploration; materials, light, physics and single-action success rates are unchanged.

## Behavior and bounds

Open Views, enter a name, then Save current view; Restore enters free camera and deletion requires inline confirmation.
Names are trimmed/NFC and limited to64 Unicode code points; storage permits8 views per source,16 sources and65,536 UTF-16 JSON units.
Duplicate/full/corrupt/unavailable storage fails visibly without overwriting or evicting records.
Mutations re-read under an origin Web Lock and recheck source/version/generation/operation ownership after queuing.
Saving is unavailable during follow, animation, editing, inspection, lessons or disconnected/stale frames.
An optional opaque `view_scope` scene field derives once per model version from the captured canonical existing source path.
Verified case aliases and symlinks agree, while distinct same-title files remain separate.
Memory/lesson sources and old server messages cannot persist views; original camera controls continue working.
Storage belongs to the browser origin/profile/source, with manual restore only; replacing contents at the same path retains coordinates.

## Frozen source and actual package

Base main is `d196caa970034004b86976b906c053e5dcd42ad0`; baseline runtime is801 inputs, candidate805.
Evidence root: `/Users/yusenthebot/dev/robotforge-evidence/2026-11-01-saved-camera-views`.
`candidate-source-v1.json` SHA256 is `f803349c1f9b031159994f955c494dbda114c212f95ebec234bbf18594defa93`.
Guards run before application imports and recheck sources, native inputs and protected workspaces after each accepted stage.
All13 original WIP files,791 retained CMF inputs,796 unpromoted mesh inputs,799 PR68 inputs, both801-input accepted/report workspaces and78 native files remain exact.

Actual wheel4,145,465 B increases6,014 B, below the original4,194,304 B cap with48,839 B remaining.
SHA256: `3c7c82c5a8eeb27af2e49cf084af1d20970b5c64bad2befec76b1994052dcb9b`.
All427 production files match source/wheel/fresh locked offline installation; actual installed imports,60-tool MCP listing, viewer, movement, script and PNG smoke pass.
The actual source archive rebuilt through bundled offline uv from extracted contents produces identical wheel bytes and467 ZIP members.
Direct isolated PEP517 archive build lacked the offline `uv_build` backend; that failure remains, without an isolated-PEP517 certification.
No new asset, dependency, frontend build chain or downloaded robot mesh enters the package.

## Tests and unchanged budgets

JUnit identities establish4,438 distinct passes:4,305 full default,124 scoped actual Chrome and9 actual graphics/sensor cases.
The129 default skips are excluded; other deselected slow/browser cases are outside this round's claim.
The browser cohort includes11 new shelf cases plus existing presets/Back, driving, lessons, startup, inspection, screen controls and layouts.
Seven same-runtime framework cases pass, retaining differing source versions as provenance.
Source/fresh installed smoke, Ruff,787-file format check and independent source/Python review pass.
Optional bandit, pip-audit, mypy and pyright are unavailable; none was installed and no such scan pass is claimed.

All four original install budgets pass:4,048.3057 KiB wheel,164.0549 MiB installed,52.5149 MiB locked download and42 packages.
All seven original viewer budgets pass in every fixed baseline/candidate/candidate/baseline run.
These existing fixtures include template-dog cold start, Go2 warehouse and combined asset rendering, not Malo cold-start measurements.

| Metric | Baseline0 | Candidate1 | Candidate2 | Baseline3 |
| --- | ---: | ---: | ---: | ---: |
| Cold start, s | 0.8204 | 0.8231 | 0.8289 | 0.8220 |
| Warehouse FPS | 60.1329 | 60.1009 | 60.0890 | 60.2069 |
| Warehouse frame, ms | 3.1 | 4.2 | 3.0 | 2.9 |
| Combined asset FPS | 60.1309 | 60.2751 | 60.1209 | 60.3193 |
| Combined asset frame, ms | 5.2 | 5.2 | 5.3 | 5.3 |
| Dog server peak, MiB | 133.6 | 133.9 | 134.0 | 134.1 |
| Idle CPU, percent of one core | 4.6 | 4.8 | 4.4 | 4.6 |

Separate warehouse peaks are560.2/559.1/558.0/558.8 MiB in the same order.
No budget was changed and no speed improvement is claimed.

## Actual Malo and Go2 workflows

Original authored worlds/native assets run paused at time zero,1280×800 CSS pixels,DPR2 and2560×1600 raw canvas pixels.
The fresh installed candidate saves two views using real browser input, explores, restores, uses Back, refreshes and manually restores again.
New server and physical Chrome processes reopen the same world/home/origin/retained profile, then manually restore both choices without injecting bookmark storage or auto-restoring.
Six server/browser phases retain24 static stages and20 non-self exact full-RGBA comparisons across baseline/candidate and restored/refresh/reopen views.
The declared capture control pins the canonical render after checking actual restored pose within1e-9 and applies the existing product fog formula; it does not certify unmodified transient fog.
World bytes, native state, materials, texture pixels/sharing and canonical rendering controls are preserved.
Go2's481 raw model arrays match bitwise; Malo's479 match, while `_sizes`/`mesh_pathadr` differ solely through the proved isolated-home generated path, including all offsets/sizes and actual matching STL payloads.
No array is silently dropped; retained-home reopen snapshots remain exact.

Comparator v1 passed Malo but rejected Go2 diagnostic target z0.2 versus0.19999999999999996.
Post-capture comparator v2 corrects that extra exact-metadata demand for `camera.aim` and `canonicalFog.actualTarget` only, applying the already-declared1e-9 actual-pose gate.
Each finite target is bound to same-stage pre-pin, authored and saved targets;168 audit rows retain raw values/deltas/IEEE hex.
Twelve Go2 audit rows are numerically nonexact, maximum5.551115123125783e-17 m; Malo's84 rows are exact.
All other camera matrices/projection/eye and canonical fog uniforms/distance/radius remain exact, as does every full RGBA comparison.
The original helper/captures/failure remain retained; this metadata correction was not preregistered before capture and adds no pixel allowance or camera repair.

Six full films retain2,239 raw JPEGs/timestamps and2,200 fully decoded CFR frames.
Root inspected six decoded contacts,36 decoded samples and four full native UI/action-endpoint screenshots.
Continuous playback was not watched because available perception tools expose still images; no smoothness or complete-UI-readability claim is made.

| Candidate film | Duration | Path under evidence root |
| --- | ---: | --- |
| Malo workflow | 22.433333 s | `media-candidate-malo-v1/first/workflow.mp4` |
| Malo new-process reopen | 8.2 s | `media-candidate-malo-v1/reopen/workflow.mp4` |
| Go2 workflow | 21.3 s | `media-candidate-go2-v1/first/workflow.mp4` |
| Go2 new-process reopen | 7.4 s | `media-candidate-go2-v1/reopen/workflow.mp4` |

## Retained limits and next direction

Initial browser setup failed through a proxy before explicit loopback exclusions; corrected checks pass without a product workaround.
The first default run was deliberately stopped at1,190 passes before its900-second wrapper limit; the complete3,600-second-bound rerun passes in1,821.87 seconds.
The wrapper timeout self-check reaped its direct child but group probing/signaling returned EPERM, so it does not prove detached-descendant cleanup.
Two native desktop attempts still report Mac locked despite the unlock reply; no fresh native desktop zoom pass is claimed.
Acceptance is local macOS actual headless Chrome including emulated mobile, without physical-phone, Windows or new remote CI certification.

A preceding fixed DEFLATE9 screen saved16,530 B, below its prospective32,768 B gate, so no packaging backend change was promoted.
Transparent policy compression was not implemented because consumers require real shipped `.onnx` paths.
These decisions do not reverse the rejected compact-concrete result or establish a shared visual improvement.
The broad material-quality/lightweight/creative-play goal remains active, with single-action success tuning deferred.
Next assess shared resources/content against actual package/load costs and ordinary-view benefit, without retuning rejected content to manufacture a pass.
Measurements and stage receipt hashes are indexed in [the JSON report](2026-11-01-saved-camera-views.json).
