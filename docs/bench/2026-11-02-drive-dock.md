# Driving hints and playback controls (2026-11-02)

Opening Scene & build moved the keyboard hint horizontally over Reset and Play.
At 821px the closed inventory could also leave Play covered.
Removing the wide-screen low-bottom override keeps the hint above the dock; on-screen controls and visible Undo already used that height.
Camera framing does not inspect the HUD, and no JavaScript, physics, material, dependency or interface changes are included.

## Verification scope

The baseline worktree at `8ebf237` has the same complete tree as merged main `093037d`.
Candidate source/scripts/tests/lock are frozen as 809 inputs, SHA `a925f661a0f6377d4894d0b56ad873854390458e6bf1144478d4cd012e387b71`.
The linked [JSON report](2026-11-02-drive-dock.json) binds receipts, methods, raw captures and media by SHA.

- 59 actual Chrome driving/camera/regression cases and 65 layout/row/saved-layout cases passed, with 124 distinct JUnit cases and no overlap or skips.
- The 12 new cases cover English/Chinese, widths 320/700/820/821/1280/1920 at 800px height, every visible dock button's bounds/hit/obstruction, and trusted Reset with inventory open and closed.
- This is scoped browser verification, not a fresh full-default or full-browser-suite claim.
- Offline wheel build, Ruff lint and format checks passed.
- A fresh locked offline installation passed dependency checks and the actual viewer/WebSocket/MCP smoke test; all 428 product files match source, wheel and installation.
- Type/security scanners mypy, pyright, bandit and pip-audit were unavailable and are not reported as passing.

## Original-world controls

Four fresh owned homes, Chrome profiles and servers loaded the actual Malo URDF/STLs and Menagerie Go2 with original workshop/warehouse YAML bytes.
No asset download, repair or staging was permitted; cloned Malo cache bookkeeping changed only after dependency payload SHA checks, and the normal cache validator accepted it.
Controls are English, desktop 1280x800/DPR2, paused at zero, with fixed ordinary front cameras.
Baseline and candidate both reproduce the original scene; the baseline's trusted click at the obstructed Reset centre sends no Reset.
Each candidate completes two trusted button Resets, one with inventory open and one closed, with exactly one command and one generation advance per click.
Both remain paused at zero and retain all 481 native model arrays, other recorded state and world bytes within each run.

Eight native 2560x1600 canvas images form four exact before/after pairs and two exact cross-source pairs.
Camera projection and full RGBA equality are strict; no pixel tolerance, crop, shader change or image-quality improvement is claimed.
Each capture has the same normalized multiset of 33 resource entries; per-request HTTP transfer bytes were not measured.
Original 78 native inputs and protected 13 WIP/791 CMF/796 mesh/799 arrangement/two 801/source 808/camera 805 inventories stayed exact.
All owned application/browser/encoder processes were reaped.

## Retained failures and protocol corrections

The new baseline regression failed in both languages at 1280px as expected.
A first partial CSS fix passed ten cases but failed two at 821px; the final rule removal passed all 124 scoped cases.
The first actual-world baseline capture failed exact pixels because the final panel projection was still easing.
The next method observes the existing bounded camera-settle helper and adds exact projection equality; it does not change projection, raw pixels or tolerances.
A later candidate capture failed an overly broad ban on all world requests after Reset, and its failed increment was not persisted; that old increment's action is unknown.
Fresh v4 captures independently prove the precise existing read-only protocol: one `layout_list` after open-panel Reset, zero when closed, exact request shape/ID/order and a unique successful reply with current version, generation and world SHA.
Backend source confirms this branch returns before execution mutation and reads layouts without creating or writing them; arrangement invalidation sends no query.
Before/after Reset witnesses and final wire audit are durable, hash-bound and rechecked during media export.

The old encoder and a microsecond-input-clock revision both failed the Malo candidate duration gate: 9.766667s versus raw span 9.54544825s plus the declared tail, exceeding the original 0.15s threshold.
The higher input clock alone did not fix output synchronization; both failed exports remain rejected.
Actual null/debug traces instead established an explicit fps-filter EOF at 289, 31 inputs and 289 contiguous output PTS, yielding 9.633333s within the unchanged threshold.
The final encoder uses the same explicit 30fps filter/passthrough chain for each bounded null preflight and formal H.264 export, with no trim, frame cap, resize or tolerance change.
The formal decoded frame count must match that film's observed preflight count.
A source-only parser draft rejected harmless zero-frame initialization summaries; the final parser retains these separately and requires one nonzero processing summary matching the unique EOF instance.
All earlier helper versions and failed receipts remain outside the repository.

## Cost and package

Only `viewer/static/style.css` differs among the 428 product files: 125 fewer source bytes.
Wheel size is 4,148,876 B, 21 B below baseline and 45,428 B below the original 4MiB cap.
The four original install budgets pass: 164.0655 MB installed, 42 packages and 52.5149 MB dependency download.
Four full viewer measurements ran in fixed A/B/B/A order without recording; all seven original viewer budgets pass in every run.

| Run | Cold start s | FPS | Frame ms | Assets FPS / ms | Server MB | Idle % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.9247 | 60.0551 | 1.6 | 60.1569 / 2.2 | 143.7 | 4.6 |
| B1 | 0.9426 | 60.1129 | 1.6 | 59.9780 / 2.0 | 143.5 | 4.8 |
| B2 | 0.9282 | 60.0930 | 1.9 | 60.1489 / 2.3 | 143.7 | 4.8 |
| A3 | 0.9366 | 60.1049 | 1.6 | 60.1469 / 2.2 | 144.0 | 4.6 |

These standard benchmark results do not establish a speedup or native-model cold-load timing.

## Watchable evidence

All films are full-frame 2560x1600 H.264, exported at 30 CFR from at most 5Hz real screenshot samples.
All 100 raw JPEGs and acquisition intervals are retained, with no screenshot/input concurrency.
The sampled span includes inventory reopen; the final close is a separate native UI still.
All four films passed complete decode, strict duration and frame-count checks.
Root inspected four decoded contact sheets (24 samples) and four actual inventory-open UI PNGs through resized previews; no continuous-playback, smoothness or every-frame-readability certification is claimed.

| Film | Seconds | Decoded frames | Raw samples |
| --- | ---: | ---: | ---: |
| [Malo before](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-drive-dock/malo-baseline-media-v3/workflow.mp4) | 7.2 | 216 | 20 |
| [Malo after](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-drive-dock/malo-candidate-media-v3/workflow.mp4) | 9.633333 | 289 | 30 |
| [Go2 before](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-drive-dock/go2-baseline-media-v3/workflow.mp4) | 7.2 | 216 | 20 |
| [Go2 after](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-drive-dock/go2-candidate-media-v3/workflow.mp4) | 9.5 | 285 | 30 |

Native desktop zoom, Windows and remote CI were not verified this round.
Push/merge remain pending at report creation and must be checked against a separate delivery receipt.
The broad visual/lightweight/play goal remains active, with no action-success tuning.
Next, screen thin decorative edges and joints in the existing Go2 warehouse and harbor crates, with Malo as a non-target control, before choosing a shared geometry correction.
