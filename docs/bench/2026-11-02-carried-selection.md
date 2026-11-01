# Include items on top (2026-11-02)

Selecting a table or another rigid support now offers an explicit **Include items on top** action.
It selects the complete supported layout, including recursively stacked items, for the existing group move, copy and arrangement-save controls.
A single remaining member keeps the single selection and shows a short message.
Existing interfaces, placement rules and action-success behavior remain compatible.
This milestone improves scene building; it makes no material-quality improvement claim.

The query checks paused state, version, generation and world SHA, compares complete saved/live support edges, and returns at most 16 members in world order.
Unsupported members, robots, cycles, incomplete support and oversized layouts reject the complete query.
Private model/data copies keep forward/contact calculations away from live state.
The client validates the complete reply before replacing outlines and rejects replies superseded by selection, pointer, scene, pause, reset or socket changes.
There is no new idle polling or dependency.

## Verification and provenance

The [JSON report](2026-11-02-carried-selection.json) binds receipts, raw captures, methods, media and retained failures by SHA.
Final source/scripts/tests/lock are frozen as 812 inputs, SHA `9c04395b07b24c81cb5c3b1d5a213072394638a633b69baee958008a681c2f70`.
The seven candidate freezes differ only in tests; all product-file hashes are identical.
The baseline checkout at `85bd0ac` has the complete tree of merged main `1bae581`.

- 308 unit/support/edit/copy/group/move/arrangement cases passed.
- 20 new actual Chrome cases passed, including English desktop, Chinese 320×740/DPR2, button bounds/hit testing, move/copy/save/Undo, eleven malformed replies and five held-reply interruptions.
- 60 existing Chrome cases passed, including browser reload, map changes, arrangement insertion, occupied retries, rotation and latest Undo.
- These are 388 distinct JUnit cases without skips, plus seven same-runtime framework comparisons.
- Ruff lint/format, original viewer/WebSocket/MCP smoke, and fresh locked offline installed-package selection smoke passed.
- All 429 product files match source, wheel and installation.
- Bandit, pip-audit, mypy and pyright were unavailable; they are not reported as passing.

Browser plugin was unavailable, so verification used the repository's real Chrome/CDP harness.
Actual pages rendered the intended worlds, controls and native models without captured browser errors or blank/error screens.
Desktop captures are 1280×800/DPR1; standard viewer performance measurements use the existing DPR2 configuration.
This is scoped local macOS verification, not a fresh full-default/full-browser suite, Windows or remote-CI claim.
The 60 compatibility cases cover browser reload and cross-map insertion; this round does not claim a new actual-model save/reopen-in-a-new-server film.

## Original-model workflow evidence

Fresh owned homes/profiles load the actual cached Malo URDF/STLs and Menagerie Go2, preserving original workshop/warehouse descriptors and explicitly appending the declared stack fixture.
No model download, repair or staging was allowed.
Each fixture contains a table, two tabletop objects and one stacked upper block.
Within each run, all 481 model arrays and the recorded native state remain exact during selection.
Candidate queries additionally preserve complete `mjSTATE_INTEGRATION`, model/data identity, files and edit receipts.

Each baseline reproduces the server rejection of moving the support alone, then uses four manual member clicks and enters group-move preparation before cancelling.
Baselines do **not** commit a group move or execute Undo.
Each candidate uses two declared native button queries: a read-only proof followed by Escape, then a second query leading directly into a real four-member move and Undo.
Every query returns the exact four names, preserves the camera and creates four highlights.
Move retains member offsets and authored properties; Undo restores the original world bytes and integration state while version/generation advance normally.

Ten diagnostic full-canvas frames form eight exact RGBA/camera comparisons: two baseline within-run pairs, four candidate within-run pairs and two initial cross-source pairs.
These comparisons are correlated controls, not eight independent visual-quality trials.
Selection is cleared through real Escape before comparison; only the source-identified robot selection ring is temporarily hidden in the separate diagnostic framebuffer and restored immediately.
Actual UI screenshots and video samples retain the normal ring and visible selection highlights.
Initial draw counts and renderer memory counts match across sources.

Go2 baseline v3 uses `[3,0,0]` for its rejected single-prop move.
The first candidate's actual group request at that point correctly fails because the table footprint overlaps the dock ramp.
Fresh candidate v2 declares `[2,0,0]`, preserving camera/fixture/hit gates, and completes move/Undo without an in-run retry.
Go2 before/after films therefore do **not** demonstrate an identical destination.
The group anchor is the arithmetic mean of all member origins, rather than the table origin.

Actual four-member query round trips were 35.7/18.2 ms for Malo and 74.9/69.8 ms for Go2.
These are two UI samples per model, not a latency distribution or a 2/16-member performance benchmark.

## Retained failures and test corrections

Initial backend runs retain 26 failed/9 passed and 3 failed/32 passed, with original logs/XML/receipts.
Test-helper corrections remove duplicate name arguments and use the existing `LayoutError` type for custom-hook rejection.
The blanket MjData getter snapshot also compared independent NumPy allocations returned for absent native arena pointers.
The final test compares every model array and native-backed data array, complete integration state, files and receipts; detached MjData getters retain exact ownership classification, shape and dtype checks.
Native-backed/detached transitions still fail, and live forward/compile/load/apply/move-admission sentinels remain active.
There is no field-name exclusion list.

The separate no-op/invalid-parse/valid-query/getter diagnostic observed **no array drift** in any phase.
It did not reproduce the earlier failure.
The ownership contract instead follows the pinned [MuJoCo arena getter](https://github.com/google-deepmind/mujoco/blob/3.15.0/python/mujoco/structs.cc#L856-L864), [InitPyArray wrapper](https://github.com/google-deepmind/mujoco/blob/3.15.0/python/mujoco/structs.h#L980-L995), [locked dependency](https://github.com/google-deepmind/mujoco/blob/3.15.0/python/mujoco/CMakeLists.txt#L276-L288) and [pybind11 constructor](https://github.com/pybind/pybind11/blob/97bf890db679505a14dfe547a5e77bb2bd05dc90/include/pybind11/numpy.h#L967-L1014).
Changing bytes in detached allocations are not described as live simulation mutation.

Browser v1 used an incorrect table-origin expectation for the four-member mean anchor.
Browser v2's 320px destination was outside the viewport; the next diagnostic preserves x=322.677 and its native PNG before clicking.
The final phone flow uses declared visible floor point `[0,1,0]`, with unchanged camera, bounds/canvas-hit gates and matching explicit offset assertions.
Browser v4 incorrectly expected Undo visible while selection mode intentionally hides it; final tests check receipt/state preservation, Escape, visible enabled Undo and the actual inverse.
All earlier failures remain rejected; the final complete 20-case run is v5.

Go2 baseline v1 emitted no edit request and saved no hit diagnostic, so its precise cause remains unknown.
Fresh v2 records the placement hint obstructing the declared CSS point and fails the canvas-hit gate; v3 uses a declared clear point.
The later candidate dock-overlap rejection is a separate, actual backend collision result.
The first report assembler also rejected partial benchmark records because it incorrectly expected every section to run; corrected aggregation checks the four install/seven viewer budgets and retains all other sections as `not run`.
No benchmark threshold or runtime result changed.

## Cost

Wheel size is **4,151,437 B**, **+2,561 B** against baseline, with **42,867 B** remaining below the original 4 MiB cap.
The four original install budgets pass: 164.0739 MB installed, 42 packages and 52.5149 MB dependency download.
No new textures, models or dependencies are included.
Four full viewer measurements ran serially in fixed A/B/B/A order without recording; all seven original viewer budgets pass in every run.

| Run | Cold start s | FPS | Frame ms | Assets FPS / ms | Server MB | Idle % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.9243 | 60.3468 | 1.6 | 60.1589 / 4.1 | 143.8 | 4.8 |
| B1 | 0.9312 | 60.0989 | 1.8 | 60.1529 / 4.3 | 143.8 | 4.6 |
| B2 | 0.9168 | 60.1029 | 1.6 | 60.1489 / 4.0 | 143.7 | 4.8 |
| A3 | 0.9252 | 60.1249 | 1.5 | 60.1549 / 4.0 | 143.9 | 4.8 |

Server MB is the existing benchmark metric; its separately recorded warehouse peaks were 568.5/570.5/567.9/569.5 MB.
These figures establish budget compliance, not a universal speedup or native-model cold-load timing.
Actual-model resource lists contain 33 baseline entries and 35 candidate entries; the extra paths are `/mesh/2.bin` and `/mesh/3.bin`.
Existing scene installation fetches meshes by model version, so these correspond to the candidate's move/Undo rebuilds by source inspection.
The endpoint flows differ; transfer bytes were not measured and equal bandwidth is not claimed.

## Watchable evidence

All four films are native 1280×800 H.264, exported at 30 CFR from ten real, sequential, at-most-5Hz screenshot samples each.
All 40 original JPEGs and acquisition intervals remain unchanged; there is no trim, resize or duration cap.
Each film passes complete decode, strict duration/frame-count checks and six decoded contacts.
Root inspected all four contact sheets (24 samples), actual desktop selection/move UI images and the 320px selected-control image.
This is discrete inspection, not continuous playback or smoothness certification.

| Film | Seconds | Decoded frames |
| --- | ---: | ---: |
| [Malo before](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-carried-selection/malo-baseline-media-v1/workflow.mp4) | 6.633333 | 199 |
| [Malo after](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-carried-selection/malo-candidate-media-v1/workflow.mp4) | 8.933333 | 268 |
| [Go2 before](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-carried-selection/go2-baseline-media-v1/workflow.mp4) | 7.066667 | 212 |
| [Go2 after](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-carried-selection/go2-candidate-media-v1/workflow.mp4) | 9.966667 | 299 |

Native desktop zoom remains unverified: after the user's unlock reply, CUA still reported a locked Mac.
No repeat unlock request was made.
All owned test/application/browser/encoder processes were reaped, and original WIP/native inputs and retained experiment checkouts remained exact.
Feature commit `12d5eea` is pushed; final report commit and PR delivery must be verified through the separate delivery receipt.
The broader visual/lightweight/play goal remains active, with action-success tuning deferred.
Next, assess material-editing coverage for common furniture such as tables and bins, reusing existing finishes and checking real views and original budgets before extending it.
