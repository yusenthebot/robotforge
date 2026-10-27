# Wood color and relief screen (2026-10-28)

The frozen external candidate is rejected at the ordinary-distance table-and-bench gate.
The table is smoother but flatter and less distinct; the bench changes subtly without clearly improving wood recognition.
Root and an independent reviewer inspected the original comparisons and agree that blurred soft bands are insufficient.
Production remains byte-identical on `b58ddc20ac481b2aafc0629cab54afa0f58ae9c0b5e29e9d815738ab71f2961c`; no candidate shader ships.

## Frozen representation

Only external `look.js` changes (+772 source bytes).
This is a different representation from the rejected pith/ring field: phase-free longitudinal color with raw domain warp, and independent fine relief/roughness.
The two existing color octaves use their original amplitudes; a raw slowly varying noise bends their transverse coordinates.
A fourth fine noise supplies only relief and roughness; four noise evaluations replace two.
The parameters in the JSON companion were fixed before the first capture and never retuned.
Mapped wood retains the exact previous field, zero strength bypasses it, and ground/unrelated materials stay unchanged.
The fully filtered color and roughness means remain 0.575; finite-image means are not guaranteed equal.
Static review found no shader derivative/cache/map blocker, which does not establish visual benefit or cost.

## Actual controlled comparison

Five worlds with 12 total views per cohort produce 24 original 2560×1600 PNGs in actual Chrome at CSS1280×800, DPR2.
The same studio lighting, camera/projection, reset-paused state and hidden empty-world note are used for table, bench, kitchen, three-axis solids and rover control.
All 2542 controls pass: world files, full scenes, all five model-array sets, camera error below 1e-10, draw/resource/program counts and GL0.
Native-image, zero-strength and rubber GPU readbacks are nonblank and byte-exact across cohorts.
These checks establish comparison integrity, not appearance acceptance.
There are no added assets, samplers, passes or dependencies; unchanged resource counts do not establish unchanged GPU work.
No ABBA, install/loading/FPS/memory budget or candidate lifecycle certification followed the appearance rejection.
Sequential draw samples are descriptive and include values below clock precision.
The completed analyzer tail overlapped the start of movie acquisition; none of those jobs is claimed as an isolated cost experiment.

## Actual movies and product checks

Four paused scripted-camera movies cover ordinary-to-grazing table and bench views.
All fully decode and preserve paused qpos/qvel/ctrl, zero simulation time, GL/page error checks, source identities and stopped Chrome/server threads.
Root inspected six decoded samples per movie, 24 total; this is not a frame-by-frame motion audit or a shimmer/swimming improvement claim.
The corrected recording helper is the actual helper used in this round; no fault-injection certification is claimed.

baseline-wood_props: 6.567s, 197 decoded frames, 387 original JPEGs; SHA256 `4980a8befa4c014a7d0793c970af57b08d508cc226a34db050ede574fa0f2552`.

baseline-park: 6.567s, 197 decoded frames, 385 original JPEGs; SHA256 `6fa9bb62048cb498c704a69fb07ae8639198bcce134a41f05a94f6ca9b3709d2`.

candidate-wood_props: 6.567s, 197 decoded frames, 386 original JPEGs; SHA256 `8e4999d69b7e799fd128859e22a54cbac32f8cea3ea5047a7bc9a8c43b3f5669`.

candidate-park: 6.567s, 197 decoded frames, 385 original JPEGs; SHA256 `dba98a3f2beaad2ec8c764cde0637be0509094881054329c5a6941ad5a09d45f`.

Offline wheel build, Ruff/format and 41 existing experiment-record tests pass.
All 408 product files are byte-identical to baseline source and wheel; wheel size stays 3925041 B.
The 1635 retained artifacts have a separate SHA256/size index; all 13 original uncommitted files remain hash-identical.
Type/security tools absent from the approved project environment were not installed or reported as passed.
Raw sources, helpers, manifests, original images/JPEGs and movies are retained under `../robotforge-evidence/2026-10-28-wood-color-relief/`.

## Next direction

Stop procedural wood-signal tuning rather than extend this rejected candidate.
Next make existing shared finishes directly explorable on already placed primitive props, with persistence and Undo.
Keep the appearance override separate from physical `material`, which controls density-derived mass and magnetic collision affinity.
Use explicit existing creative-edit rebuild/pause/reset semantics; do not claim uninterrupted simulation state preservation.
Validate compiler physical contracts, stale/failed transaction rollback, native visual editing/reopen/Undo and original lightweight budgets.
This is the next plan, not implemented or certified in this record.
The broad material, lightweight and playability goal remains active; individual action-success tuning stays deferred.
