# Actual-world specular baseline screen (2026-10-28)

The ordinary-distance defect gate is not met, so no specular compensation candidate is implemented.
Root and an independent image reviewer do not establish a repeatable, attributable defect on two actual material paths.
This closes this unproven hypothesis; it does not claim that every surface is free of shimmer or that material quality is complete.
Product source remains `47f8fe814c536e7f24e2b8abbcc4fd2090a5101331d877262514ee4055a84939` from main `187f7ff7279dab4adbd96ad421b31a6cc81fdf3e`.

## Actual scene coverage

Unmodified authored Malo workshop, Go2 warehouse and Panda workbench worlds are copied into isolated homes.
No diagnostic props, material changes or reduced roughness are introduced.
CSS1280×800 at DPR1/2, studio/daylight, and three fixed distances produce 36 original PNGs.
The ordinary, inspection and diagnostic views use distance scales 1, 0.57 and 0.3 about each frozen target.
The ordinary view is explicitly selected, not the viewer's automatic framing; close views can crop the whole robot.
All six sessions retain exact paused qpos/qvel/ctrl, zero simulation time and all 482 model arrays, including names/paths, after every view.
These are within-session controls, not a claim of cross-session generated asset paths or pixel equality.
World files, all 748 source/test/helper/dependency inputs and 141 external input hashes remain unchanged.
All accepted pages report GL0 and no page errors, and their Chrome processes and server threads stop.

The actual complete mesh inventories contain no brushed or knurled procedural material.
The inventoried meshes with their visible flag enabled contain no procedural grain programs in Malo/Panda, and rough wood/stone grain in Go2.
Grounds use concrete in Malo/Go2 and wood in Panda.
The clipped projected AABB fractions are box estimates only, not visible pixel area, occlusion tests or masks.
Malo/Panda coatings show broad smooth shading in selected views; Go2 crates are mainly diffuse and rough.
The available DPR and light comparisons do not provide two attributable ordinary-distance specular defects.
Edges, thin debug lines and shadow differences alone do not establish a bump-variance problem.

## Static boundary and real recordings

The vendored Three shader computes geometry roughness from nonPerturbedNormal derivatives.
The existing grain filter already attenuates unresolved octaves and carries first-order slopes separately.
The metric bump helper itself uses screen derivatives.
Applying another screen derivative to that result would introduce nested derivatives, whose behavior is undefined by [GLSL ES 3.00 section8.9](https://registry.khronos.org/OpenGL/specs/es/3.0/GLSL_ES_Specification_3.00.pdf).
No global shader-chunk replacement, slope-magnitude roughness fudge or new slope-moment representation is tested or shipped.

Twelve actual paused scripted-camera movies contain three 2.4-second arcs at the three distances, with intentional cuts between them.
Encoding removes only setup gaps between these segments and has no duration cap.
Source timestamp intervals are retained subject to a 1/60-second segment tail and 30fps quantization; each encoded duration is checked within 0.15 seconds of its source-derived target.
Repeated source timestamps within a segment have zero duration; raw frames and timestamps remain retained.
All movies fully decode: 2629 encoded frames from 5179 verified accepted original JPEGs.
Root inspected all six still overview sheets, six DPR2 ordinary original files, and 72 decoded samples across all twelve movies.
The independent reviewer inspected all six still sheets, three DPR2 studio ordinary files and 36 DPR2 decoded samples.
Image display may resize original files; this is not a pixel-by-pixel audit.
Neither reviewer watched every continuous frame, so no temporal-stability or absence-of-shimmer certification is made.
These are scripted camera recordings, not native gameplay; 30fps encoding is not measured app FPS.

The initial DPR2 collector incorrectly required globally strict timestamps across stop/start segments.
Its failed manifest and raw JPEGs remain, but failed-stage timestamp metadata was not serialized before the assertion and is excluded from the accepted cohort.
The corrected collector saves timestamp metadata before timestamp validation and checks nondecreasing times within each segment.
Executed helper v1/v2 sources are retained and hash-bound to the successful manifests.
Other collector exception paths were not fault-injected; observed successful cleanup does not certify those paths.

## Verification and cost scope

Offline wheel build, Ruff/format, 41 experiment-record tests and nine actual GPU filtering/metric-normal checks pass.
All 409 product files match frozen source and wheel exactly; the 3928336 B wheel is byte-identical to the previously accepted PR40 wheel.
No runtime asset, dependency, texture, sampler or rendering pass is added by this records-only change.
There is no candidate or runtime change to benchmark, and no new loading/FPS/install/memory improvement is claimed.
The broad prior split suite and original11/11 budgets are prior PR40 evidence, not new results of this screen.
No full suite, installed smoke, slow assets/sensors suites, ABBA, physical phone or Windows run is claimed here.
Optional mypy/pyright/bandit/pip-audit are absent from the approved project environment and were not installed or counted as passing.
All 13 original uncommitted files remain hash-identical.
The closed local index binds 6314 files/1110217326 B; subsequent review, delivery and preview metadata are outside that index.
Raw data are in `../robotforge-evidence/2026-10-28-specular-baseline/`; the JSON companion binds movie hashes and exact scope.

## Next bounded direction

Move to a native demand screen for precise composition across props, local terrain and robot starts.
Try aligned obstacles, fixed gaps and tabletop arrangements in the real Malo/Go2/Panda worlds before adding controls.
If that demonstrates repeated positioning work, test one default-off planar snapping candidate through shared creationPoint/placeEntity paths.
Preview and submit must use one final coordinate, revalidate the same support after snapping and preserve current backend occupancy, reset, Undo and layout rules.
Do not slide a hit off a tabletop while retaining its old height, or silently drop to the lower floor.
No placement feature is implemented or validated in this record.
The material/lightweight/playability goal remains active; individual action-success tuning stays deferred.
