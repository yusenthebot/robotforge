# Natural grass ground imagery

Shared grass floors, heightfields and detailed chooser previews use one locally bundled CC0 Leafy Grass image pair.
The 256-square maps retain authored tint, existing 0.6 m artistic repeat, metric bump, material settings and physics.
Grass and oak preparation run concurrently with independent atomic fallback and 1,000 ms deadlines; a late completion only seeds the CPU cache for a future scene.
Existing three- and four-argument `groundMaterial` calls and the wood-only prepareGroundTextures return contract remain compatible.
No dependency or single-action-success tuning is added.

## Verification

Runtime `74ae683d092bde9c04874b3adeee01cd915fad31a4495b3e52afbd8cc5fafe5f` with 774 frozen source/script/test/dependency inputs; original 13 WIP bytes retained.
685 scoped default viewer/gallery/placement cases, 338 current non-slow browser cases and 9 GPU cases yield 1,032 distinct pytest passes.
Browser coverage is explicitly split over identical product source: initial 336 pass / 1 fail / 1 optional pxr collection skip, then corrected related modules 13/13; current collection and per-node receipts are checked.
The failed reset-order test also fails on the old product; its deterministic replacement holds actual WebSocket states and verifies both state-first/display-first orders with the original stale-receipt and Undo assertions.
Product code did not change for this test synchronization fix.
Seven canonical cases, source/fresh offline installed 60-tool smoke, 417 source/wheel/installed product files, installed mixed grass/oak viewer, Ruff/format/JS parsing and original 11 install/viewer budgets pass.
The full default/slow suite and optional unavailable type/security tools are not certified; manual paid GitHub Actions are not dispatched.

Default MALO/Go2/Panda retain 24 main-canvas exact pairs.
External admitted imagery to production has 1,447 integrity checks and 23/24 whole-window exact pairs; the remaining 290 pixels are solely the inspected clock 0.0 s / paused label, with all other pixels exact.
The initial strict whole-window failure and the intermediate misprinted 24 exact stdout are retained; final structured counts and final stdout say 23 exact plus 1 clock-label difference.
The first fresh-wheel QA helper incorrectly waited for simulation time while paused; its failed attempt is retained and the corrected paused-ready helper passes.
The first Go2 capture failed before recording because the QA incorrectly expected renderer DPR3 despite the existing cap of 2; the final guard checks actual device DPR3 / main DPR2 and 780x1688 buffer.
The first native MALO recording reached actual drive and reset but its six-second strict camera-settle check timed out during follow-height decay; raw footage remains preliminary.
The final native workflows stop following with the existing native control before fixed reset inspection, retaining the original 1e-9 camera tolerance and deadline.
The external first hot-image adapter GL1282 failure is retained; the accepted fresh-texture adapter is external only and never shipped.

Six final movies fully decode and 36 decoded samples were inspected by the root; the native MALO and narrow Chinese touch Go2 workflows cover actual grass/wood/grass choices, driving and release, pause/reset and persisted reload.
Device DPR3 portrait retains the existing capped main renderer DPR2 and is desktop Chrome touch emulation, not a physical-phone claim.
The other four clips are paused controlled park/MALO grass camera arcs; MALO grass is an explicit derived preview, not its authored concrete floor.
Source/world/model/state/UV/sampler controls and bounded relocated mesh-path handling are recorded separately in JSON.
Complete integration vectors are preserved within captures; paired stored model/qpos/qvel/ctrl arrays do not imply separately stored paired full integration vectors.
Six samples/movie and complete decode are not every-frame or flicker-free certification.

## Cost and reproduction

Wheel 4,031,203 B adds 79,500 B over the prior 3,951,703 B package and stays within the unchanged 4,096 KiB cap.
The two encoded images total 77,254 B; the lazy module is 103,620 B and is requested only for actual grass surfaces or detailed grass preview.
Default closed-editor worlds retain 31 HTTP resources with no grass-module request.
Grass uses the same two 256-square map slots and no additional rendering pass; sampled costs are in JSON, not peak GPU-memory or leak certification.
Source CC0 URLs/hashes, artistic source-scale caveat and offline fixed BOX/linear-light reproduction are in `src/robotforge/viewer/static/materials/README.md` and `scripts/prepare_ground_grass.py`.

| Fresh-profile case | Browser-cold main-ready ms before / candidate | Actual reload ms before / candidate |
|---|---:|---:|
| default malo | 513.25 / 540.60 | 271.50 / 270.75 |
| default go2 | 636.87 / 641.93 | 377.93 / 381.48 |
| grass park | 586.35 / 597.88 | 344.15 / 331.65 |
| grass malo | 524.52 / 509.15 | 266.93 / 267.90 |

ABBA followed by BAAB uses four fresh package processes/profiles per cohort/case at actual DPR2 with explicit light; server compilation is separate.
These are local GPU-finished main-frame samples, not complete sensor/UI readiness or universal/causal speed or RSS gains.
Full commands, failures, raw/decoded frames, license sources, wheel and hashes are retained at `../robotforge-evidence/2026-10-29-natural-grass/` and its sealed external-screen directory.
Native videos are `workflow-malo-v2/workflow.mp4` and `workflow-go2-v3/workflow.mp4` relative to that directory.
Broad shared visual-quality/lightweight/creative-play goal remains active; single-action success tuning stays deferred.
