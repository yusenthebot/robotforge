# Native desktop Chrome zoom observation

2026-11-02: the previously tool-blocked desktop check became executable after the unlock reply.
This is a new, scoped manual observation of the accepted product, not a product change or a replacement for earlier failed captures.
The [machine record](2026-11-02-desktop-zoom.json) pins twelve external evidence files and the unchanged source manifest.

## Inputs and method

Served the accepted `robotforge-native-capture` source at `1ba3c476ee97068d9abfdc4d785890bb5aaa7970`, whose tree equals merged main `5908f79a7d0781f59176a59c8776d2e179527cfb`.
All 817 frozen source inputs were checked before and after.
A disposable copy of the prior Malo workshop home/cache used the original world bytes and existing converted assets; import staging and downloads were forbidden.
The authoritative session remained paused at time zero.

Used the actual macOS Chrome application through native accessibility actions and keyboard shortcuts, with read-only CDP diagnostics.
No viewport or DPR emulation was applied.
Chrome's actual native control reported 110%, 125%, 150%, and 100%; 110% was only a transit step.
Saved `Zoom check` at 100%, exercised Fit, Side and Restore at 125%, Fit, Front and keyboard Restore at 150%, then Fit and Restore after returning to 100%.

| Actual zoom | CSS viewport | Browser DPR | Renderer DPR | Drawing buffer |
|---|---|---|---|---|
| 100% | 1512 × 717 | 2 | 2 | 3024 × 1434 |
| 125% | 1209 × 573 | 2.5 | 2 | 2418 × 1146 |
| 150% | 1008 × 478 | 3 | 2 | 2016 × 956 |
| Returned 100% | 1512 × 717 | 2 | 2 | 3024 × 1434 |

The browser's outer size stayed 1512 × 894 and visual viewport scale stayed one.
Renderer DPR remains capped by existing behavior; this observation does not change or certify its image quality policy.

## Result and evidence

The demonstrated Fit, axis-view and bookmark recovery path worked at the observed zoom states.
At 150%, Save/Restore initially lie below the panel's internal viewport: this is a real scrolling requirement, not an all-controls-visible result.
Six native Tab keys brought Restore fully into view with a visible focus ring; Return restored the saved camera.
The three stable restore observations at 125%, 150%, and final 100% have maximum eye/target component error `2.398081733190338e-14`, below the existing `1e-9` check, with unchanged bookmark bytes.

Native screenshots, inspected by the root and an independent frontend reviewer:

- [125% panel](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-desktop-zoom/native-125-before.png)
- [150% panel before scrolling](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-desktop-zoom/native-150-panel.png)
- [150% keyboard-reached Restore](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-desktop-zoom/native-150-keyboard-reach.png)
- [150% restored view](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-desktop-zoom/native-150-restored.png)
- [Final 100%](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-desktop-zoom/native-100-final.png)

All 481 native model arrays, qpos/qvel/ctrl, time, paused state, and owned home/world/cache file bytes match before and after.
All twelve page observations retain time zero, scene version one, no lost WebGL context and GL error zero.
Captured warning/error console entries are empty.
The dedicated tab closed, Chrome zoom returned to 100%, the server thread stopped, and the renderer closed.

## Retained failed attempts and limits

The first app binding addressed another front window; the Chrome group button then selected the actual test tab.
The early Cmd+0 is not treated as a test-page observation.
An interim desktop screenshot was unavailable and one action reported a changed application; fresh state and explicit group selection recovered the target.
The filename `chrome-menu-100.png` is retained, but that screenshot shows the page rather than a menu or zoom percentage.
The first native text-entry attempt left an empty bookmark name and Save correctly rejected it; the observation label `100-bookmark-saved` records that attempt, not a successful save.
Native setValue then visibly populated the name and Save succeeded.
The native AX scroll call at 150% reported `noWindowsAvailable`; pointer-wheel scrolling remains unverified, while keyboard reach and restoration were observed.
The first Side diagnostic was sampled during its tween and is not a settled-camera assertion.

No new pytest run, installation, package/performance budget measurement, movie, Windows, remote CI or system display-scale check is claimed.
This check covers this local macOS Chrome, Chinese UI and Malo world; it does not qualify every asset, platform, zoom level or interaction.
Earlier E83 failures remain retained and closed.
There is no material-quality gain or action-success change in this record.

## Next

Continue the shared visual/lightweight/playability goal.
Native-image anisotropy has source coverage in X2 and ANYmal C, but the historical ordinary images do not establish a shared visible defect; no sampler candidate or new runtime comparison was made here.
Any later screen must first demonstrate ordinary-view benefit and retain the existing cost gates.
