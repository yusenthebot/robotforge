# Mesh transport screen — 2026-10-28

The external gzip-1 candidate is not promoted. It reduces the actual Malo mesh
by 74.72% and Go2 by 70.20%, but Malo's median compression CPU time is
261.499 ms, above the prospectively fixed 250 ms screen. No product code changes,
compression-level search or additional samples were used to obtain a pass.
This does not establish that browser loading regresses or that an asynchronous
implementation has no value: neither was tested after the first-stage refusal.

## Frozen experiment

On merged base `622f554`, copy each existing original world into an isolated home,
load its real model through `load_playground`, then obtain the unmodified hub's
actual mesh payload. Each world runs in a separate process. The single algorithm
is `gzip.compress(blob, compresslevel=1, mtime=0)`, three sequential samples per
model, with process CPU and wall time measured separately. Every compressed
sample is decompressed and compared to the entire original bytes. Identical gzip
hashes across the three repetitions also establish deterministic output here.

These are new screening thresholds, not existing repository budgets: both
models must reduce encoded mesh bytes by at least 30%, with compression CPU
median at most 250 ms. The next stage would require fixed ABBA and BAAB browser
cohorts, cold/edit ready regression at most max(50 ms, 5%), peak server RSS
increase at most 64 MiB, event-loop extra delay at most 20 ms and no
candidate-only stall above 50 ms. Those later gates were not run or passed.

| Model | Raw bytes | Gzip bytes | Reduction | CPU samples (ms) | CPU median |
|---|---:|---:|---:|---|---:|
| Malo workshop | 47,064,672 | 11,897,380 | 74.72% | 263.101 / 261.499 / 261.076 | 261.499 ms |
| Go2 warehouse | 20,332,512 | 6,058,843 | 70.20% | 149.878 / 152.165 / 148.353 | 149.878 ms |

Python 3.12.16 on this Mac was used for both models. The raw world copies and
original world files remain exact. This is one local machine and one frozen
algorithm, not a general comparison of compression libraries or hardware.
Recorded `ru_maxrss` values are process high-water marks already affected by
model compilation; a zero change cannot establish zero compression memory cost.
Python decompression times are not browser decode or first-interaction timings.

## Compatibility and verification scope

The existing `/mesh/{version}.bin` path still returns the original representation;
no API, scene packing, shader, physics, texture, dependency or budget changes.
Any future negotiated implementation must preserve decoded bytes, stale-version
409, no-store behavior, raw clients and current-version ownership. It must also
handle cancellation without duplicate workers or publishing an old world's
result. HTTP negotiation design references
[RFC 9110 section 12.5.3](https://www.rfc-editor.org/rfc/rfc9110.html#section-12.5.3).
None of that candidate machinery is implemented by this documentation change.

All 761 source/test/script/dependency inputs remain identical to the verified
merged terrain baseline; runtime digest remains
`bd83fe3b4eee1e7ec8059194cdd4767ad54c8ee8c194b4147db8748108502b61`.
The offline wheel's 413 product files equal current source and the previous
merged wheel. The 13 original checkout WIP files remain unchanged.
Twenty-nine selected existing viewer-server/native-texture tests, Ruff check,
format and offline build pass. These checks certify the unchanged product,
not a compression candidate. Optional mypy, pyright, Bandit and pip-audit are
unavailable. No new full-suite, framework, installed-smoke, budget, GPU/pixel,
browser loading, control responsiveness or video certification is claimed.
The initial pytest command named a nonexistent scene test file and collected no
tests (exit 4); its log/XML are retained beside the corrected 29-pass run.

Evidence: `../robotforge-evidence/2026-10-28-mesh-transport/`, containing the
predeclared plan, external helper, both raw/gzip payloads, scene descriptions,
per-model timing records, exact-input/package bridge, logs and final decision.
There is no new candidate movie because the candidate stopped before browser
promotion; the prior merged continuous-building recordings remain separate.

The broad material, lightweight and creative-play goal remains active. Next
check the actual material identity of a large visible part on each original
Malo and Go2 against real references and its effective viewer mapping before
proposing another shared material rule. Names or screenshot differences alone
do not prove a material defect. Individual action success remains deferred.
