"""Pure camera-bookmark schema and serialized storage behavior, checked in Node.

The real viewer's DOM and camera integration are covered by browser acceptance.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "robotforge" / "viewer" / "static"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

PROGRAM = """
import { pathToFileURL } from "node:url";
import { webcrypto } from "node:crypto";
globalThis.crypto ??= webcrypto;
const m = await import(pathToFileURL(process.argv[1]).href);
const scenario = process.argv[2];
const scope = "a".repeat(64), other = "b".repeat(64);
const pose = {eye: [1, 2, 3], target: [0, 0, 0]};
const id = "00000000-0000-4000-8000-000000000000";
const view = {id, name: "Front", ...pose};
const store = {version: 1, sources: [{scope, views: [view]}]};
const clone = (value) => JSON.parse(JSON.stringify(value));
const code = (fn) => { try { fn(); return "ok"; } catch (error) { return error.code; } };
const settled = (promise) => promise.then(() => "ok", (error) => error.code);
const memory = (initial = null) => ({
  raw: initial, writes: 0,
  getItem(key) { if (key !== m.BOOKMARK_KEY) throw Error("wrong key"); return this.raw; },
  setItem(key, raw) { if (key !== m.BOOKMARK_KEY) throw Error("wrong key"); this.writes++; this.raw = raw; },
});
const immediate = {request: async (key, options, fn) => {
  if (key !== m.BOOKMARK_KEY || options.mode !== "exclusive") throw Error("wrong lock");
  return fn();
}};
const deferred = () => {
  const queue = [];
  return {queue, request: (key, options, fn) => new Promise((resolve, reject) => {
    queue.push(() => { try { resolve(fn()); } catch (error) { reject(error); } });
  })};
};
const options = (target, manager = immediate) => ({
  storage: () => target, locks: () => manager, isCurrent: () => true,
  scope, action: "save", name: "Side", capture: () => pose,
});
let out;
if (scenario === "schema") {
  const invalid = {};
  invalid.json = () => m.parseBookmarks("{");
  invalid.blank = () => m.parseBookmarks("");
  invalid.version = () => m.parseBookmarks('{"version":2,"sources":[]}');
  invalid.rawCap = () => m.parseBookmarks(" ".repeat(65537));
  invalid.scope = () => m.validateBookmarks({...store, sources: [{scope: "x", views: []}]});
  invalid.extra = () => m.validateBookmarks({...store, camera: 1});
  invalid.duplicateScope = () => m.validateBookmarks({...store, sources: [...store.sources, ...store.sources]});
  invalid.duplicateId = () => m.validateBookmarks({
    ...store, sources: [{scope, views: [view, {...view, name: "Side"}]}],
  });
  invalid.duplicateName = () => {
    const copy = clone(store); copy.sources[0].views.push({...view, id: "00000000-0000-4000-8000-000000000001"});
    m.validateBookmarks(copy);
  };
  invalid.untrimmed = () => m.validateBookmarks({...store, sources: [{scope, views: [{...view, name: " Front "}]}]});
  invalid.unknownViewField = () => m.validateBookmarks({...store, sources: [{scope, views: [{...view, zoom: 2}]}]});
  invalid.tooManySources = () => m.validateBookmarks({version: 1, sources: Array.from(
    {length: 17}, (_, i) => ({scope: i.toString(16).padStart(64, "0"), views: []}),
  )});
  invalid.tooManyViews = () => m.validateBookmarks({version: 1, sources: [{scope, views: Array.from(
    {length: 9}, (_, i) => ({
      ...view, id: `00000000-0000-4000-8000-${String(i).padStart(12, "0")}`, name: String(i),
    }),
  )}]});
  out = {
    empty: m.parseBookmarks(null), roundtrip: m.parseBookmarks(m.serializeBookmarks(store)),
    errors: Object.fromEntries(Object.entries(invalid).map(([key, fn]) => [key, code(fn)])),
  };
} else if (scenario === "names_poses") {
  out = {
    name: m.bookmarkName("  Cafe\\u0301  "), emoji: m.bookmarkName("😀".repeat(64)).length,
    invalidNames: ["", " ", "x\\u0000", "x\\n", "x\\u0085", "😀".repeat(65), null, 7]
      .map((name) => code(() => m.bookmarkName(name))),
    invalidPoses: [
      {eye: [NaN, 0, 1], target: [0, 0, 0]},
      {eye: [Infinity, 0, 1], target: [0, 0, 0]},
      {eye: ["1", 0, 1], target: [0, 0, 0]},
      {eye: [1e6 + 1, 0, 0], target: [1e6, 0, 0]},
      {eye: [0, 0, 0], target: [0, 0, 0]},
      {eye: [0.00001, 0, 0], target: [0, 0, 0]},
      {eye: [1e6, 1e6, 0], target: [0, 0, 0]},
      {eye: [1, 0], target: [0, 0, 0]},
      {eye: [, 0, 1], target: [0, 0, 0]},
      {eye: [1, 0, 0], target: new Array(3)}, null,
    ].map((value) => code(() => m.bookmarkPose(value))),
    low: m.bookmarkPose({eye: [1e-4, 0, 0], target: [0, 0, 0]}),
    high: m.bookmarkPose({eye: [1e6, 0, 0], target: [0, 0, 0]}),
  };
} else if (scenario === "fresh_concurrent") {
  const target = memory(), locks = deferred();
  const first = settled(m.mutateBookmarks({...options(target, locks), name: "Front"}));
  const second = settled(m.mutateBookmarks({...options(target, locks), name: "Side"}));
  locks.queue.shift()(); await first;
  locks.queue.shift()(); await second;
  const third = await settled(m.mutateBookmarks({...options(target), scope: other, name: "Front"}));
  out = {first: await first, second: await second, third, store: m.parseBookmarks(target.raw), writes: target.writes};
} else if (scenario === "stale") {
  const results = [];
  for (const change of ["identity", "generation", "disconnect", "new_action", "storage_event", "canSave"]) {
    const target = memory(), locks = deferred(); let valid = true, captures = 0;
    const queued = settled(m.mutateBookmarks({
      ...options(target, locks), isCurrent: () => valid, capture: () => { captures++; return pose; },
    }));
    valid = false; locks.queue.shift()();
    results.push({change, result: await queued, writes: target.writes, captures});
  }
  let valid = true;
  const target = memory();
  const duringCapture = await settled(m.mutateBookmarks({
    ...options(target), isCurrent: () => valid, capture: () => { valid = false; return pose; },
  }));
  out = {results, duringCapture, writes: target.writes};
} else if (scenario === "failures") {
  const corrupt = memory("{"); let captured = false;
  const badStore = await settled(m.mutateBookmarks({
    ...options(corrupt), capture: () => { captured = true; return pose; },
  }));
  const quota = memory(); quota.setItem = () => { throw Error("quota"); };
  const getter = await settled(m.mutateBookmarks({...options(memory()), storage: () => { throw Error("security"); }}));
  const read = memory(); read.getItem = () => { throw Error("security"); };
  const noLocks = await settled(m.mutateBookmarks(options(memory(), null)));
  const locksGetter = await settled(m.mutateBookmarks({
    ...options(memory()), locks: () => { throw Error("security"); },
  }));
  const badCapture = await settled(m.mutateBookmarks({
    ...options(memory()), capture: () => { throw Error("camera"); },
  }));
  out = {badStore, captured, corruptRaw: corrupt.raw, corruptWrites: corrupt.writes, getter,
    quota: await settled(m.mutateBookmarks(options(quota))), quotaRaw: quota.raw,
    read: await settled(m.mutateBookmarks(options(read))), noLocks, locksGetter, badCapture};
} else if (scenario === "caps_delete") {
  const target = memory();
  for (let i = 0; i < 8; i++) await m.mutateBookmarks({...options(target), name: String(i)});
  const duplicate = await settled(m.mutateBookmarks({...options(target), name: " 0 "}));
  const full = await settled(m.mutateBookmarks({...options(target), name: "Ninth"}));
  const before = m.parseBookmarks(target.raw);
  const missing = await settled(m.mutateBookmarks({...options(target), action: "remove", id}));
  for (const entry of before.sources[0].views) {
    await m.mutateBookmarks({...options(target), action: "remove", id: entry.id});
  }
  const deleted = m.parseBookmarks(target.raw);
  for (let i = 0; i < 16; i++) {
    await m.mutateBookmarks({...options(target), scope: i.toString(16).padStart(64, "0"), name: "Front"});
  }
  const sourceFull = await settled(m.mutateBookmarks({...options(target), scope: "f".repeat(64)}));
  out = {duplicate, full, missing, deleted, sourceFull, count: m.parseBookmarks(target.raw).sources.length};
} else throw Error("unknown scenario");
console.log(JSON.stringify(out));
"""


def evaluate(scenario: str) -> dict:
    run = subprocess.run(
        [NODE, "--input-type=module", "-e", PROGRAM, str(STATIC / "camera-bookmarks.js"), scenario],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_whole_store_schema_rejects_corruption_and_caps_without_treating_it_as_empty():
    got = evaluate("schema")
    assert got["empty"] == {"version": 1, "sources": []}
    assert got["roundtrip"]["sources"][0]["views"][0]["name"] == "Front"
    assert set(got["errors"].values()) == {"corrupt"}


def test_names_normalize_without_truncating_and_poses_are_finite_and_bounded():
    got = evaluate("names_poses")
    assert got["name"] == "Café" and got["emoji"] == 128
    assert set(got["invalidNames"]) == {"invalid_name"}
    assert set(got["invalidPoses"]) == {"invalid_pose"}
    assert got["low"]["eye"][0] == 1e-4
    assert got["high"]["eye"][0] == 1e6


def test_queued_mutations_read_fresh_store_and_keep_sources_separate():
    got = evaluate("fresh_concurrent")
    assert got["first"] == got["second"] == got["third"] == "ok"
    assert got["writes"] == 3
    sources = got["store"]["sources"]
    assert [view["name"] for view in sources[0]["views"]] == ["Front", "Side"]
    assert [view["name"] for view in sources[1]["views"]] == ["Front"]
    ids = [view["id"] for source in sources for view in source["views"]]
    assert len(set(ids)) == 3


def test_invalidated_queued_operations_never_capture_or_write_and_recheck_before_write():
    got = evaluate("stale")
    assert all(case["result"] == "stale" and case["writes"] == case["captures"] == 0 for case in got["results"])
    assert got["duringCapture"] == "stale" and got["writes"] == 0


def test_unavailable_or_failed_storage_preserves_unsaved_data_and_reports_cause():
    got = evaluate("failures")
    assert got["badStore"] == "corrupt" and got["corruptRaw"] == "{"
    assert got["corruptWrites"] == 0 and got["captured"] is False
    assert got["getter"] == got["quota"] == got["read"] == "storage_error"
    assert got["quotaRaw"] is None
    assert got["noLocks"] == got["locksGetter"] == "unavailable"
    assert got["badCapture"] == "invalid_pose"


def test_duplicates_and_capacity_never_overwrite_or_evict_and_delete_uses_id():
    got = evaluate("caps_delete")
    assert got["duplicate"] == "duplicate"
    assert got["full"] == got["sourceFull"] == "full"
    assert got["missing"] == "stale"
    assert got["deleted"] == {"version": 1, "sources": []}
    assert got["count"] == 16
