// Named main-camera poses, scoped to an opaque source identity and this browser origin.
export const BOOKMARK_KEY = "robotforge.camera-bookmarks.v1";
const SCOPE = /^[0-9a-f]{64}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const fail = (code) => { throw new BookmarkError(code); };
export class BookmarkError extends Error {
  constructor(code) { super(code); this.code = code; }
}

export function bookmarkName(value) {
  if (typeof value !== "string" || /\p{Cc}/u.test(value)) fail("invalid_name");
  const name = value.trim().normalize("NFC");
  if (!name || [...name].length > 64) fail("invalid_name");
  return name;
}

export function bookmarkPose(value) {
  const vector = (v) => Array.isArray(v) && v.length === 3
    && [...v].every((x) => typeof x === "number" && Number.isFinite(x) && Math.abs(x) <= 1e6);
  if (!value || !vector(value.eye) || !vector(value.target)) fail("invalid_pose");
  const distance = Math.hypot(...value.eye.map((x, i) => x - value.target[i]));
  if (!Number.isFinite(distance) || distance < 1e-4 || distance > 1e6) fail("invalid_pose");
  return { eye: [...value.eye], target: [...value.target] };
}

export function validateBookmarks(value) {
  const object = (v) => v && typeof v === "object" && !Array.isArray(v);
  const keys = (v, expected) => Object.keys(v).sort().join() === expected.sort().join();
  if (!object(value) || !keys(value, ["version", "sources"]) || value.version !== 1
    || !Array.isArray(value.sources) || value.sources.length > 16) fail("corrupt");
  const scopes = new Set(), ids = new Set();
  const sources = value.sources.map((source) => {
    if (!object(source) || !keys(source, ["scope", "views"]) || typeof source.scope !== "string" || !SCOPE.test(source.scope)
      || scopes.has(source.scope) || !Array.isArray(source.views) || source.views.length > 8) fail("corrupt");
    scopes.add(source.scope);
    const names = new Set();
    const views = source.views.map((view) => {
      if (!object(view) || !keys(view, ["id", "name", "eye", "target"])
        || typeof view.id !== "string" || !UUID.test(view.id) || ids.has(view.id)) fail("corrupt");
      let name, pose;
      try { name = bookmarkName(view.name); pose = bookmarkPose(view); } catch { fail("corrupt"); }
      if (name !== view.name || names.has(name)) fail("corrupt");
      names.add(name); ids.add(view.id);
      return { id: view.id, name, ...pose };
    });
    return { scope: source.scope, views };
  });
  const result = { version: 1, sources };
  if (JSON.stringify(result).length > 65536) fail("corrupt");
  return result;
}

export function parseBookmarks(raw) {
  if (raw === null) return { version: 1, sources: [] };
  if (typeof raw !== "string" || raw.length > 65536) fail("corrupt");
  let value;
  try { value = JSON.parse(raw); } catch { fail("corrupt"); }
  return validateBookmarks(value);
}

export function serializeBookmarks(value) {
  return JSON.stringify(validateBookmarks(value));
}

function readBookmarks(storage) {
  try { return parseBookmarks(storage.getItem(BOOKMARK_KEY)); }
  catch (error) { if (error instanceof BookmarkError) throw error; fail("storage_error"); }
}

// Every mutation re-reads under the same origin lock; a queued operation owns no snapshot of the store.
export async function mutateBookmarks({ storage, locks, isCurrent, scope, action, name, id, capture }) {
  const check = () => { if (!SCOPE.test(scope) || !isCurrent()) fail("stale"); };
  check();
  let manager;
  try { manager = locks(); } catch { fail("unavailable"); }
  if (!manager || typeof manager.request !== "function") fail("unavailable");
  try {
    return await manager.request(BOOKMARK_KEY, { mode: "exclusive" }, () => {
      check();
      let target;
      try { target = storage(); } catch { fail("storage_error"); }
      const store = readBookmarks(target);
      let source = store.sources.find((entry) => entry.scope === scope);
      if (action === "save") {
        const clean = bookmarkName(name);
        if (source?.views.some((view) => view.name === clean)) fail("duplicate");
        if ((source?.views.length ?? 0) >= 8 || (!source && store.sources.length >= 16)) fail("full");
        check();
        let captured;
        try { captured = capture(); } catch { fail("invalid_pose"); }
        const pose = bookmarkPose(captured);
        const view = { id: globalThis.crypto.randomUUID(), name: clean, ...pose };
        if (!source) { source = { scope, views: [] }; store.sources.push(source); }
        source.views.push(view);
      } else if (action === "remove") {
        if (!source?.views.some((view) => view.id === id)) fail("stale");
        source.views = source.views.filter((view) => view.id !== id);
        if (!source.views.length) store.sources = store.sources.filter((entry) => entry !== source);
      } else fail("stale");
      const raw = serializeBookmarks(store);
      check();
      try { target.setItem(BOOKMARK_KEY, raw); } catch { fail("storage_error"); }
      return store;
    });
  } catch (error) {
    if (error instanceof BookmarkError) throw error;
    fail("storage_error");
  }
}

export function createCameraBookmarks({ element, context, capture, restore, t,
  storage = () => window.localStorage, locks = () => navigator.locks }) {
  const doc = element.ownerDocument;
  element.tabIndex = -1;
  const make = (tag, className, text) => {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const word = (key, vars) => t(`view_saved_${key}`, vars);
  const title = make("h3", "camera-bookmarks-title");
  const note = make("p", "camera-bookmarks-note");
  const form = make("form", "camera-bookmarks-form");
  const label = make("label", "camera-bookmarks-label");
  const input = make("input", "camera-bookmarks-name");
  input.type = "text"; input.autocomplete = "off"; input.maxLength = 128;
  label.append(input);
  const save = make("button", "pill camera-bookmarks-save"); save.type = "submit";
  form.append(label, save);
  const list = make("ul", "camera-bookmarks-list");
  const status = make("p", "camera-bookmarks-status");
  status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
  element.append(title, note, form, list, status);
  let epoch = 0, identity = "", opened = false, pending = null, busy = false, problem = null;
  let entries = [], controls = [];
  const current = () => ({ ...context() });
  const token = (c) => JSON.stringify([c.scope, c.version, c.generation]);
  const usable = (c) => c.ready === true && typeof c.scope === "string" && SCOPE.test(c.scope);
  const say = (key, vars) => {
    const text = word(key, vars);
    if (status.textContent !== text) status.textContent = text;
  };
  const available = () => {
    try { const manager = locks(); return Boolean(manager && typeof manager.request === "function"); }
    catch { return false; }
  };
  function clearPending() {
    epoch++; busy = false; pending = null;
  }
  function cancel() {
    const removedId = pending?.id;
    const cancelled = pending !== null;
    clearPending();
    if (cancelled) {
      render(); say(entries.length ? "title" : "empty");
      controls.find((control) => control.dataset.removeId === removedId)?.focus();
    }
    update();
    return cancelled;
  }
  function invalidate() {
    clearPending(); input.value = ""; entries = []; list.replaceChildren(); controls = [];
    problem = null; say("stale");
  }
  function guard(expected, saving = false) {
    const c = current();
    return usable(c) && token(c) === expected && (!saving || c.canSave === true);
  }
  function load() {
    try {
      const c = current();
      if (!usable(c) || !available()) { problem = "unavailable"; entries = []; return; }
      const store = readBookmarks(storage());
      entries = store.sources.find((source) => source.scope === c.scope)?.views ?? [];
      problem = null;
    } catch (error) {
      problem = error instanceof BookmarkError ? error.code : "storage_error";
      entries = [];
    }
  }
  function render() {
    title.textContent = word("title"); note.textContent = word("note");
    input.setAttribute("aria-label", word("name")); input.placeholder = word("name");
    save.textContent = word("save");
    list.replaceChildren(); controls = [];
    const renderedToken = token(current());
    for (const entry of entries) {
      const row = make("li", "camera-bookmarks-row");
      row.append(make("span", "camera-bookmarks-view-name", entry.name));
      const button = (key, handler, named = false) => {
        const node = make("button", `pill camera-bookmarks-${key}`, word(key)); node.type = "button";
        node.setAttribute("aria-label", word(named ? `${key}_named` : key, { name: entry.name }));
        node.dataset.bookmarkAction = key; node.dataset.bookmarkId = entry.id;
        node.addEventListener("click", handler); row.append(node); controls.push(node);
        return node;
      };
      button("restore", () => {
        const expected = renderedToken;
        if (!guard(expected)) return;
        clearPending();
        load();
        const fresh = entries.find((view) => view.id === entry.id);
        if (problem || !fresh || !guard(expected)) { render(); say(problem ?? "stale"); update(); return; }
        try { restore(bookmarkPose(fresh)); render(); say("restore_named", { name: fresh.name }); }
        catch { say("invalid_pose"); }
        update();
      }, true);
      const removeButton = button("remove", () => {
        const c = current(); if (!guard(renderedToken) || problem || !available()) return;
        clearPending(); pending = { id: entry.id, name: entry.name, token: token(c) };
        render(); say("remove_confirm", { name: entry.name }); update();
        controls.find((control) => control.dataset.bookmarkAction === "cancel"
          && control.dataset.bookmarkId === entry.id)?.focus();
      }, true);
      removeButton.dataset.removeId = entry.id;
      if (pending?.id === entry.id) {
        row.append(make("span", "camera-bookmarks-confirmation", word("remove_confirm", { name: entry.name })));
        button("confirm", () => {
          if (!pending || !guard(pending.token)) return;
          const deletion = pending;
          void mutate("remove", deletion.token, { id: deletion.id, name: deletion.name });
        });
        button("cancel", () => { if (usable(current())) cancel(); });
      }
      list.append(row);
    }
  }
  async function mutate(action, expected, values) {
    const saving = action === "save";
    if (!guard(expected, saving) || problem || !available()) { say(problem ?? "unavailable"); return; }
    const removedIndex = saving ? -1 : entries.findIndex((entry) => entry.id === values.id);
    let removed = false, focusId;
    const operation = ++epoch;
    pending = null; busy = true; render(); say("busy"); update();
    try {
      await mutateBookmarks({ storage, locks, scope: current().scope, action, ...values, capture,
        isCurrent: () => operation === epoch && element.isConnected && guard(expected, saving) });
      if (operation !== epoch || !guard(expected)) return;
      busy = false; if (saving) input.value = "";
      load(); render(); say(problem ?? (saving ? "success" : "removed"), { name: values.name });
      if (!saving && !problem) {
        removed = true;
        focusId = entries[Math.min(Math.max(removedIndex, 0), entries.length - 1)]?.id;
      }
    } catch (error) {
      if (operation !== epoch) return;
      busy = false;
      const code = error instanceof BookmarkError ? error.code : "storage_error";
      if (["corrupt", "storage_error", "unavailable"].includes(code)) problem = code;
      render(); say(code);
    }
    update();
    if (removed && guard(expected)) {
      const target = controls.find((control) => control.dataset.bookmarkAction === "restore"
        && control.dataset.bookmarkId === focusId) ?? input;
      if (!target.disabled) target.focus();
      else element.focus();
    }
  }
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!guard(identity, true)) return;
    void mutate("save", identity, { name: input.value });
  });
  function update() {
    const c = current(), next = token(c);
    if (identity !== next || (!c.ready && (pending || busy || input.value))) {
      identity = next; invalidate();
    }
    const persistence = available();
    const enabled = usable(c) && !problem && persistence && !busy;
    const saveDisabled = !enabled || c.canSave !== true;
    if (input.disabled !== saveDisabled) input.disabled = saveDisabled;
    if (save.disabled !== saveDisabled) save.disabled = saveDisabled;
    for (const control of controls) {
      if (control.disabled !== !enabled) control.disabled = !enabled;
    }
    if (!usable(c) || !persistence) say("unavailable");
    else if (problem) say(problem);
  }
  function open() {
    opened = true; clearPending(); identity = token(current());
    load(); render(); say(problem ?? (entries.length ? "title" : "empty")); update();
  }
  doc.defaultView?.addEventListener("storage", (event) => {
    if (event.key !== BOOKMARK_KEY && event.key !== null) return;
    invalidate();
    if (opened) { load(); render(); say(problem ?? (entries.length ? "title" : "empty")); }
    update();
  });
  render(); identity = token(current()); update();
  return { open, update, invalidate, cancel };
}
