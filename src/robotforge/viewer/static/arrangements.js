// Persistent prop shelf. World operations and scene ownership remain with app.js.
import { t, tOr } from "./i18n.js";

export function createArrangementShelf({ context, request, prepare, selection, notice }) {
  const $ = (id) => document.getElementById(id);
  let catalog = null, epoch = 0, loading = false, pending = false, error = "", confirmation = null;
  let saving = false, saveSelection = null, saveEpoch = null, listKey = null;
  const guard = (value) => ({ version: value.version, generation: value.generation, world_sha256: value.world_sha256 });
  const stamp = () => {
    const c = context();
    return `${c.version}:${c.generation}:${c.world_sha256}:${c.session}:${c.selectionEpoch}:${c.pointerEpoch}`;
  };
  const ready = () => {
    const c = context();
    return c.ready && catalog?.available && catalog.version === c.version
      && catalog.generation === c.generation && catalog.world_sha256 === c.world_sha256;
  };
  const displayName = (entry) => entry.damaged && entry.name === "Damaged arrangement" ? t("arrangement_damaged_name") : entry.name;
  const visible = () => !$("world-panel").hidden && !$("world-body-objects").hidden;
  const node = (tag, text, className) => {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    if (className) n.className = className;
    return n;
  };
  function render() {
    const c = context(), blocked = c.busy || loading || pending;
    $("arrangement-refresh").disabled = blocked || !c.connected;
    $("arrangement-status").textContent = error || (pending ? t("arrangement_working") : loading ? t("arrangement_loading")
      : !c.supported ? t("arrangement_needs_world") : !ready() ? tOr(catalog?.reason, t("arrangement_refresh_hint"))
        : !c.paused ? t("arrangement_pause") : catalog.entries.length ? t("arrangement_count", { count: catalog.entries.length, limit: catalog.limit }) : t("arrangement_empty"));
    const nextListKey = JSON.stringify([catalog?.entries, ready(), blocked, c.paused]);
    if (nextListKey !== listKey) {
      listKey = nextListKey;
      $("arrangement-list").replaceChildren(...(catalog?.entries || []).map((entry) => {
      const row = node("div", undefined, "arrangement-row"); row.dataset.arrangement = entry.id;
      const label = node("span");
      label.append(node("strong", displayName(entry)), node("small", t(entry.damaged ? "arrangement_damaged" : "arrangement_members", { count: entry.count })));
      row.append(label);
      for (const action of ["insert", "remove"]) {
        const button = node("button", t(`arrangement_${action}`), "pill"); button.type = "button";
        button.dataset.arrangementAction = action;
        button.setAttribute("aria-label", t(`arrangement_${action}_named`, { name: displayName(entry) }));
        button.disabled = !ready() || blocked || (action === "insert" && (!c.paused || entry.damaged)) || !/^[0-9a-f]{64}$/.test(entry.sha256 || "");
        button.addEventListener("click", () => action === "insert" ? insert(entry) : confirm(entry));
        row.append(button);
      }
      return row;
      }));
    }
    $("arrangement-confirm").hidden = !confirmation;
    if (confirmation) $("arrangement-confirm-text").textContent = t("arrangement_remove_confirm", { name: displayName(confirmation) });
    $("arrangement-remove-apply").disabled = blocked || !ready();
    $("arrangement-remove-cancel").disabled = pending;
    const selected = selection();
    if (saving && (saveEpoch !== c.selectionEpoch || JSON.stringify(selected.names) !== JSON.stringify(saveSelection))) {
      saving = false; saveSelection = null; saveEpoch = null;
      $("arrangement-name").value = "";
      $("arrangement-save-error").textContent = "";
    }
    $("arrangement-save-open").hidden = !selected.group;
    $("arrangement-save-open").disabled = c.busy || pending || !selected.ready;
    $("arrangement-save-form").hidden = !saving;
    $("arrangement-name").disabled = pending;
    $("arrangement-save").disabled = pending || c.busy || !selected.ready;
    $("arrangement-save-cancel").disabled = pending;
  }
  function invalidate({ clear = false } = {}) {
    epoch++; loading = pending = false; error = ""; confirmation = null; saving = false; saveSelection = null; saveEpoch = null;
    if (clear) catalog = null;
    render();
  }
  async function refresh() {
    if (!context().connected || pending) return;
    const serial = ++epoch, current = stamp();
    loading = true; error = ""; confirmation = null; render();
    try {
      const data = await request({ type: "world", action: "arrangement_list" });
      if (serial !== epoch || current !== stamp()) return;
      if (!Array.isArray(data.entries) || !Number.isInteger(data.limit)) throw new Error(t("arrangement_invalid_reply"));
      catalog = data;
    } catch (cause) { if (serial === epoch && current === stamp()) error = cause.message; }
    finally { if (serial === epoch) { loading = false; render(); } }
  }
  async function operation(action, payload, done) {
    const c = context();
    if (pending || c.busy || !c.ready) return;
    const serial = ++epoch, current = stamp();
    loading = false; pending = true; error = ""; $("arrangement-save-error").textContent = ""; render();
    try {
      const data = await request({ type: "world", action: `arrangement_${action}`, ...payload });
      if (serial !== epoch || current !== stamp()) return;
      const now = context();
      if (action === "prepare" && (!now.ready || now.busy || !now.paused || !visible())) return;
      done(data);
    } catch (cause) {
      if (serial !== epoch || current !== stamp()) return;
      error = cause.message;
      if (action === "save") $("arrangement-save-error").textContent = error;
      if (action === "prepare") {
        // A fresh explicit retry must obtain a fresh catalogue identity.
        catalog = null;
      }
    } finally {
      if (serial === epoch) { pending = false; render(); }
    }
  }
  function confirm(entry) { confirmation = entry; render(); $("arrangement-remove-cancel").focus(); }
  function cancelConfirmation() {
    const id = confirmation?.id; confirmation = null; render();
    [...$("arrangement-list").children].find((row) => row.dataset.arrangement === id)?.querySelector('[data-arrangement-action="remove"]')?.focus();
  }
  function insert(entry) {
    if (!ready() || entry.damaged || !context().paused) return;
    operation("prepare", { ...guard(catalog), arrangement_id: entry.id, sha256: entry.sha256 }, (data) => {
      if (data.mode !== "arrangement" || data.arrangement_id !== entry.id || data.sha256 !== entry.sha256
          || data.version !== catalog.version || data.generation !== catalog.generation || data.world_sha256 !== catalog.world_sha256
          || !/^[0-9a-f]{32}$/.test(data.token || "") || !Array.isArray(data.members) || data.members.length !== entry.count
          || data.members.length < 2 || data.members.length > 16 || new Set(data.members.map((member) => member.name)).size !== data.members.length
          || data.members.some((member) => typeof member.name !== "string" || !["box", "sphere", "cylinder", "table", "bin"].includes(member.shape)
            || !Array.isArray(member.size) || member.size.length !== (member.shape === "sphere" ? 1 : member.shape === "cylinder" ? 2 : 3)
            || !member.size.every((v) => Number.isFinite(v) && v > 0 && v <= 10000)
            || !Array.isArray(member.offset) || member.offset.length !== 3 || !member.offset.every(Number.isFinite) || !Number.isFinite(member.yaw))) {
        throw new Error(t("arrangement_invalid_reply"));
      }
      pending = false; prepare(data);
    });
  }
  $("arrangement-refresh").addEventListener("click", refresh);
  $("arrangement-save-open").addEventListener("click", () => {
    if (!selection().ready || pending) return;
    saving = true; saveSelection = selection().names.slice(); saveEpoch = context().selectionEpoch; error = "";
    $("arrangement-save-error").textContent = ""; $("arrangement-name").value = ""; render(); $("arrangement-name").focus();
  });
  function cancelSave() { saving = false; saveSelection = null; saveEpoch = null; render(); $("arrangement-save-open").focus(); }
  $("arrangement-save-cancel").addEventListener("click", cancelSave);
  $("arrangement-save-form").addEventListener("keydown", (event) => {
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); if (!pending) cancelSave(); }
  });
  $("arrangement-confirm").addEventListener("keydown", (event) => {
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); if (!pending) cancelConfirmation(); }
  });
  $("arrangement-save-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const s = selection(), name = $("arrangement-name").value.trim();
    if (!s.ready || !saveSelection || saveEpoch !== context().selectionEpoch || JSON.stringify(s.names) !== JSON.stringify(saveSelection)) {
      saving = false; saveSelection = null; saveEpoch = null; render();
      notice(t("arrangement_selection_changed")); return;
    }
    if (!name || name.length > 64 || /[\u0000-\u001f\u007f]/.test(name)) {
      $("arrangement-save-error").textContent = t("arrangement_invalid_name"); $("arrangement-name").focus(); return;
    }
    operation("save", { ...guard(context()), names: s.names, name }, (data) => {
      if (!Array.isArray(data.entries) || !data.saved) throw new Error(t("arrangement_invalid_reply"));
      catalog = data; saving = false; saveSelection = null; notice(t("arrangement_saved", { name: data.saved.name }));
      $("arrangement-save-open").focus();
    });
  });
  $("arrangement-remove-cancel").addEventListener("click", cancelConfirmation);
  $("arrangement-remove-apply").addEventListener("click", () => {
    const entry = confirmation;
    if (!entry || !ready()) return;
    operation("remove", { ...guard(catalog), arrangement_id: entry.id, sha256: entry.sha256 }, (data) => {
      if (!Array.isArray(data.entries)) throw new Error(t("arrangement_invalid_reply"));
      catalog = data; confirmation = null; notice(t("arrangement_removed", { name: displayName(entry) })); $("arrangement-refresh").focus();
    });
  });
  return { render, invalidate, refresh, opened() { if (visible()) refresh(); } };
}
