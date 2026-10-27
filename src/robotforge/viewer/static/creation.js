// Local starter picker. Placement/raycasting and request ownership stay with the viewer.
import { t, tOr } from "./i18n.js";

const paths = {
  rover: "M9 15h30v14H9z M9 19H5v14h7 M39 19h5v14h-7 M17 15V9h14v6 M8 34h7m18 0h7",
  quadruped: "M12 13h25v13H12z M14 26l-6 10h9 M33 26l7 10h-9 M16 26l4 10m10-10-4 10 M37 15l7 3v7h-7",
  arm: "M8 37h30 M17 37v-9l9-11-7-9 M14 7h9v7h-9z M23 16h7v7h-7z M28 22l8 4v7m0-7 6 1v6",
  humanoid: "M20 4h10v10H20z M18 17h14v13H18z M18 19 10 29m22-10 8 10 M21 30l-4 11h-5 M29 30l4 11h5",
  ball: "M25 7a16 16 0 1 0 0 32 16 16 0 0 0 0-32 M12 13l26 20 M25 7c-9 10-9 22 0 32",
  block: "M9 15l16-8 16 8v20l-16 8-16-8z M9 15l16 8 16-8 M25 23v20",
  pin: "M17 11c0-5 16-5 16 0v24c0 6-16 6-16 0z M17 11c0 5 16 5 16 0",
  barrier: "M5 19l32-8 8 5v15l-32 8-8-5z M5 19l8 5 32-8 M13 24v15",
  table: "M6 15l26-7 13 8-27 8z M8 18v19m10-13v19m24-24v17 M32 10v17",
  ramp: "M4 34l15-20h12l15 20H4 M19 14v20m12-20v20 M4 34l7 5h29l6-5",
  stairs: "M4 34h6v-7h6v-7h7v-6h8v6h7v7h6v7H4 M4 34l6 6h29l5-6",
};

// Observe pointers without taking them away from OrbitControls. Once a gesture becomes a drag,
// multi-touch or canceled, lifting one finger must never turn the remaining finger into a tap.
export function createPlacementGesture() {
  const pointers = new Set();
  let candidate = null, epoch = 0;
  const modified = (e) => e.ctrlKey || e.shiftKey || e.altKey || e.metaKey;
  const moved = (e) => candidate && Math.hypot(e.clientX - candidate.x, e.clientY - candidate.y) > 8;
  return {
    active: () => pointers.size > 0,
    epoch: () => epoch,
    invalidate() { candidate = null; epoch++; },
    reset() { pointers.clear(); candidate = null; epoch++; },
    down(e, context) {
      epoch++;
      if (pointers.has(e.pointerId)) { candidate = null; return; }
      pointers.add(e.pointerId);
      candidate = pointers.size === 1 && context && e.isPrimary && e.button === 0 && e.buttons === 1 && !modified(e)
        ? { id: e.pointerId, x: e.clientX, y: e.clientY, context } : null;
    },
    move(e) {
      if (pointers.has(e.pointerId) && (moved(e) || modified(e) || e.buttons !== 1)) candidate = null;
    },
    up(e, canceled = false) {
      if (!pointers.has(e.pointerId)) return null;
      const result = !canceled && e.button === 0 && pointers.size === 1 && candidate?.id === e.pointerId
        && !moved(e) && !modified(e) ? candidate.context : null;
      pointers.delete(e.pointerId);
      candidate = null;
      return result;
    },
  };
}

export function initCreation({ start, submit, cancel, change, coordinateChange = () => {} }) {
  const $ = (id) => document.getElementById(id);
  let item = null, point = null, sending = false, size = "normal", yaw = 0, turns = 0, session = 0;
  const repeat = $("place-repeat");
  const material = $("place-material");
  const spacing = $("place-spacing");
  spacing.addEventListener("change", coordinateChange);
  function coordinates(value, status = null) {
    const readout = $("placement-position"), note = $("placement-coordinate-status");
    const text = value ? t("placement_position", { x: value[0].toFixed(4), y: value[1].toFixed(4), z: value[2].toFixed(4) })
      : t("placement_position_empty");
    if (readout.textContent !== text) readout.textContent = text;
    const message = status ? t(status) : "";
    if (note.textContent !== message) note.textContent = message;
    note.hidden = !message;
  }
  const syncRepeat = () => { $("place-cancel").textContent = t(repeat.checked ? "done_placing" : "cancel_place"); };
  repeat.addEventListener("change", syncRepeat);
  const scales = { small: 0.75, normal: 1, large: 1.5 };
  const send = () => {
    if (point && !sending && item) submit({ ...item, scale: size, yaw,
      ...(item.kind === "terrain-edit" ? { turns } : {}),
      ...(item.kind === "object" && item.name !== "table" && material.value ? { material: material.value } : {}) }, point);
  };
  const setOpen = (open) => {
    $("create-options").hidden = !open;
    $("create-toggle").setAttribute("aria-expanded", String(open));
  };
  $("create-toggle").addEventListener("click", () => setOpen($("create-options").hidden));
  $("place-here").addEventListener("click", send);
  $("place-cancel").addEventListener("click", () => { if (!sending) cancel(); });
  const sizeButtons = $("object-placement-controls").querySelectorAll("[data-size]");
  function setSize(value) {
    size = value;
    for (const b of sizeButtons) b.setAttribute("aria-pressed", String(b.dataset.size === size));
  }
  for (const b of sizeButtons) b.addEventListener("click", () => { setSize(b.dataset.size); change(); });
  $("place-rotate").addEventListener("click", () => {
    if (item?.kind === "terrain-edit") { turns = (turns + 1) % 4; yaw = item.yaw + 90 * turns; }
    else yaw = (yaw + 90) % 360;
    $("place-rotate").textContent = t("rotate_place", { angle: yaw }); change();
  });
  function begin(entry, kind, title) {
    item = { ...entry, kind }; point = null; sending = false; size = "normal";
    spacing.value = "free"; coordinates(null);
    // Authored worlds may contain many full turns; copy the same heading within the request's angle bounds.
    turns = 0;
    yaw = kind === "edit" || kind === "terrain-edit" ? entry.yaw : kind === "copy" || kind === "start" ? entry.yaw % 360 : 0; session++;
    repeat.checked = false; syncRepeat();
    material.value = "";
    $("prop-material-control").hidden = kind !== "object" || entry.name === "table";
    $("placement-hint").textContent = t(kind === "terrain-edit" ? "terrain_edit_hint" : kind === "terrain" ? "terrain_place_hint" : kind === "model" ? "model_place_hint" : kind === "start" ? "robot_start_hint" : kind === "edit" ? "edit_place_hint" : kind === "copy" ? "copy_place_hint" : "place_hint");
    setSize(size);
    $("place-rotate").textContent = t("rotate_place", { angle: yaw });
    $("object-placement-controls").hidden = kind === "robot" || kind === "model";
    for (const node of document.querySelectorAll(".prop-size-control")) node.hidden = kind === "edit" || kind === "copy" || kind === "start" || kind === "terrain" || kind === "terrain-edit";
    $("placement-bar").querySelector(".repeat-placement").hidden = kind === "edit" || kind === "start" || kind === "model" || kind === "terrain" || kind === "terrain-edit";
    $("placement-title").textContent = t(kind === "terrain-edit" ? "move_named_terrain" : kind === "start" ? "robot_start_named" : kind === "edit" ? "move_named_prop" : kind === "copy" ? "copy_named_prop" : "place_robot", { name: title });
    $("placement-bar").hidden = false;
    $("place-here").disabled = true;
    start();
  }
  function card(entry, kind, busy) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = kind === "robot" || kind === "model" ? "create-card" : "object-card";
    button.dataset[kind === "terrain" ? "terrain" : kind === "model" ? "model" : kind === "robot" ? "template" : "object"] = entry.name;
    if (kind === "model") button.dataset.source = entry.source;
    button.disabled = busy;
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("viewBox", "0 0 50 46"); icon.setAttribute("aria-hidden", "true");
    const path = document.createElementNS(icon.namespaceURI, "path");
    path.setAttribute("d", paths[entry.name] || paths[entry.kind === "quadruped" ? "quadruped" : "arm"] || paths.block); icon.append(path);
    const copy = document.createElement("span"), title = document.createElement("strong"), desc = document.createElement("span");
    const key = kind === "terrain" ? "local_terrain" : kind === "robot" ? "starter" : "prop";
    title.textContent = kind === "model" ? entry.title : tOr(`${key}_${entry.name}`, entry.name);
    desc.textContent = kind === "model" ? t(entry.local ? "model_local" : entry.downloaded ? "model_cached" : "model_download", { size: entry.download_mb }) : tOr(`${key}_${entry.name}_desc`, entry.summary || "");
    copy.append(title, desc); button.append(icon, copy);
    button.addEventListener("click", () => begin(entry, kind, title.textContent));
    return button;
  }
  return {
    active: () => item !== null,
    editing: () => item?.kind === "edit" || item?.kind === "start" || item?.kind === "terrain-edit" ? item : null,
    edit: (entry) => begin(entry, "edit", entry.name),
    editTerrain: (entry) => begin(entry, "terrain-edit", entry.name),
    repositionRobot: (entry) => begin(entry, "start", entry.name),
    heading: () => item?.kind === "start" || item?.kind === "terrain" || item?.kind === "terrain-edit" ? yaw : null,
    placingTerrain: () => item?.kind === "terrain" || item?.kind === "terrain-edit",
    duplicate: (entry) => begin(entry, "copy", entry.name),
    session: () => session,
    spacing: () => spacing.value,
    repeating: () => (item?.kind === "object" || item?.kind === "copy") && repeat.checked,
    next(guard = {}) {
      item = { ...item, ...guard };
      point = null; coordinates(null); $("place-here").disabled = true;
      $("placement-hint").textContent = t("place_next_hint");
    },
    preview: () => item?.kind === "terrain" || item?.kind === "terrain-edit" ? { boxes: item.boxes, yaw }
      : item && item.kind !== "robot" && item.kind !== "start" && item.kind !== "model" ? { shape: item.shape, size: item.size.map((v) => v * scales[size]), yaw } : null,
    open() { setOpen(true); $("model-list").querySelector("button")?.focus(); },
    render(entries, editable, busy, objects = [], models = [], terrains = []) {
      $("create-section").hidden = !editable || !entries.length;
      $("create-list").replaceChildren(...entries.map((entry) => card(entry, "robot", busy)));
      $("model-list").replaceChildren(...models.map((entry) => card(entry, "model", busy)));
      $("object-list").replaceChildren(...(editable ? objects : []).map((entry) => card(entry, "object", busy)));
      $("terrain-create-list").replaceChildren(...(editable ? terrains : []).map((entry) => card(entry, "terrain", busy)));
      $("local-terrain-section").hidden = !editable;
      $("objects-note").hidden = editable;
      $("build-pick-map").hidden = editable;
      $("build-add-robot").hidden = !editable || !entries.length;
      $("build-add-robot").dataset.off = editable ? "0" : "1";
      $("build-add-robot").disabled = busy || !editable;
    },
    point(value, status = null) { point = value; coordinates(value, status); $("place-here").disabled = !value || sending; },
    busy(value) {
      sending = value;
      $("place-here").disabled = !point || value;
      $("place-cancel").disabled = value;
      for (const b of sizeButtons) b.disabled = value;
      $("place-rotate").disabled = value;
      repeat.disabled = value;
      material.disabled = value;
      spacing.disabled = value;
      $("placement-bar").setAttribute("aria-busy", String(value));
    },
    progress(stage, fraction) {
      if (!sending || item?.kind !== "model") return;
      const percent = Number.isFinite(fraction) ? ` ${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%` : "";
      $("placement-hint").textContent = tOr(`model_stage_${stage}`, t("model_stage_prepare")) + percent;
    },
    clear() {
      session++; item = point = null;
      sending = false;
      coordinates(null);
      $("placement-bar").hidden = true;
      $("place-cancel").disabled = false;
    },
    place: send,
  };
}
