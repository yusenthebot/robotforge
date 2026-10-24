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

export function initCreation({ start, submit, cancel, change }) {
  const $ = (id) => document.getElementById(id);
  let item = null, point = null, sending = false, size = "normal", yaw = 0, session = 0;
  const repeat = $("place-repeat");
  const syncRepeat = () => { $("place-cancel").textContent = t(repeat.checked ? "done_placing" : "cancel_place"); };
  repeat.addEventListener("change", syncRepeat);
  const scales = { small: 0.75, normal: 1, large: 1.5 };
  const send = () => {
    if (point && !sending && item) submit({ ...item, scale: size, yaw }, point);
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
    yaw = (yaw + 90) % 360;
    $("place-rotate").textContent = t("rotate_place", { angle: yaw }); change();
  });
  function begin(entry, kind, title) {
    item = { ...entry, kind }; point = null; sending = false; size = "normal";
    // Authored worlds may contain many full turns; copy the same heading within the request's angle bounds.
    yaw = kind === "edit" ? entry.yaw : kind === "copy" || kind === "start" ? entry.yaw % 360 : 0; session++;
    repeat.checked = false; syncRepeat();
    $("placement-hint").textContent = t(kind === "start" ? "robot_start_hint" : kind === "edit" ? "edit_place_hint" : kind === "copy" ? "copy_place_hint" : "place_hint");
    setSize(size);
    $("place-rotate").textContent = t("rotate_place", { angle: yaw });
    $("object-placement-controls").hidden = kind === "robot";
    for (const node of document.querySelectorAll(".prop-size-control")) node.hidden = kind === "edit" || kind === "copy" || kind === "start";
    $("placement-bar").querySelector(".repeat-placement").hidden = kind === "edit" || kind === "start";
    $("placement-title").textContent = t(kind === "start" ? "robot_start_named" : kind === "edit" ? "move_named_prop" : kind === "copy" ? "copy_named_prop" : "place_robot", { name: title });
    $("placement-bar").hidden = false;
    $("place-here").disabled = true;
    start();
  }
  function card(entry, kind, busy) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = kind === "robot" ? "create-card" : "object-card";
    button.dataset[kind === "robot" ? "template" : "object"] = entry.name;
    button.disabled = busy;
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("viewBox", "0 0 50 46"); icon.setAttribute("aria-hidden", "true");
    const path = document.createElementNS(icon.namespaceURI, "path");
    path.setAttribute("d", paths[entry.name] || paths.block); icon.append(path);
    const copy = document.createElement("span"), title = document.createElement("strong"), desc = document.createElement("span");
    const key = kind === "robot" ? "starter" : "prop";
    title.textContent = tOr(`${key}_${entry.name}`, entry.name);
    desc.textContent = tOr(`${key}_${entry.name}_desc`, entry.summary || "");
    copy.append(title, desc); button.append(icon, copy);
    button.addEventListener("click", () => begin(entry, kind, title.textContent));
    return button;
  }
  return {
    active: () => item !== null,
    editing: () => item?.kind === "edit" || item?.kind === "start" ? item : null,
    edit: (entry) => begin(entry, "edit", entry.name),
    repositionRobot: (entry) => begin(entry, "start", entry.name),
    heading: () => item?.kind === "start" ? yaw : null,
    duplicate: (entry) => begin(entry, "copy", entry.name),
    session: () => session,
    repeating: () => (item?.kind === "object" || item?.kind === "copy") && repeat.checked,
    next(guard = {}) {
      item = { ...item, ...guard };
      point = null; $("place-here").disabled = true;
      $("placement-hint").textContent = t("place_next_hint");
    },
    preview: () => item && item.kind !== "robot" && item.kind !== "start" ? { shape: item.shape, size: item.size.map((v) => v * scales[size]), yaw } : null,
    open() { setOpen(true); $("create-list").querySelector("button")?.focus(); },
    render(entries, editable, busy, objects = []) {
      $("create-section").hidden = !editable || !entries.length;
      $("create-list").replaceChildren(...entries.map((entry) => card(entry, "robot", busy)));
      $("object-list").replaceChildren(...(editable ? objects : []).map((entry) => card(entry, "object", busy)));
      $("objects-note").hidden = editable;
      $("build-pick-map").hidden = editable;
      $("build-add-robot").hidden = !editable || !entries.length;
      $("build-add-robot").dataset.off = editable ? "0" : "1";
      $("build-add-robot").disabled = busy || !editable;
    },
    point(value) { point = value; $("place-here").disabled = !value || sending; },
    busy(value) {
      sending = value;
      $("place-here").disabled = !point || value;
      $("place-cancel").disabled = value;
      for (const b of sizeButtons) b.disabled = value;
      $("place-rotate").disabled = value;
      repeat.disabled = value;
      $("placement-bar").setAttribute("aria-busy", String(value));
    },
    clear() {
      session++; item = point = null;
      sending = false;
      $("placement-bar").hidden = true;
      $("place-cancel").disabled = false;
    },
    place: send,
  };
}
