// Local starter picker. Placement/raycasting and request ownership stay with the viewer.
import { t, tOr } from "./i18n.js";

const paths = {
  rover: "M9 15h30v14H9z M9 19H5v14h7 M39 19h5v14h-7 M17 15V9h14v6 M8 34h7m18 0h7",
  quadruped: "M12 13h25v13H12z M14 26l-6 10h9 M33 26l7 10h-9 M16 26l4 10m10-10-4 10 M37 15l7 3v7h-7",
  arm: "M8 37h30 M17 37v-9l9-11-7-9 M14 7h9v7h-9z M23 16h7v7h-7z M28 22l8 4v7m0-7 6 1v6",
  humanoid: "M20 4h10v10H20z M18 17h14v13H18z M18 19 10 29m22-10 8 10 M21 30l-4 11h-5 M29 30l4 11h5",
};

export function initCreation({ start, submit, cancel }) {
  const $ = (id) => document.getElementById(id);
  let template = null, point = null, sending = false;
  const setOpen = (open) => {
    $("create-options").hidden = !open;
    $("create-toggle").setAttribute("aria-expanded", String(open));
  };
  $("create-toggle").addEventListener("click", () => setOpen($("create-options").hidden));
  $("place-here").addEventListener("click", () => {
    if (point && !sending) submit(template, point);
  });
  $("place-cancel").addEventListener("click", () => { if (!sending) cancel(); });
  return {
    active: () => template !== null,
    open() { setOpen(true); $("create-list").querySelector("button")?.focus(); },
    render(entries, editable, busy) {
      $("create-section").hidden = !editable || !entries.length;
      $("create-list").replaceChildren(...entries.map((entry) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "create-card";
        button.dataset.template = entry.name;
        button.disabled = busy;
        const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        icon.setAttribute("viewBox", "0 0 50 46");
        icon.setAttribute("aria-hidden", "true");
        const path = document.createElementNS(icon.namespaceURI, "path");
        path.setAttribute("d", paths[entry.name] || paths.rover); icon.append(path);
        const copy = document.createElement("span"), title = document.createElement("strong"), desc = document.createElement("span");
        title.textContent = tOr(`starter_${entry.name}`, entry.name);
        desc.textContent = tOr(`starter_${entry.name}_desc`, entry.summary);
        copy.append(title, desc); button.append(icon, copy);
        button.addEventListener("click", () => {
          template = entry.name; point = null; sending = false;
          $("placement-title").textContent = t("place_robot", { name: title.textContent });
          $("placement-bar").hidden = false;
          $("place-here").disabled = true;
          start(entry.name);
        });
        return button;
      }));
    },
    point(value) { point = value; $("place-here").disabled = !value || sending; },
    busy(value) {
      sending = value;
      $("place-here").disabled = !point || value;
      $("place-cancel").disabled = value;
      $("placement-bar").setAttribute("aria-busy", String(value));
    },
    clear() {
      template = point = null;
      sending = false;
      $("placement-bar").hidden = true;
      $("place-cancel").disabled = false;
    },
    place() { if (point && !sending) submit(template, point); },
  };
}
