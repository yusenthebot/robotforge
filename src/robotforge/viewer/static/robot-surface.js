// Explicit part styling uses the current world's server-owned body/geometry identities.
import { t } from "./i18n.js";

export function initRobotSurface({ current, ready, apply }) {
  const $ = (id) => document.getElementById(id);
  let key = null, robot = null, selected = new Set(), listKey = null;
  const multiple = $("robot-surface-multiple"), list = $("robot-surface-part-list"), material = $("robot-surface-material");
  function part() {
    return current()?.parts.find((p) => p.body === $("robot-surface-body").value) || null;
  }
  function finish() {
    let value = part()?.surface_material || "";
    if (multiple.checked) {
      const values = new Set((current()?.parts || []).filter((p) => selected.has(p.body)).map((p) => p.surface_material || ""));
      value = values.size === 1 ? [...values][0] : "mixed";
    }
    let placeholder = material.querySelector('option[value="mixed"]');
    if (value === "mixed" && !placeholder) {
      placeholder = document.createElement("option"); placeholder.value = "mixed";
      placeholder.disabled = true; placeholder.hidden = true; placeholder.textContent = t("surface_choose");
      material.prepend(placeholder);
    } else if (value !== "mixed") placeholder?.remove();
    material.value = value;
  }
  function batchControls(entry) {
    $("robot-surface-batch-controls").hidden = !multiple.checked;
    $("robot-surface-body").parentElement.hidden = multiple.checked;
    if (!multiple.checked) return;
    if (listKey !== key) {
      list.replaceChildren(...(entry?.parts || []).map((p) => {
        const label = document.createElement("label"), checkbox = document.createElement("input"), name = document.createElement("span");
        label.className = "check"; checkbox.type = "checkbox"; checkbox.value = p.body;
        const style = p.surface_material ? t(`material_${p.surface_material}`) : t("surface_original");
        name.textContent = `${p.body} · ${style}`;
        checkbox.addEventListener("change", () => {
          if (!ready()) { render(); return; }
          if (checkbox.checked && selected.size < 64) selected.add(p.body);
          else selected.delete(p.body);
          finish(); render();
        });
        label.append(checkbox, name); return label;
      }));
      listKey = key;
    }
    for (const checkbox of list.querySelectorAll("input")) {
      checkbox.checked = selected.has(checkbox.value);
      checkbox.disabled = !ready() || (!checkbox.checked && selected.size >= 64);
    }
    $("robot-surface-count").textContent = t("robot_surface_count", { count: selected.size });
    $("robot-surface-clear").disabled = !ready() || !selected.size;
  }
  function render() {
    const entry = current();
    $("robot-surface").hidden = !entry;
    if (!entry || robot !== entry.name) {
      $("robot-surface-controls").hidden = true;
      $("robot-surface").setAttribute("aria-expanded", "false");
    }
    robot = entry?.name || null;
    const next = JSON.stringify(entry);
    if (key !== next) {
      const previous = $("robot-surface-body").value;
      $("robot-surface-body").replaceChildren(...(entry?.parts || []).map((p) => {
        const option = document.createElement("option"); option.value = p.body;
        option.textContent = p.body; return option;
      }));
      if (entry?.parts.some((p) => p.body === previous)) $("robot-surface-body").value = previous;
      key = next; listKey = null; selected.clear(); multiple.checked = false;
      list.replaceChildren(); finish();
    }
    for (const id of ["robot-surface", "robot-surface-body", "robot-surface-material", "robot-surface-apply", "robot-surface-multiple"]) {
      $(id).disabled = !entry || !ready();
    }
    batchControls(entry);
    if (multiple.checked && (!selected.size || material.value === "mixed")) $("robot-surface-apply").disabled = true;
  }
  $("robot-surface").addEventListener("click", () => {
    if (!ready() || !current()) return;
    const open = $("robot-surface-controls").hidden;
    $("robot-surface-controls").hidden = !open;
    $("robot-surface").setAttribute("aria-expanded", String(open)); finish();
  });
  $("robot-surface-body").addEventListener("change", finish);
  multiple.addEventListener("change", () => {
    selected.clear();
    if (multiple.checked && part()) selected.add(part().body);
    finish(); render();
  });
  $("robot-surface-clear").addEventListener("click", () => {
    if (!ready()) return;
    selected.clear(); finish(); render();
  });
  material.addEventListener("change", () => {
    if (material.value !== "mixed") material.querySelector('option[value="mixed"]')?.remove();
    render();
  });
  $("robot-surface-apply").addEventListener("click", async () => {
    const entry = current(), chosen = part();
    if (!ready() || !entry || !chosen || (multiple.checked && (!selected.size || material.value === "mixed"))) return;
    const names = multiple.checked ? [...selected] : chosen.body;
    const result = await apply(entry, names, material.value || null, $("robot-surface-apply"));
    const now = current();
    // Only a successful acknowledgement for this exact displayed generation may restore a batch.
    if (Array.isArray(names) && result && ready() && now?.name === entry.name && now.version === result.version
        && now.generation === result.generation && now.world_sha256 === result.world_sha256) {
      selected = new Set(names.filter((body) => now.parts.some((p) => p.body === body)));
      multiple.checked = true; finish(); render();
    }
  });
  return { render, highlight: () => !$("drawer").hidden && !$("robot-surface-controls").hidden
    && current() && ready() ? multiple.checked
      ? [...new Set(current().parts.filter((p) => selected.has(p.body)).flatMap((p) => p.geoms))]
      : part()?.geoms || [] : [] };
}
