// Explicit part styling uses the current world's server-owned body/geometry identities.
export function initRobotSurface({ current, ready, apply }) {
  const $ = (id) => document.getElementById(id);
  let key = null, robot = null;
  function part() {
    return current()?.parts.find((p) => p.body === $("robot-surface-body").value) || null;
  }
  function finish() {
    $("robot-surface-material").value = part()?.surface_material || "";
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
      key = next; finish();
    }
    for (const id of ["robot-surface", "robot-surface-body", "robot-surface-material", "robot-surface-apply"]) {
      $(id).disabled = !entry || !ready();
    }
  }
  $("robot-surface").addEventListener("click", () => {
    if (!ready() || !current()) return;
    const open = $("robot-surface-controls").hidden;
    $("robot-surface-controls").hidden = !open;
    $("robot-surface").setAttribute("aria-expanded", String(open)); finish();
  });
  $("robot-surface-body").addEventListener("change", finish);
  $("robot-surface-apply").addEventListener("click", () => {
    const entry = current(), chosen = part();
    if (ready() && entry && chosen) apply(entry, chosen.body, $("robot-surface-material").value || null,
      $("robot-surface-apply"));
  });
  return { render, highlight: () => !$("drawer").hidden && !$("robot-surface-controls").hidden
    && current() && ready() ? part()?.geoms || [] : [] };
}
