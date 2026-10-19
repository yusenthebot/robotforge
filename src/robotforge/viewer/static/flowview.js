// The behavior tree a script runs, as a panel above the dock: one row per node, indented under its parent, with a dot
// that is amber and pulsing while the node runs, green when it succeeded, red when it failed and grey before it ran.
//
// The server sends a "flow" message whenever the script says the tree moved (viewer/server.py, flow_board.py):
// {"name", "nodes": [{"id", "parent", "kind", "name"}], "status" (a letter per node: . r s f), "notes" (by id)}.
// No nodes takes the panel away. The header folds the tree down to the node that runs now.

import { escapeHtml, t } from "./i18n.js";

const $ = (id) => document.getElementById(id);

// What a node's kind looks like in front of its name (the usual behavior tree symbols).
const GLYPH = {
  sequence: "→",
  fallback: "?",
  parallel: "⇉",
  retry: "↻",
  repeat: "↻",
  inverter: "!",
  timeout: "⧖",
  condition: "◇",
  wait: "⧖",
  machine: "▣",
  state: "○",
};
const WORD = { r: "running", s: "done", f: "failed", ".": "" };

let api = null;
let tree = null; // the last message
let drawnKey = ""; // the structure on screen: rows are rebuilt only when it changes
let folded = false;

export function initFlow(context) {
  api = context;
  folded = api.storageGet("robotforge.flow-folded") === "1";
  $("flow-head").addEventListener("click", () => {
    folded = !folded;
    api.storageSet("robotforge.flow-folded", folded ? "1" : "0");
    render();
  });
}

export function flowUpdate(msg) {
  tree = msg.nodes && msg.nodes.length ? msg : null;
  render();
}

// The tree as scripted checks read it: its name and, per node, name and state.
export function flowState() {
  if (!tree) return null;
  return {
    name: tree.name,
    shown: !$("flow-panel").hidden,
    nodes: tree.nodes.map((n, i) => ({ name: n.name, kind: n.kind, status: tree.status[i], note: tree.notes[i] ?? "" })),
  };
}

function render() {
  const panel = $("flow-panel");
  if (!tree || api.inLesson()) {
    panel.hidden = true;
    drawnKey = "";
    return;
  }
  panel.hidden = false;
  panel.classList.toggle("folded", folded);
  $("flow-title").textContent = tree.name || t("flow_title");
  const key = JSON.stringify(tree.nodes);
  if (key !== drawnKey) {
    drawnKey = key;
    const depth = [];
    $("flow-rows").innerHTML = tree.nodes
      .map((n, i) => {
        depth[i] = n.parent < 0 ? 0 : depth[n.parent] + 1;
        return (
          `<li class="flow-node" data-id="${i}" data-kind="${escapeHtml(n.kind)}" style="--depth:${depth[i]}">` +
          `<span class="flow-dot" aria-hidden="true"></span>` +
          `<span class="flow-glyph" aria-hidden="true">${GLYPH[n.kind] ?? "▪"}</span>` +
          `<span class="flow-name">${escapeHtml(n.name)}</span><span class="flow-note"></span></li>`
        );
      })
      .join("");
  }
  const rows = $("flow-rows").children;
  let running = "";
  tree.nodes.forEach((n, i) => {
    const row = rows[i];
    const s = tree.status[i];
    if (row.dataset.s !== s) {
      row.dataset.s = s;
      row.setAttribute("aria-label", `${n.name} ${WORD[s] ?? ""}`.trim());
    }
    row.querySelector(".flow-note").textContent = tree.notes[i] ?? "";
    // The node that runs now is the deepest running one: only its row is lit, the running ones above it just have the dot.
    const leaf = s === "r" && !tree.nodes.some((m, j) => m.parent === i && tree.status[j] === "r");
    row.classList.toggle("now", leaf);
    if (leaf) running = n.name;
  });
  const ended = tree.status.length && tree.status[0] !== "r" && tree.status[0] !== ".";
  const state = ended ? (tree.status[0] === "s" ? "done" : "failed") : "running";
  $("flow-state").textContent = ended ? t(`flow_${state}`) : running;
  panel.dataset.state = state;
}
