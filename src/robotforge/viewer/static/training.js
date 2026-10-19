// The training view: a reinforcement learning or search run, live. The server sends a "training" message about twice
// a second while a run changes (viewer/server.py, training/live.py list its fields): its state and progress, and one
// entry per trial with the algorithm, its settings, its learning curve, its episode returns, its losses and its test
// score. This panel draws the curves of all trials on one chart (the leader bold, pruned trials faded), the leader's
// episode returns and a loss, a table that sets every trial's settings side by side, and the buttons that stop the run
// or play the best policy in the world.
//
// No imports of the viewer's internals: app.js hands over what it needs (initTraining), so Node can parse this file.

import { formatTick, niceTicks } from "./plot.js";

const $ = (id) => document.getElementById(id);

const TEXT = {
  en: {
    title: "Training",
    button: "Training",
    state: { starting: "starting", running: "training", done: "done", stopped: "stopped", failed: "failed", testing: "testing" },
    reward: "Reward per episode (average of the latest episodes)",
    returns: "Episode returns, trial {id}",
    metric: "Loss and other numbers, trial {id}",
    none_metric: "this algorithm reports no losses",
    steps: "steps",
    leader: "leading",
    trial: "Trial",
    algo: "Algorithm",
    score: "Test return",
    success: "Success",
    progress: "{pct}% of the work",
    eta: "about {time} left",
    elapsed: "{time} so far",
    play_best: "Play the best policy",
    play: "Play",
    stop: "Stop",
    stopping: "Stopping",
    played: "{name}: trial {id} is playing",
    stopped_note: "Stopping: the best policy so far is kept",
    pruned: "dropped",
    queued: "waiting",
    failed: "failed",
    searching: "{method} search, {n} trials",
    trials: "{n} trials",
    waiting: "waiting for the first episodes",
    no_policy: "no policy to play yet",
  },
  zh: {
    title: "训练",
    button: "训练",
    state: { starting: "启动中", running: "训练中", done: "完成", stopped: "已停止", failed: "失败", testing: "测试中" },
    reward: "每局回报（最近几局的平均）",
    returns: "每局回报，试验 {id}",
    metric: "损失等数值，试验 {id}",
    none_metric: "这个算法没有损失可看",
    steps: "步数",
    leader: "领先",
    trial: "试验",
    algo: "算法",
    score: "测试回报",
    success: "成功率",
    progress: "完成 {pct}%",
    eta: "还需约 {time}",
    elapsed: "已用 {time}",
    play_best: "播放最好的策略",
    play: "播放",
    stop: "停止",
    stopping: "正在停止",
    played: "{name}：试验 {id} 正在播放",
    stopped_note: "正在停止：保留目前最好的策略",
    pruned: "已淘汰",
    queued: "等待",
    failed: "失败",
    searching: "{method} 搜索，{n} 个试验",
    trials: "{n} 个试验",
    waiting: "等待第一批回合",
    no_policy: "还没有可播放的策略",
  },
};

let api = null;
let run = null; // the latest "training" message
let selected = null; // trial id whose returns and losses are drawn
let metric = null; // the loss series shown
let userClosed = false;
let lastName = null;

const lang = () => (document.documentElement.lang || "en").startsWith("zh") ? "zh" : "en";
const tx = (key, vars = {}) => {
  const text = TEXT[lang()][key] ?? TEXT.en[key] ?? key;
  return String(text).replace(/\{(\w+)\}/g, (_, k) => String(vars[k] ?? ""));
};
const stateText = (s) => TEXT[lang()].state[s] ?? s;

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export function trainingColor(index, dark) {
  const hue = (index * 137.5 + 210) % 360; // far apart, whatever the number of trials
  return `hsl(${Math.round(hue)} ${dark ? 70 : 62}% ${dark ? 66 : 46}%)`;
}

const isDark = () => getComputedStyle(document.documentElement).colorScheme.includes("dark");

export function formatValue(v) {
  if (v === null || v === undefined) return "-";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "string") return v;
  if (!Number.isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 100000)) return v.toExponential(1);
  if (Number.isInteger(v)) return String(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toPrecision(2);
}

export function formatSteps(n) {
  if (!Number.isFinite(n)) return "-";
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e5 ? 0 : 1)}k`;
  return String(Math.round(n));
}

export function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "-";
  const s = Math.max(0, Math.round(seconds));
  return s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min` : s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`;
}

// Settings that differ between trials, in the order they first appear (the table's columns); when nothing differs,
// every setting of the first trial.
export function settingColumns(trials, limit = 5) {
  const seen = new Map();
  for (const t of trials) for (const [k, v] of Object.entries(t.settings || {})) {
    if (!seen.has(k)) seen.set(k, new Set());
    seen.get(k).add(JSON.stringify(v));
  }
  const varying = [...seen].filter(([, values]) => values.size > 1).map(([k]) => k);
  const all = [...seen.keys()];
  const chosen = varying.length ? varying : all;
  return chosen.slice(0, limit);
}

export function initTraining(context) {
  api = context;
  const toggle = $("training-toggle");
  const panel = $("training-view");
  if (!toggle || !panel) return;
  toggle.addEventListener("click", () => {
    const open = panel.hidden;
    panel.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    userClosed = !open;
  });
  panel.addEventListener("click", onClick);
  panel.addEventListener("change", (e) => {
    if (e.target.matches("select.metric")) {
      metric = e.target.value;
      draw();
    }
  });
  window.addEventListener("resize", () => draw());
}

export function trainingUpdate(msg) {
  const toggle = $("training-toggle");
  const panel = $("training-view");
  if (!toggle || !panel) return;
  run = msg;
  toggle.hidden = false;
  const dot = toggle.querySelector(".dot");
  if (dot) dot.dataset.state = msg.state;
  if (msg.name !== lastName) {
    lastName = msg.name;
    selected = null;
    metric = null;
    if (msg.state === "running" || msg.state === "starting") userClosed = false;
  }
  if (!userClosed && panel.hidden) {
    panel.hidden = false;
    toggle.setAttribute("aria-expanded", "true");
  }
  render();
}

export function trainingState() {
  return run ? { name: run.name, state: run.state, leader: run.leader, selected, trials: (run.trials || []).length } : null;
}

function onClick(e) {
  const button = e.target.closest("[data-act]");
  if (button) {
    const act = button.dataset.act;
    if (act === "play") play(button.dataset.trial || null);
    else if (act === "stop") stop();
    return;
  }
  const row = e.target.closest("tr[data-trial]");
  if (row) {
    selected = row.dataset.trial;
    metric = null;
    render();
  }
}

async function play(trial) {
  if (!run || !api) return;
  try {
    const data = await api.request({ type: "training_play", run: run.name, trial });
    api.toast(tx("played", { name: run.name, id: data?.trial ?? trial ?? run.leader }));
  } catch (err) {
    api.toast(err.message);
  }
}

async function stop() {
  if (!run || !api) return;
  try {
    await api.request({ type: "training_stop", run: run.name });
    api.toast(tx("stopped_note"));
  } catch (err) {
    api.toast(err.message);
  }
}

function leaderTrial() {
  return (run.trials || []).find((t) => t.id === run.leader) || (run.trials || [])[0] || null;
}

function selectedTrial() {
  const trials = run.trials || [];
  return trials.find((t) => t.id === selected) || leaderTrial();
}

function render() {
  const panel = $("training-view");
  if (!run || !panel) return;
  const trials = run.trials || [];
  const lead = leaderTrial();
  const live = run.state === "running" || run.state === "starting" || run.state === "testing";
  const pct = Math.round(100 * (run.progress ?? 0));
  const search = run.search || {};
  const subtitle = [
    run.task && run.robot ? `${run.task} · ${run.robot}` : "",
    search.method ? tx("searching", { method: search.method, n: trials.length }) : tx("trials", { n: trials.length }),
  ].filter(Boolean).join(" · ");
  const time = live && run.eta_s ? tx("eta", { time: formatTime(run.eta_s) }) : tx("elapsed", { time: formatTime(run.elapsed_s) });
  const playable = trials.some((t) => t.playable) || run.playable;
  const columns = settingColumns(trials);
  const dark = isDark();
  const sel = selectedTrial();
  const metrics = sel ? Object.keys(sel.metrics || {}) : [];
  if (metrics.length && !metrics.includes(metric)) metric = metrics.includes("loss") ? "loss" : metrics[0];
  const rows = trials
    .map((t, i) => {
      const score = t.score ? formatValue(t.score.mean_return) : formatValue(t.mean_return);
      const ok = t.score ? t.score.success_rate : t.success_rate;
      const state = t.state === "pruned" ? tx("pruned") : t.state === "queued" ? tx("queued") : t.state === "failed" ? tx("failed") : "";
      const cells = columns.map((c) => `<td>${esc(formatValue((t.settings || {})[c]))}</td>`).join("");
      const mark = t.id === run.leader ? `<span class="lead-badge">${esc(tx("leader"))}</span>` : "";
      return `<tr data-trial="${esc(t.id)}" class="${t.id === run.leader ? "lead" : ""} ${t.id === (sel && sel.id) ? "sel" : ""} ${t.state === "pruned" || t.state === "failed" ? "faded" : ""}">
        <td><span class="swatch" style="background:${trainingColor(i, dark)}"></span>${esc(t.id)} ${mark}</td>
        <td>${esc(t.algo)}</td>${cells}
        <td class="num">${esc(score)}</td><td class="num">${ok === null || ok === undefined ? "-" : Math.round(100 * ok) + "%"}</td>
        <td class="num">${esc(formatSteps(t.steps))}</td>
        <td class="state">${esc(state)}</td>
        <td><button type="button" class="mini" data-act="play" data-trial="${esc(t.id)}" ${t.playable ? "" : "disabled"} aria-label="${esc(tx("play"))} ${esc(t.id)}">&#9654;</button></td></tr>`;
    })
    .join("");
  const head = columns.map((c) => `<th>${esc(c.replace(/_/g, " "))}</th>`).join("");
  const metricPicker = metrics.length
    ? `<select class="metric" aria-label="metric">${metrics.map((m) => `<option value="${esc(m)}" ${m === metric ? "selected" : ""}>${esc(m.replace(/_/g, " "))}</option>`).join("")}</select>`
    : `<span class="hint">${esc(tx("none_metric"))}</span>`;
  panel.innerHTML = `
    <header class="tv-head">
      <div class="tv-title"><h2>${esc(tx("title"))}: ${esc(run.name)}</h2><span class="chip" data-state="${esc(run.state)}">${esc(stateText(run.state))}</span></div>
      <div class="tv-sub">${esc(subtitle)}</div>
      <div class="tv-bar" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><span style="width:${pct}%"></span></div>
      <div class="tv-meta"><span>${esc(tx("progress", { pct }))}</span><span>${esc(time)}</span>
        <span class="tv-actions">
          ${live ? `<button type="button" class="pill-button" data-act="stop">${esc(tx("stop"))}</button>` : ""}
          <button type="button" class="primary-button" data-act="play" ${playable ? "" : "disabled"} title="${playable ? "" : esc(tx("no_policy"))}">&#9654; ${esc(tx("play_best"))}</button>
        </span></div>
    </header>
    <section class="tv-chart"><h3>${esc(tx("reward"))}</h3><canvas id="tv-reward" height="170"></canvas><div class="tv-legend" id="tv-legend"></div></section>
    <div class="tv-two">
      <section class="tv-chart"><h3>${esc(tx("returns", { id: sel ? sel.id : "" }))}</h3><canvas id="tv-returns" height="110"></canvas></section>
      <section class="tv-chart"><h3>${esc(tx("metric", { id: sel ? sel.id : "" }))} ${metricPicker}</h3><canvas id="tv-metric" height="110"></canvas></section>
    </div>
    <div class="tv-table"><table>
      <thead><tr><th>${esc(tx("trial"))}</th><th>${esc(tx("algo"))}</th>${head}<th>${esc(tx("score"))}</th><th>${esc(tx("success"))}</th><th>${esc(tx("steps"))}</th><th></th><th></th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
  const legend = $("tv-legend");
  if (legend) {
    legend.innerHTML = trials
      .map((t, i) => `<span class="${t.id === run.leader ? "lead" : ""} ${t.state === "pruned" ? "faded" : ""}"><i style="background:${trainingColor(i, dark)}"></i>${esc(t.id)} ${esc(t.algo)}</span>`)
      .join("");
  }
  draw();
}

// ---- drawing ----

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

function prepare(canvas) {
  if (!canvas) return null;
  const ratio = window.devicePixelRatio || 1;
  const w = Math.max(120, Math.floor(canvas.parentElement.clientWidth));
  const h = Number(canvas.getAttribute("height")) || 120;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  canvas.width = Math.round(w * ratio);
  canvas.height = Math.round(h * ratio);
  const g = canvas.getContext("2d");
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { g, w, h };
}

function bounds(seriesList) {
  let x1 = 1, lo = Infinity, hi = -Infinity;
  for (const s of seriesList) for (const [x, y] of s.points) {
    if (!Number.isFinite(y)) continue;
    x1 = Math.max(x1, x);
    lo = Math.min(lo, y);
    hi = Math.max(hi, y);
  }
  if (!Number.isFinite(lo)) [lo, hi] = [-1, 1];
  if (hi - lo < 1e-9) [lo, hi] = [lo - 0.5, hi + 0.5];
  const pad = (hi - lo) * 0.08;
  return { x0: 0, x1, y0: lo - pad, y1: hi + pad };
}

export function plotSeries(c, seriesList, { dots = false, empty = "" } = {}) {
  const ctx = prepare(c);
  if (!ctx) return;
  const { g, w, h } = ctx;
  g.clearRect(0, 0, w, h);
  const left = 40, right = 8, top = 6, bottom = 18;
  g.font = `10px ${token("--mono")}`;
  const any = seriesList.some((s) => s.points.length > 0);
  if (!any) {
    g.fillStyle = token("--muted");
    g.fillText(empty, left, h / 2);
    return;
  }
  const b = bounds(seriesList);
  const x = (v) => left + (v / b.x1) * (w - left - right);
  const y = (v) => top + (1 - (v - b.y0) / (b.y1 - b.y0)) * (h - top - bottom);
  g.strokeStyle = token("--line");
  g.fillStyle = token("--muted");
  g.lineWidth = 1;
  const yt = niceTicks(b.y0, b.y1, h < 130 ? 2 : 4);
  for (const v of yt) {
    g.beginPath();
    g.moveTo(left, Math.round(y(v)) + 0.5);
    g.lineTo(w - right, Math.round(y(v)) + 0.5);
    g.stroke();
    g.fillText(formatTick(v, yt), 2, y(v) + 3);
  }
  const xt = niceTicks(0, b.x1, 4);
  for (const v of xt) g.fillText(formatSteps(v), Math.min(x(v), w - 28), h - 4);
  g.save();
  g.beginPath();
  g.rect(left, 0, w - left, h - bottom + 2);
  g.clip();
  for (const s of seriesList) {
    if (!s.points.length) continue;
    g.globalAlpha = s.alpha ?? 1;
    if (dots) {
      g.fillStyle = s.color;
      for (const [px, py] of s.points) {
        g.beginPath();
        g.arc(x(px), y(py), 1.8, 0, 2 * Math.PI);
        g.fill();
      }
      continue;
    }
    g.strokeStyle = s.color;
    g.lineWidth = s.width ?? 1.4;
    g.lineJoin = "round";
    g.beginPath();
    s.points.forEach(([px, py], i) => (i ? g.lineTo(x(px), y(py)) : g.moveTo(x(px), y(py))));
    g.stroke();
  }
  g.restore();
  g.globalAlpha = 1;
}

function draw() {
  if (!run) return;
  const trials = run.trials || [];
  const dark = isDark();
  const sel = selectedTrial();
  plotSeries(
    $("tv-reward"),
    trials.map((t, i) => ({
      points: t.curve || [],
      color: trainingColor(i, dark),
      width: t.id === run.leader ? 2.6 : 1.4,
      alpha: t.state === "pruned" || t.state === "failed" ? 0.35 : t.id === run.leader ? 1 : 0.8,
    })),
    { empty: tx("waiting") },
  );
  const i = sel ? trials.indexOf(sel) : 0;
  plotSeries($("tv-returns"), [{ points: sel ? sel.returns || [] : [], color: trainingColor(Math.max(i, 0), dark), alpha: 0.75 }], { dots: true, empty: tx("waiting") });
  plotSeries(
    $("tv-metric"),
    [{ points: sel && metric ? (sel.metrics || {})[metric] || [] : [], color: trainingColor(Math.max(i, 0), dark), width: 1.8 }],
    { empty: tx("none_metric") },
  );
}
