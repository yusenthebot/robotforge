// Lessons in the viewer: the course picker, the step-by-step panel, live plots and the overlays that
// draw what the physics is doing (angles, errors, torques, forces). Checks and grading live in Python;
// this module only shows the lesson's state and sends the learner's input.
import * as THREE from "three";
import { LANG, t } from "./i18n.js";

const $ = (id) => document.getElementById(id);
const pick = (text) => (text ? (text[LANG] ?? text.en ?? "") : "");

let api = null; // hooks into the main viewer, see initLessons
let catalog = []; // lessons the server offers
let lesson = null; // the running lesson's description
let state = null; // its latest live update
let activeSlider = null;
let needCamera = false; // the lesson's camera still has to be applied once its scene has arrived

export function initLessons(hooks) {
  api = hooks;
  $("learn-toggle").addEventListener("click", () => setPicker($("learn-panel").hidden));
  $("lesson-exit").addEventListener("click", () => exitLesson());
  $("lesson-finished-exit").addEventListener("click", () => exitLesson());
  $("lesson-finished-stay").addEventListener("click", () => ($("lesson-finished").hidden = true));
  $("lesson-next").addEventListener("click", () => {
    const last = lesson && state && state.step === lesson.steps.length - 1;
    if (last && passedAll()) finishLesson();
    else api.send({ type: "lesson", action: "next" });
  });
}

export const lessonActive = () => lesson !== null;
export const lessonTitle = () => (lesson ? pick(lesson.title) : null);
export const lessonDriveSpec = () => lesson?.drive ?? null;
let followWanted = null; // the body the lesson last asked the camera to follow
let viewWanted = null; // the framing the lesson last asked for, as JSON
export const lessonKind = () => lesson?.kind ?? null;
export const lessonState = () => (lesson ? { id: lesson.id, ...state } : null); // for scripted checks

// ---------------------------------------------------------------- picker

function setPicker(open) {
  $("learn-panel").hidden = !open;
  $("learn-toggle").setAttribute("aria-expanded", String(open));
  if (open) api.closePanels();
}

function progress() {
  try {
    return JSON.parse(localStorage.getItem("robotforge.lessons") || "{}");
  } catch {
    return {};
  }
}

function saveProgress(id, entry) {
  try {
    const all = progress();
    all[id] = { ...all[id], ...entry };
    localStorage.setItem("robotforge.lessons", JSON.stringify(all));
  } catch {
    // storage may be unavailable; progress just is not remembered
  }
}

function renderPicker() {
  const done = progress();
  const list = $("lesson-list");
  const groups = [
    ["lesson", t("lessons_heading")],
    ["bench", t("bench_heading")],
    ["compare", t("compare_heading")],
    ["challenge", t("challenges_heading")],
  ];
  const sections = [];
  for (const [kind, heading] of groups) {
    const items = catalog.filter((c) => (c.kind ?? "lesson") === kind);
    if (!items.length) continue;
    const h = document.createElement("h3");
    h.className = "course-group";
    h.textContent = heading;
    sections.push(h, ...items.map((c) => courseCard(c, done)));
  }
  list.replaceChildren(...sections);
}

function courseCard(c, done) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "course";
  const chapter = document.createElement("span");
  chapter.className = "course-chapter";
  chapter.textContent = pick(c.chapter);
  const title = document.createElement("span");
  title.className = "course-title";
  title.textContent = pick(c.title);
  const summary = document.createElement("span");
  summary.className = "course-summary";
  summary.textContent = pick(c.summary);
  const meta = document.createElement("span");
  meta.className = "course-meta";
  const stars = done[c.id]?.stars ?? 0;
  meta.textContent = [t("lesson_minutes", { n: c.minutes }), done[c.id]?.done ? t("lesson_done_badge") : ""]
    .filter(Boolean)
    .join(" · ");
  if (stars) meta.append(starSpan(stars));
  b.append(chapter, title, summary, meta);
  b.addEventListener("click", () => startLesson(c.id, b));
  return b;
}

function starSpan(n) {
  const s = document.createElement("span");
  s.className = "stars";
  s.setAttribute("aria-label", t("stars_label", { n }));
  s.textContent = " " + "★".repeat(n) + "☆".repeat(3 - n);
  return s;
}

async function startLesson(id, button) {
  button.classList.add("busy");
  try {
    const desc = await api.request({ type: "lesson_open", action: "start", lesson: id });
    openLesson(desc);
  } catch (err) {
    api.toast(err.message);
  } finally {
    button.classList.remove("busy");
  }
}

async function exitLesson() {
  if (!lesson) return;
  api.wantFrame();
  try {
    await api.request({ type: "lesson_open", action: "exit" });
  } catch (err) {
    api.toast(err.message);
  }
  closeLesson();
}

function finishLesson() {
  const stars = state?.results?.stars ?? state?.results?.test?.stars ?? 0;
  saveProgress(lesson.id, { done: true, stars: Math.max(stars, progress()[lesson.id]?.stars ?? 0) });
  $("lesson-finished").hidden = false;
  $("lesson-finished-body").textContent = t("lesson_done_body", { title: pick(lesson.title) });
}

// ---------------------------------------------------------------- running a lesson

function openLesson(desc) {
  lesson = desc;
  state = null;
  followWanted = viewWanted = null;
  api.followBody(null);
  plots.reset(desc.plots);
  bench.setup(desc.bench, desc.bench_state);
  renderTable(null);
  overlays.build(desc.rig);
  setPicker(false);
  document.body.classList.add("in-lesson");
  $("lesson-panel").hidden = false;
  $("lesson-plots").hidden = desc.plots.length === 0;
  $("lesson-finished").hidden = true;
  $("lesson-chapter").textContent = pick(desc.chapter);
  $("lesson-title").textContent = pick(desc.title);
  $("lesson-drive").hidden = !desc.drive;
  pip.setup(desc.rig);
  renderControls();
  renderStep();
  needCamera = true;
  if (desc.camera) api.moveCamera(desc.camera.eye, desc.camera.target);
  api.shift();
  api.retitle();
}

function closeLesson() {
  lesson = null;
  state = null;
  followWanted = viewWanted = null;
  api.followBody(null);
  overlays.clear();
  pip.teardown();
  document.body.classList.remove("in-lesson");
  $("lesson-panel").hidden = true;
  $("lesson-plots").hidden = true;
  api.shift();
  api.retitle();
}

const passedAll = () => lesson && state && lesson.steps.every((s) => state.passed.includes(s.id));

// The scene description says which lessons exist and whether one is running (e.g. after a reconnect).
export function lessonScene(desc) {
  catalog = desc.lessons || [];
  renderPicker();
  $("learn-toggle").hidden = catalog.length === 0;
  if (desc.lesson && (!lesson || lesson.id !== desc.lesson.id)) openLesson(desc.lesson);
  else if (!desc.lesson && lesson) closeLesson();
  else if (lesson) overlays.build(lesson.rig); // the scene was rebuilt; overlays hang on new bodies
  // A new scene resets the camera's motion, so aim it at the lesson's rig once that scene is here.
  if (lesson && desc.lesson && needCamera) {
    needCamera = false;
    if (lesson.camera) api.moveCamera(lesson.camera.eye, lesson.camera.target);
  }
}

export function lessonUpdate(msg) {
  if (msg.ended) {
    closeLesson();
    return;
  }
  if (!lesson || msg.lesson !== lesson.id) return;
  const stepChanged = !state || state.step !== msg.step || state.passed.length !== msg.passed.length;
  const unlockChanged = !state || state.unlocked.join() !== msg.unlocked.join();
  if (msg.bench) bench.update(msg.bench);
  const quizChanged = JSON.stringify(state?.quiz) !== JSON.stringify(msg.quiz);
  const resultChanged = JSON.stringify(state?.results) !== JSON.stringify(msg.results);
  state = msg;
  plots.add(msg.samples);
  if (unlockChanged) {
    renderControls();
    bench.syncLocks();
  } else syncControls();
  if (stepChanged || quizChanged || resultChanged) renderStep();
  $("lesson-readout").textContent = readout(msg.live);
  // A lesson may frame each robot its own way (a small one close up, a tall one from further back).
  const view = msg.live?.view ? JSON.stringify(msg.live.view) : null;
  if (view && view !== viewWanted) {
    viewWanted = view;
    followWanted = null;
    api.followBody(null);
    api.moveCamera(msg.live.view.eye, msg.live.view.target);
  }
  const follow = msg.live?.follow ?? null;
  // Only when the lesson asks for a different body, so orbiting the camera away stays yours.
  if (follow !== followWanted && api.followBody(follow)) followWanted = follow;
  renderFormula(msg.live?.formula ?? null);
  renderTable(msg.live?.table ?? null);
}

// The controller's sum with live numbers, one row per term; a row appears once its step shows it.
function renderFormula(rows) {
  const box = $("lesson-formula");
  const shown = new Set(state?.shown ?? []);
  const visible = (rows ?? []).filter((r) => shown.has(r.show));
  box.hidden = visible.length === 0;
  if (box.hidden) return;
  const text = (x) => (typeof x === "string" ? x : pick(x));
  const list = $("lesson-formula-rows");
  if (list.childElementCount !== visible.length) {
    list.replaceChildren(
      ...visible.map(() => {
        const row = document.createElement("div");
        row.className = "formula-row";
        for (const part of ["label", "expr", "value", "note"]) {
          const cell = document.createElement("span");
          cell.className = `formula-${part}`;
          row.append(cell);
        }
        return row;
      })
    );
  }
  visible.forEach((r, k) => {
    const row = list.children[k];
    row.style.setProperty("--term", `var(${COLORS[r.color] ?? "--ink"})`);
    const [label, expr, value, note] = row.children;
    label.textContent = text(r.label);
    expr.textContent = text(r.expr);
    value.textContent = r.value;
    note.textContent = r.note ? text(r.note) : "";
    note.hidden = !r.note;
  });
}

function readout(live) {
  if (!live) return "";
  if (live.readout) return pick(live.readout);
  if (live.angle !== undefined) {
    return t("readout_joint", { angle: live.angle.toFixed(1), error: live.error.toFixed(1), torque: live.torque.toFixed(2) });
  }
  if (live.ranges !== undefined) {
    const front = live.ranges[0] >= 0 ? live.ranges[0].toFixed(2) : "> 4";
    return t("readout_rover", { odom: live.odom.toFixed(2), heading: live.heading_gyro.toFixed(0), front, touch: live.touch.toFixed(1) });
  }
  if (live.err_cm !== undefined) {
    return t("readout_arm", { q1: live.q1.toFixed(0), q2: live.q2.toFixed(0), err: live.err_cm.toFixed(1), d: (live.d * 100).toFixed(0) });
  }
  return "";
}

function renderStep() {
  const i = state ? state.step : 0;
  const step = lesson.steps[i];
  const passed = state ? state.passed.includes(step.id) : false;
  $("lesson-step-count").textContent = t("step_of", { i: i + 1, n: lesson.steps.length });
  const dots = $("lesson-dots");
  dots.replaceChildren(
    ...lesson.steps.map((s, k) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "dot";
      const done = state?.passed.includes(s.id);
      b.dataset.state = k === i ? "current" : done ? "done" : "todo";
      b.disabled = !(done || k === i || (k > 0 && state?.passed.includes(lesson.steps[k - 1].id)));
      b.setAttribute("aria-label", `${t("step_of", { i: k + 1, n: lesson.steps.length })}: ${pick(s.title)}`);
      b.addEventListener("click", () => api.send({ type: "lesson", action: "goto", step: k }));
      return b;
    })
  );
  $("lesson-step-title").textContent = pick(step.title);
  const body = $("lesson-step-body");
  body.replaceChildren(...pick(step.body).split("\n").map((line) => paragraph(line)));
  renderQuiz(step);
  renderResult();
  const doneBox = $("lesson-step-done");
  doneBox.hidden = !passed;
  if (passed) $("lesson-step-done-text").textContent = pick(step.done);
  const last = i === lesson.steps.length - 1;
  const next = $("lesson-next");
  next.hidden = !(passed || step.manual);
  next.textContent = last ? t(lesson.kind === "challenge" ? "finish_challenge" : "finish_lesson") : t("next_step");
  const guessed = state?.quiz?.chosen !== undefined;
  $("lesson-waiting").hidden = passed || step.manual || (!!step.quiz && !(step.quiz.predict && guessed));
}

function paragraph(line) {
  const p = document.createElement("p");
  p.textContent = line;
  return p;
}

function renderQuiz(step) {
  const box = $("lesson-quiz");
  box.hidden = !step.quiz;
  if (!step.quiz) return;
  const quiz = state?.quiz;
  const chosen = quiz?.chosen;
  const pending = !!quiz?.pending; // a guess, waiting for the experiment
  const locked = step.quiz.predict && chosen !== undefined; // a guess is made once
  box.dataset.kind = step.quiz.predict ? "predict" : "quiz";
  $("lesson-quiz-question").textContent = pick(step.quiz.question);
  $("lesson-quiz-options").replaceChildren(
    ...step.quiz.options.map((option, k) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "quiz-option";
      b.textContent = pick(option);
      if (chosen === k) b.dataset.verdict = pending ? "guess" : quiz.right ? "right" : "wrong";
      else if (!pending && quiz?.answer === k) b.dataset.verdict = "right"; // after a wrong guess, show the answer
      if (locked) b.disabled = true;
      else b.addEventListener("click", () => api.send({ type: "lesson", action: "answer", choice: k }));
      return b;
    })
  );
  const feedback = $("lesson-quiz-feedback");
  feedback.hidden = chosen === undefined;
  if (chosen === undefined) return;
  if (pending) feedback.textContent = t("predict_pending");
  else if (step.quiz.predict) {
    const verdict = document.createElement("strong");
    verdict.textContent = t(quiz.right ? "predict_right" : "predict_wrong");
    feedback.replaceChildren(verdict, pick(quiz.explain));
  } else feedback.textContent = quiz.right ? pick(quiz.explain) : t("quiz_try_again");
}

function renderResult() {
  const box = $("lesson-result");
  const summary = state?.results?.summary;
  if (summary) {
    box.hidden = false;
    box.replaceChildren();
    const line = document.createElement("span");
    line.textContent = pick(summary);
    box.append(line);
    if (state.results.stars !== undefined) box.append(starSpan(state.results.stars)); // lessons report without stars
    return;
  }
  const test = state?.results?.test;
  box.hidden = !test;
  if (!test) return;
  box.replaceChildren();
  if (test.running) {
    box.textContent = t("test_running");
    return;
  }
  const line = document.createElement("span");
  if (test.settle_s !== null) {
    const off = test.end_err >= 0.5 ? t("test_result_off", { err: test.end_err }) : ""; // inside the band, but not on target
    line.textContent = t("test_result", { settle: test.settle_s, overshoot: test.overshoot_pct }) + off;
  }
  else if (test.end_err !== null && test.end_err !== undefined) line.textContent = t("test_stuck", { err: test.end_err });
  else line.textContent = t("test_not_settled", { overshoot: test.overshoot_pct });
  box.append(line, starSpan(test.stars));
  if (test.note) {
    const note = document.createElement("p");
    note.className = "lesson-result-note";
    note.textContent = pick(test.note);
    box.append(note);
  }
}

// ---------------------------------------------------------------- experiment record

// Every finished test keeps its settings, its metrics and its curve: up to five runs laid over each
// other with a table to compare them, plus sweeps that try one knob across its range (metric vs setting).
const RUN_COLORS = ["--plot-blue", "--plot-violet", "--plot-green", "--amber", "--ink"];
const runColor = (id) => RUN_COLORS[(id - 1) % RUN_COLORS.length];

const bench = {
  spec: null,
  state: null,
  box: null,
  setup(spec, initial) {
    this.box?.remove();
    this.box = null;
    this.spec = spec;
    this.state = initial ?? null;
    this.zoom = false; // the run chart shows the whole climb, or only the last degrees near the target
    if (!spec) return;
    this.box = document.createElement("section");
    this.box.className = "bench";
    $("lesson-plots").append(this.box);
    this.render();
  },
  update(next) {
    this.state = next;
    this.render();
  },
  label(control) {
    const k = this.spec.params.indexOf(control);
    if (k >= 0 && this.spec.short[k]) return pick(this.spec.short[k]);
    const c = lesson?.controls.find((x) => x.id === control);
    return c ? pick(c.label) : control;
  },
  value(control, v) {
    const c = lesson?.controls.find((x) => x.id === control);
    if (c?.kind === "toggle") return t(v ? "on" : "off");
    if (c?.kind === "choice" && c.options?.[v]) return pick(c.options[v]);
    return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(3)));
  },
  render() {
    const box = this.box;
    if (!box) return;
    const spec = this.spec;
    const runs = this.state?.runs ?? [];
    const head = document.createElement("div");
    head.className = "plot-head";
    const title = document.createElement("span");
    title.className = "plot-title";
    title.textContent = pick(spec.title);
    head.append(title);
    if (runs.length && spec.zoom && spec.reference !== null) {
      const zoom = document.createElement("button");
      zoom.type = "button";
      zoom.className = "link-button";
      zoom.textContent = t(this.zoom ? "bench_zoom_out" : "bench_zoom_in", { span: `${spec.zoom}${spec.y_unit}` });
      zoom.setAttribute("aria-pressed", String(this.zoom));
      zoom.addEventListener("click", () => {
        this.zoom = !this.zoom;
        this.render();
      });
      head.append(zoom);
    }
    if (runs.length || this.state?.sweep) {
      const clear = document.createElement("button");
      clear.type = "button";
      clear.className = "link-button";
      clear.textContent = t("bench_clear");
      clear.addEventListener("click", () => api.send({ type: "lesson", action: "clear_runs" }));
      head.append(clear);
    }
    const parts = [head];
    if (!runs.length) {
      const empty = document.createElement("p");
      empty.className = "bench-empty";
      empty.textContent = t("bench_empty");
      parts.push(empty);
    } else {
      const chart = document.createElement("canvas");
      chart.className = "bench-chart";
      chart.dataset.kind = "runs";
      parts.push(chart, this.table(runs));
    }
    if (spec.sweeps.length) parts.push(this.sweeps());
    box.replaceChildren(...parts);
    this.syncLocks();
    this.draw();
  },
  table(runs) {
    const spec = this.spec;
    const grid = document.createElement("table");
    grid.className = "bench-table";
    const head = document.createElement("tr");
    const first = document.createElement("th");
    first.scope = "col";
    first.textContent = t("bench_run");
    head.append(first);
    for (const m of spec.metrics) {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = m.unit ? `${pick(m.label)} (${m.unit})` : pick(m.label);
      head.append(th);
    }
    const best = spec.metrics.map((m) => {
      if (!m.ranked) return null;
      const values = runs.map((r) => r.metrics[m.key]).filter((v) => v !== null && v !== undefined);
      if (values.length < 2 || values.every((v) => v === values[0])) return null;
      return m.lower_is_better ? Math.min(...values) : Math.max(...values);
    });
    const body = document.createElement("tbody");
    for (const run of runs) {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.scope = "row";
      const chip = document.createElement("span");
      chip.className = "run-chip";
      chip.style.setProperty("--c", `var(${runColor(run.id)})`);
      chip.textContent = `#${run.id}`;
      const settings = document.createElement("span");
      settings.className = "run-settings";
      settings.textContent = spec.params.map((p) => `${this.label(p)} ${this.value(p, run.params[p])}`).join(" · ");
      th.append(chip, settings);
      tr.append(th);
      spec.metrics.forEach((m, k) => {
        const td = document.createElement("td");
        const v = run.metrics[m.key];
        td.textContent = v === null || v === undefined ? "-" : String(v);
        if (best[k] !== null && v === best[k]) td.dataset.best = "true";
        tr.append(td);
      });
      body.append(tr);
    }
    const thead = document.createElement("thead");
    thead.append(head);
    grid.append(thead, body);
    return grid;
  },
  sweeps() {
    const wrap = document.createElement("div");
    wrap.className = "bench-sweeps";
    const row = document.createElement("div");
    row.className = "sweep-buttons";
    for (const sw of this.spec.sweeps) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "pill-button";
      b.dataset.control = sw.control;
      b.textContent = t("sweep_button", { name: this.label(sw.control) });
      b.addEventListener("click", async () => {
        b.classList.add("busy");
        try {
          await api.request({ type: "lesson_bench", action: "sweep", control: sw.control });
        } catch (err) {
          api.toast(err.message);
        } finally {
          b.classList.remove("busy");
        }
      });
      row.append(b);
    }
    wrap.append(row);
    const sweep = this.state?.sweep;
    const spec = sweep ? this.spec.sweeps.find((w) => w.control === sweep.control) : null;
    if (sweep && spec) {
      const xs = sweep.points.map((p) => p.x);
      const note = document.createElement("p");
      note.className = "sweep-title";
      note.textContent = t("sweep_title", { name: this.label(sweep.control), lo: xs[0], hi: xs[xs.length - 1] });
      wrap.append(note);
      for (const key of spec.metrics) {
        const m = this.spec.metrics.find((x) => x.key === key);
        const label = document.createElement("span");
        label.className = "sweep-metric";
        label.textContent = m ? (m.unit ? `${pick(m.label)} (${m.unit})` : pick(m.label)) : key;
        const chart = document.createElement("canvas");
        chart.className = "sweep-chart";
        chart.dataset.kind = "sweep";
        chart.dataset.metric = key;
        wrap.append(label, chart);
      }
    }
    return wrap;
  },
  syncLocks() {
    if (!this.box) return;
    const open = new Set(state?.unlocked ?? []);
    for (const b of this.box.querySelectorAll(".sweep-buttons button")) {
      b.disabled = !open.has(b.dataset.control);
      b.title = b.disabled ? t("sweep_locked") : "";
    }
  },
  draw() {
    if (!this.box) return;
    const runs = this.state?.runs ?? [];
    const newest = runs.length ? runs[runs.length - 1].id : null;
    for (const canvas of this.box.querySelectorAll("canvas")) {
      if (canvas.dataset.kind === "runs") {
        const lines = runs.map((r) => ({ xs: r.x, ys: r.y, color: runColor(r.id), width: r.id === newest ? 2.2 : 1.3 }));
        const ref = this.spec.reference;
        const yWindow = this.zoom && ref !== null ? [ref - this.spec.zoom, ref + this.spec.zoom] : null;
        drawXY(canvas, lines, { reference: ref, height: 120, xUnit: this.spec.x_unit, yWindow });
      } else {
        const sweep = this.state.sweep;
        const key = canvas.dataset.metric;
        const lines = [{ xs: sweep.points.map((p) => p.x), ys: sweep.points.map((p) => p[key] ?? null), color: "--ink", width: 1.6, dots: true }];
        drawXY(canvas, lines, { height: 74 });
      }
    }
  },
};

window.addEventListener("resize", () => bench.draw());

// A static chart: lines of (x, y) with gaps where y is null, optional dots and a dashed reference level.
function drawXY(canvas, lines, { reference = null, height = 100, xUnit = "", yWindow = null } = {}) {
  const css = getComputedStyle(document.documentElement);
  const token = (name) => css.getPropertyValue(name).trim();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 300;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(height * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, height);
  let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
  for (const line of lines) {
    line.xs.forEach((x, i) => {
      const y = line.ys[i];
      if (y === null || y === undefined) return;
      [x0, x1, y0, y1] = [Math.min(x0, x), Math.max(x1, x), Math.min(y0, y), Math.max(y1, y)];
    });
  }
  if (!Number.isFinite(x0)) return;
  if (reference !== null) [y0, y1] = [Math.min(y0, reference), Math.max(y1, reference)];
  if (x1 - x0 < 1e-9) [x0, x1] = [x0 - 1, x1 + 1];
  if (y1 - y0 < 1e-9) [y0, y1] = [y0 - 1, y1 + 1];
  const pad = (y1 - y0) * 0.1;
  [y0, y1] = yWindow ?? [y0 - pad, y1 + pad]; // a fixed window clips whatever lies outside it
  const X = (v) => 36 + ((v - x0) / (x1 - x0)) * (w - 46);
  const Y = (v) => 6 + (1 - (v - y0) / (y1 - y0)) * (height - 22);
  g.font = `10px ${token("--mono") || "monospace"}`;
  g.fillStyle = token("--muted");
  g.strokeStyle = token("--line");
  g.lineWidth = 1;
  for (const v of niceTicks(y0, y1, 3)) {
    g.beginPath();
    g.moveTo(36, Math.round(Y(v)) + 0.5);
    g.lineTo(w - 8, Math.round(Y(v)) + 0.5);
    g.stroke();
    g.fillText(formatTick(v), 2, Y(v) + 3);
  }
  const xt = niceTicks(x0, x1, 4);
  xt.forEach((v, i) => {
    const text = formatTick(v) + (i === xt.length - 1 && xUnit ? ` ${xUnit}` : "");
    g.fillText(text, Math.min(X(v) - 6, w - g.measureText(text).width - 2), height - 3);
  });
  if (reference !== null) {
    g.strokeStyle = token("--muted");
    g.setLineDash([4, 4]);
    g.beginPath();
    g.moveTo(36, Y(reference));
    g.lineTo(w - 8, Y(reference));
    g.stroke();
    g.setLineDash([]);
  }
  g.save();
  g.beginPath();
  g.rect(36, 0, w - 44, height - 16);
  g.clip();
  for (const line of lines) {
    g.strokeStyle = token(line.color);
    g.fillStyle = token(line.color);
    g.lineWidth = line.width ?? 1.5;
    g.beginPath();
    let open = false;
    line.xs.forEach((x, i) => {
      const y = line.ys[i];
      if (y === null || y === undefined) {
        open = false;
        return;
      }
      if (!open) g.moveTo(X(x), Y(y));
      else g.lineTo(X(x), Y(y));
      open = true;
    });
    g.stroke();
    if (line.dots) {
      line.xs.forEach((x, i) => {
        const y = line.ys[i];
        g.beginPath();
        if (y === null || y === undefined) {
          g.strokeStyle = token("--muted"); // never reached: a hollow mark along the top
          g.arc(X(x), 9, 2.5, 0, Math.PI * 2);
          g.stroke();
          g.strokeStyle = token(line.color);
        } else {
          g.arc(X(x), Y(y), 2.5, 0, Math.PI * 2);
          g.fill();
        }
      });
    }
  }
  g.restore();
}

// ---------------------------------------------------------------- face-off table

// Robots side by side: one column each, the best value in a row marked. It sits above the plots.
const tableBox = document.createElement("section");
tableBox.className = "faceoff-table";
tableBox.hidden = true;
let tableKey = "";

function renderTable(table) {
  const key = table ? LANG + JSON.stringify(table) : "";
  if (key === tableKey) return;
  tableKey = key;
  tableBox.hidden = !table;
  if (!table) return;
  $("lesson-plots").hidden = false;
  const head = document.createElement("tr");
  head.append(document.createElement("td"));
  for (const c of table.columns) {
    const th = document.createElement("th");
    th.scope = "col";
    th.style.setProperty("--c", `var(${COLORS[c.color]})`);
    const name = document.createElement("span");
    name.className = "who";
    name.textContent = pick(c.label);
    th.append(name);
    if (c.sub !== null) {
      const kind = document.createElement("span"); // twins have no subtitle: they are the same robot
      kind.className = "kind";
      kind.textContent = c.sub ? pick(c.sub) : t(c.real ? "real_robot" : "our_template");
      th.append(kind);
    }
    head.append(th);
  }
  const thead = document.createElement("thead");
  thead.append(head);
  const tbody = document.createElement("tbody");
  for (const row of table.rows) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    if (row.heading) {
      tr.className = "group";
      th.colSpan = table.columns.length + 1;
      th.scope = "colgroup";
      th.textContent = pick(row.heading);
      tr.append(th);
    } else {
      th.scope = "row";
      th.textContent = pick(row.label);
      tr.append(th);
      row.cells.forEach((cell, k) => {
        const td = document.createElement("td");
        td.textContent = typeof cell === "string" ? cell : pick(cell);
        if (row.best === k) td.dataset.best = "true";
        tr.append(td);
      });
    }
    tbody.append(tr);
  }
  const grid = document.createElement("table");
  grid.append(thead, tbody);
  tableBox.replaceChildren(grid);
}

// ---------------------------------------------------------------- controls

function renderControls() {
  const box = $("lesson-controls");
  const unlocked = new Set(state ? state.unlocked : []);
  const usable = lesson.controls.filter((c) => unlocked.has(c.id));
  $("lesson-controls-wrap").hidden = usable.length === 0;
  const buttons = usable.filter((c) => c.kind === "button");
  const rows = [];
  let group = null;
  for (const c of usable.filter((x) => x.kind !== "button")) {
    if ((c.group ?? "") !== group) {
      group = c.group ?? "";
      const info = lesson.groups?.[group];
      if (info) rows.push(groupHeading(info)); // one twin's own settings start here
    }
    rows.push(controlRow(c));
  }
  const bar = document.createElement("div");
  bar.className = "lesson-buttons";
  bar.append(
    ...buttons.map((c) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "pill-button";
      b.textContent = pick(c.label);
      b.addEventListener("click", () => api.send({ type: "lesson", action: "press", control: c.id }));
      return b;
    })
  );
  const twins = Object.keys(lesson.groups ?? {}).length > 0; // the shared test goes first, then each twin
  const bars = buttons.length ? [bar] : [];
  box.replaceChildren(...(twins ? [...bars, ...rows] : [...rows, ...bars]));
  syncControls();
}

function groupHeading(info) {
  const h = document.createElement("p");
  h.className = "control-group";
  h.style.setProperty("--c", `var(${COLORS[info.color]})`);
  h.textContent = pick(info.label);
  return h;
}

function controlRow(c) {
  const value = state?.values[c.id] ?? c.value;
  if (c.kind === "choice") {
    const row = document.createElement("div");
    row.className = "lesson-choice";
    const name = document.createElement("span");
    name.className = "joint-name";
    name.textContent = pick(c.label);
    const seg = document.createElement("div");
    seg.className = "seg";
    seg.dataset.choice = c.id;
    c.options.forEach((option, k) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = pick(option);
      b.setAttribute("aria-pressed", String(k === value));
      b.addEventListener("click", () => api.send({ type: "lesson", action: "set", control: c.id, value: k }));
      seg.append(b);
    });
    row.append(name, seg);
    return row;
  }
  if (c.kind === "toggle") {
    const label = document.createElement("label");
    label.className = "check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.control = c.id;
    input.checked = !!value;
    input.addEventListener("change", () => api.send({ type: "lesson", action: "set", control: c.id, value: input.checked ? 1 : 0 }));
    const span = document.createElement("span");
    span.textContent = pick(c.label);
    label.append(input, span);
    return label;
  }
  const row = document.createElement("label");
  row.className = "joint";
  const name = document.createElement("span");
  name.className = "joint-name";
  name.textContent = pick(c.label);
  const shown = document.createElement("span");
  shown.className = "joint-value";
  const input = document.createElement("input");
  input.type = "range";
  input.min = c.min;
  input.max = c.max;
  input.step = c.step;
  input.value = value;
  input.dataset.control = c.id;
  const show = () => {
    shown.textContent = `${Number(input.value).toFixed(c.step < 1 ? 2 : 0)}${c.unit}`;
    const p = ((Number(input.value) - c.min) / (c.max - c.min)) * 100;
    input.style.setProperty("--p", `${p}%`);
  };
  show();
  input.addEventListener("pointerdown", () => (activeSlider = input));
  input.addEventListener("pointerup", () => (activeSlider = null));
  input.addEventListener("input", () => {
    show();
    api.send({ type: "lesson", action: "set", control: c.id, value: Number(input.value) });
  });
  input.showValue = show;
  row.append(name, shown, input);
  return row;
}

function syncControls() {
  if (!state) return;
  for (const seg of $("lesson-controls").querySelectorAll("[data-choice]")) {
    const v = state.values[seg.dataset.choice];
    [...seg.children].forEach((b, k) => b.setAttribute("aria-pressed", String(k === v)));
  }
  for (const input of $("lesson-controls").querySelectorAll("input[data-control]")) {
    if (input === activeSlider) continue;
    const v = state.values[input.dataset.control];
    if (v === undefined) continue;
    if (input.type === "checkbox") input.checked = !!v;
    else if (Number(input.value) !== v) {
      input.value = v;
      input.showValue?.();
    }
  }
}

// ---------------------------------------------------------------- live plots

const COLORS = { ink: "--ink", amber: "--amber", blue: "--plot-blue", violet: "--plot-violet", green: "--plot-green", muted: "--muted" };

const plots = {
  specs: [],
  data: [], // samples, oldest first
  canvases: [],
  ranges: [],
  reset(specs) {
    this.specs = specs;
    this.data = [];
    this.ranges = specs.map(() => null);
    const box = $("lesson-plots");
    box.replaceChildren(tableBox);
    this.canvases = specs.map((spec) => {
      const wrap = document.createElement("section");
      wrap.className = "plot";
      const head = document.createElement("div");
      head.className = "plot-head";
      const title = document.createElement("span");
      title.className = "plot-title";
      title.textContent = spec.unit ? `${pick(spec.title)} (${spec.unit})` : pick(spec.title);
      const legend = document.createElement("span");
      legend.className = "plot-legend";
      const named = spec.kind === "footfall" ? [] : spec.series; // footfall rows carry their own names
      for (const s of named) {
        const item = document.createElement("span");
        item.className = "legend-item";
        item.style.setProperty("--c", `var(${COLORS[s.color]})`);
        item.dataset.dashed = String(s.dashed);
        item.textContent = pick(s.label);
        legend.append(item);
      }
      head.append(title, legend);
      const canvas = document.createElement("canvas");
      canvas.height = 116;
      wrap.append(head, canvas);
      box.append(wrap);
      return canvas;
    });
  },
  add(samples) {
    if (!samples?.length) return;
    const last = this.data.length ? this.data[this.data.length - 1].t : -Infinity;
    if (samples[0].t < last - 0.5) this.data = []; // time went back: the learner pressed Reset
    this.data.push(...samples);
    const keep = Math.max(...this.specs.map((s) => s.window), 1) + 1;
    const end = this.data[this.data.length - 1].t;
    while (this.data.length && this.data[0].t < end - keep) this.data.shift();
  },
  draw() {
    if (!lesson || $("lesson-plots").hidden) return;
    const css = getComputedStyle(document.documentElement);
    const token = (name) => css.getPropertyValue(name).trim();
    this.specs.forEach((spec, k) => this.drawOne(spec, k, this.canvases[k], token));
  },
  drawOne(spec, k, canvas, token) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight || 116;
    if (canvas.width !== Math.round(w * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const end = this.data.length ? this.data[this.data.length - 1].t : 0;
    const start = end - spec.window;
    const shown = this.data.filter((s) => s.t >= start);
    if (spec.kind === "footfall") {
      this.drawFootfall(spec, g, w, h, shown, start, token);
      return;
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of shown) for (const series of spec.series) {
      const v = s[series.key];
      if (v === undefined) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    if (!Number.isFinite(lo)) [lo, hi] = [-1, 1];
    if (hi - lo < 1e-6) [lo, hi] = [lo - 1, hi + 1];
    const pad = (hi - lo) * 0.12;
    // Grow at once, shrink slowly, so the axes do not jump around while you watch.
    const prev = this.ranges[k];
    const target = [lo - pad, hi + pad];
    const range = prev ? [Math.min(target[0], prev[0] + (target[0] - prev[0]) * 0.05), Math.max(target[1], prev[1] + (target[1] - prev[1]) * 0.05)] : target;
    this.ranges[k] = range;
    const x = (t) => 36 + ((t - start) / spec.window) * (w - 44);
    const y = (v) => 8 + (1 - (v - range[0]) / (range[1] - range[0])) * (h - 22);
    g.font = `10px ${token("--mono") || "monospace"}`;
    g.fillStyle = token("--muted");
    g.strokeStyle = token("--line");
    g.lineWidth = 1;
    for (const v of niceTicks(range[0], range[1], 3)) {
      g.beginPath();
      g.moveTo(36, Math.round(y(v)) + 0.5);
      g.lineTo(w - 8, Math.round(y(v)) + 0.5);
      g.stroke();
      g.fillText(formatTick(v), 2, y(v) + 3);
    }
    for (const series of spec.series) {
      g.strokeStyle = token(COLORS[series.color]);
      g.lineWidth = series.color === "ink" ? 1.8 : 1.4;
      g.setLineDash(series.dashed ? [5, 4] : []);
      g.beginPath();
      let started = false;
      for (const s of shown) {
        const v = s[series.key];
        if (v === undefined) continue;
        if (!started) g.moveTo(x(s.t), y(v));
        else g.lineTo(x(s.t), y(v));
        started = true;
      }
      g.stroke();
    }
    g.setLineDash([]);
  },
  // One row per foot, filled while it is on the ground: the classic gait diagram, newest on the right.
  drawFootfall(spec, g, w, h, shown, start, token) {
    const rows = spec.series.length;
    const left = 44;
    const rowH = (h - 10) / rows;
    const x = (t) => left + ((t - start) / spec.window) * (w - left - 8);
    g.font = `10px ${token("--font") || "sans-serif"}`;
    spec.series.forEach((series, r) => {
      const top = 4 + r * rowH;
      g.fillStyle = token("--muted");
      g.fillText(pick(series.label), 2, top + rowH / 2 + 3);
      g.fillStyle = token("--track");
      g.fillRect(left, top + 2, w - left - 8, rowH - 4);
      g.fillStyle = token(COLORS[series.color]);
      let from = null;
      shown.forEach((s, i) => {
        const on = s[series.key] > 0.5;
        if (on && from === null) from = s.t;
        const last = i === shown.length - 1;
        if (from !== null && (!on || last)) {
          g.fillRect(x(from), top + 2, Math.max(1, x(s.t) - x(from)), rowH - 4);
          from = null;
        }
      });
    });
  },
};

function niceTicks(lo, hi, count) {
  const span = hi - lo;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(Math.abs(v) < step / 1e6 ? 0 : v);
  return out;
}

function formatTick(v) {
  return Math.abs(v) >= 10 || v === 0 ? v.toFixed(0) : v.toFixed(1);
}

// ---------------------------------------------------------------- overlays in the 3D view

// Each rig kind draws its own explanation on top of the physics. A rig overlay is
// { group, parts: {name: Object3D}, frame() } and lives until the lesson or its scene ends.
const overlays = {
  current: null,
  clear() {
    const o = this.current;
    if (o) {
      o.group.traverse((x) => {
        x.geometry?.dispose();
        x.material?.map?.dispose();
        x.material?.dispose();
      });
      api.scene.remove(o.group);
    }
    this.current = null;
  },
  build(rig) {
    this.clear();
    const make = RIGS[rig?.kind];
    if (!make) return;
    this.current = make(rig);
    for (const part of Object.values(this.current.parts)) part.visible = false;
    api.scene.add(this.current.group);
  },
  frame() {
    const o = this.current;
    if (!o || !state) return;
    const shown = new Set(state.shown);
    for (const [name, part] of Object.entries(o.parts)) part.visible = shown.has(name);
    o.frame();
  },
  target() {
    return this.current?.dragTarget ?? null;
  },
};

// One motor joint: a protractor, the target as a ghost arm, the error as a wedge, the motor's torque as a
// curved arrow and gravity as a straight one.
function jointRig(rig) {
  const pivot = new THREE.Vector3().fromArray(rig.pivot);
  const zero = new THREE.Vector3().fromArray(rig.zero).normalize();
  const swing = new THREE.Vector3().fromArray(rig.swing).normalize();
  // The arm's direction in its own body frame; the simple rig's body frame is the world's at 0 degrees.
  const local = rig.local ? new THREE.Vector3().fromArray(rig.local).normalize() : zero.clone();
  const L = rig.length;
  const point = (deg, radius) => {
    const a = THREE.MathUtils.degToRad(deg);
    return pivot.clone().addScaledVector(zero, Math.cos(a) * radius).addScaledVector(swing, Math.sin(a) * radius);
  };
  const group = new THREE.Group();
  const amber = api.color("--amber");
  const ink = api.color("--ink");
  const blue = api.color("--plot-blue");
  const scale = new THREE.Group();
  {
    const r = L * 1.22;
    const [lo, hi] = rig.range ?? [-90, 180]; // the part of the dial worth drawing
    const pts = [];
    for (let deg = Math.ceil(lo / 15) * 15; deg <= hi; deg += 15) {
      const long = deg % 45 === 0;
      pts.push(point(deg, r), point(deg, r - (long ? 0.04 : 0.02)));
      if (long) {
        const tag = label(`${deg}°`, ink);
        tag.position.copy(point(deg, r + 0.045));
        scale.add(tag);
      }
    }
    const arc = [];
    for (let deg = lo; deg <= hi; deg += 3) arc.push(point(deg, r));
    scale.add(
      new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat(ink, 0.55)),
      new THREE.Line(new THREE.BufferGeometry().setFromPoints(arc), lineMat(ink, 0.35))
    );
  }
  const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 12), basic(amber, 0.45));
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.045, 24, 12), basic(amber, 0.35));
  const target = new THREE.Group();
  target.add(rod, ball);
  const error = new THREE.Mesh(new THREE.BufferGeometry(), basic(amber, 0.22, true));
  const torqueLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: ink, depthTest: false, transparent: true }));
  const torqueHead = new THREE.Mesh(new THREE.ConeGeometry(0.016, 0.04, 16), basic(ink, 0.95));
  const torque = new THREE.Group();
  torque.add(torqueLine, torqueHead);
  const gravity = arrow(new THREE.Vector3(0, 0, -1), 0.16, blue);
  const mg = label("mg", blue);
  mg.position.set(0.035, 0, -0.1);
  gravity.add(mg);
  const parts = { scale, target, error, torque, gravity };
  group.add(...Object.values(parts));
  const weight = new THREE.Vector3().fromArray(rig.weight);
  const angleNow = () => {
    const node = api.bodyNode(rig.body);
    if (!node) return 0;
    const dir = local.clone().applyQuaternion(node.quaternion);
    return THREE.MathUtils.radToDeg(Math.atan2(dir.dot(swing), dir.dot(zero)));
  };
  return {
    group,
    parts,
    frame() {
      const now = angleNow();
      const goal = state.values.target ?? 0;
      const tip = point(goal, L);
      rod.position.copy(pivot.clone().lerp(tip, 0.5));
      rod.scale.set(1, L, 1);
      rod.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tip.clone().sub(pivot).normalize());
      ball.position.copy(tip);
      const wedge = [];
      const steps = Math.max(2, Math.ceil(Math.abs(goal - now) / 3));
      for (let k = 1; k <= steps; k++) {
        const a0 = now + ((goal - now) * (k - 1)) / steps;
        const a1 = now + ((goal - now) * k) / steps;
        wedge.push(pivot.clone(), point(a0, L * 0.82), point(a1, L * 0.82));
      }
      setPoints(error, wedge);
      const sweep = ((state.live?.torque ?? 0) / (state.live?.torque_max ?? 1)) * 140;
      const arc = [];
      for (let k = 0; k <= 24; k++) arc.push(point(now + (sweep * k) / 24, 0.11));
      setPoints(torqueLine, arc);
      const end = arc[arc.length - 1];
      torqueHead.position.copy(end);
      torqueHead.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), end.clone().sub(arc[arc.length - 2]).normalize());
      torqueHead.visible = Math.abs(sweep) > 4;
      const node = api.bodyNode(rig.body);
      if (node) gravity.position.copy(weight.clone().applyQuaternion(node.quaternion).add(node.position));
    },
  };
}

// A two-link arm in a plane: the draggable target, the joint angles, the L1-L2-d triangle, the workspace
// ring, the other elbow solution and the error arrow of iterative IK.
function planarArmRig(rig) {
  const S = new THREE.Vector3().fromArray(rig.shoulder);
  const X = new THREE.Vector3().fromArray(rig.x_axis).normalize();
  const Z = new THREE.Vector3().fromArray(rig.z_axis).normalize();
  const toward = new THREE.Vector3().crossVectors(X, Z).normalize(); // from the arm toward the camera side
  const front = toward.clone().multiplyScalar(0.035); // overlays float just in front of the links
  const at = (x, z, lift = front) => S.clone().addScaledVector(X, x).addScaledVector(Z, z).add(lift);
  const amber = api.color("--amber");
  const ink = api.color("--ink");
  const blue = api.color("--plot-blue");
  const violet = api.color("--plot-violet");
  const group = new THREE.Group();

  const targetBall = new THREE.Mesh(new THREE.SphereGeometry(0.028, 24, 12), basic(amber, 0.95));
  const targetHalo = new THREE.Mesh(new THREE.RingGeometry(0.04, 0.05, 40), basic(amber, 0.6, true));
  targetHalo.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), toward);
  const target = new THREE.Group();
  target.add(targetBall, targetHalo);

  const angles = new THREE.Group();
  const ref1 = dashedLine(ink, 0.5);
  const arc1 = new THREE.Line(new THREE.BufferGeometry(), lineMat(ink, 0.9));
  const ref2 = dashedLine(ink, 0.5);
  const arc2 = new THREE.Line(new THREE.BufferGeometry(), lineMat(violet, 0.95));
  const t1 = label("θ1", ink);
  const t2 = label("θ2", violet);
  angles.add(ref1, arc1, ref2, arc2, t1, t2);

  const triangle = new THREE.Group();
  const sideD = dashedLine(ink, 0.7);
  const lab1 = label("L1", ink);
  const lab2 = label("L2", ink);
  const labD = label("d", ink);
  triangle.add(sideD, lab1, lab2, labD);

  const workspace = new THREE.Group();
  const behind = toward.clone().multiplyScalar(-0.03);
  const ring = new THREE.Mesh(new THREE.RingGeometry(Math.abs(rig.l1 - rig.l2), rig.l1 + rig.l2, 96), basic(amber, 0.1, true));
  ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), toward);
  ring.position.copy(S.clone().add(behind));
  const rim = [];
  for (let k = 0; k <= 96; k++) {
    const a = (k / 96) * Math.PI * 2;
    rim.push(at(Math.cos(a) * (rig.l1 + rig.l2), Math.sin(a) * (rig.l1 + rig.l2), behind));
  }
  workspace.add(ring, new THREE.Line(new THREE.BufferGeometry().setFromPoints(rim), lineMat(amber, 0.6)));

  const other = dashedLine(violet, 0.85);
  const error = new THREE.Group();
  const errLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: blue, depthTest: false, transparent: true }));
  const errHead = new THREE.Mesh(new THREE.ConeGeometry(0.014, 0.035, 16), basic(blue, 0.95));
  error.add(errLine, errHead);

  const parts = { target, angles, triangle, workspace, other, error };
  group.add(...Object.values(parts));

  const joints = () => {
    const upper = api.bodyNode(rig.upper);
    const fore = api.bodyNode(rig.fore);
    if (!upper || !fore) return null;
    const elbow = fore.position.clone().add(front);
    const hand = new THREE.Vector3(rig.l2, 0, 0).applyQuaternion(fore.quaternion).add(fore.position).add(front);
    return { shoulder: S.clone().add(front), elbow, hand };
  };
  const arcPoints = (center, from, to, radius) => {
    // from and to are directions in the plane; sweep the short way between them
    const a0 = Math.atan2(from.dot(Z), from.dot(X));
    let a1 = Math.atan2(to.dot(Z), to.dot(X));
    while (a1 - a0 > Math.PI) a1 -= 2 * Math.PI;
    while (a1 - a0 < -Math.PI) a1 += 2 * Math.PI;
    const pts = [];
    for (let k = 0; k <= 24; k++) {
      const a = a0 + ((a1 - a0) * k) / 24;
      pts.push(center.clone().addScaledVector(X, Math.cos(a) * radius).addScaledVector(Z, Math.sin(a) * radius));
    }
    return { pts, mid: a0 + (a1 - a0) / 2 };
  };

  return {
    group,
    parts,
    dragTarget: { mesh: targetBall, plane: new THREE.Plane().setFromNormalAndCoplanarPoint(toward, S), origin: S, X, Z },
    frame() {
      const tx = state.values.tx ?? 0;
      const tz = state.values.tz ?? 0;
      target.position.copy(at(tx, tz));
      const j = joints();
      if (!j) return;
      const dir1 = j.elbow.clone().sub(j.shoulder).normalize();
      const dir2 = j.hand.clone().sub(j.elbow).normalize();
      // angles: theta1 from the x axis, theta2 from the upper link's line
      setPoints(ref1, [j.shoulder, j.shoulder.clone().addScaledVector(X, 0.13)]);
      const a1 = arcPoints(j.shoulder, X, dir1, 0.085);
      setPoints(arc1, a1.pts);
      t1.position.copy(j.shoulder.clone().addScaledVector(X, Math.cos(a1.mid) * 0.12).addScaledVector(Z, Math.sin(a1.mid) * 0.12));
      setPoints(ref2, [j.elbow, j.elbow.clone().addScaledVector(dir1, 0.11)]);
      const a2 = arcPoints(j.elbow, dir1, dir2, 0.065);
      setPoints(arc2, a2.pts);
      t2.position.copy(j.elbow.clone().addScaledVector(X, Math.cos(a2.mid) * 0.1).addScaledVector(Z, Math.sin(a2.mid) * 0.1));
      // triangle: the two links and the distance to the target
      const tgt = at(tx, tz);
      setPoints(sideD, [j.shoulder, tgt]);
      lab1.position.copy(j.shoulder.clone().lerp(j.elbow, 0.5).addScaledVector(Z, 0.035));
      lab2.position.copy(j.elbow.clone().lerp(j.hand, 0.5).addScaledVector(Z, 0.035));
      labD.position.copy(j.shoulder.clone().lerp(tgt, 0.5).addScaledVector(Z, -0.03));
      // the other solution, from the server's numbers
      const oe = state.live?.other_elbow;
      const oh = state.live?.other_hand;
      if (oe && oh) setPoints(other, [j.shoulder, at(oe[0], oe[1]), at(oh[0], oh[1])]);
      // iterative IK: the error the next step will shrink
      const gap = tgt.clone().sub(j.hand);
      const len = gap.length();
      error.visible = error.visible && len > 0.012;
      if (len > 0.012) {
        const end = j.hand.clone().addScaledVector(gap, (len - 0.03) / len);
        setPoints(errLine, [j.hand, end]);
        errHead.position.copy(end.clone().addScaledVector(gap, 0.0175 / len));
        errHead.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), gap.clone().normalize());
      }
    },
  };
}


// A rover with sensors: lidar rays coloured by distance, the bumper lighting up on contact, the measured
// acceleration as an arrow, and the camera's view cone.
function roverRig(rig) {
  const group = new THREE.Group();
  const amber = api.color("--amber");
  const muted = api.color("--muted");
  const blue = api.color("--plot-blue");
  const bad = api.color("--bad");
  const ink = api.color("--ink");
  const n = rig.rays.length;
  const rayGeo = new THREE.BufferGeometry();
  rayGeo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(n * 6), 3));
  rayGeo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(n * 6), 3));
  const rayLines = new THREE.LineSegments(rayGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85, depthTest: true }));
  const hitGeo = new THREE.BufferGeometry();
  hitGeo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
  hitGeo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
  const hits = new THREE.Points(hitGeo, new THREE.PointsMaterial({ size: 0.035, vertexColors: true, transparent: true, depthTest: true }));
  const rays = new THREE.Group();
  rays.add(rayLines, hits);
  const bumperGlow = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.18, 0.05), basic(bad, 0.55));
  const bumper = new THREE.Group();
  bumper.add(bumperGlow);
  const imuArrow = arrow(new THREE.Vector3(1, 0, 0), 0.2, blue);
  const imuLabel = label("a", blue);
  imuLabel.position.set(0.22, 0, 0.03);
  imuArrow.add(imuLabel);
  const imu = new THREE.Group();
  imu.add(imuArrow);
  const cone = new THREE.LineSegments(new THREE.BufferGeometry(), lineMat(ink, 0.55));
  const camera = new THREE.Group();
  camera.add(cone);
  const parts = { rays, bumper, imu, camera };
  group.add(...Object.values(parts));

  const lidar = new THREE.Vector3().fromArray(rig.lidar);
  const camPos = new THREE.Vector3().fromArray(rig.camera.pos);
  const camQuat = new THREE.Quaternion(rig.camera.quat[1], rig.camera.quat[2], rig.camera.quat[3], rig.camera.quat[0]);
  const near = new THREE.Color().copy(bad);
  const far = new THREE.Color().copy(muted);
  return {
    group,
    parts,
    camera: { pos: camPos, quat: camQuat, fovy: rig.camera.fovy, body: rig.body },
    frame() {
      const node = api.bodyNode(rig.body);
      if (!node) return;
      const ranges = state.live?.ranges;
      if (ranges && rays.visible) {
        const origin = lidar.clone().applyQuaternion(node.quaternion).add(node.position);
        const pos = rayGeo.attributes.position;
        const col = rayGeo.attributes.color;
        const hp = hitGeo.attributes.position;
        const hc = hitGeo.attributes.color;
        rig.rays.forEach((angle, i) => {
          const dir = new THREE.Vector3(Math.cos(angle), Math.sin(angle), 0).applyQuaternion(node.quaternion);
          const r = ranges[i];
          const len = r >= 0 ? r : 4;
          const end = origin.clone().addScaledVector(dir, len);
          pos.setXYZ(2 * i, origin.x, origin.y, origin.z);
          pos.setXYZ(2 * i + 1, end.x, end.y, end.z);
          const c = new THREE.Color().lerpColors(near, far, Math.min(1, Math.max(0, (len - 0.2) / 1.6)));
          col.setXYZ(2 * i, c.r, c.g, c.b);
          col.setXYZ(2 * i + 1, c.r, c.g, c.b);
          hp.setXYZ(i, end.x, end.y, end.z);
          const h = r >= 0 ? c : new THREE.Color(0, 0, 0);
          hc.setXYZ(i, h.r, h.g, h.b);
        });
        pos.needsUpdate = col.needsUpdate = hp.needsUpdate = hc.needsUpdate = true;
        rayGeo.computeBoundingSphere();
        hitGeo.computeBoundingSphere();
      }
      bumperGlow.position.copy(new THREE.Vector3().fromArray(rig.bumper).applyQuaternion(node.quaternion).add(node.position));
      bumperGlow.quaternion.copy(node.quaternion);
      bumperGlow.visible = (state.live?.touch ?? 0) > 0.05;
      // the accelerometer, minus gravity, in the floor's plane
      const a = state.live?.accel;
      if (a) {
        const world = new THREE.Vector3(a[0], a[1], 0).applyQuaternion(node.quaternion);
        world.z = 0;
        const mag = world.length();
        imuArrow.visible = mag > 0.15;
        imuArrow.position.copy(node.position).add(new THREE.Vector3(0, 0, 0.16));
        imuArrow.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), world.clone().normalize());
        imuArrow.scale.setScalar(Math.min(2.5, 0.4 + mag * 0.35));
      }
      if (camera.visible) {
        const eye = camPos.clone().applyQuaternion(node.quaternion).add(node.position);
        const q = node.quaternion.clone().multiply(camQuat);
        const h = Math.tan(THREE.MathUtils.degToRad(rig.camera.fovy / 2)) * 0.45;
        const w = h * 1.6;
        const corners = [[-w, -h], [w, -h], [w, h], [-w, h]].map(([x, y]) => new THREE.Vector3(x, y, -0.45).applyQuaternion(q).add(eye));
        const pts = [];
        for (let k = 0; k < 4; k++) pts.push(eye, corners[k], corners[k], corners[(k + 1) % 4]);
        setPoints(cone, pts);
      }
    },
  };
}

// What the robot's camera sees: the scene drawn again from the camera on the rover, a dozen times a second.
const pip = {
  wrap: null,
  canvas: null,
  ctx: null,
  target: null,
  cam: null,
  pixels: null,
  image: null,
  last: 0,
  setup(rig) {
    this.teardown();
    if (!rig?.camera) return;
    const w = 320;
    const h = 200;
    this.wrap = document.createElement("section");
    this.wrap.className = "plot pip";
    const head = document.createElement("div");
    head.className = "plot-head";
    const title = document.createElement("span");
    title.className = "plot-title";
    title.textContent = t("robot_camera");
    head.append(title);
    this.canvas = document.createElement("canvas");
    this.canvas.width = w;
    this.canvas.height = h;
    this.ctx = this.canvas.getContext("2d");
    this.wrap.append(head, this.canvas);
    this.wrap.hidden = true;
    $("lesson-plots").append(this.wrap);
    this.target = new THREE.WebGLRenderTarget(w, h);
    this.target.texture.colorSpace = THREE.SRGBColorSpace;
    this.cam = new THREE.PerspectiveCamera(rig.camera.fovy, w / h, 0.02, 50);
    this.pixels = new Uint8Array(w * h * 4);
    this.image = this.ctx.createImageData(w, h);
  },
  teardown() {
    this.wrap?.remove();
    this.target?.dispose();
    this.wrap = this.canvas = this.ctx = this.target = this.cam = null;
  },
  frame() {
    const rigCam = overlays.current?.camera;
    if (!this.cam || !rigCam || !state) return;
    const show = state.shown.includes("camera");
    this.wrap.hidden = !show;
    if (!show) return;
    const now = performance.now();
    if (now - this.last < 80) return;
    this.last = now;
    const node = api.bodyNode(rigCam.body);
    if (!node) return;
    this.cam.position.copy(rigCam.pos.clone().applyQuaternion(node.quaternion).add(node.position));
    this.cam.quaternion.copy(node.quaternion.clone().multiply(rigCam.quat));
    this.cam.updateMatrixWorld();
    const renderer = api.renderer;
    const helpers = overlays.current.group;
    helpers.visible = false; // the robot does not see our drawings
    renderer.setRenderTarget(this.target);
    renderer.render(api.scene, this.cam);
    renderer.setRenderTarget(null);
    helpers.visible = true;
    const { width: w, height: h } = this.target;
    renderer.readRenderTargetPixels(this.target, 0, 0, w, h, this.pixels);
    const rows = this.image.data;
    for (let y = 0; y < h; y++) rows.set(this.pixels.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
    this.ctx.putImageData(this.image, 0, 0);
  },
};

// Robots side by side: a name tag over each, its workspace as a cloud of reachable points, and the
// targets of the running test (the one each robot is on drawn larger, done ones faded).
function faceoffRig(rig) {
  const group = new THREE.Group();
  const labels = new THREE.Group();
  const workspace = new THREE.Group();
  const targets = new THREE.Group();
  const amber = api.color("--amber");
  const balls = [];
  const tags = [];
  rig.entrants.forEach((e, i) => {
    const color = api.color(COLORS[e.color]);
    const sub = e.sub === null ? "" : e.sub ? pick(e.sub) : t(e.real ? "real_robot" : "our_template");
    const tag = nameTag(pick(e.label), sub, color);
    tag.position.set(e.base[0], e.base[1], e.top + 0.12);
    labels.add(tag);
    tags.push([tag, e]);
    if (e.cloud.length) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(e.cloud.flat(), 3));
      const material = new THREE.PointsMaterial({
        color,
        size: 0.024,
        map: dotTexture(),
        alphaTest: 0.05,
        transparent: true,
        opacity: 0.7,
        depthWrite: false,
        toneMapped: false, // keep the robot's colour, as in the table and the plot
      });
      workspace.add(new THREE.Points(geometry, material));
    }
    balls.push(
      (rig.targets[i] ?? []).map((p) => {
        const ball = new THREE.Mesh(new THREE.SphereGeometry(0.014, 18, 10), basic(amber, 0.8));
        ball.position.fromArray(p);
        targets.add(ball);
        return ball;
      })
    );
  });
  const parts = { labels, workspace, targets };
  group.add(labels, workspace, targets);
  return {
    group,
    parts,
    frame() {
      for (const [tag, e] of tags) {
        const node = e.body !== undefined ? api.bodyNode(e.body) : null;
        if (node) tag.position.set(node.position.x, node.position.y, e.top + 0.12); // walking robots take theirs along
      }
      workspace.visible = (state.values.workspace ?? 0) >= 1;
      const active = state.live?.active ?? [];
      balls.forEach((row, i) =>
        row.forEach((ball, k) => {
          const now = active[i] === k;
          ball.scale.setScalar(now ? 1.7 : 1);
          ball.material.opacity = now ? 1 : active[i] > k ? 0.25 : 0.8;
        })
      );
    },
  };
}

// A name tag that faces the camera: a coloured dot, the name and a small second line, on a pill.
function nameTag(name, sub, color) {
  const px = 2; // canvas pixels per CSS pixel, for crisp text
  const canvas = document.createElement("canvas");
  const g = canvas.getContext("2d");
  const font = (weight, size) => `${weight} ${size * px}px -apple-system, "Segoe UI", "PingFang SC", sans-serif`;
  g.font = font(650, 15);
  const nameWidth = g.measureText(name).width;
  g.font = font(500, 11);
  const subWidth = g.measureText(sub).width;
  const pad = 11 * px;
  const dot = 8 * px;
  const width = Math.ceil(pad * 2 + dot + 7 * px + Math.max(nameWidth, subWidth));
  const height = (sub ? 44 : 30) * px; // one line when there is no subtitle
  canvas.width = width;
  canvas.height = height;
  g.fillStyle = `#${api.color("--panel-solid").getHexString()}`;
  g.globalAlpha = 0.92;
  g.beginPath();
  g.roundRect(0, 0, width, height, height / 2);
  g.fill();
  g.globalAlpha = 1;
  g.fillStyle = `#${color.getHexString()}`;
  g.beginPath();
  g.arc(pad + dot / 2, height / 2, dot / 2, 0, Math.PI * 2);
  g.fill();
  const x = pad + dot + 7 * px;
  g.textBaseline = "alphabetic";
  g.fillStyle = `#${api.color("--ink").getHexString()}`;
  g.font = font(650, 15);
  g.fillText(name, x, (sub ? 20 : 20.5) * px);
  if (sub) {
    g.fillStyle = `#${api.color("--muted").getHexString()}`;
    g.font = font(500, 11);
    g.fillText(sub, x, 35 * px);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false, sizeAttenuation: false, toneMapped: false })
  );
  const tall = (0.034 * height) / (44 * px); // of the view's height (sizeAttenuation off): ~30 px for two lines
  sprite.scale.set((tall * width) / height, tall, 1);
  sprite.renderOrder = 10;
  return sprite;
}

// A round point for THREE.Points (a fresh one per overlay: clearing an overlay disposes its textures).
function dotTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 32;
  const g = c.getContext("2d");
  g.fillStyle = "#fff";
  g.beginPath();
  g.arc(16, 16, 14, 0, Math.PI * 2);
  g.fill();
  return new THREE.CanvasTexture(c);
}

// Six-joint arms in 3D: the draggable target, the line from hand to target, and the manipulability
// ellipsoid at the hand (its axes are the directions the hand moves easily, its radii how easily).
function arm3dRig(rig) {
  const tags = faceoffRig(rig); // the name tags over the arms
  const amber = api.color("--amber");
  const blue = api.color("--plot-blue");
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.03, 24, 12), basic(amber, 0.95));
  const halo = new THREE.Mesh(new THREE.SphereGeometry(0.05, 24, 12), basic(amber, 0.18));
  const target = new THREE.Group();
  target.add(ball, halo);
  const error = dashedLine(amber, 0.9);
  const shell = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), basic(blue, 0.22, true));
  const rim = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.SphereGeometry(1, 12, 6)), lineMat(blue, 0.35));
  const ellipsoid = new THREE.Group();
  ellipsoid.add(shell, rim);
  ellipsoid.matrixAutoUpdate = false;
  const parts = { ...tags.parts, target, error, ellipsoid };
  tags.group.add(target, error, ellipsoid);
  const round = (v) => Number(v.toFixed(3));
  return {
    group: tags.group,
    parts,
    dragTarget: {
      mesh: ball,
      // Back to the sliders: forward and sideways from the active arm's base, height from the floor.
      toControls(hit) {
        const base = rig.bases[state.live?.arm ?? "ours"];
        return { tx: round(hit.x - base[0]), ty: round(hit.y - base[1]), tz: round(hit.z) };
      },
    },
    frame() {
      tags.frame();
      const live = state.live;
      if (!live?.target) return;
      const goal = new THREE.Vector3().fromArray(live.target);
      const hand = new THREE.Vector3().fromArray(live.hand);
      target.position.copy(goal);
      setPoints(error, [hand, goal]);
      const [a, b, c] = [0, 1, 2].map((k) => new THREE.Vector3(...live.ellipsoid.axes.map((row) => row[k])));
      const [ra, rb, rc] = live.ellipsoid.radii.map((r) => Math.max(r, 0.004));
      ellipsoid.matrix.makeBasis(a.multiplyScalar(ra), b.multiplyScalar(rb), c.multiplyScalar(rc)).setPosition(hand);
    },
  };
}

// Robot vacuums: the floor each one has covered, painted square by square in its colour.
function cleanerRig(rig) {
  const tags = faceoffRig(rig);
  const floor = new THREE.Group();
  const painted = {};
  for (const e of rig.entrants) {
    const material = basic(api.color(COLORS[e.color]), 0.38, true);
    Object.assign(material, { polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 });
    const tile = new THREE.PlaneGeometry(rig.cell * 0.92, rig.cell * 0.92);
    const mesh = new THREE.InstancedMesh(tile, material, rig.cells * rig.cells);
    mesh.count = 0;
    floor.add(mesh);
    painted[e.name] = { mesh, shown: -1, trial: -1, origin: rig.arenas[e.name] };
  }
  const parts = { ...tags.parts, floor };
  tags.group.add(floor);
  const place = new THREE.Matrix4();
  return {
    group: tags.group,
    parts,
    frame() {
      tags.frame();
      const cells = state.live?.cells ?? {};
      const trial = state.live?.trial ?? 0;
      for (const [name, p] of Object.entries(painted)) {
        const list = cells[name] ?? [];
        if (list.length === p.shown && trial === p.trial) continue; // nothing new to paint
        list.forEach(([ix, iy], k) => {
          place.makeTranslation(p.origin[0] + (ix + 0.5) * rig.cell, p.origin[1] + (iy + 0.5) * rig.cell, 0.004);
          p.mesh.setMatrixAt(k, place);
        });
        p.mesh.count = list.length;
        p.mesh.instanceMatrix.needsUpdate = true;
        [p.shown, p.trial] = [list.length, trial];
      }
    },
  };
}

// Walking dogs: name tags that follow them, and under the dog being tested the support (the line or
// patch between the feet on the ground) with the centre of mass dropped onto the floor.
function gaitRig(rig) {
  const tags = faceoffRig(rig);
  const support = new THREE.Group();
  const violet = api.color("--plot-violet");
  const amber = api.color("--amber");
  const onTop = (material) => Object.assign(material, { depthTest: false }); // seen through the dog's body
  const patch = new THREE.Mesh(new THREE.BufferGeometry(), onTop(basic(violet, 0.32, true)));
  const feet = new THREE.Group();
  const footDisc = new THREE.CircleGeometry(0.022, 20);
  const footMat = onTop(basic(violet, 0.9, true));
  for (let k = 0; k < 4; k++) feet.add(new THREE.Mesh(footDisc, footMat));
  const com = new THREE.Mesh(new THREE.CircleGeometry(0.03, 28), onTop(basic(amber, 1, true)));
  const drop = new THREE.Line(new THREE.BufferGeometry(), lineMat(amber, 0.7));
  // The capture point (balance lesson): a hollow ring, joined to the centre of mass.
  const capture = new THREE.Mesh(new THREE.RingGeometry(0.03, 0.042, 32), onTop(basic(amber, 1, true)));
  const reach = new THREE.Line(new THREE.BufferGeometry(), lineMat(amber, 0.85));
  support.add(patch, feet, com, drop, capture, reach);
  support.traverse((o) => {
    o.renderOrder = 5; // after the robots, so the overlay draws over their legs
  });
  const z = 0.004; // just above the floor
  const parts = { ...tags.parts, support };
  tags.group.add(support);
  const WIDTH = 0.03; // two feet make a line: drawn as a strip this wide
  return {
    group: tags.group,
    parts,
    frame() {
      tags.frame();
      const pts = (state.live?.support ?? []).map(([px, py]) => new THREE.Vector3(px, py, z));
      feet.children.forEach((disc, k) => {
        disc.visible = k < pts.length;
        if (disc.visible) disc.position.copy(pts[k]);
      });
      const tri = [];
      if (pts.length === 2) {
        const along = pts[1].clone().sub(pts[0]);
        const side = new THREE.Vector3(-along.y, along.x, 0).normalize().multiplyScalar(WIDTH / 2);
        const [a, b] = pts;
        tri.push(a.clone().add(side), b.clone().add(side), b.clone().sub(side), a.clone().add(side), b.clone().sub(side), a.clone().sub(side));
      }
      for (let k = 1; k + 1 < pts.length; k++) tri.push(pts[0], pts[k], pts[k + 1]);
      patch.visible = tri.length > 0;
      if (tri.length) setPoints(patch, tri);
      const c = state.live?.com;
      com.visible = drop.visible = !!c;
      if (c) {
        com.position.set(c[0], c[1], z + 0.001);
        const trunk = api.bodyNode(state.live?.follow); // the robot being tested
        const top = trunk ? trunk.position.z : 0.3;
        setPoints(drop, [new THREE.Vector3(c[0], c[1], top), new THREE.Vector3(c[0], c[1], z)]);
      }
      const cp = state.live?.capture;
      capture.visible = reach.visible = !!(cp && c);
      if (cp && c) {
        capture.position.set(cp[0], cp[1], z + 0.002);
        setPoints(reach, [new THREE.Vector3(c[0], c[1], z + 0.002), new THREE.Vector3(cp[0], cp[1], z + 0.002)]);
      }
    },
  };
}

const RIGS = {
  gait: gaitRig,
  joint: jointRig,
  planar_arm: planarArmRig,
  rover: roverRig,
  faceoff: faceoffRig,
  arm3d: arm3dRig,
  cleaner: cleanerRig,
};

function lineMat(color, opacity) {
  return new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthTest: false });
}

function dashedLine(color, opacity) {
  const line = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: 0.018, gapSize: 0.012, depthTest: false })
  );
  line.userData.dashed = true;
  return line;
}

function setPoints(obj, points) {
  obj.geometry.dispose();
  obj.geometry = new THREE.BufferGeometry().setFromPoints(points);
  if (obj.userData.dashed) obj.computeLineDistances();
}

// ---------------------------------------------------------------- dragging a lesson's target

let targetDrag = null; // { last send time }

function startTargetDrag(e) {
  const t = overlays.target();
  if (!t || !lesson || !state || !state.unlocked.includes("tx") || !overlays.current.parts.target.visible) return;
  if (e.target !== api.canvas || e.button !== 0) return;
  const ray = api.rayAt(e.clientX, e.clientY);
  if (!ray.intersectObject(t.mesh, false).length) return;
  e.stopPropagation(); // not a camera orbit, and not a body drag
  e.preventDefault();
  if (t.toControls) {
    // A target in 3D slides on the plane facing the camera through it, where the pointer can follow it.
    const at = t.mesh.getWorldPosition(new THREE.Vector3());
    t.plane = new THREE.Plane().setFromNormalAndCoplanarPoint(ray.ray.direction.clone().negate(), at);
  }
  targetDrag = { sent: 0 };
  api.canvas.classList.add("dragging");
}

function moveTargetDrag(e) {
  if (!targetDrag) return;
  const t = overlays.target();
  const hit = new THREE.Vector3();
  if (!api.rayAt(e.clientX, e.clientY).ray.intersectPlane(t.plane, hit)) return;
  if (t.toControls) targetDrag.at = t.toControls(hit);
  else {
    const rel = hit.sub(t.origin);
    targetDrag.at = { tx: Number(rel.dot(t.X).toFixed(3)), tz: Number(rel.dot(t.Z).toFixed(3)) };
  }
  const now = performance.now();
  if (now - targetDrag.sent < 33) return; // about 30 updates a second is plenty
  targetDrag.sent = now;
  sendTarget(targetDrag.at);
}

function sendTarget(values) {
  for (const [control, value] of Object.entries(values)) api.send({ type: "lesson", action: "set", control, value });
}

function endTargetDrag() {
  if (!targetDrag) return;
  if (targetDrag.at) sendTarget(targetDrag.at); // the last move may have been skipped by the throttle
  targetDrag = null;
  api.canvas.classList.remove("dragging");
}

window.addEventListener("pointerdown", startTargetDrag, { capture: true });
window.addEventListener("pointermove", moveTargetDrag);
window.addEventListener("pointerup", endTargetDrag);
window.addEventListener("pointercancel", endTargetDrag);

function basic(color, opacity, doubleSided = false) {
  return new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
    side: doubleSided ? THREE.DoubleSide : THREE.FrontSide,
  });
}

function arrow(dir, length, color) {
  const g = new THREE.Group();
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, length - 0.04, 10), basic(color, 0.95));
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.018, 0.04, 16), basic(color, 0.95));
  const up = new THREE.Vector3(0, 1, 0);
  const q = new THREE.Quaternion().setFromUnitVectors(up, dir.clone().normalize());
  shaft.quaternion.copy(q);
  head.quaternion.copy(q);
  shaft.position.copy(dir.clone().multiplyScalar((length - 0.04) / 2));
  head.position.copy(dir.clone().multiplyScalar(length - 0.02));
  g.add(shaft, head);
  return g;
}

function label(text, color) {
  const c = document.createElement("canvas");
  c.width = 128;
  c.height = 48;
  const g = c.getContext("2d");
  g.font = "600 30px -apple-system, 'Segoe UI', sans-serif";
  g.fillStyle = `#${color.getHexString()}`;
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, 64, 24);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sprite.scale.set(0.1, 0.0375, 1);
  return sprite;
}

// Called every animation frame by the main viewer.
export function lessonFrame() {
  overlays.frame();
  plots.draw();
  pip.frame();
}
