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
  list.replaceChildren(
    ...catalog.map((c) => {
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
    })
  );
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
  const stars = state?.results?.test?.stars ?? 0;
  saveProgress(lesson.id, { done: true, stars: Math.max(stars, progress()[lesson.id]?.stars ?? 0) });
  $("lesson-finished").hidden = false;
  $("lesson-finished-body").textContent = t("lesson_done_body", { title: pick(lesson.title) });
}

// ---------------------------------------------------------------- running a lesson

function openLesson(desc) {
  lesson = desc;
  state = null;
  plots.reset(desc.plots);
  overlays.build(desc.rig);
  setPicker(false);
  document.body.classList.add("in-lesson");
  $("lesson-panel").hidden = false;
  $("lesson-plots").hidden = desc.plots.length === 0;
  $("lesson-finished").hidden = true;
  $("lesson-chapter").textContent = pick(desc.chapter);
  $("lesson-title").textContent = pick(desc.title);
  renderControls();
  renderStep();
  needCamera = true;
  if (desc.camera) api.moveCamera(desc.camera.eye, desc.camera.target);
  api.shift();
}

function closeLesson() {
  lesson = null;
  state = null;
  overlays.clear();
  document.body.classList.remove("in-lesson");
  $("lesson-panel").hidden = true;
  $("lesson-plots").hidden = true;
  api.shift();
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
  const quizChanged = JSON.stringify(state?.quiz) !== JSON.stringify(msg.quiz);
  const resultChanged = JSON.stringify(state?.results) !== JSON.stringify(msg.results);
  state = msg;
  plots.add(msg.samples);
  if (unlockChanged) renderControls();
  else syncControls();
  if (stepChanged || quizChanged || resultChanged) renderStep();
  $("lesson-readout").textContent = readout(msg.live);
}

function readout(live) {
  if (!live || live.angle === undefined) return "";
  return t("readout_joint", { angle: live.angle.toFixed(1), error: live.error.toFixed(1), torque: live.torque.toFixed(2) });
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
  next.textContent = last ? t("finish_lesson") : t("next_step");
  $("lesson-waiting").hidden = passed || step.manual || !!step.quiz;
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
  const chosen = state?.quiz?.chosen;
  $("lesson-quiz-question").textContent = pick(step.quiz.question);
  $("lesson-quiz-options").replaceChildren(
    ...step.quiz.options.map((option, k) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "quiz-option";
      b.textContent = pick(option);
      if (chosen === k) b.dataset.verdict = state.quiz.right ? "right" : "wrong";
      b.addEventListener("click", () => api.send({ type: "lesson", action: "answer", choice: k }));
      return b;
    })
  );
  const feedback = $("lesson-quiz-feedback");
  feedback.hidden = chosen === undefined;
  if (chosen !== undefined) feedback.textContent = state.quiz.right ? pick(state.quiz.explain) : t("quiz_try_again");
}

function renderResult() {
  const box = $("lesson-result");
  const test = state?.results?.test;
  box.hidden = !test;
  if (!test) return;
  box.replaceChildren();
  if (test.running) {
    box.textContent = t("test_running");
    return;
  }
  const line = document.createElement("span");
  line.textContent = test.settle_s === null
    ? t("test_not_settled", { overshoot: test.overshoot_pct })
    : t("test_result", { settle: test.settle_s, overshoot: test.overshoot_pct });
  box.append(line, starSpan(test.stars));
}

// ---------------------------------------------------------------- controls

function renderControls() {
  const box = $("lesson-controls");
  const unlocked = new Set(state ? state.unlocked : []);
  const usable = lesson.controls.filter((c) => unlocked.has(c.id));
  $("lesson-controls-wrap").hidden = usable.length === 0;
  const buttons = usable.filter((c) => c.kind === "button");
  const rows = usable.filter((c) => c.kind !== "button").map(controlRow);
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
  box.replaceChildren(...rows, ...(buttons.length ? [bar] : []));
  syncControls();
}

function controlRow(c) {
  const value = state?.values[c.id] ?? c.value;
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
    box.replaceChildren();
    this.canvases = specs.map((spec) => {
      const wrap = document.createElement("section");
      wrap.className = "plot";
      const head = document.createElement("div");
      head.className = "plot-head";
      const title = document.createElement("span");
      title.className = "plot-title";
      title.textContent = `${pick(spec.title)} (${spec.unit})`;
      const legend = document.createElement("span");
      legend.className = "plot-legend";
      for (const s of spec.series) {
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

// For a "joint" rig: a protractor around the pivot, the target as a ghost arm, the error as a
// shaded wedge, the motor's torque as a curved arrow and gravity as a straight one.
const overlays = {
  group: null,
  rig: null,
  parts: {},
  clear() {
    if (this.group) {
      this.group.traverse((o) => {
        o.geometry?.dispose();
        o.material?.map?.dispose();
        o.material?.dispose();
      });
      api.scene.remove(this.group);
    }
    this.group = null;
    this.rig = null;
    this.parts = {};
  },
  build(rig) {
    this.clear();
    if (!rig || rig.kind !== "joint") return;
    this.rig = {
      pivot: new THREE.Vector3().fromArray(rig.pivot),
      zero: new THREE.Vector3().fromArray(rig.zero).normalize(),
      swing: new THREE.Vector3().fromArray(rig.swing).normalize(),
      length: rig.length,
      body: rig.body,
      weight: new THREE.Vector3().fromArray(rig.weight),
    };
    this.group = new THREE.Group();
    this.group.renderOrder = 5;
    api.scene.add(this.group);
    const amber = api.color("--amber");
    const ink = api.color("--ink");
    this.parts.scale = this.protractor(ink);
    this.parts.target = this.ghostArm(amber);
    this.parts.error = new THREE.Mesh(new THREE.BufferGeometry(), basic(amber, 0.22, true));
    this.parts.torque = new THREE.Group();
    this.parts.torque.add(new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: ink, depthTest: false, transparent: true })));
    this.parts.torque.add(new THREE.Mesh(new THREE.ConeGeometry(0.016, 0.04, 16), basic(ink, 0.95)));
    this.parts.gravity = arrow(new THREE.Vector3(0, 0, -1), 0.16, api.color("--plot-blue"));
    this.parts.gravity.add(label("mg", api.color("--plot-blue")));
    this.parts.gravity.children[this.parts.gravity.children.length - 1].position.set(0.035, 0, -0.1);
    for (const p of Object.values(this.parts)) {
      p.visible = false;
      this.group.add(p);
    }
  },
  point(angleDeg, radius) {
    const a = THREE.MathUtils.degToRad(angleDeg);
    return this.rig.pivot.clone()
      .addScaledVector(this.rig.zero, Math.cos(a) * radius)
      .addScaledVector(this.rig.swing, Math.sin(a) * radius);
  },
  protractor(color) {
    const g = new THREE.Group();
    const r = this.rig.length * 1.22;
    const pts = [];
    for (let deg = -90; deg <= 180; deg += 15) {
      const long = deg % 45 === 0;
      pts.push(this.point(deg, r), this.point(deg, r - (long ? 0.04 : 0.02)));
      if (long) {
        const tag = label(`${deg}°`, color);
        tag.position.copy(this.point(deg, r + 0.045));
        g.add(tag);
      }
    }
    const ticks = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }));
    const arc = [];
    for (let deg = -90; deg <= 180; deg += 3) arc.push(this.point(deg, r));
    const rim = new THREE.Line(new THREE.BufferGeometry().setFromPoints(arc), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 }));
    g.add(ticks, rim);
    return g;
  },
  ghostArm(color) {
    const g = new THREE.Group();
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 12), basic(color, 0.45));
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.045, 24, 12), basic(color, 0.35));
    g.add(rod, ball);
    g.userData = { rod, ball };
    return g;
  },
  angleNow() {
    const node = api.bodyNode(this.rig.body);
    if (!node) return 0;
    const dir = this.rig.zero.clone().applyQuaternion(node.quaternion);
    return THREE.MathUtils.radToDeg(Math.atan2(dir.dot(this.rig.swing), dir.dot(this.rig.zero)));
  },
  frame() {
    if (!this.rig || !state) return;
    const shown = new Set(state.shown);
    for (const [name, part] of Object.entries(this.parts)) part.visible = shown.has(name);
    const now = this.angleNow();
    const target = state.values.target ?? 0;
    const L = this.rig.length;
    // ghost arm at the target angle
    const { rod, ball } = this.parts.target.userData;
    const tip = this.point(target, L);
    rod.position.copy(this.rig.pivot.clone().lerp(tip, 0.5));
    rod.scale.set(1, L, 1);
    rod.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tip.clone().sub(this.rig.pivot).normalize());
    ball.position.copy(tip);
    // the error: a wedge from where the arm is to where it should be
    const wedge = [];
    const steps = Math.max(2, Math.ceil(Math.abs(target - now) / 3));
    for (let k = 0; k <= steps; k++) {
      const a = now + ((target - now) * k) / steps;
      if (k > 0) wedge.push(this.rig.pivot, this.point(a - (target - now) / steps, L * 0.82), this.point(a, L * 0.82));
    }
    this.parts.error.geometry.dispose();
    this.parts.error.geometry = new THREE.BufferGeometry().setFromPoints(wedge.map((p) => p.clone()));
    // torque: a curved arrow around the pivot, as long as the push is strong
    const torque = state.live?.torque ?? 0;
    const max = state.live?.torque_max ?? 1;
    const sweep = (torque / max) * 140;
    const [line, head] = this.parts.torque.children;
    const arc = [];
    for (let k = 0; k <= 24; k++) arc.push(this.point(now + (sweep * k) / 24, 0.11));
    line.geometry.dispose();
    line.geometry = new THREE.BufferGeometry().setFromPoints(arc);
    const end = arc[arc.length - 1];
    const before = arc[arc.length - 2];
    head.position.copy(end);
    head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), end.clone().sub(before).normalize());
    head.visible = Math.abs(sweep) > 4;
    // gravity on the weight
    const node = api.bodyNode(this.rig.body);
    if (node) this.parts.gravity.position.copy(this.rig.weight.clone().applyQuaternion(node.quaternion).add(node.position));
  },
};

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
  sprite.scale.set(0.08, 0.03, 1);
  return sprite;
}

// Called every animation frame by the main viewer.
export function lessonFrame() {
  overlays.frame();
  plots.draw();
}
