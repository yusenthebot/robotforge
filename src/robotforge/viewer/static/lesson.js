// Lessons in the viewer: the course picker, the step-by-step panel, live plots and the overlays that
// draw what the physics is doing (angles, errors, torques, forces). Checks and grading live in Python;
// this module only shows the lesson's state and sends the learner's input.
import * as THREE from "three";
import { LANG, t } from "./i18n.js";

const $ = (id) => document.getElementById(id);
const pick = (text) => (text ? (text[LANG] ?? text.en ?? "") : "");
// A title as shown: a hyphenated word stays on one line ("six-joint" broke as "six- / joint arms").
const titleText = (text) => pick(text).replace(/(\p{L})-(?=\p{L})/gu, "$1\u2011");
// A value and its unit with a space between, as lessons write " m" ("2.00cm" and "0.10m/s" sat beside "0.25 m");
// degrees and percent stay on the number.
const withUnit = (value, unit) => (!unit || /^[\s°%]/.test(unit) ? `${value}${unit}` : `${value}\u00a0${unit}`);

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
  $("lesson-fold").addEventListener("click", () => foldLesson(!$("lesson-panel").classList.contains("folded")));
  $("lesson-finished-exit").addEventListener("click", () => exitLesson());
  $("lesson-finished-stay").addEventListener("click", () => ($("lesson-finished").hidden = true));
  $("lesson-next").addEventListener("click", () => {
    const last = lesson && state && state.step === lesson.steps.length - 1;
    if (last && passedAll()) finishLesson();
    else api.send({ type: "lesson", action: "next" });
  });
}

export const lessonActive = () => lesson !== null;
// Where a lesson aims the camera at its rig, if it does: the viewer's whole-scene framing must not override it.
export const lessonCamera = () => lesson?.camera ?? null;
export const lessonTitle = () => (lesson ? pick(lesson.title) : null);
export const lessonDriveSpec = () => lesson?.drive ?? null;
let followWanted = null; // the body the lesson last asked the camera to follow
let viewWanted = null; // the framing the lesson last asked for, as JSON
export const lessonState = () => (lesson ? { id: lesson.id, ...state } : null); // for scripted checks
// Each live plot's y axis as last drawn, for scripted checks: [lo, hi] in the plot's unit (null before its first draw).
export const lessonPlotAxes = () => plots.specs.map((s, k) => ({ id: s.id, unit: s.unit, range: plots.ranges[k] }));

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
  b.dataset.lesson = c.id;
  const chapter = document.createElement("span");
  chapter.className = "course-chapter";
  chapter.textContent = pick(c.chapter);
  const title = document.createElement("span");
  title.className = "course-title";
  title.textContent = titleText(c.title);
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
  const box = $("lesson-finished");
  box.hidden = false;
  $("lesson-finished-body").textContent = t("lesson_done_body", { title: pick(lesson.title) });
  // It sits at the foot of the panel, under the live numbers: below the fold the Finish button seemed to do nothing.
  const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  box.scrollIntoView({ block: "nearest", behavior: smooth ? "smooth" : "auto" });
}

// ---------------------------------------------------------------- running a lesson

function openLesson(desc) {
  lesson = desc;
  state = null;
  followWanted = viewWanted = null;
  api.followBody(null);
  plots.reset(desc.plots);
  bench.setup(desc.bench, desc.bench_state);
  plotTabs.pick = 0; // a new lesson starts on its first plot
  plotTabs.sync();
  stepShown = -1;
  renderTable(null);
  overlays.build(desc.rig);
  setPicker(false);
  document.body.classList.add("in-lesson");
  $("lesson-panel").hidden = false;
  $("lesson-plots").hidden = desc.plots.length === 0;
  $("lesson-finished").hidden = true;
  $("lesson-chapter").textContent = pick(desc.chapter);
  $("lesson-title").textContent = titleText(desc.title);
  $("lesson-drive").hidden = !desc.drive;
  // A lesson may say what its keys move ("Move the gripper"); otherwise they drive its robot.
  $("lesson-drive").querySelector("[data-i18n]").textContent = desc.drive?.label ? pick(desc.drive.label) : t("lesson_drive");
  pip.setup(desc.rig);
  camview.setup(desc.rig);
  renderControls();
  renderStep();
  needCamera = true;
  if (desc.camera) api.moveCamera(desc.camera.eye, desc.camera.target, desc.camera.fit);
  foldLesson(false); // a new lesson opens unfolded (and shifts the view for its panels)
  api.retitle();
}

function closeLesson() {
  lesson = null;
  state = null;
  early = null;
  followWanted = viewWanted = null;
  api.followBody(null);
  overlays.clear();
  pip.teardown();
  camview.teardown();
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
  followWanted = null; // a new scene drops the camera's follow: ask again with the next update
  renderPicker();
  $("learn-toggle").hidden = catalog.length === 0;
  if (desc.lesson && (!lesson || lesson.id !== desc.lesson.id)) openLesson(desc.lesson);
  else if (!desc.lesson && lesson) closeLesson();
  else if (lesson) overlays.build(lesson.rig); // the scene was rebuilt; overlays hang on new bodies
  // A new scene resets the camera's motion, so aim it at the lesson's rig once that scene is here.
  if (lesson && desc.lesson && needCamera) {
    needCamera = false;
    if (lesson.camera) api.moveCamera(lesson.camera.eye, lesson.camera.target, lesson.camera.fit);
  }
}

// What a lesson sends only when it changes (its drawing, its camera's pictures), from updates that came before the
// lesson's scene had loaded here: the motion planning lesson's first drawing was lost, so its target never showed.
let early = null; // { lesson, drawing, pictures }

export function lessonUpdate(msg) {
  if (msg.ended) {
    closeLesson();
    return;
  }
  if (!lesson || msg.lesson !== lesson.id) {
    if (msg.lesson) {
      const kept = early?.lesson === msg.lesson ? early : null;
      early = {
        lesson: msg.lesson,
        drawing: msg.live?.drawing ?? kept?.drawing ?? null,
        pictures: msg.live?.vision?.pictures ?? kept?.pictures ?? null,
      };
    }
    return;
  }
  const before = early?.lesson === msg.lesson ? early : null;
  early = null;
  const stepChanged = !state || state.step !== msg.step || state.passed.length !== msg.passed.length;
  const unlockChanged = !state || state.unlocked.join() !== msg.unlocked.join();
  if (msg.bench) bench.update(msg.bench);
  const quizChanged = JSON.stringify(state?.quiz) !== JSON.stringify(msg.quiz);
  const resultChanged = JSON.stringify(state?.results) !== JSON.stringify(msg.results);
  // A drawing comes with one update only (when it changes): keep it, or a slow frame may miss it.
  const drawing = msg.live?.drawing ?? state?.drawing ?? before?.drawing ?? null;
  const pictures = msg.live?.vision?.pictures ?? state?.pictures ?? before?.pictures ?? null; // a lesson camera's, likewise
  state = msg;
  state.drawing = drawing;
  state.pictures = pictures;
  plots.add(msg.samples);
  if (unlockChanged) {
    renderControls();
    bench.syncLocks();
  } else syncControls();
  if (stepChanged || quizChanged || resultChanged) renderStep();
  showReadout(readout(msg.live));
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
  renderFormula(msg.live?.formula ?? null, msg.live?.formula_title ?? null);
  renderTable(msg.live?.table ?? null);
  camview.update();
}

// The controller's sum with live numbers, one row per term; a row appears once its step shows it. A lesson whose
// rows are not a controller's says what they are instead (the imitation lesson: a pair being recorded).
function renderFormula(rows, title) {
  const box = $("lesson-formula");
  const shown = new Set(state?.shown ?? []);
  const visible = (rows ?? []).filter((r) => shown.has(r.show));
  box.hidden = visible.length === 0;
  if (box.hidden) return;
  const text = (x) => (typeof x === "string" ? x : pick(x));
  const heading = title ? text(title) : t("formula_title");
  const head = box.querySelector("[data-i18n]");
  if (head.textContent !== heading) head.textContent = heading;
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
    value.textContent = text(r.value);
    note.textContent = r.note ? text(r.note) : "";
    note.hidden = !r.note;
  });
}

// The readout keeps its line breaks and wraps a line too long for the panel. A line of items ("odometry 0.00 m
// ahead 2.08 m", "angle 3.0° · target 5°") wraps between items only: each item is one unbreakable piece, so
// neither a label and its value nor the characters of a Chinese label are split. A plain sentence wraps as text.
let readoutShown = null;
function showReadout(text) {
  if (text === readoutShown) return;
  readoutShown = text;
  const el = $("lesson-readout");
  el.replaceChildren();
  text.split("\n").forEach((line, k) => {
    if (k) el.append("\n");
    const parts = line.split(/( {2,}| · )/);
    if (parts.length < 3) {
      el.append(line);
      return;
    }
    parts.forEach((part, i) => {
      if (i % 2) {
        el.append(part.replace(/^ /, "\u00a0")); // the dot stays with the item before it; the line breaks after
        return;
      }
      const item = document.createElement("span");
      item.className = "readout-item";
      item.textContent = part;
      el.append(item);
    });
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

let stepShown = -1; // the step the panel shows, so a new one can start at its top
function renderStep() {
  const i = state ? state.step : 0;
  const step = lesson.steps[i];
  if (i !== stepShown) {
    // A new step (or lesson) starts at the top of the panel: it kept the last one's place, so a lesson opened after
    // another's Finish showed its title half scrolled away, and a step after a long one began at its foot.
    stepShown = i;
    $("lesson-panel").scrollTop = 0;
  }
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
  $("lesson-step-title").textContent = titleText(step.title);
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
  p.textContent = line.replaceAll("W A S D", "W A S D"); // the four keys never split over two lines
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

// Below this width (or this height) the lesson, its plots and the 3D view cannot all be shown in full.
const ROOMY = 1100;
const ROOMY_HEIGHT = 640;

// In such a window the plots box shows one thing at a time (a plot, the face-off table or the experiment record),
// picked from a row of tabs, so the robot keeps most of the view. A big window shows them all, as before.
const plotTabs = {
  strip: null,
  pick: 0,
  // The sections shown, in tab order. Live plots first: they are short, so the robots keep the view; the table and
  // the record are a tap away.
  sections() {
    const order = (el) => (el.classList.contains("plot") ? 0 : el.classList.contains("faceoff-table") ? 1 : 2);
    return [...$("lesson-plots").children]
      .filter((el) => el.tagName === "SECTION" && !el.hidden)
      .sort((a, b) => order(a) - order(b));
  },
  // Bring one section's tab to the front: a picture the step has just switched on.
  show(section) {
    const k = this.sections().indexOf(section);
    if (k >= 0) this.pick = k;
    this.sync();
  },
  sync() {
    const box = $("lesson-plots");
    const sections = this.sections();
    const tabbed = sections.length > 1 && (window.innerWidth <= ROOMY || window.innerHeight <= ROOMY_HEIGHT);
    box.classList.toggle("tabbed", tabbed);
    if (!tabbed) {
      this.strip?.remove();
      for (const s of sections) s.classList.remove("off");
      return;
    }
    if (!this.strip) {
      this.strip = document.createElement("div");
      this.strip.className = "plot-tabs";
      this.strip.setAttribute("role", "tablist");
    }
    if (box.firstChild !== this.strip) box.prepend(this.strip);
    this.pick = Math.min(this.pick, sections.length - 1);
    const labels = sections.map(tabLabel);
    if (this.strip.dataset.labels !== labels.join("|") || this.strip.childElementCount !== labels.length) {
      this.strip.dataset.labels = labels.join("|");
      this.strip.replaceChildren(
        ...labels.map((label, k) => {
          const b = document.createElement("button");
          b.type = "button";
          b.setAttribute("role", "tab");
          b.textContent = label;
          b.title = label;
          b.addEventListener("click", () => {
            this.pick = k;
            this.sync();
            plots.draw();
            bench.draw();
          });
          return b;
        })
      );
    }
    [...this.strip.children].forEach((b, k) => b.setAttribute("aria-selected", String(k === this.pick)));
    sections.forEach((s, k) => s.classList.toggle("off", k !== this.pick));
  },
};

function tabLabel(section) {
  if (section.classList.contains("faceoff-table")) return section.dataset.tab || t("plots_table"); // a table may name its tab
  if (section.classList.contains("bench")) return t("plots_record", { n: bench.state?.runs?.length ?? 0 });
  const title = section.querySelector(".plot-title")?.textContent ?? "";
  // The unit and any aside in brackets stay on the plot itself; Chinese titles use full-width brackets.
  return title.replace(/(\s*[(（][^()（）]*[)）])+$/, "");
}
window.addEventListener("resize", () => plotTabs.sync());

// Fold the lesson panel down to its title, for a bigger view of the robot (on a phone it is a sheet along the bottom).
function foldLesson(folded) {
  $("lesson-panel").classList.toggle("folded", folded);
  const button = $("lesson-fold");
  button.setAttribute("aria-expanded", String(!folded));
  button.dataset.i18nLabel = folded ? "unfold_lesson" : "fold_lesson";
  button.setAttribute("aria-label", t(button.dataset.i18nLabel));
  button.title = t(button.dataset.i18nLabel);
  api.shift();
}

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
    const actions = document.createElement("span"); // Zoom and Clear, kept together at the right
    actions.className = "plot-actions";
    head.append(title, actions);
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
      actions.append(zoom);
    }
    if (runs.length || this.state?.sweep) {
      const clear = document.createElement("button");
      clear.type = "button";
      clear.className = "link-button";
      clear.textContent = t("bench_clear");
      clear.addEventListener("click", () => api.send({ type: "lesson", action: "clear_runs" }));
      actions.append(clear);
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
    plotTabs.sync(); // the record's tab shows how many runs it holds
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
      // Each setting stays on one line with its value; lines break after a dot, never before one. A setting the run
      // did not use is left out (the hand-written pick learns from no demonstrations).
      const used = spec.params.filter((p) => run.params[p] !== undefined);
      settings.textContent = used.map((p) => `${this.label(p)}\u00a0${this.value(p, run.params[p])}`).join("\u00a0· ");
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
  const yt = niceTicks(y0, y1, 3);
  for (const v of yt) {
    g.beginPath();
    g.moveTo(36, Math.round(Y(v)) + 0.5);
    g.lineTo(w - 8, Math.round(Y(v)) + 0.5);
    g.stroke();
    g.fillText(formatTick(v, yt), 2, Y(v) + 3);
  }
  const xt = niceTicks(x0, x1, 4);
  xt.forEach((v, i) => {
    const text = formatTick(v, xt) + (i === xt.length - 1 && xUnit ? ` ${xUnit}` : "");
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
  tableBox.dataset.tab = table?.tab ? pick(table.tab) : ""; // e.g. a list of demonstrations, not robots side by side
  plotTabs.sync();
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
      b.dataset.control = c.id; // for scripted checks, as the sliders and toggles have
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
    shown.textContent = withUnit(Number(input.value).toFixed(c.step < 1 ? 2 : 0), c.unit);
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
    // Nothing shown below zero, or no more than a rover's settling (a distance, a share of the floor): the axis stops
    // at zero too. Widened evenly around the data, a converged distance drew ticks at 2, 0 and -2 mm.
    const floor = Number.isFinite(lo) && lo >= -0.02 * smallestSpan(spec) ? Math.min(lo, 0) : null;
    if (!Number.isFinite(lo)) [lo, hi] = [-1, 1];
    if (hi - lo < 1e-6) [lo, hi] = [lo - 1, hi + 1];
    const short = smallestSpan(spec) - (hi - lo); // widen around the data up to the plot's smallest span
    if (short > 0) [lo, hi] = [lo - short / 2, hi + short / 2];
    if (floor !== null && lo < floor) [lo, hi] = [floor, hi + floor - lo]; // past zero: widen upward instead
    const pad = (hi - lo) * 0.12;
    // Grow at once, shrink slowly, so the axes do not jump around while you watch.
    const prev = this.ranges[k];
    const target = [floor === null ? lo - pad : Math.max(floor, lo - pad), hi + pad];
    const range = prev ? [Math.min(target[0], prev[0] + (target[0] - prev[0]) * 0.05), Math.max(target[1], prev[1] + (target[1] - prev[1]) * 0.05)] : target;
    this.ranges[k] = range;
    const x = (t) => 36 + ((t - start) / spec.window) * (w - 44);
    const y = (v) => 8 + (1 - (v - range[0]) / (range[1] - range[0])) * (h - 22);
    g.font = `10px ${token("--mono") || "monospace"}`;
    g.fillStyle = token("--muted");
    g.strokeStyle = token("--line");
    g.lineWidth = 1;
    const yt = niceTicks(range[0], range[1], h < 80 ? 2 : 3); // a compact plot has room for two labels
    for (const v of yt) {
      g.beginPath();
      g.moveTo(36, Math.round(y(v)) + 0.5);
      g.lineTo(w - 8, Math.round(y(v)) + 0.5);
      g.stroke();
      g.fillText(formatTick(v, yt), 2, y(v) + 3);
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
    g.font = `10px ${token("--font") || "sans-serif"}`;
    // The bars start after the longest name: a fixed 44 px cut "front right" to "front righ" under its bar.
    const left = Math.ceil(Math.max(36, ...spec.series.map((s) => g.measureText(pick(s.label)).width)) + 8);
    const rowH = (h - 10) / rows;
    const x = (t) => left + ((t - start) / spec.window) * (w - left - 8);
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

// The smallest value range a plot draws, so a robot standing still does not fill it with its wobble: the lesson's own
// `span`, else one for the plot's unit. Without it the sensors lesson drew its odometry's last 0.4 mm across the whole
// plot, between two ticks that both read 0. Never so small that three decimals cannot tell the ticks apart.
const UNIT_SPAN = { "°": 4, "°/s": 10, "N·m": 1, cm: 2, mm: 4, m: 0.2, "m/s": 0.2, "%": 10 };
const LABELLED_SPAN = 0.004;
function smallestSpan(spec) {
  return Math.max(spec.span || UNIT_SPAN[spec.unit] || 0, LABELLED_SPAN);
}

function niceTicks(lo, hi, count) {
  const span = hi - lo;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  // The roundest step near span / count; a finer one when that leaves a single line, which gives no scale.
  const steps = [0.5, 1, 2, 5, 10].map((m) => m * mag);
  let k = steps.findIndex((s) => s >= step0);
  if (k < 0) k = steps.length - 1;
  for (; k >= 0; k--) {
    const step = steps[k];
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(Math.abs(v) < step / 1e6 ? 0 : v);
    if (out.length >= 2 || k === 0) return out;
  }
  return [];
}

// As many decimals as the tick spacing needs (0.02 apart reads 0.02 and 0.04, not 0.0 and 0.0), and never "-0".
function formatTick(v, ticks = []) {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : Math.abs(v);
  const decimals = step > 0 ? Math.min(3, Math.max(0, -Math.floor(Math.log10(step) + 1e-9))) : 0;
  const text = v.toFixed(decimals);
  return Number(text) === 0 ? "0" : text;
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
    if (this.wrap.hidden === show) {
      // In a small window the picture is a tab of its own, brought up when the step switches it on: under the tabs,
      // it and the lesson sheet covered the whole phone, the rover too.
      this.wrap.hidden = !show;
      if (show) plotTabs.show(this.wrap);
      else plotTabs.sync();
    }
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

// What the lesson's own camera sees, as the physics side rendered it: the picture or the color finder's mask,
// the blobs it found boxed and the biggest one's centroid crossed. Pictures come only when they change.
const camview = {
  wrap: null,
  canvas: null,
  ctx: null,
  note: null,
  seg: null,
  size: [320, 240],
  view: "picture", // "picture" or "mask"
  chosen: false, // the learner picked a view during this step: the step's own suggestion no longer applies
  step: -1,
  images: { picture: null, mask: null },
  shown: null, // the pictures last decoded
  setup(rig) {
    this.teardown();
    if (rig?.kind !== "vision") return;
    this.size = rig.image;
    this.wrap = document.createElement("section");
    this.wrap.className = "plot pip camview";
    const head = document.createElement("div");
    head.className = "plot-head";
    const title = document.createElement("span");
    title.className = "plot-title";
    title.textContent = t("robot_camera");
    this.seg = document.createElement("div");
    this.seg.className = "seg";
    for (const view of ["picture", "mask"]) {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.view = view;
      b.textContent = t(`camera_${view}`);
      b.addEventListener("click", () => {
        this.view = view;
        this.chosen = true;
        this.syncSeg();
        this.draw();
      });
      this.seg.append(b);
    }
    head.append(title, this.seg);
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.size[0] * 2; // twice the picture, so the boxes and the cross stay sharp
    this.canvas.height = this.size[1] * 2;
    this.canvas.style.aspectRatio = `${this.size[0]} / ${this.size[1]}`;
    this.ctx = this.canvas.getContext("2d");
    this.note = document.createElement("p");
    this.note.className = "camview-note";
    this.note.hidden = true;
    this.wrap.append(head, this.canvas, this.note);
    $("lesson-plots").prepend(this.wrap); // what the robot sees comes first
    plotTabs.sync(); // and in a small window it is the first tab, not a panel above the tabs
    this.view = "picture";
    this.chosen = false;
    this.step = -1;
    this.syncSeg();
    this.draw();
  },
  teardown() {
    this.wrap?.remove();
    this.wrap = this.canvas = this.ctx = this.note = this.seg = null;
    this.images = { picture: null, mask: null };
    this.shown = null;
  },
  syncSeg() {
    for (const b of this.seg?.children ?? []) b.setAttribute("aria-pressed", String(b.dataset.view === this.view));
  },
  update() {
    if (!this.wrap || !state) return;
    const v = state.live?.vision;
    if (!v) return;
    if (state.step !== this.step) {
      this.step = state.step;
      this.chosen = false;
    }
    if (!this.chosen && v.view && v.view !== this.view) {
      this.view = v.view;
      this.syncSeg();
    }
    this.note.hidden = !v.note;
    this.canvas.hidden = !!v.note;
    this.seg.hidden = !!v.note;
    if (v.note) this.note.textContent = pick(v.note);
    const pictures = state.pictures;
    if (pictures && pictures !== this.shown) {
      this.shown = pictures;
      for (const key of ["picture", "mask"]) {
        const img = new Image();
        img.onload = () => {
          if (this.shown !== pictures) return; // a newer frame is on its way
          this.images[key] = img;
          this.draw();
        };
        img.src = pictures[key];
      }
    }
    this.draw();
  },
  draw() {
    const g = this.ctx;
    if (!g) return;
    const { width: W, height: H } = this.canvas;
    const s = W / this.size[0];
    const img = this.images[this.view];
    g.fillStyle = `#${api.color("--hover").getHexString()}`;
    g.fillRect(0, 0, W, H);
    if (!img) {
      g.fillStyle = `#${api.color("--muted").getHexString()}`;
      g.font = `${12 * s}px -apple-system, "Segoe UI", "PingFang SC", sans-serif`;
      g.textAlign = "center";
      g.fillText(t("camera_waiting"), W / 2, H / 2);
      return;
    }
    g.imageSmoothingEnabled = this.view === "picture"; // the mask stays a grid of pixels
    g.drawImage(img, 0, 0, W, H);
    const blobs = state?.live?.vision?.blobs ?? [];
    const amber = `#${api.color("--amber").getHexString()}`;
    const other = this.view === "mask" ? "rgba(122, 162, 247, 0.95)" : "rgba(255, 255, 255, 0.85)";
    blobs.forEach((b, k) => {
      const [x0, y0, x1, y1] = b.box;
      g.lineWidth = k === 0 ? 2.5 : 1.5;
      g.strokeStyle = k === 0 ? amber : other;
      g.strokeRect(x0 * s - 2, y0 * s - 2, (x1 - x0 + 1) * s + 4, (y1 - y0 + 1) * s + 4);
    });
    const best = blobs[0];
    if (best) {
      const [x, y, r] = [best.u * s, best.v * s, 7 * s];
      g.strokeStyle = "rgba(0, 0, 0, 0.55)"; // a dark edge keeps the cross readable on the white mat
      g.lineWidth = 4.5;
      cross(g, x, y, r);
      g.strokeStyle = amber;
      g.lineWidth = 2.2;
      cross(g, x, y, r);
    }
  },
};

function cross(g, x, y, r) {
  g.beginPath();
  g.moveTo(x - r, y);
  g.lineTo(x + r, y);
  g.moveTo(x, y - r);
  g.lineTo(x, y + r);
  g.stroke();
}

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
    tag.position.set(e.base[0], e.base[1], e.top + TAG_RISE);
    tag.userData.foot = new THREE.Vector3(e.base[0], e.base[1], 0); // where its robot stands
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
    // `topOf(entrant)`: how high its robot reaches now, when that changes (an arm), else the rig's fixed top;
    // `avoid`: what else the tags must not hide, as { at, r } (a point and a radius in m).
    frame(topOf = (e) => e.top, avoid = []) {
      for (const [tag, e] of tags) {
        const node = e.body !== undefined ? api.bodyNode(e.body) : null;
        if (node) {
          tag.position.set(node.position.x, node.position.y, 0); // walking robots take theirs along
          tag.userData.foot.set(node.position.x, node.position.y, 0);
        }
        tag.position.z = topOf(e) + TAG_RISE;
        tag.userData.meshes = api.robotMeshes(e.name); // what the other robots' tags must leave in sight
      }
      if (labels.visible) declutter(tags.map(([tag]) => tag), avoid);
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

// A tag stands on its point, a little above it, rather than being centred on it: seen from overhead (the robot
// vacuums' camera) a tag centred over a robot hid the robot it names. TAG_RISE (m) is the point above the robot's top.
const TAG_LIFT = 0.6; // the tag's middle above its point, in tag heights
const TAG_RISE = 0.06;

// Name tags of robots standing in a row, seen down the row, land on top of each other. Keep the nearest
// where it is and move any farther tag that would cover one already placed: above it, beside it or below it.
// A tag whose robot stands in the free part of the view is kept inside it: seen from above, the PiPER's tag in the
// motion planning lesson stood so high over the arm that only its bottom edge showed, at the top of the window.
// Wherever it goes, it leaves in sight the lower half of its own robot, what the rig marks (`avoid`: the IK lesson's
// target ball) and the other robots: seen down the row in the gait lesson, the template dog's tag stood on the front
// legs of the Go2 behind it. Where it would hide one of them over its robot, it hangs under its robot instead, or else
// stands beside what it would hide.
const tagPoint = new THREE.Vector3();
const footPoint = new THREE.Vector3();
const topPoint = new THREE.Vector3();
const TAG_EDGE = 6; // px a kept tag stays clear of the free view's edge, and of a tag beside it
const TAG_STAY = 6; // px a tag must clear the other robots by to change place: a walking robot's legs swing in and out
const SPINE_BASE = 0.05; // m, the lowest point of a robot a tag must not hide (its base, not the floor)

// Each part of a robot as the camera sees it: the screen box round the corners of the part's own box, in the view's
// -1 to 1 units. A part with a corner behind the camera is left out.
const corner = new THREE.Vector3();
function screenParts(meshes, cam) {
  const parts = [];
  for (const mesh of meshes ?? []) {
    const g = mesh.geometry;
    if (!g.boundingBox) g.computeBoundingBox();
    const [lo, hi] = [g.boundingBox.min, g.boundingBox.max];
    const r = { left: Infinity, right: -Infinity, bottom: Infinity, top: -Infinity };
    let behind = false;
    for (let k = 0; k < 8 && !behind; k++) {
      corner.set(k & 1 ? hi.x : lo.x, k & 2 ? hi.y : lo.y, k & 4 ? hi.z : lo.z).applyMatrix4(mesh.matrixWorld).project(cam);
      behind = corner.z > 1;
      [r.left, r.right] = [Math.min(r.left, corner.x), Math.max(r.right, corner.x)];
      [r.bottom, r.top] = [Math.min(r.bottom, corner.y), Math.max(r.top, corner.y)];
    }
    if (!behind) parts.push(r);
  }
  return parts;
}

function declutter(sprites, avoid = []) {
  const cam = api.camera;
  const [sx, sy] = [cam.projectionMatrix.elements[0], cam.projectionMatrix.elements[5]];
  const r = api.freeRect();
  const [W, H] = [window.innerWidth, window.innerHeight];
  const free = { left: (2 * r.left) / W - 1, right: (2 * r.right) / W - 1, top: 1 - (2 * r.top) / H, bottom: 1 - (2 * r.bottom) / H };
  const inside = (p) => p.z < 1 && p.x > free.left && p.x < free.right && p.y < free.top && p.y > free.bottom;
  const [padX, padY] = [(2 * TAG_EDGE) / W, (2 * TAG_EDGE) / H]; // in the view's -1 to 1 units
  // What the rig marks, as circles on the screen: the middle (-1 to 1) and the radius in px.
  const across = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0);
  const marks = avoid.map(({ at, r: radius }) => {
    const c = at.clone().project(cam);
    const edge = at.clone().addScaledVector(across, radius).project(cam);
    return { x: c.x, y: c.y, r: (Math.abs(edge.x - c.x) * W) / 2 };
  });
  // Does a tag with its middle at (x, y) hide the circle m? Measured in px, the nearest point of the tag to its middle.
  const hides = (x, y, w, h, m) => {
    const dx = (Math.max(Math.abs(m.x - x) - w / 2, 0) * W) / 2;
    const dy = (Math.max(Math.abs(m.y - y) - h / 2, 0) * H) / 2;
    return dx * dx + dy * dy < m.r * m.r;
  };
  // How much of the screen box b, grown by `grow` px, does it cover? In the view's units, 0 when it is clear.
  const covers = (x, y, w, h, b, grow) => {
    const [gx, gy] = [(2 * grow) / W, (2 * grow) / H];
    const across = Math.min(x + w / 2, b.right + gx) - Math.max(x - w / 2, b.left - gx);
    const down = Math.min(y + h / 2, b.top + gy) - Math.max(y - h / 2, b.bottom - gy);
    return Math.max(across, 0) * Math.max(down, 0);
  };
  // Each tag's robot as the camera sees it: the lower half of its middle as points, its parts and the box round them.
  const robots = sprites.map((s) => {
    const [low, high] = [Math.min(SPINE_BASE, s.position.z - TAG_RISE), s.position.z - TAG_RISE];
    const spine = [0, 0.25, 0.5]
      .map((f) => topPoint.copy(s.position).setZ(low + f * (high - low)).project(cam).clone())
      .filter((q) => q.z < 1)
      .map((q) => ({ x: q.x, y: q.y, r: 2 }));
    const parts = screenParts(s.userData.meshes, cam);
    const box = parts.length
      ? parts.reduce((a, b) => ({ left: Math.min(a.left, b.left), right: Math.max(a.right, b.right), bottom: Math.min(a.bottom, b.bottom), top: Math.max(a.top, b.top) }))
      : null;
    return { spine, parts, box };
  });
  const items = sprites
    .map((s, i) => {
      s.center.set(0.5, 0.5 - TAG_LIFT);
      const p = tagPoint.copy(s.position).project(cam); // screen position, -1 to 1 across the view
      const [w, h] = [s.scale.x * sx, s.scale.y * sy];
      const y0 = p.y + TAG_LIFT * h; // the tag's middle
      const it = { s, px: p.x, x: p.x, y: y0, y0, z: p.z, w, h, fits: null };
      if (!s.userData.foot || !inside(footPoint.copy(s.userData.foot).project(cam))) return it;
      const room = { left: free.left + w / 2 + padX, right: free.right - w / 2 - padX, top: free.top - h / 2 - padY, bottom: free.bottom + h / 2 + padY };
      // A tag held down over a tall robot's head still names it; one held down onto its base (the planning arm on a
      // phone) does not, and one on another robot names the wrong one.
      const others = robots.filter((_, k) => k !== i);
      const keep = [...robots[i].spine, ...marks, ...others.filter((o) => !o.parts.length).flatMap((o) => o.spine)];
      const boxes = others.flatMap((o) => o.parts);
      it.inRoom = ([cx, cy]) => cx >= room.left && cx <= room.right && cy >= room.bottom && cy <= room.top;
      const clear = (c) => it.inRoom(c) && !keep.some((m) => hides(c[0], c[1], w, h, m));
      const hidden = (c, grow = 0) => boxes.reduce((sum, b) => sum + covers(c[0], c[1], w, h, b, grow), 0);
      it.fits = (c, grow = 0) => clear(c) && hidden(c, grow) === 0;
      const own = robots[i].box;
      const high = Math.min(y0, room.top);
      // Over the whole of its robot as the camera sees it: from overhead (the robot vacuums) the point over a robot's
      // middle is inside its outline, and the tag stood on the far half of it.
      const over = Math.min(own ? Math.max(y0, own.top + TAG_LIFT * h) : y0, room.top);
      const held = Math.min(Math.max(p.x, room.left), room.right);
      const under = Math.min(footPoint.y, own?.bottom ?? Infinity) - padY - h / 2; // under its feet
      const beside = [
        ...keep.flatMap((m) => [-1, 1].map((k) => m.x + k * (w / 2 + (2 * m.r) / W + padX))),
        ...boxes.flatMap((b) => [b.left - w / 2 - padX, b.right + w / 2 + padX]),
      ]
        .map((bx) => ["beside", [bx, high]])
        .sort((a, b) => Math.abs(a[1][0] - held) - Math.abs(b[1][0] - held));
      // Over its robot's outline, or else over its point, held in the free view; else under its robot; else beside what
      // it would hide. A place other than last frame's must clear the other robots by TAG_STAY, so the tag does not hop
      // to and fro as they walk.
      const places = [["over", [held, over]], ["above", [held, high]], ["under", [held, under]], ...beside];
      // Where nothing fits (four robots in lanes on a phone), the place in the free view that hides the least of them.
      const [kind, at] =
        places.find(([k, c]) => it.fits(c, k === s.userData.place ? 0 : TAG_STAY)) ??
        places.filter(([, c]) => clear(c)).sort((a, b) => hidden(a[1]) - hidden(b[1]))[0] ??
        places[0];
      s.userData.place = kind;
      [it.x, it.y] = at;
      return it;
    })
    .filter((it) => it.z < 1)
    .sort((a, b) => a.z - b.z);
  const placed = [];
  const meets = (it, x, y) => placed.some((p) => Math.abs(p.x - x) < (p.w + it.w) / 2 && Math.abs(p.y - y) < (p.h + it.h) / 2);
  for (const it of items) {
    let [x, y] = [it.x, it.y];
    if (meets(it, x, y)) {
      const gap = it.h * 0.08;
      const column = placed.filter((p) => Math.abs(p.x - x) < (p.w + it.w) / 2);
      const row = placed.filter((p) => Math.abs(p.y - y) < (p.h + it.h) / 2);
      const lifts = column.map((p) => [x, p.y + (p.h + it.h) / 2 + gap]).filter((c) => c[1] > y).sort((a, b) => a[1] - b[1]);
      const sides = row
        .flatMap((p) => [-1, 1].map((k) => [p.x + k * ((p.w + it.w) / 2 + padX), y]))
        .sort((a, b) => Math.abs(a[0] - it.x) - Math.abs(b[0] - it.x));
      const drops = column.map((p) => [x, p.y - (p.h + it.h) / 2 - gap]).filter((c) => c[1] < y).sort((a, b) => b[1] - a[1]);
      const open = (c) => !meets(it, c[0], c[1]);
      const all = [...lifts, ...sides, ...drops];
      // Clear of the other tags and of all a tag leaves in sight; else clear of the tags in the free view; else lifted
      // over the others, as a tag whose robot is out of the free view always is.
      [x, y] = (it.fits && (all.find((c) => open(c) && it.fits(c)) ?? all.find((c) => open(c) && it.inRoom(c)))) ?? lifts.find(open) ?? [x, y];
    }
    it.s.center.x = 0.5 - (x - it.px) / it.w; // a smaller anchor draws the sprite further right
    it.s.center.y = 0.5 - TAG_LIFT - (y - it.y0) / it.h; // a lower anchor draws the sprite higher
    placed.push({ ...it, x, y });
  }
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
const TARGET_HALO = 0.05; // m, the glow round the target ball
function arm3dRig(rig) {
  const tags = faceoffRig(rig); // the name tags over the arms
  const amber = api.color("--amber");
  const blue = api.color("--plot-blue");
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.03, 24, 12), basic(amber, 0.95));
  const halo = new THREE.Mesh(new THREE.SphereGeometry(TARGET_HALO, 24, 12), basic(amber, 0.18));
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
      const live = state.live;
      // Each tag over its arm as it stands now, and the working arm's over its target as well: the target goes up to
      // 0.75 m and tags draw over everything, so a tag at a fixed 0.5 m hid the ball step 1 asks you to drag.
      const above = (e) => (e.name === live?.arm && live?.target ? live.target[2] + TARGET_HALO : -Infinity);
      const ball = live?.target ? [{ at: new THREE.Vector3().fromArray(live.target), r: TARGET_HALO }] : [];
      tags.frame((e) => Math.max(api.robotTop(e.name) ?? e.top, above(e)), ball);
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
      // Over each dog as it stands: at the rig's fixed 0.5 m the template dog's tag sat on the Go2 behind it.
      tags.frame((e) => api.robotTop(e.name) ?? e.top);
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

// Motion planning: the straight joint-space sweep (and where it hits), the two trees growing in, the path
// as a tube, all drawn where the hand would be. Drawings arrive only when they change; the newest stays.
function planRig(rig) {
  const tags = faceoffRig(rig);
  const amber = api.color("--amber");
  const violet = api.color("--plot-violet");
  const blue = api.color("--plot-blue");
  const red = api.color("--bad");
  const straight = new THREE.Group();
  const trees = new THREE.Group();
  const path = new THREE.Group();
  const sweep = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: violet, dashSize: 0.02, gapSize: 0.012, transparent: true, opacity: 0.9, depthTest: false }));
  sweep.userData.dashed = true;
  // Dots along the sweep as well: a one-pixel dashed line is hard to see.
  const sweepDots = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color: violet, size: 0.012, map: dotTexture(), alphaTest: 0.05, transparent: true, depthTest: false, toneMapped: false }));
  const clash = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 10), Object.assign(basic(red, 0.95), { depthTest: false }));
  clash.renderOrder = 6; // seen through the fence it is inside of
  const goal = new THREE.Mesh(new THREE.SphereGeometry(0.018, 20, 12), basic(amber, 0.9));
  clash.visible = goal.visible = false; // until a drawing places them (not as two dots at the world's origin)
  straight.add(sweep, sweepDots, clash, goal);
  const branches = [blue, violet].map((c) => new THREE.LineSegments(new THREE.BufferGeometry(), lineMat(c, 0.75)));
  // The nodes as dots too: one-pixel lines alone vanish at this distance.
  const nodes = [blue, violet].map(
    (c) => new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color: c, size: 0.02, map: dotTexture(), alphaTest: 0.05, transparent: true, depthTest: false, toneMapped: false }))
  );
  trees.add(...branches, ...nodes);
  const tube = new THREE.Mesh(new THREE.BufferGeometry(), basic(amber, 0.85));
  path.add(tube);
  const parts = { ...tags.parts, straight, trees, path };
  tags.group.add(straight, trees, path);
  let shown = null; // the drawing on screen
  let grownAt = 0; // when its trees started growing in
  const GROW_MS = 1400;
  const draw = (d) => {
    shown = d;
    grownAt = performance.now();
    goal.position.fromArray(d.goal);
    goal.visible = true;
    const pts = (d.straight ?? []).map((p) => new THREE.Vector3().fromArray(p));
    sweep.visible = sweepDots.visible = pts.length > 1;
    if (sweep.visible) {
      setPoints(sweep, pts);
      sweepDots.geometry.dispose();
      sweepDots.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    }
    clash.visible = !!d.clash;
    if (d.clash) clash.position.fromArray(d.clash);
    (d.trees ?? [[], []]).forEach((edges, k) => {
      const flat = edges.flatMap(([a, b]) => [new THREE.Vector3().fromArray(a), new THREE.Vector3().fromArray(b)]);
      setPoints(branches[k], flat);
      branches[k].geometry.setDrawRange(0, 0);
      branches[k].userData.count = flat.length;
      const tips = edges.map(([, b]) => new THREE.Vector3().fromArray(b));
      nodes[k].geometry.dispose();
      nodes[k].geometry = new THREE.BufferGeometry().setFromPoints(tips);
      nodes[k].geometry.setDrawRange(0, 0);
      nodes[k].userData.count = tips.length;
    });
    tube.geometry.dispose();
    const way = (d.path ?? []).map((p) => new THREE.Vector3().fromArray(p));
    tube.geometry = way.length > 1 ? new THREE.TubeGeometry(new THREE.CatmullRomCurve3(way), Math.min(400, way.length * 3), 0.005, 8) : new THREE.BufferGeometry();
    tube.visible = false;
  };
  return {
    group: tags.group,
    parts,
    frame() {
      tags.frame((e) => api.robotTop(e.name) ?? e.top); // over each arm as it stands, not where it might reach
      const d = state.drawing;
      if (d && d !== shown) draw(d);
      if (!shown) return;
      // The trees grow in edge by edge, in the order they were found; the path appears once they meet.
      const u = Math.min(1, (performance.now() - grownAt) / GROW_MS);
      for (const b of branches) b.geometry.setDrawRange(0, 2 * Math.floor(((b.userData.count ?? 0) / 2) * u));
      for (const n of nodes) n.geometry.setDrawRange(0, Math.floor((n.userData.count ?? 0) * u));
      tube.visible = u >= 1 && (shown.path?.length ?? 0) > 1;
    },
  };
}

// A camera on a post: what it sees as a pyramid down to the floor, the ray through the centroid to where it
// meets the plane (drawn too) or to the middle of the depth camera's box, the estimate as an amber ball and the
// cube's true middle as a blue ring, both drawn through the cube.
function visionRig(rig) {
  const group = new THREE.Group();
  const ink = api.color("--ink");
  const amber = api.color("--amber");
  const blue = api.color("--plot-blue");
  const eye = new THREE.Vector3().fromArray(rig.eye);
  const [R, U, F] = [rig.right, rig.up, rig.forward].map((v) => new THREE.Vector3().fromArray(v));
  const tanV = Math.tan(THREE.MathUtils.degToRad(rig.fovy / 2));
  const tanH = (tanV * rig.image[0]) / rig.image[1];
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => {
    const dir = F.clone().addScaledVector(R, a * tanH).addScaledVector(U, b * tanV);
    return eye.clone().addScaledVector(dir, -eye.z / dir.z); // where the corner's ray meets the floor
  });
  const pts = [];
  corners.forEach((c, k) => pts.push(eye, c, c, corners[(k + 1) % 4]));
  const frustum = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat(ink, 0.45));
  const onTop = (m) => Object.assign(m, { depthTest: false });
  // The ray as a thin rod: a one-pixel line is lost against the mat from across the room.
  const line = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 8), onTop(basic(amber, 0.9)));
  const along = new THREE.Vector3(0, 1, 0);
  const estimate = new THREE.Mesh(new THREE.SphereGeometry(0.007, 20, 12), onTop(basic(amber, 1)));
  const truth = new THREE.Mesh(new THREE.RingGeometry(0.009, 0.013, 32), onTop(basic(blue, 1, true)));
  const gap = dashedLine(blue, 0.9);
  const ray = new THREE.Group();
  ray.add(line, estimate, truth, gap);
  const [mw, mh] = rig.mat.size;
  const sheet = new THREE.Mesh(new THREE.PlaneGeometry(mw, mh), basic(blue, 0.14, true));
  const rim = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(mw, mh)), lineMat(blue, 0.6));
  const plane = new THREE.Group();
  plane.add(sheet, rim);
  const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), lineMat(amber, 0.9));
  const parts = { frustum, ray, plane, box };
  group.add(frustum, ray, plane, box);
  group.traverse((o) => {
    o.renderOrder = 6; // over the cubes, which the estimate sits inside of
  });
  return {
    group,
    parts,
    frame() {
      const v = state.live?.vision ?? {};
      const est = v.estimate ? new THREE.Vector3().fromArray(v.estimate) : null;
      line.visible = estimate.visible = gap.visible = !!est;
      if (est) {
        const span = est.clone().sub(eye);
        line.position.copy(eye).addScaledVector(span, 0.5);
        line.quaternion.setFromUnitVectors(along, span.clone().normalize());
        line.scale.set(0.0018, span.length(), 0.0018);
        estimate.position.copy(est);
      }
      truth.visible = !!v.truth;
      if (v.truth) {
        const at = new THREE.Vector3().fromArray(v.truth);
        truth.position.copy(at);
        truth.quaternion.copy(api.camera.quaternion); // the ring faces the viewer
        if (est) setPoints(gap, [est, at]);
      }
      if (plane.visible) plane.visible = v.plane !== undefined;
      // On the table the sheet lies just over the mat, which is a few millimetres thick, so it shows.
      if (plane.visible) plane.position.set(rig.mat.center[0], rig.mat.center[1], Math.max(v.plane, rig.mat.top + 0.001));
      if (box.visible) box.visible = !!(v.box && est);
      if (box.visible) {
        box.position.copy(est);
        box.scale.set(...v.box.map((x) => Math.max(x, 0.002)));
      }
    },
  };
}

// Teaching by showing: the amber target the gripper follows (drag it), where cubes can appear, every
// demonstration (a ring where the cube was, the gripper's path), the last test's spots (green: picked, red:
// missed) and the arm's own try (violet). Demonstrations and tests arrive only when they change.
function imitateRig(rig) {
  const amber = api.color("--amber");
  const blue = api.color("--plot-blue");
  const green = api.color("--plot-green");
  const red = api.color("--bad");
  const violet = api.color("--plot-violet");
  const ink = api.color("--ink");
  const z0 = rig.desk + 0.002; // just above the desk
  const onDesk = (x, y) => new THREE.Vector3(x, y, z0);
  const onTop = (m) => Object.assign(m, { depthTest: false });

  // Seen through the fingers it sits between: it is the handle the learner drags.
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.014, 24, 12), onTop(basic(amber, 0.95)));
  const halo = new THREE.Mesh(new THREE.SphereGeometry(0.026, 24, 12), onTop(basic(amber, 0.22)));
  const marker = new THREE.Group();
  marker.add(ball, halo);
  const error = dashedLine(amber, 0.9);
  const target = new THREE.Group();
  target.add(marker, error);

  // The ring sector cubes appear in: its outline and a faint fill on the desk.
  const [r0, r1] = rig.region.r;
  const [d0, d1] = rig.region.deg.map((d) => THREE.MathUtils.degToRad(d));
  const arc = (r, a, b) => {
    const out = [];
    for (let k = 0; k <= 48; k++) {
      const t = a + ((b - a) * k) / 48;
      out.push(onDesk(rig.base[0] + r * Math.cos(t), rig.base[1] + r * Math.sin(t)));
    }
    return out;
  };
  const outline = [...arc(r1, d0, d1), ...arc(r0, d1, d0)];
  const region = new THREE.Group();
  const shape = new THREE.Shape(outline.map((p) => new THREE.Vector2(p.x, p.y)));
  const fill = new THREE.Mesh(new THREE.ShapeGeometry(shape), basic(ink, 0.05, true));
  fill.position.z = z0 - 0.001;
  const edge = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(outline), lineMat(ink, 0.3));
  region.add(fill, edge);

  const demos = new THREE.Group();
  const tests = new THREE.Group();
  const trialPath = new THREE.Line(new THREE.BufferGeometry(), onTop(lineMat(violet, 0.95)));
  const trialRing = new THREE.Mesh(new THREE.RingGeometry(0.026, 0.034, 40), onTop(basic(violet, 0.95, true)));
  const trial = new THREE.Group();
  trial.add(trialPath, trialRing);
  const parts = { target, region, demos, tests, trial };
  const group = new THREE.Group();
  group.add(...Object.values(parts));
  group.traverse((o) => {
    o.renderOrder = 5; // after the desk and the arm: drawn over them
  });

  const clear = (g) => {
    for (const child of [...g.children]) {
      child.geometry?.dispose();
      child.material?.dispose();
      g.remove(child);
    }
  };
  let shown = null; // the drawing on screen
  const draw = (d) => {
    shown = d;
    clear(demos);
    for (const demo of d.demos ?? []) {
      const color = demo.fix ? violet : blue; // a correction, taken over from the arm's own try
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.022, 0.03, 40), onTop(basic(color, 0.9, true)));
      ring.position.copy(onDesk(...demo.spot));
      const path = new THREE.Line(new THREE.BufferGeometry().setFromPoints(demo.path.map((p) => new THREE.Vector3(...p))), lineMat(color, 0.7));
      ring.renderOrder = path.renderOrder = 5;
      demos.add(ring, path);
    }
    clear(tests);
    for (const t of d.tests ?? []) {
      const dot = new THREE.Mesh(new THREE.CircleGeometry(0.011, 24), onTop(basic(t.ok ? green : red, 0.95, true)));
      dot.position.copy(onDesk(...t.spot));
      dot.renderOrder = 6;
      tests.add(dot);
    }
  };
  let trialShown = "";
  return {
    group,
    parts,
    dragTarget: {
      mesh: ball,
      level: true, // over the desk at its height (Shift: up and down), so it lands right over what the pointer is on
      // Back to the sliders: forward and sideways from the arm's base, height over the desk.
      toControls(hit) {
        const round = (v) => Number(v.toFixed(3));
        return { tx: round(hit.x - rig.base[0]), ty: round(hit.y - rig.base[1]), tz: round(hit.z - rig.desk) };
      },
    },
    frame() {
      const live = state.live;
      if (!live?.target) return;
      const goal = new THREE.Vector3().fromArray(live.target);
      marker.position.copy(goal);
      marker.visible = !live.trying; // during a try the arm steers itself: nothing to drag
      setPoints(error, [new THREE.Vector3().fromArray(live.hand), goal]);
      const d = state.drawing;
      if (d && d !== shown) draw(d);
      const tr = live.trial;
      trial.visible = trial.visible && !!tr;
      if (tr) {
        const key = `${tr.spot}:${tr.path.length}:${tr.ok}`;
        if (key !== trialShown) {
          trialShown = key;
          trialRing.position.copy(onDesk(...tr.spot));
          if (tr.path.length > 1) setPoints(trialPath, tr.path.map((p) => new THREE.Vector3(...p)));
          trialPath.visible = tr.path.length > 1;
        }
      }
    },
  };
}

const RIGS = {
  vision: visionRig,
  imitate: imitateRig,
  gait: gaitRig,
  plan: planRig,
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
  let lift = null; // where a Shift-drag started: only the height changes
  if (t.toControls) {
    // A target in 3D slides on the plane facing the camera through it, where the pointer can follow it. A level
    // target slides on the level plane through it instead: on the facing plane a drag that looked right on screen
    // left it 2 to 3 cm off the cube along the line of sight. With Shift it slides up and down a wall facing the
    // camera.
    const at = t.mesh.getWorldPosition(new THREE.Vector3());
    let normal = ray.ray.direction.clone().negate();
    if (t.level) {
      const wall = normal.clone().setZ(0);
      if (e.shiftKey && wall.lengthSq() > 1e-6) {
        normal = wall.normalize();
        lift = at;
      } else if (!e.shiftKey) normal = new THREE.Vector3(0, 0, 1);
    }
    t.plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, at);
  }
  targetDrag = { sent: 0, lift };
  api.canvas.classList.add("dragging");
}

function moveTargetDrag(e) {
  if (!targetDrag) return;
  const t = overlays.target();
  const hit = new THREE.Vector3();
  if (!api.rayAt(e.clientX, e.clientY).ray.intersectPlane(t.plane, hit)) return;
  if (targetDrag.lift) hit.setX(targetDrag.lift.x).setY(targetDrag.lift.y);
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
