// robotforge viewer: a thin three.js client. Physics runs in the Python server;
// this page draws the bodies it streams and sends user input back.
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- renderer setup

const canvas = $("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene3 = new THREE.Scene();
const BG_TOP = "#2b323c";
const BG_BOTTOM = "#1a1f26";
const BG_HORIZON = "#232931";
scene3.background = makeGradientTexture(BG_TOP, BG_BOTTOM);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 500);
camera.up.set(0, 0, 1); // MuJoCo is Z-up
camera.position.set(1.6, -1.6, 1.0);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x3a3328, 1.1);
hemi.position.set(0, 0, 1);
scene3.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.02;
scene3.add(sun, sun.target);

function makeGradientTexture(top, bottom) {
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 256;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, top);
  grad.addColorStop(1, bottom);
  g.fillStyle = grad;
  g.fillRect(0, 0, 4, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// ---------------------------------------------------------------- world state

let sceneDesc = null; // last scene description from the server
let bodyNodes = []; // THREE.Group per body, index = MuJoCo body id
let worldRoot = null;
let firstScene = true;
let pendingFit = null; // "camera" | "lights" | null
let showCollision = false;

function rgbaColor(rgba) {
  return new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);
}

function checkerTexture(rgba) {
  // Low-contrast checker: enough to read motion and scale, quiet enough not to dominate.
  const base = rgbaColor(rgba);
  const light = base.clone().offsetHSL(0, 0, 0.018).getStyle();
  const dark = base.clone().offsetHSL(0, 0, -0.018).getStyle();
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  g.fillStyle = light;
  g.fillRect(0, 0, 128, 128);
  g.fillStyle = dark;
  g.fillRect(0, 0, 64, 64);
  g.fillRect(64, 64, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return tex;
}

function geometryFor(geom, meshGeometries) {
  const s = geom.size;
  switch (geom.type) {
    case "sphere":
      return new THREE.SphereGeometry(s[0], 32, 16);
    case "ellipsoid": {
      const g = new THREE.SphereGeometry(1, 32, 16);
      g.scale(s[0], s[1], s[2]);
      return g;
    }
    case "capsule": {
      const g = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 24);
      g.rotateX(Math.PI / 2); // three.js capsules run along Y, MuJoCo along Z
      return g;
    }
    case "cylinder": {
      const g = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 32);
      g.rotateX(Math.PI / 2);
      return g;
    }
    case "box":
      return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case "mesh":
      return meshGeometries.get(geom.mesh) ?? null;
    default:
      return null;
  }
}

function planeObject(geom, extent) {
  const half = geom.size[0] > 0 ? geom.size[0] : Math.max(20, extent * 20);
  const halfY = geom.size[1] > 0 ? geom.size[1] : half;
  const tex = checkerTexture(geom.rgba);
  const cell = Math.max(0.25, extent / 4);
  tex.repeat.set((2 * half) / (2 * cell), (2 * halfY) / (2 * cell));
  const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95, metalness: 0 });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2 * half, 2 * halfY), mat);
  mesh.receiveShadow = true;
  return mesh;
}

function buildMeshGeometries(desc, buffer) {
  const floats = new Float32Array(buffer);
  const out = new Map();
  for (const entry of desc.meshes) {
    const n = entry.count;
    const pos = floats.subarray(entry.offset, entry.offset + 3 * n);
    const nor = floats.subarray(entry.offset + 3 * n, entry.offset + 6 * n);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    let hasNormals = false;
    for (let i = 0; i < Math.min(nor.length, 30); i++) if (nor[i] !== 0) hasNormals = true;
    if (hasNormals) g.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
    else g.computeVertexNormals();
    out.set(entry.id, g);
  }
  return out;
}

function disposeWorld() {
  if (!worldRoot) return;
  worldRoot.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      if (o.material.map) o.material.map.dispose();
      o.material.dispose();
    }
  });
  scene3.remove(worldRoot);
  worldRoot = null;
}

async function applyScene(desc) {
  let buffer = new ArrayBuffer(0);
  if (desc.meshes.length > 0) {
    const res = await fetch(`mesh/${desc.version}.bin`);
    if (!res.ok) return; // a newer scene is already on its way
    buffer = await res.arrayBuffer();
  }
  const meshGeometries = buildMeshGeometries(desc, buffer);
  disposeWorld();
  sceneDesc = desc;
  worldRoot = new THREE.Group();
  bodyNodes = desc.bodies.map((b) => {
    const node = new THREE.Group();
    node.userData.body = b.id;
    worldRoot.add(node);
    return node;
  });

  const extent = Math.max(desc.stat.extent, 0.1);
  for (const geom of desc.geoms) {
    let obj;
    if (geom.type === "plane") {
      obj = planeObject(geom, extent);
    } else {
      const g = geometryFor(geom, meshGeometries);
      if (!g) continue;
      const transparent = geom.rgba[3] < 1;
      const mat = new THREE.MeshStandardMaterial({
        color: rgbaColor(geom.rgba),
        roughness: 0.55,
        metalness: 0.05,
        transparent,
        opacity: geom.rgba[3],
        depthWrite: !transparent,
      });
      obj = new THREE.Mesh(g, mat);
      obj.castShadow = true;
      obj.receiveShadow = true;
    }
    obj.position.fromArray(geom.pos);
    obj.quaternion.set(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]);
    obj.userData.body = geom.body;
    obj.userData.group = geom.group;
    obj.userData.isPlane = geom.type === "plane";
    obj.visible = geom.group < desc.visibleGroupLimit || showCollision;
    bodyNodes[geom.body].add(obj);
  }
  scene3.add(worldRoot);

  // Lights and (first time) camera are fitted once real poses arrive, see applyPoses.
  pendingFit = firstScene ? "camera" : "lights";
  firstScene = false;
  renderActuators(desc.actuators);
  renderDrive(desc.robots || [], desc.learned || {});
  follow = null;
}

function contentSphere() {
  const box = new THREE.Box3();
  if (worldRoot) {
    worldRoot.updateMatrixWorld(true);
    worldRoot.traverse((o) => {
      if (o.isMesh && o.visible && !o.userData.isPlane) box.expandByObject(o);
    });
  }
  if (box.isEmpty() && sceneDesc) {
    return new THREE.Sphere(new THREE.Vector3().fromArray(sceneDesc.stat.center), sceneDesc.stat.extent);
  }
  return box.getBoundingSphere(new THREE.Sphere());
}

function fitLights(sphere) {
  const c = sphere.center;
  const r = Math.max(sphere.radius, 0.2) * 1.5;
  // Fade the (often infinite) floor into the background instead of a hard horizon.
  scene3.fog = new THREE.Fog(BG_HORIZON, r * 5, r * 18);
  sun.position.copy(c).add(new THREE.Vector3(0.6, -0.8, 1.6).multiplyScalar(r * 1.5));
  sun.target.position.copy(c);
  const cam = sun.shadow.camera;
  cam.left = cam.bottom = -r;
  cam.right = cam.top = r;
  cam.near = 0.01;
  cam.far = r * 6;
  cam.updateProjectionMatrix();
}

function frameCamera() {
  const sphere = contentSphere();
  const r = Math.max(sphere.radius, 0.08);
  const vfov = THREE.MathUtils.degToRad(camera.fov);
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  const d = (r / Math.sin(Math.min(vfov, hfov) / 2)) * 1.05;
  camera.position.copy(sphere.center).add(new THREE.Vector3(1.0, -1.3, 0.8).normalize().multiplyScalar(d));
  controls.target.copy(sphere.center);
  camera.near = Math.max(d / 1000, 0.001);
  camera.far = d * 200;
  camera.updateProjectionMatrix();
  controls.update();
}

function applyPoses(buffer) {
  if (!sceneDesc) return;
  const f = new Float32Array(buffer);
  const nb = sceneDesc.nbody;
  if (f.length !== 1 + 7 * nb) return; // frame from a different model version
  latestTime = f[0];
  const qOff = 1 + 3 * nb;
  for (let b = 0; b < nb; b++) {
    const node = bodyNodes[b];
    node.position.set(f[1 + 3 * b], f[2 + 3 * b], f[3 + 3 * b]);
    node.quaternion.set(f[qOff + 4 * b + 1], f[qOff + 4 * b + 2], f[qOff + 4 * b + 3], f[qOff + 4 * b]);
  }
  if (pendingFit) {
    const sphere = contentSphere();
    fitLights(sphere);
    if (pendingFit === "camera") frameCamera();
    pendingFit = null;
  }
}

// ---------------------------------------------------------------- connection

let ws = null;
let retryMs = 500;
let latestTime = 0;

function setConn(state, text) {
  const el = $("conn");
  el.classList.toggle("ok", state === "ok");
  el.classList.toggle("bad", state === "bad");
  $("conn-text").textContent = text;
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const base = location.pathname.replace(/[^/]*$/, "");
  ws = new WebSocket(`${proto}://${location.host}${base}ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    retryMs = 500;
    setConn("ok", "live");
  };
  ws.onclose = () => {
    setConn("bad", "reconnecting");
    setTimeout(connect, retryMs);
    retryMs = Math.min(retryMs * 2, 5000);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") {
      applyPoses(ev.data);
      return;
    }
    const msg = JSON.parse(ev.data);
    if (msg.type === "scene") applyScene(msg);
    else if (msg.type === "state") applyState(msg);
    else if (msg.type === "error") toast(msg.message);
  };
}

// ---------------------------------------------------------------- UI

let paused = false;
const sliderEls = new Map();
let activeSlider = null;

function applyState(s) {
  paused = s.paused;
  const active = (s.behaviors || []).find((b) => b.robot === $("drive-robot").value);
  $("drive-status").textContent =
    active && "speed" in active ? `${active.speed >= 0 ? " " : ""}${active.speed.toFixed(2)} m/s` : "";
  syncWalker(active);
  $("btn-play").textContent = paused ? "Play" : "Pause";
  $("stat-rtf").textContent = paused ? "paused" : `${s.rtf.toFixed(2)}x`;
  const src = s.source || "";
  $("source").textContent = src.split(/[\\/]/).pop();
  $("source").title = src;
  $("speed").value = String(s.speed);
  s.ctrl.forEach((v, i) => {
    const el = sliderEls.get(i);
    if (el && el.input !== activeSlider) {
      el.input.value = v;
      el.value.textContent = v.toFixed(2);
    }
  });
}

function sliderRow(a, label) {
  const row = document.createElement("div");
  row.className = "slider-row";
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = label.replaceAll("_", " ");
  name.title = a.name;
  const value = document.createElement("span");
  value.className = "value";
  value.textContent = "0.00";
  const input = document.createElement("input");
  input.type = "range";
  input.min = a.range[0];
  input.max = a.range[1];
  input.step = (a.range[1] - a.range[0]) / 500 || 0.001;
  input.value = Math.min(Math.max(0, a.range[0]), a.range[1]);
  input.setAttribute("aria-label", a.name);
  input.addEventListener("pointerdown", () => (activeSlider = input));
  input.addEventListener("pointerup", () => (activeSlider = null));
  input.addEventListener("input", () => {
    value.textContent = Number(input.value).toFixed(2);
    send({ type: "ctrl", id: a.id, value: Number(input.value) });
  });
  row.append(name, value, input);
  sliderEls.set(a.id, { input, value });
  return row;
}

function renderActuators(actuators) {
  const box = $("actuators");
  box.replaceChildren();
  sliderEls.clear();
  $("act-count").textContent = actuators.length ? String(actuators.length) : "";
  if (!actuators.length) {
    const p = document.createElement("div");
    p.className = "empty";
    p.textContent = "Nothing to control yet. Robots you add will show their joints here.";
    box.append(p);
    return;
  }
  // "dog/front_left_hip" -> group "dog"; names without a prefix go in one unnamed group.
  const groups = new Map();
  for (const a of actuators) {
    const cut = a.name.indexOf("/");
    const group = cut > 0 ? a.name.slice(0, cut) : "";
    const label = cut > 0 ? a.name.slice(cut + 1) : a.name;
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push([a, label]);
  }
  if (groups.size === 1 && groups.has("")) {
    for (const [a, label] of groups.get("")) box.append(sliderRow(a, label));
    return;
  }
  let first = true;
  for (const [group, items] of groups) {
    const details = document.createElement("details");
    details.className = "group";
    details.open = first;
    first = false;
    const summary = document.createElement("summary");
    const title = document.createElement("span");
    title.textContent = group || "other";
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = String(items.length);
    summary.append(title, n);
    details.append(summary);
    for (const [a, label] of items) details.append(sliderRow(a, label));
    box.append(details);
  }
}

function setPanelCollapsed(collapsed) {
  $("panel").classList.toggle("collapsed", collapsed);
  $("panel-toggle").setAttribute("aria-expanded", String(!collapsed));
  try {
    localStorage.setItem("robotforge.panelCollapsed", collapsed ? "1" : "0");
  } catch {
    // storage may be unavailable (private mode, sandboxed iframe); the panel still works
  }
}

function initialPanelCollapsed() {
  try {
    const saved = localStorage.getItem("robotforge.panelCollapsed");
    if (saved !== null) return saved === "1";
  } catch {
    // ignore and fall back to the width rule
  }
  return window.innerWidth < 900;
}

setPanelCollapsed(initialPanelCollapsed());
$("panel-toggle").addEventListener("click", () => setPanelCollapsed(!$("panel").classList.contains("collapsed")));

// ---------------------------------------------------------------- keyboard driving

const DRIVE_KEYS = {
  w: "fwd", arrowup: "fwd", s: "back", arrowdown: "back", a: "left", arrowleft: "left", d: "right", arrowright: "right",
};
const pressed = new Set();
let mobileRobots = [];
let lastDrive = "";

let learnedWalkers = {}; // robot -> {policy, max_speed, active}: robots that can walk with a learned policy

function renderDrive(robots, learned = {}) {
  learnedWalkers = learned;
  mobileRobots = robots.filter((r) => ["quadruped", "rover", "humanoid"].includes(r.kind));
  const box = $("drive");
  box.hidden = mobileRobots.length === 0;
  const select = $("drive-robot");
  const previous = select.value;
  select.replaceChildren(
    ...mobileRobots.map((r) => {
      const o = document.createElement("option");
      o.value = r.name;
      o.textContent = r.name;
      return o;
    })
  );
  if (mobileRobots.some((r) => r.name === previous)) select.value = previous;
  lastDrive = "";
  renderWalker();
}

let walkerPendingUntil = 0; // after a switch, trust the user's pick until the server has answered

function renderWalker() {
  const learned = learnedWalkers[$("drive-robot").value];
  const pick = $("drive-walker");
  pick.hidden = !learned;
  if (learned) {
    pick.value = learned.active ? "rl" : "gait";
    pick.options[1].title = learned.policy;
  }
}

function syncWalker(active) {
  // Show what really drives the robot (a switch can fail, e.g. without the optional onnxruntime).
  const learned = learnedWalkers[$("drive-robot").value];
  if (!learned || performance.now() < walkerPendingUntil) return;
  learned.active = active?.walker === "rl";
  $("drive-walker").value = learned.active ? "rl" : "gait";
}

function setWalker(walker) {
  const robot = mobileRobots.find((r) => r.name === $("drive-robot").value);
  const learned = learnedWalkers[robot?.name];
  if (!robot || !learned) return;
  learned.active = walker === "rl";
  walkerPendingUntil = performance.now() + 1000;
  lastDrive = "";
  send({ type: "teleop", robot: robot.name, speed: 0, turn: 0, walker });
}

function driveCommand() {
  const robot = mobileRobots.find((r) => r.name === $("drive-robot").value);
  if (!robot) return null;
  const fwd = (pressed.has("fwd") ? 1 : 0) - (pressed.has("back") ? 1 : 0);
  const left = (pressed.has("left") ? 1 : 0) - (pressed.has("right") ? 1 : 0);
  const learned = learnedWalkers[robot.name];
  const top = (learned?.active ? learned.max_speed : robot.max_speed) ?? 0.3;
  const turnRate = robot.kind === "rover" ? 2.0 : 1.0;
  return { type: "teleop", robot: robot.name, speed: fwd * top * 0.6, turn: left * turnRate };
}

function sendDrive() {
  const cmd = driveCommand();
  if (!cmd) return;
  const key = `${cmd.robot}:${cmd.speed}:${cmd.turn}`;
  if (key === lastDrive) return;
  lastDrive = key;
  send(cmd);
  for (const k of document.querySelectorAll("#drive kbd")) {
    const dir = { W: "fwd", A: "left", S: "back", D: "right" }[k.textContent];
    k.classList.toggle("on", pressed.has(dir));
  }
}

// Follow the driven robot with the camera until the user orbits by hand.
let follow = null; // { body, last: Vector3 }

function rootBodyOf(robot) {
  return sceneDesc?.bodies.find((b) => b.name.startsWith(`${robot}/`) && b.dynamic)?.id ?? null;
}

function startFollow() {
  const body = rootBodyOf($("drive-robot").value);
  if (body === null || !bodyNodes[body]) return;
  if (follow && follow.body === body) return;
  follow = { body, last: bodyNodes[body].position.clone() };
}

function updateFollow() {
  if (!follow || !bodyNodes[follow.body]) return;
  const now = bodyNodes[follow.body].position;
  const delta = now.clone().sub(follow.last);
  delta.z = 0; // follow across the floor, keep the user's camera height
  camera.position.add(delta);
  controls.target.add(delta);
  follow.last.copy(now);
}

controls.addEventListener("start", () => {
  follow = null;
});

function driveKey(e, down) {
  if (mobileRobots.length === 0) return false;
  const dir = DRIVE_KEYS[e.key.toLowerCase()];
  if (!dir) return false;
  e.preventDefault();
  if (down) startFollow();
  if (down) pressed.add(dir);
  else pressed.delete(dir);
  sendDrive();
  return true;
}

window.addEventListener("keyup", (e) => driveKey(e, false));
window.addEventListener("blur", () => {
  pressed.clear();
  sendDrive();
});
$("drive-robot").addEventListener("change", (e) => {
  e.target.blur(); // keep arrow keys for driving, not for changing the selection
  lastDrive = "";
  follow = null;
  renderWalker();
});
$("drive-walker").addEventListener("change", (e) => {
  e.target.blur();
  setWalker(e.target.value);
});

let toastTimer = null;
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 4000);
}

$("btn-play").addEventListener("click", () => send({ type: "cmd", cmd: paused ? "play" : "pause" }));
$("btn-step").addEventListener("click", () => send({ type: "cmd", cmd: "step" }));
$("btn-reset").addEventListener("click", () => send({ type: "cmd", cmd: "reset" }));
$("btn-frame").addEventListener("click", () => frameCamera());
$("speed").addEventListener("change", (e) => send({ type: "speed", value: Number(e.target.value) }));
$("show-collision").addEventListener("change", (e) => {
  showCollision = e.target.checked;
  if (!worldRoot || !sceneDesc) return;
  worldRoot.traverse((o) => {
    if (o.isMesh && !o.userData.isPlane) {
      o.visible = o.userData.group < sceneDesc.visibleGroupLimit || showCollision;
    }
  });
});
$("show-shadows").addEventListener("change", (e) => {
  sun.castShadow = e.target.checked;
});

window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (driveKey(e, true)) return;
  if (e.code === "Space") {
    e.preventDefault();
    send({ type: "cmd", cmd: paused ? "play" : "pause" });
  } else if (e.key === "r" || e.key === "R") {
    send({ type: "cmd", cmd: "reset" });
  } else if (e.key === ".") {
    send({ type: "cmd", cmd: "step" });
  } else if (e.key === "f" || e.key === "F") {
    frameCamera();
  }
});

// ---------------------------------------------------------------- dragging bodies

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane();
let drag = null; // { body, local: Vector3 }
const dragLine = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineBasicMaterial({ color: 0x5aa9ff, depthTest: false, transparent: true })
);
dragLine.renderOrder = 10;
dragLine.visible = false;
const dragDot = new THREE.Mesh(
  new THREE.SphereGeometry(1, 16, 8),
  new THREE.MeshBasicMaterial({ color: 0x5aa9ff, depthTest: false, transparent: true })
);
dragDot.renderOrder = 11;
dragDot.visible = false;
scene3.add(dragLine, dragDot);

function setPointer(e) {
  // Matrices are normally refreshed by render(); refresh here too so picking never
  // uses stale transforms (e.g. when the tab is throttled and frames are skipped).
  camera.updateMatrixWorld();
  if (worldRoot) worldRoot.updateMatrixWorld();
  const rect = canvas.getBoundingClientRect();
  pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
}

function pickBody(e) {
  if (!worldRoot || !sceneDesc) return null;
  setPointer(e);
  const hits = raycaster.intersectObjects(worldRoot.children, true);
  for (const h of hits) {
    if (!h.object.visible || h.object.userData.isPlane) continue;
    const body = h.object.userData.body;
    if (body > 0 && sceneDesc.bodies[body]?.dynamic) return { body, point: h.point };
    return null; // first visible hit is static: don't reach through it
  }
  return null;
}

canvas.addEventListener(
  "pointerdown",
  (e) => {
    if (e.button !== 0) return;
    const hit = pickBody(e);
    if (!hit) return;
    controls.enabled = false; // must happen before OrbitControls sees this event
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      // synthetic or already-released pointers cannot be captured; dragging still works
    }
    canvas.classList.add("dragging");
    const node = bodyNodes[hit.body];
    const local = node.worldToLocal(hit.point.clone());
    drag = { body: hit.body, local };
    dragPlane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), hit.point);
    send({ type: "drag_start", body: hit.body, point: hit.point.toArray() });
    updateDragVisual(hit.point);
  },
  { capture: true }
);

canvas.addEventListener("pointermove", (e) => {
  if (!drag) {
    canvas.classList.toggle("hover-body", !!pickBody(e));
    return;
  }
  setPointer(e);
  const target = new THREE.Vector3();
  if (raycaster.ray.intersectPlane(dragPlane, target)) {
    send({ type: "drag_move", target: target.toArray() });
    updateDragVisual(target);
  }
});

function endDrag(e) {
  if (!drag) return;
  drag = null;
  controls.enabled = true;
  canvas.classList.remove("dragging");
  if (e && canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  dragLine.visible = dragDot.visible = false;
  send({ type: "drag_end" });
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

let dragTarget = new THREE.Vector3();
function updateDragVisual(target) {
  dragTarget.copy(target);
  dragLine.visible = dragDot.visible = true;
}

function refreshDragVisual() {
  if (!drag) return;
  const anchor = bodyNodes[drag.body].localToWorld(drag.local.clone());
  const pos = dragLine.geometry.attributes.position;
  pos.setXYZ(0, anchor.x, anchor.y, anchor.z);
  pos.setXYZ(1, dragTarget.x, dragTarget.y, dragTarget.z);
  pos.needsUpdate = true;
  dragLine.geometry.computeBoundingSphere();
  dragDot.position.copy(dragTarget);
  dragDot.scale.setScalar(Math.max(camera.position.distanceTo(dragTarget), 0.1) * 0.006);
}

// ---------------------------------------------------------------- render loop

let frames = 0;
let fpsT0 = performance.now();

function loop() {
  requestAnimationFrame(loop);
  updateFollow();
  controls.update();
  refreshDragVisual();
  renderer.render(scene3, camera);
  frames++;
  const now = performance.now();
  if (now - fpsT0 > 500) {
    $("stat-fps").textContent = String(Math.round((frames * 1000) / (now - fpsT0)));
    $("stat-time").textContent = latestTime.toFixed(2);
    frames = 0;
    fpsT0 = now;
  }
}

connect();
loop();

// Expose a tiny hook for automated checks (headless verification reads this).
window.__robotforge = {
  get scene() { return sceneDesc; },
  get time() { return latestTime; },
  bodyPosition: (id) => bodyNodes[id]?.position.toArray(),
  pick: (clientX, clientY) => pickBody({ clientX, clientY })?.body ?? null,
  cameraTarget: () => controls.target.toArray(),
  // CSS-pixel position of a body's origin, for scripted pointer tests.
  screenOf: (id) => {
    const node = bodyNodes[id];
    if (!node) return null;
    camera.updateMatrixWorld();
    worldRoot.updateMatrixWorld();
    const v = node.getWorldPosition(new THREE.Vector3()).project(camera);
    const rect = canvas.getBoundingClientRect();
    return [rect.left + ((v.x + 1) / 2) * rect.width, rect.top + ((1 - v.y) / 2) * rect.height];
  },
};
