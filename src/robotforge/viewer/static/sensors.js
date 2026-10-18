// The viewer draws what the robots' sensors sense. In the world: a lidar's rays and the points it hit, colored by
// distance; a camera's frustum on the robot; touch pads that light up; force and contact arrows; a rangefinder's
// beam. Beside the view: what the selected camera sees (rendered here in the browser from the camera's pose, so
// it works where the server has no OpenGL) and live plots of IMUs, force-torque sensors and encoders.
//
// The server sends a "sensors" message ten times a second while the world has sensors (viewer/server.py lists its
// fields, sensorkit.js the ones read here). What sits on a robot follows its body in every frame drawn; what a
// sensor measured (a scan's points) stays where it was measured. Sensors put on with add_sensor show from the
// start; a robot's own (a template's IMU and foot pads) are listed in the robots drawer, off until switched on.

import * as THREE from "three";
import { t } from "./i18n.js";
import { drawLines, plotRange } from "./plot.js";
import {
  arrowLength,
  depthPicture,
  frustumCorners,
  jointGroups,
  nearColor,
  pullBack,
  quantileSpan,
  rayPicks,
  spanOf,
  steadySpan,
  thingColor,
  thingName,
  unpackDepth,
} from "./sensorkit.js";

const $ = (id) => document.getElementById(id);

const MAX_POINTS = 1200; // hit points the server sends per lidar at most (sensors/lidar.py VIEW_POINTS)
const MAX_RAYS = { lidar: 360, lidar3d: 150 }; // rays drawn: a 3D scan's every ray would be a solid fog
const RAY_ALPHA = { lidar: [0.6, 0.12], lidar3d: [0.45, 0.04] }; // opacity of a ray at the sensor and at its hit
const POINT_PX = 3.2; // CSS px across a hit point
const PULL_BACK = 0.02; // m: a point's dot sits this far in front of the surface it hit, not half inside it
const FRUSTUM_HALF = 0.045; // m: half the height of a camera's frustum where it ends, whatever its field of view
const ARROW_REACH = 0.22; // m: the arrow of the largest force seen lately
const ARROW_FLOOR = { touch: 5, force_torque: 2 }; // N: a scale never finer than this, so a whisper stays short
const PEAK_DECAY = 0.985; // per message: the largest force seen fades with a half-life of about five seconds
const PLOT_WINDOW = 8; // s of history in a plot
const PIP_EVERY_MS = { rgb: 66, depth: 100, segmentation: 100 }; // the picture is drawn again this often
const PIP_MAX = [640, 480]; // px: a bigger camera is drawn at this size in the browser
const SMALL = () => window.innerWidth <= 600 || window.innerHeight <= 560; // one view at a time, in tabs

// What each plotted type shows, a channel at a time. span is the smallest height of its axis, so a sensor's noise
// at rest is not drawn as a swing.
const PLOTS = {
  imu: [
    { key: "accel", field: "accel_m_s2", unit: "m/s²", span: 2, axes: ["x", "y", "z"] },
    { key: "gyro", field: "gyro_rad_s", unit: "rad/s", span: 0.5, axes: ["x", "y", "z"] },
    { key: "tilt", field: "orientation_deg", unit: "°", span: 4, axes: ["roll", "pitch"] },
  ],
  force_torque: [
    { key: "force", field: "force_n", unit: "N", span: 2, axes: ["x", "y", "z"] },
    { key: "torque", field: "torque_nm", unit: "N·m", span: 0.2, axes: ["x", "y", "z"] },
  ],
  encoders: [{ key: "angles", field: "rad", unit: "rad", span: 0.2, axes: [] }], // its joints come from the sensor
};
const AXIS_COLORS = ["--bad", "--plot-green", "--plot-blue"]; // x, y, z drawn red, green, blue as axes always are
const JOINT_COLORS = ["--plot-blue", "--plot-violet", "--plot-green", "--amber", "--ink", "--muted"];

let api = null; // set by initSensors
const root = new THREE.Group(); // everything drawn in the world; hidden in a lesson
root.name = "sensors";
let version = -1; // the model the page has loaded
let entries = new Map(); // sensor name -> its latest entry
let simTime = 0;
const drawings = new Map(); // sensor name -> { group, update(entry), frame() }
const histories = new Map(); // sensor name -> [[t, entry]], oldest first
const peaks = new Map(); // sensor name -> the largest force seen lately (N)
const chosen = {}; // sensor name -> shown or not, as switched in the drawer; otherwise it shows if it was added
let arrowsOn = true;
let listKey = ""; // what the drawer's list shows, to rebuild it only when that changes

export function initSensors(context) {
  api = context;
  api.scene.add(root);
  $("sensor-views").setAttribute("aria-label", t("sensor_views"));
  arrowsOn = api.storageGet("robotforge.sensor-arrows") !== "0";
  $("show-forces").checked = arrowsOn;
  $("show-forces").addEventListener("change", (e) => {
    arrowsOn = e.target.checked;
    api.storageSet("robotforge.sensor-arrows", arrowsOn ? "1" : "0");
  });
  window.addEventListener("resize", () => views.layout());
}

// ---------------------------------------------------------------- state

const robotKey = () => api.selected() ?? "";
const mine = () => [...entries.values()].filter((e) => e.robot === robotKey());

function shown(entry) {
  if (!(entry.name in chosen)) {
    const saved = api.storageGet(`robotforge.sensor.${entry.name}`);
    chosen[entry.name] = saved === null ? !!entry.added : saved === "1";
  }
  return chosen[entry.name];
}

function setShown(name, on) {
  chosen[name] = on;
  api.storageSet(`robotforge.sensor.${name}`, on ? "1" : "0");
  const entry = entries.get(name);
  if (entry?.type === "camera" && on) pip.name = name; // the camera just switched on is the one you want to see
  sensorsSelect();
}

// A new model: what was drawn belongs to the old one.
export function sensorsScene(desc) {
  version = desc.version;
  for (const d of drawings.values()) dispose(d.group);
  drawings.clear();
  histories.clear();
  peaks.clear();
  entries = new Map();
  pip.meshes = pip.geoms = null;
  sight.boxes = null;
  sight.memo.clear();
  pip.name = null;
  sensorsSelect();
}

export function sensorsUpdate(msg) {
  if (msg.version !== undefined && msg.version !== version) return; // about a model the page has not loaded yet
  if (msg.time < simTime - 0.5) {
    histories.clear(); // time went back: the world was reset
    peaks.clear();
  }
  simTime = msg.time;
  const next = new Map();
  for (const e of msg.sensors) next.set(e.name, e);
  const changed = next.size !== entries.size || [...next.keys()].some((k) => !entries.has(k));
  entries = next;
  for (const e of entries.values()) {
    if (e.status === "waiting") continue;
    remember(e);
    if (e.type === "touch" || e.type === "force_torque") {
      const force = e.type === "touch" ? Math.max(...e.force_n, 0) : Math.hypot(...e.force_n);
      peaks.set(e.name, Math.max(ARROW_FLOOR[e.type], force, (peaks.get(e.name) ?? 0) * PEAK_DECAY));
    }
    if (shown(e) && !api.inLesson()) drawing(e)?.update(e);
  }
  for (const [name, d] of drawings) {
    if (!entries.has(name)) {
      dispose(d.group);
      drawings.delete(name);
    }
  }
  if (changed) sensorsSelect();
  else renderSummaries();
  views.plot();
}

function remember(e) {
  if (!PLOTS[e.type]) return;
  const list = histories.get(e.name) ?? [];
  if (list.length && list[list.length - 1][0] === simTime) return; // paused: the same sample again
  list.push([simTime, e]);
  while (list.length && list[0][0] < simTime - PLOT_WINDOW - 1) list.shift();
  histories.set(e.name, list);
}

// ---------------------------------------------------------------- the robots drawer: the selected robot's sensors

const TYPE_KEYS = { camera: (e) => `sensor_camera_${e.mode}` };

function typeLabel(e) {
  return t(TYPE_KEYS[e.type]?.(e) ?? `sensor_${e.type}`);
}

function summary(e) {
  if (e.status === "waiting") return t("sensor_waiting");
  const num = (v, d = 2) => Number(v).toFixed(d);
  switch (e.type) {
    case "lidar":
    case "lidar3d":
      return t(e.type === "lidar3d" ? "sensor_points" : "sensor_hits", { n: e.hits ?? e.points.length / 3 });
    case "camera":
      return `${e.size[0]} x ${e.size[1]}, ${Math.round(e.fov_deg)}°`;
    case "imu":
      return `${num(Math.hypot(...e.accel_m_s2))} m/s²`;
    case "force_torque":
      return e.payload_kg !== undefined ? t("sensor_holds", { kg: num(Math.max(0, e.payload_kg)) }) : `${num(Math.hypot(...e.force_n), 1)} N`;
    case "touch":
      return e.total_n > 0.05 ? `${num(e.total_n, 1)} N` : t("sensor_no_contact");
    case "encoders":
      return t("sensor_joints", { n: Object.keys(e.rad ?? {}).length + Object.keys(e.m ?? {}).length });
    case "rangefinder":
      return e.distance_m === null ? t("sensor_nothing", { m: e.range_m }) : `${num(e.distance_m)} m`;
    case "gps":
    case "odometry":
      return `x ${num(e.position_m[0])}, y ${num(e.position_m[1])} m`;
    default:
      return "";
  }
}

// The selected robot changed, or its sensors did: list them in the drawer and lay out the views.
export function sensorsSelect() {
  if (!api) return;
  const list = mine();
  const key = `${robotKey()}|${list.map((e) => e.name).join(",")}`;
  if (key !== listKey) {
    listKey = key;
    $("sensor-list").replaceChildren(
      ...list.map((e) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "choice";
        b.dataset.sensor = e.name;
        b.setAttribute("aria-pressed", String(shown(e)));
        const name = document.createElement("span");
        name.className = "choice-title";
        name.textContent = e.name.slice(e.robot ? e.robot.length + 1 : 0);
        const desc = document.createElement("span");
        desc.className = "choice-desc";
        b.append(name, desc);
        b.addEventListener("click", () => setShown(e.name, !shown(e)));
        return b;
      })
    );
    $("sensor-count").textContent = list.length ? String(list.length) : "";
    $("sensor-section").hidden = list.length === 0;
    $("sensor-arrows").hidden = !list.some((e) => e.type === "touch" || e.type === "force_torque");
  }
  renderSummaries();
  views.build(list.filter(shown));
}

function renderSummaries() {
  for (const b of $("sensor-list").children) {
    const e = entries.get(b.dataset.sensor);
    if (!e) continue;
    b.setAttribute("aria-pressed", String(shown(e))); // in place: a rebuilt list would lose the focus just clicked
    const text = [typeLabel(e), summary(e)].filter(Boolean).join(" · ");
    const desc = b.querySelector(".choice-desc");
    if (desc.textContent !== text) desc.textContent = text;
  }
}

// ---------------------------------------------------------------- drawings in the world

const tmp = new THREE.Vector3();
const tmpQ = new THREE.Quaternion();
const Z = new THREE.Vector3(0, 0, 1);
const Y = new THREE.Vector3(0, 1, 0);
const srgb = new THREE.Color();

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawing(e) {
  let d = drawings.get(e.name);
  if (d) return d;
  const make = MAKERS[e.type];
  if (!make) return null;
  d = make(e);
  root.add(d.group);
  drawings.set(e.name, d);
  return d;
}

function dispose(group) {
  root.remove(group);
  group.traverse((o) => {
    o.geometry?.dispose();
    o.material?.map?.dispose();
    o.material?.dispose();
  });
}

// Where something on a body is now: the body's pose in this frame, times the place on it.
function onBody(e, localPos, localQuat, outPos, outQuat) {
  const node = api.bodyNode(e.body);
  if (!node) return false;
  outPos.fromArray(localPos).applyQuaternion(node.quaternion).add(node.position);
  if (outQuat) outQuat.copy(node.quaternion).multiply(tmpQ.set(localQuat[1], localQuat[2], localQuat[3], localQuat[0]));
  return true;
}

// A round, soft-edged dot for points (THREE.Points draws squares otherwise).
let dotTex = null;
function dotTexture() {
  if (dotTex) return dotTex;
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.6, "rgba(255,255,255,1)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  dotTex = new THREE.CanvasTexture(c);
  return dotTex;
}

function linearColor(rgb) {
  srgb.setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace); // stored linear, as three.js works in
  return srgb;
}

function lidarDrawing(first) {
  const group = new THREE.Group();
  const points = new THREE.BufferGeometry();
  points.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * MAX_POINTS), 3));
  points.setAttribute("color", new THREE.BufferAttribute(new Float32Array(3 * MAX_POINTS), 3));
  const dots = new THREE.Points(
    points,
    new THREE.PointsMaterial({
      size: POINT_PX,
      sizeAttenuation: false,
      vertexColors: true,
      map: dotTexture(),
      alphaTest: 0.4,
      transparent: true,
      depthWrite: false,
      toneMapped: false,
    })
  );
  dots.frustumCulled = false; // its bounds change with every scan
  const rays = new THREE.BufferGeometry();
  const most = MAX_RAYS[first.type];
  rays.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6 * most), 3));
  rays.setAttribute("color", new THREE.BufferAttribute(new Float32Array(8 * most), 4)); // with alpha: rays fade out
  const lines = new THREE.LineSegments(
    rays,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, toneMapped: false })
  );
  lines.frustumCulled = false;
  dots.renderOrder = 6;
  lines.renderOrder = 5;
  group.add(lines, dots);
  let span = null;
  return {
    group,
    update(e) {
      const flat = e.points;
      const n = Math.min(flat.length / 3, MAX_POINTS);
      const o = e.origin;
      const dist = new Float32Array(n);
      for (let i = 0; i < n; i++) dist[i] = Math.hypot(flat[3 * i] - o[0], flat[3 * i + 1] - o[1], flat[3 * i + 2] - o[2]);
      span = steadySpan(span, quantileSpan(dist));
      const pos = points.attributes.position.array;
      const col = points.attributes.color.array;
      for (let i = 0; i < n; i++) {
        const p = pullBack(o, [flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]], PULL_BACK);
        pos.set(p, 3 * i);
        const c = linearColor(nearColor(dist[i], span[0], span[1]));
        col[3 * i] = c.r;
        col[3 * i + 1] = c.g;
        col[3 * i + 2] = c.b;
      }
      points.setDrawRange(0, n);
      points.attributes.position.needsUpdate = points.attributes.color.needsUpdate = true;
      const picks = rayPicks(n, most);
      const rpos = rays.attributes.position.array;
      const rcol = rays.attributes.color.array;
      // Each ray strong at the sensor and faint at its far end, where its point shows the hit: a hall's worth of
      // lines drawn evenly crossed the whole view.
      const [near, far] = RAY_ALPHA[first.type];
      picks.forEach((i, k) => {
        rpos.set(o, 6 * k);
        rpos.set(pos.subarray(3 * i, 3 * i + 3), 6 * k + 3);
        rcol.set(col.subarray(3 * i, 3 * i + 3), 8 * k);
        rcol[8 * k + 3] = near;
        rcol.set(col.subarray(3 * i, 3 * i + 3), 8 * k + 4);
        rcol[8 * k + 7] = far;
      });
      rays.setDrawRange(0, 2 * picks.length);
      rays.attributes.position.needsUpdate = rays.attributes.color.needsUpdate = true;
    },
    frame() {},
  };
}

function cameraDrawing(first) {
  const group = new THREE.Group();
  // As tall wherever it looks: a wide lens's frustum drawn as deep as a narrow one's reached past a small robot.
  const depth = FRUSTUM_HALF / Math.tan((first.fov_deg * Math.PI) / 360);
  const corners = frustumCorners(first.fov_deg, first.size[0] / first.size[1], depth);
  const v = [];
  for (const c of corners) v.push(0, 0, 0, ...c);
  for (let i = 0; i < 4; i++) v.push(...corners[i], ...corners[(i + 1) % 4]);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
  const blue = api.color("--plot-blue");
  const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: blue, transparent: true, opacity: 0.9, toneMapped: false }));
  const face = new THREE.Mesh(
    new THREE.BufferGeometry().setFromPoints(corners.map((c) => new THREE.Vector3(...c))).setIndex([0, 1, 2, 0, 2, 3]),
    new THREE.MeshBasicMaterial({ color: blue, transparent: true, opacity: 0.14, side: THREE.DoubleSide, depthWrite: false, toneMapped: false })
  );
  group.add(lines, face);
  group.renderOrder = 7;
  let last = first;
  return {
    group,
    update(e) {
      last = e;
    },
    frame() {
      group.visible = onBody(last, last.local_pos, last.local_quat, group.position, group.quaternion);
      face.visible = pip.name === last.name && views.showsPicture(); // the camera whose picture is up
    },
  };
}

// Pads and arrows are drawn over everything, so a robot's own leg does not hide the force in it. Scenery between them
// and the eye (a rack, a wall) does: they are left out while it stands in the way, judged a few times a second
// against the boxes of the world's fixed shapes (drawn over everything, they showed through the racks).
const SIGHT_EVERY_MS = 150;
const SIGHT_MARGIN = 0.25; // m: scenery this near the point is what it stands on, not something in the way
const sight = { boxes: null, ray: new THREE.Ray(), hit: new THREE.Vector3(), memo: new Map() };

function sceneryBoxes() {
  if (sight.boxes) return sight.boxes;
  const desc = api.desc();
  sight.boxes = [];
  api.world()?.updateMatrixWorld(true);
  api.world()?.traverse((o) => {
    const body = o.userData.body;
    if (!o.isMesh || o.userData.isPlane || o.userData.lift || body === undefined || desc?.bodies[body]?.dynamic) return;
    sight.boxes.push(new THREE.Box3().setFromObject(o));
  });
  return sight.boxes;
}

function behindScenery(key, point) {
  const now = performance.now();
  const memo = sight.memo.get(key);
  if (memo && now - memo.t < SIGHT_EVERY_MS) return memo.hidden;
  const eye = api.camera.position;
  const far = eye.distanceTo(point) - SIGHT_MARGIN;
  let hidden = false;
  if (far > 0) {
    sight.ray.origin.copy(eye);
    sight.ray.direction.copy(point).sub(eye).normalize();
    for (const box of sceneryBoxes()) {
      if (sight.ray.intersectBox(box, sight.hit) && eye.distanceTo(sight.hit) < far && !box.containsPoint(point)) {
        hidden = true;
        break;
      }
    }
  }
  sight.memo.set(key, { t: now, hidden });
  return hidden;
}

// An arrow along +y, `length` long: a shaft and a head, drawn over everything so a force inside a leg still shows.
const SHAFT = new THREE.CylinderGeometry(1, 1, 1, 12).translate(0, 0.5, 0);
const HEAD = new THREE.ConeGeometry(1, 1, 16).translate(0, 0.5, 0);
function arrow(color) {
  const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.92, depthTest: false, depthWrite: false, toneMapped: false });
  const group = new THREE.Group();
  const shaft = new THREE.Mesh(SHAFT, material);
  const head = new THREE.Mesh(HEAD, material);
  shaft.renderOrder = head.renderOrder = 12;
  group.add(shaft, head);
  group.userData = { shaft, head };
  return group;
}

function pointArrow(a, from, dir, length, thick = 0.006) {
  a.visible = length > 0.004;
  if (!a.visible) return;
  a.position.copy(from);
  a.quaternion.setFromUnitVectors(Y, dir);
  const head = Math.min(length * 0.4, thick * 5);
  a.userData.shaft.scale.set(thick, length - head, thick);
  a.userData.head.position.y = length - head;
  a.userData.head.scale.set(thick * 2.4, head, thick * 2.4);
}

function forceDrawing() {
  const group = new THREE.Group();
  const force = arrow(api.color("--plot-violet"));
  group.add(force);
  let last = null;
  const at = new THREE.Vector3();
  const dir = new THREE.Vector3();
  return {
    group,
    update(e) {
      last = e;
    },
    frame() {
      group.visible = !!last && arrowsOn && onBody(last, last.local_pos, last.local_quat, at, null) && !behindScenery(last.name, at);
      if (!group.visible) return;
      const size = Math.hypot(...last.force_n);
      dir.fromArray(last.force_n).normalize();
      pointArrow(force, at, dir, arrowLength(size, peaks.get(last.name), ARROW_REACH));
    },
  };
}

function touchDrawing(first) {
  const group = new THREE.Group();
  const pad = new THREE.Group(); // in the body's frame
  const green = api.color("--plot-green");
  const idle = new THREE.Color(0xffffff);
  const [hx, hy] = first.cell_half;
  const shape = first.cell_shape === "box" ? new THREE.PlaneGeometry(2 * hx * 0.9, 2 * hy * 0.9) : new THREE.SphereGeometry(hx * 1.15, 16, 10);
  const padQuat = new THREE.Quaternion(first.pad_quat[1], first.pad_quat[2], first.pad_quat[3], first.pad_quat[0]);
  const cells = first.cells_local.map((p) => {
    const m = new THREE.Mesh(
      shape,
      // Over everything: a pad lies under a foot, where the foot itself and the floor would hide it.
      new THREE.MeshBasicMaterial({ color: idle.clone(), transparent: true, opacity: 0.3, side: THREE.DoubleSide, depthTest: false, depthWrite: false, toneMapped: false })
    );
    m.position.fromArray(p);
    m.quaternion.copy(padQuat);
    m.renderOrder = 11;
    pad.add(m);
    return m;
  });
  const contact = arrow(green);
  group.add(pad, contact);
  const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(padQuat); // outward, in the body's frame
  let last = first;
  const center = new THREE.Vector3();
  const up = new THREE.Vector3();
  return {
    group,
    update(e) {
      last = e;
      const top = peaks.get(e.name) ?? ARROW_FLOOR.touch;
      e.force_n.forEach((f, i) => {
        const k = Math.min(1, f / Math.max(top, 1e-6));
        cells[i].material.color.copy(idle).lerp(green, Math.min(1, k * 1.6));
        cells[i].material.opacity = 0.3 + 0.65 * Math.min(1, k * 1.6);
      });
    },
    frame() {
      const node = api.bodyNode(last.body);
      group.visible = !!node && !behindScenery(last.name, cells[0].getWorldPosition(center));
      if (!group.visible) return;
      pad.position.copy(node.position);
      pad.quaternion.copy(node.quaternion);
      const total = last.total_n;
      contact.visible = arrowsOn && total > 0.05;
      if (!contact.visible) return;
      center.set(0, 0, 0); // where the pressure is: the zones weighed by their force
      last.force_n.forEach((f, i) => center.addScaledVector(cells[i].position, f / total));
      center.applyQuaternion(node.quaternion).add(node.position);
      // The ground pushes into the pad: against its outward normal (a ball foot's is the world's up).
      if (last.cell_shape === "box") up.copy(normal).applyQuaternion(node.quaternion).negate();
      else up.copy(Z);
      pointArrow(contact, center, up, arrowLength(total, peaks.get(last.name), ARROW_REACH), 0.005);
    },
  };
}

function rangeDrawing() {
  const group = new THREE.Group();
  const blue = api.color("--plot-blue");
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const beam = new THREE.Line(g, new THREE.LineBasicMaterial({ color: blue, transparent: true, opacity: 0.85, toneMapped: false }));
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), new THREE.MeshBasicMaterial({ color: blue, toneMapped: false }));
  beam.frustumCulled = false;
  group.add(beam, dot);
  let last = null;
  const at = new THREE.Vector3();
  const q = new THREE.Quaternion();
  return {
    group,
    update(e) {
      last = e;
    },
    frame() {
      group.visible = !!last && onBody(last, last.local_pos, last.local_quat, at, q);
      if (!group.visible) return;
      const hit = last.distance_m;
      const end = tmp.copy(Z).applyQuaternion(q).multiplyScalar(hit ?? last.range_m).add(at); // a beam looks along its +z
      const p = g.attributes.position;
      p.setXYZ(0, at.x, at.y, at.z);
      p.setXYZ(1, end.x, end.y, end.z);
      p.needsUpdate = true;
      beam.material.opacity = hit === null ? 0.3 : 0.85;
      dot.visible = hit !== null;
      dot.position.copy(end);
    },
  };
}

const MAKERS = { lidar: lidarDrawing, lidar3d: lidarDrawing, camera: cameraDrawing, force_torque: forceDrawing, touch: touchDrawing, rangefinder: rangeDrawing };

// ---------------------------------------------------------------- the picture a camera sees, drawn in the browser

const DEPTH_VERTEX = `
  varying float vDepth;
  void main() {
    vec4 eye = modelViewMatrix * vec4(position, 1.0);
    vDepth = -eye.z;
    gl_Position = projectionMatrix * eye;
  }`;
// Distance along the view (as MuJoCo's depth camera gives it) in 16 bits of the range; blue marks a drawn pixel.
const DEPTH_FRAGMENT = `
  uniform float range;
  varying float vDepth;
  void main() {
    if (vDepth > range) { gl_FragColor = vec4(0.0); return; }
    float q = floor(clamp(vDepth / range, 0.0, 1.0) * 65535.0 + 0.5);
    float hi = floor(q / 256.0);
    gl_FragColor = vec4(hi / 255.0, (q - hi * 256.0) / 255.0, 1.0, 1.0);
  }`;
// The camera's light turned into a picture as the main view's is: Khronos PBR Neutral tone mapping (three.js's
// NeutralToneMapping) and the sRGB curve. Rendering into a texture skips both, and highlights burned out.
const TONE_FRAGMENT = `
  uniform sampler2D light;
  varying vec2 vUv;
  vec3 neutral(vec3 color) {
    const float start = 0.76;
    const float desaturation = 0.15;
    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < start) return color;
    float d = 1.0 - start;
    float newPeak = 1.0 - d * d / (peak + d - start);
    color *= newPeak / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3(newPeak), g);
  }
  vec3 encode(vec3 c) {
    return mix(c * 12.92, pow(c, vec3(1.0 / 2.4)) * 1.055 - 0.055, step(vec3(0.0031308), c));
  }
  void main() {
    vec3 c = clamp(neutral(texture2D(light, vUv).rgb), 0.0, 1.0);
    gl_FragColor = vec4(encode(c), 1.0);
  }`;

const pip = {
  name: null, // the camera whose picture is shown
  cam: new THREE.PerspectiveCamera(60, 4 / 3, 0.02, 50),
  size: [0, 0],
  light: null, // rgb: the light, half floats, before tone mapping
  target: null, // what is read back: 8 bits a channel
  bytes: null,
  depth: null,
  image: null,
  ctx: null,
  span: null,
  seen: [], // segmentation: the things in view, the biggest first
  last: 0,
  busy: false,
  renders: 0,
  ms: 0, // time the last picture took in the page, ms
  meshes: null,
  geoms: null, // geom id -> the scene's geom, for naming what a segmentation picture shows
  depthMaterial: null,
  flat: new Map(), // thing name -> its flat segmentation material
  tone: null,
  setSize(w, h) {
    if (this.size[0] === w && this.size[1] === h) return;
    this.size = [w, h];
    this.light?.dispose();
    this.target?.dispose();
    this.light = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, samples: 4 }); // smooth edges, as the view's
    this.target = new THREE.WebGLRenderTarget(w, h, { depthBuffer: true });
    this.bytes = new Uint8Array(4 * w * h);
    this.depth = new Float32Array(w * h);
    this.image = null;
  },
  meshList() {
    if (this.meshes) return this.meshes;
    this.meshes = [];
    api.world()?.traverse((o) => {
      if (o.isMesh) this.meshes.push(o);
    });
    return this.meshes;
  },
  flatMaterial(mesh) {
    if (mesh.userData.thing === undefined) {
      const desc = api.desc();
      this.geoms ??= new Map((desc?.geoms ?? []).map((g) => [g.id, g]));
      const geom = this.geoms.get(mesh.userData.geom);
      mesh.userData.thing = geom ? thingName(desc.bodies, geom) : "floor";
    }
    const name = mesh.userData.thing;
    let m = this.flat.get(name);
    if (!m) {
      const [r, g, b] = thingColor(name);
      // Linear in, no conversion out (the target has no color space): the bytes are the server's palette exactly.
      m = new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.LinearSRGBColorSpace), toneMapped: false });
      m.userData.thing = name;
      this.flat.set(name, m);
    }
    return m;
  },
  toneQuad() {
    if (this.tone) return this.tone;
    const material = new THREE.ShaderMaterial({
      uniforms: { light: { value: null } },
      vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: TONE_FRAGMENT,
      depthTest: false,
      depthWrite: false,
    });
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
    quad.frustumCulled = false;
    const scene = new THREE.Scene();
    scene.add(quad);
    this.tone = { scene, material, camera: new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1) };
    return this.tone;
  },
  // Draw what camera `e` sees now, into `canvas` (asynchronously read back, so the page never waits on the GPU).
  draw(e, canvas) {
    const now = performance.now();
    if (this.busy || now - this.last < PIP_EVERY_MS[e.mode] || document.hidden) return;
    const node = api.bodyNode(e.body);
    if (!node) return;
    this.last = now;
    const t0 = performance.now();
    const w = Math.min(e.size[0], PIP_MAX[0]);
    const h = Math.round((w * e.size[1]) / e.size[0]);
    this.setSize(w, Math.min(h, PIP_MAX[1]));
    const cam = this.cam;
    cam.fov = e.fov_deg;
    cam.aspect = e.size[0] / e.size[1];
    cam.near = 0.01;
    cam.far = e.mode === "rgb" ? 200 : e.range_m * 1.02;
    cam.updateProjectionMatrix();
    onBody(e, e.local_pos, e.local_quat, cam.position, cam.quaternion);
    cam.updateMatrixWorld();
    const renderer = api.renderer;
    const scene = api.scene;
    const world = api.world();
    // The camera sees the world, not our drawings over it (sensors, the selection ring, paths, a lesson's rig).
    const hidden = [];
    for (const o of scene.children) {
      if (o !== world && !o.isLight && o.visible) {
        o.visible = false;
        hidden.push(o);
      }
    }
    const before = { background: scene.background, fog: scene.fog, target: renderer.getRenderTarget(), clear: renderer.getClearColor(new THREE.Color()), alpha: renderer.getClearAlpha() };
    const shadows = renderer.shadowMap.autoUpdate;
    renderer.shadowMap.autoUpdate = false; // the sun's shadows are the view's, drawn already this frame
    const swapped = [];
    try {
      if (e.mode === "rgb") {
        scene.fog = null; // the view's haze is for its own distances: a camera on a robot saw the far half white
        renderer.setRenderTarget(this.light);
        renderer.render(scene, cam);
        const tone = this.toneQuad();
        tone.material.uniforms.light.value = this.light.texture;
        renderer.setRenderTarget(this.target);
        renderer.render(tone.scene, tone.camera);
      } else {
        scene.background = null;
        scene.fog = null;
        renderer.setClearColor(0x000000, 0);
        if (e.mode === "depth") {
          this.depthMaterial ??= new THREE.ShaderMaterial({ uniforms: { range: { value: 10 } }, vertexShader: DEPTH_VERTEX, fragmentShader: DEPTH_FRAGMENT });
          this.depthMaterial.uniforms.range.value = e.range_m;
        }
        for (const m of this.meshList()) {
          swapped.push([m, m.material]);
          m.material = e.mode === "depth" ? this.depthMaterial : this.flatMaterial(m);
        }
        renderer.setRenderTarget(this.target);
        renderer.clear();
        renderer.render(scene, cam);
      }
    } finally {
      for (const [m, material] of swapped) m.material = material;
      for (const o of hidden) o.visible = true;
      scene.background = before.background;
      scene.fog = before.fog;
      renderer.setClearColor(before.clear, before.alpha);
      renderer.setRenderTarget(before.target);
      renderer.shadowMap.autoUpdate = shadows;
    }
    const [W, H] = this.size;
    this.busy = true;
    const done = () => {
      this.busy = false;
      this.renders++;
      this.develop(e, canvas);
      this.ms = performance.now() - t0;
    };
    if (renderer.readRenderTargetPixelsAsync) {
      renderer.readRenderTargetPixelsAsync(this.target, 0, 0, W, H, this.bytes).then(done, () => (this.busy = false));
    } else {
      renderer.readRenderTargetPixels(this.target, 0, 0, W, H, this.bytes);
      done();
    }
  },
  // The bytes read back (bottom row first) as the picture: colors for rgb and segmentation, a colormap for depth.
  develop(e, canvas) {
    const [W, H] = this.size;
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }
    // Another camera picked builds a new canvas the same size: drawing on with the old one's context left it blank.
    if (this.ctx?.canvas !== canvas) this.ctx = canvas.getContext("2d");
    if (!this.image || this.image.width !== W || this.image.height !== H) this.image = this.ctx.createImageData(W, H);
    const rows = this.image.data;
    const src = this.bytes;
    if (e.mode === "depth") {
      unpackDepth(src, e.range_m, this.depth);
      this.span = steadySpan(this.span, spanOf(this.depth));
      depthPicture(this.depth, ...(this.span ?? [0, e.range_m]), rows, { width: W, flip: true });
    } else {
      for (let y = 0; y < H; y++) rows.set(src.subarray((H - 1 - y) * W * 4, (H - y) * W * 4), y * W * 4);
      if (e.mode === "segmentation") {
        for (let i = 3; i < rows.length; i += 4) rows[i] = 255;
        this.seen = this.census(rows);
      }
    }
    this.ctx.putImageData(this.image, 0, 0);
    views.legend(e);
  },
  // What a segmentation picture shows, the biggest share first (camera.py's "seen").
  census(rgba) {
    const counts = new Map();
    for (let i = 0; i < rgba.length; i += 4) {
      const key = (rgba[i] << 16) | (rgba[i + 1] << 8) | rgba[i + 2];
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const names = new Map();
    for (const [name] of this.flat) {
      const [r, g, b] = thingColor(name);
      names.set((r << 16) | (g << 8) | b, name);
    }
    const total = rgba.length / 4;
    return [...counts]
      .filter(([key]) => names.has(key))
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([key, n]) => ({ name: names.get(key), share: n / total, color: key }));
  },
};

// ---------------------------------------------------------------- the views panel: the picture and the plots

const views = {
  key: "",
  sections: [], // { el, kind: "pip" | "plot", entry name, ... }
  tab: 0,
  channel: {}, // sensor name -> the plot channel picked (accel, gyro, ...)
  ranges: new Map(), // sensor name -> the plot's y range
  build(list) {
    const panel = $("sensor-views");
    const cams = list.filter((e) => e.type === "camera");
    if (!cams.some((e) => e.name === pip.name)) pip.name = cams[0]?.name ?? null;
    const plotted = list.filter((e) => PLOTS[e.type]);
    const key = `${cams.map((e) => e.name).join(",")}|${pip.name}|${plotted.map((e) => e.name).join(",")}`;
    if (key !== this.key) {
      this.key = key;
      this.sections = [];
      if (pip.name) this.sections.push(this.pictureSection(entries.get(pip.name), cams));
      for (const e of plotted) this.sections.push(this.plotSection(e));
      this.tab = Math.min(this.tab, Math.max(0, this.sections.length - 1));
      panel.replaceChildren(...this.sections.map((s) => s.el));
      pip.span = null;
    }
    this.layout();
    this.plot();
  },
  layout() {
    const panel = $("sensor-views");
    const off = !this.sections.length || api.inLesson();
    panel.hidden = off;
    root.visible = !api.inLesson();
    if (off) return api.shift();
    // A small window shows one view at a time, picked in a strip of tabs at the top.
    const tabbed = SMALL() && this.sections.length > 1;
    panel.classList.toggle("tabbed", tabbed);
    let strip = panel.querySelector(".plot-tabs");
    if (tabbed && !strip) {
      strip = document.createElement("div");
      strip.className = "plot-tabs";
      strip.setAttribute("role", "tablist");
      panel.prepend(strip);
    }
    if (!tabbed) strip?.remove();
    if (tabbed) {
      strip.replaceChildren(
        ...this.sections.map((s, k) => {
          const b = document.createElement("button");
          b.type = "button";
          b.setAttribute("role", "tab");
          b.setAttribute("aria-selected", String(k === this.tab));
          // A tab names the sensor's type (IMU, 关节编码器); two of a type are told apart by their names.
          const twin = this.sections.some((o) => o !== s && o.label === s.label);
          b.textContent = twin ? s.name.slice(s.name.indexOf("/") + 1) : s.label;
          b.addEventListener("click", () => {
            this.tab = k;
            this.layout();
            this.plot();
          });
          return b;
        })
      );
    }
    this.sections.forEach((s, k) => s.el.classList.toggle("off", tabbed && k !== this.tab));
    api.shift();
  },
  showsPicture() {
    const s = this.sections[0];
    return !!s && s.kind === "pip" && !$("sensor-views").hidden && !s.el.classList.contains("off");
  },
  pictureSection(e, cams) {
    const el = document.createElement("section");
    el.className = "plot pip sensor-pip";
    const head = document.createElement("div");
    head.className = "plot-head";
    const title = document.createElement("span");
    title.className = "plot-title";
    title.textContent = `${typeLabel(e)} · ${e.robot || e.name}`;
    head.append(title);
    if (cams.length > 1) {
      const seg = document.createElement("div");
      seg.className = "seg";
      for (const c of cams) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = c.name.slice(c.robot.length + 1);
        b.setAttribute("aria-pressed", String(c.name === e.name));
        b.addEventListener("click", () => {
          pip.name = c.name;
          sensorsSelect();
        });
        seg.append(b);
      }
      head.append(seg);
    }
    const canvas = document.createElement("canvas");
    canvas.width = e.size[0];
    canvas.height = e.size[1];
    canvas.style.aspectRatio = `${e.size[0]} / ${e.size[1]}`;
    const legend = document.createElement("div");
    legend.className = "sensor-legend";
    el.append(head, canvas, legend);
    return { el, kind: "pip", name: e.name, canvas, legend, label: t("sensor_view_picture") };
  },
  plotSection(e) {
    const el = document.createElement("section");
    el.className = "plot sensor-plot";
    const head = document.createElement("div");
    head.className = "plot-head";
    const title = document.createElement("span");
    title.className = "plot-title";
    const right = document.createElement("div");
    right.className = "sensor-plot-tools";
    const legend = document.createElement("span");
    legend.className = "plot-legend";
    head.append(title, right);
    const channels = this.channelsOf(e);
    if (channels.length > 1) {
      const seg = document.createElement("div");
      seg.className = "seg";
      for (const c of channels) {
        const b = document.createElement("button");
        b.type = "button";
        b.dataset.channel = c.key;
        b.textContent = c.label;
        b.addEventListener("click", () => {
          this.channel[e.name] = c.key;
          this.ranges.delete(e.name);
          this.plot();
        });
        seg.append(b);
      }
      right.append(seg);
    }
    const note = document.createElement("span");
    note.className = "sensor-note";
    const row = document.createElement("div");
    row.className = "sensor-plot-row";
    row.append(legend, note);
    const canvas = document.createElement("canvas");
    el.append(head, row, canvas);
    return { el, kind: "plot", name: e.name, canvas, title, legend, note, right, label: typeLabel(e) };
  },
  // What a plot can show, one at a time: an IMU's acceleration, turn rate or tilt; a force-torque sensor's force
  // or torque; an encoder's joints, a leg (or a run of joints) at a time.
  channelsOf(e) {
    if (e.type !== "encoders") return PLOTS[e.type].map((c) => ({ ...c, label: t(`plot_${c.key}`), title: t(`plot_${c.key}`) }));
    const [angles] = PLOTS.encoders;
    return jointGroups(Object.keys(e.rad ?? {})).map((g) => ({ ...angles, key: g.label || "all", label: g.label, title: t("plot_angles"), joints: g.joints }));
  },
  channelOf(e) {
    const channels = this.channelsOf(e);
    return channels.find((c) => c.key === this.channel[e.name]) ?? channels[0];
  },
  // The plots, drawn again with what came in (ten times a second).
  plot() {
    if ($("sensor-views").hidden) return;
    for (const s of this.sections) {
      if (s.kind !== "plot" || s.el.classList.contains("off")) continue;
      const e = entries.get(s.name);
      if (e) this.drawPlot(s, e);
    }
  },
  drawPlot(s, e) {
    const c = this.channelOf(e);
    if (!c) return;
    for (const b of s.right.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.channel === c.key));
    const history = histories.get(e.name) ?? [];
    let names = c.axes.map((a) => t(`axis_${a}`));
    let pick = (entry) => entry[c.field] ?? [];
    let colors = AXIS_COLORS;
    if (c.joints) {
      names = c.joints.map((j) => j.short);
      pick = (entry) => c.joints.map((j) => entry.rad?.[j.name]);
      colors = JOINT_COLORS;
    }
    s.title.textContent = `${c.title} (${c.unit})`;
    // What it holds, beside the force it measures (the weight of everything below the wrist taken off).
    const holds = e.type === "force_torque" && e.payload_kg !== undefined ? t("sensor_holds", { kg: Math.max(0, e.payload_kg).toFixed(2) }) : "";
    if (s.note.textContent !== holds) s.note.textContent = holds;
    const key = `${c.key}|${names.join(",")}`;
    if (s.legend.dataset.key !== key) {
      s.legend.dataset.key = key;
      s.legend.replaceChildren(
        ...names.map((n, i) => {
          const item = document.createElement("span");
          item.className = "legend-item";
          item.style.setProperty("--c", `var(${colors[i]})`);
          item.textContent = n;
          return item;
        })
      );
    }
    const series = names.map((_, i) => ({ values: history.map(([time, entry]) => [time, pick(entry)[i]]), color: colors[i] }));
    let lo = Infinity;
    let hi = -Infinity;
    for (const ser of series) for (const [time, v] of ser.values) if (time >= simTime - PLOT_WINDOW && Number.isFinite(v)) [lo, hi] = [Math.min(lo, v), Math.max(hi, v)];
    const range = plotRange(this.ranges.get(e.name), lo, hi, c.span);
    this.ranges.set(e.name, range);
    const canvas = s.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawLines(g, w, h, { series, end: simTime, window: PLOT_WINDOW, range, token });
  },
  // Under the picture: the depth colors' scale, or what a segmentation picture shows.
  legend(e) {
    const s = this.sections[0];
    if (!s || s.kind !== "pip") return;
    if (e.mode === "depth" && pip.span) {
      const [low, high] = pip.span;
      const key = `${low.toFixed(1)}|${high.toFixed(1)}`;
      if (s.legend.dataset.key === key) return;
      s.legend.dataset.key = key;
      const bar = document.createElement("canvas");
      bar.className = "depth-bar";
      bar.width = 128;
      bar.height = 1;
      const g = bar.getContext("2d");
      const img = g.createImageData(128, 1);
      for (let i = 0; i < 128; i++) {
        const [r, gg, b] = nearColor(low + ((high - low) * i) / 127, low, high);
        img.data.set([r * 255, gg * 255, b * 255, 255], 4 * i);
      }
      g.putImageData(img, 0, 0);
      const near = Object.assign(document.createElement("span"), { textContent: t("sensor_near", { m: low.toFixed(1) }) });
      const far = Object.assign(document.createElement("span"), { textContent: t("sensor_far", { m: high.toFixed(1) }) });
      s.legend.replaceChildren(near, bar, far);
    } else if (e.mode === "segmentation") {
      const key = pip.seen.map((x) => x.name).join(",");
      if (s.legend.dataset.key === key) return;
      s.legend.dataset.key = key;
      s.legend.replaceChildren(
        ...pip.seen.map((x) => {
          const item = document.createElement("span");
          item.className = "legend-item swatchy";
          item.style.setProperty("--c", `#${x.color.toString(16).padStart(6, "0")}`);
          item.textContent = x.name;
          return item;
        })
      );
    } else if (s.legend.childElementCount) {
      s.legend.dataset.key = "";
      s.legend.replaceChildren();
    }
  },
};

// ---------------------------------------------------------------- every frame

export function sensorsFrame() {
  if (!api || !entries.size) return;
  const lesson = api.inLesson();
  root.visible = !lesson;
  if (lesson) return;
  for (const [name, d] of drawings) {
    const e = entries.get(name);
    d.group.visible = !!e && shown(e);
    if (d.group.visible) d.frame();
  }
  if (pip.name && views.showsPicture()) {
    const e = entries.get(pip.name);
    if (e) pip.draw(e, views.sections[0].canvas);
  }
}

// For scripted checks (window.__robotforge.sensors()).
export function sensorsState() {
  const lidar = [...drawings.entries()].find(([name]) => entries.get(name)?.type?.startsWith("lidar"));
  return {
    sensors: [...entries.keys()],
    shown: [...entries.values()].filter(shown).map((e) => e.name),
    picture: pip.name,
    pictures: pip.renders,
    pictureMs: Math.round(pip.ms * 10) / 10,
    points: lidar ? lidar[1].group.children[1].geometry.drawRange.count : 0,
    views: views.sections.map((s) => s.name),
    plotRanges: Object.fromEntries(views.ranges),
  };
}

export function sensorsShow(name, on = true) {
  if (entries.has(name)) setShown(name, on);
}
