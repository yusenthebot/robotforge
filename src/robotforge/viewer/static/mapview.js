// The viewer draws the maps robots build as they move: the occupancy grid on the floor (free space teal and see-
// through, what the robot found in the way hot pink), the particles of its localization filter as blue dots, the pose
// it believes as a blue arrow with the path it believes it walked, and the odometry's own pose as a grey arrow with
// its own path, which drifts away from the blue one until the filter pulls it back.
//
// The server sends a "map" message four times a second while any robot maps (viewer/server.py lists its fields).
// The grid arrives as a base64 of a zlib-deflated byte per cell (0 unknown, 1 free, 2 occupied), now and then, not
// in every message. Nothing here is drawn in a lesson.

import * as THREE from "three";
import { t } from "./i18n.js";

const $ = (id) => document.getElementById(id);

const LIFT = 0.012; // m above the floor: the map is a sheet laid on it
const PARTICLE_LIFT = 0.03;
const TRAIL_LIMIT = 600; // poses kept per trail
const TRAIL_STEP = 0.04; // m between trail points
const ARROW = 0.5; // m, the length of a pose arrow
// RGBA of a cell by state; unknown is clear so the floor shows through, free is a teal about 30 percent opaque (clear
// enough to read the map's reach against the floor), and what was found in the way is a hot pink no shelf or crate
// is painted in
const PALETTE = [
  [0, 0, 0, 0],
  [30, 150, 190, 80],
  [236, 52, 112, 255],
];

let api = null;
const root = new THREE.Group();
root.name = "map";
const drawn = new Map(); // robot -> { group, sheet, texture, particles, pose, odometry, trail, odometryTrail, gridKey }
let shown = true;
let version = -1;

export function initMap(context) {
  api = context;
  api.scene.add(root);
  shown = api.storageGet("robotforge.map") !== "0";
  const box = $("show-map");
  if (box) {
    box.checked = shown;
    box.addEventListener("change", (e) => {
      shown = e.target.checked;
      api.storageSet("robotforge.map", shown ? "1" : "0");
      root.visible = shown && !api.inLesson();
    });
  }
}

function arrowMesh(color) {
  const shape = new THREE.Shape();
  shape.moveTo(ARROW, 0);
  shape.lineTo(-ARROW * 0.45, ARROW * 0.42);
  shape.lineTo(-ARROW * 0.2, 0);
  shape.lineTo(-ARROW * 0.45, -ARROW * 0.42);
  shape.closePath();
  const mesh = new THREE.Mesh(
    new THREE.ShapeGeometry(shape),
    new THREE.MeshBasicMaterial({ color, depthWrite: false, depthTest: false, transparent: true, opacity: 0.95 }),
  );
  mesh.renderOrder = 9;
  mesh.visible = false;
  return mesh;
}

function trailLine(color) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL_LIMIT * 3), 3));
  geometry.setDrawRange(0, 0);
  const line = new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color, depthWrite: false, depthTest: false, transparent: true, opacity: 0.9 }),
  );
  line.frustumCulled = false;
  line.renderOrder = 8;
  line.userData.points = [];
  return line;
}

function entryFor(robot) {
  let e = drawn.get(robot);
  if (e) return e;
  const group = new THREE.Group();
  group.name = `map:${robot}`;
  const particles = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({
      color: api.color("--accent"),
      size: 4,
      sizeAttenuation: false,
      depthWrite: false,
      depthTest: false,
      transparent: true,
      opacity: 0.7,
    }),
  );
  particles.frustumCulled = false;
  particles.renderOrder = 7;
  e = {
    group,
    sheet: null,
    texture: null,
    particles,
    pose: arrowMesh(api.color("--accent")),
    odometry: arrowMesh(new THREE.Color("#8a8f98")),
    trail: trailLine(api.color("--accent")),
    odometryTrail: trailLine(new THREE.Color("#8a8f98")),
    gridKey: "",
  };
  group.add(particles, e.pose, e.odometry, e.trail, e.odometryTrail);
  root.add(group);
  drawn.set(robot, e);
  return e;
}

async function inflate(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function drawGrid(e, grid) {
  const key = `${grid.x0},${grid.y0},${grid.rows},${grid.cols},${grid.cells.length},${grid.cells.slice(0, 64)}`;
  if (key === e.gridKey) return;
  e.gridKey = key;
  const cells = await inflate(grid.cells);
  if (e.gridKey !== key) return; // a newer map arrived while this one was being inflated
  const rgba = new Uint8Array(grid.rows * grid.cols * 4);
  for (let i = 0; i < cells.length; i++) rgba.set(PALETTE[cells[i]] ?? PALETTE[0], i * 4);
  const texture = new THREE.DataTexture(rgba, grid.cols, grid.rows, THREE.RGBAFormat);
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  // Cell (row, col) is centered at (x0 + col * res, y0 + row * res), so the sheet reaches half a cell past them.
  const w = grid.cols * grid.res;
  const h = grid.rows * grid.res;
  if (!e.sheet) {
    e.sheet = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
        toneMapped: false,
        polygonOffset: true,
        polygonOffsetFactor: -2,
        polygonOffsetUnits: -2,
      }),
    );
    e.sheet.renderOrder = 5;
    e.group.add(e.sheet);
  } else {
    e.sheet.material.map?.dispose();
    e.sheet.material.map = texture;
    e.sheet.material.needsUpdate = true;
  }
  e.sheet.scale.set(w, h, 1);
  e.sheet.position.set(grid.x0 - grid.res / 2 + w / 2, grid.y0 - grid.res / 2 + h / 2, LIFT);
}

function placeArrow(mesh, pose) {
  if (!pose) {
    mesh.visible = false;
    return;
  }
  mesh.visible = true;
  mesh.position.set(pose[0], pose[1], 0.05);
  mesh.rotation.set(0, 0, pose[2]);
}

function extendTrail(line, pose) {
  if (!pose) return;
  const pts = line.userData.points;
  const last = pts[pts.length - 1];
  if (last && Math.hypot(pose[0] - last[0], pose[1] - last[1]) < TRAIL_STEP) return;
  pts.push([pose[0], pose[1]]);
  if (pts.length > TRAIL_LIMIT) pts.shift();
  const attr = line.geometry.getAttribute("position");
  pts.forEach((p, i) => attr.setXYZ(i, p[0], p[1], 0.04));
  attr.needsUpdate = true;
  line.geometry.setDrawRange(0, pts.length);
}

export function mapUpdate(msg) {
  if (msg.version !== version) {
    version = msg.version;
    clearMaps(); // a new model: the maps of the old one go
  }
  const present = new Set();
  for (const r of msg.robots) {
    present.add(r.robot);
    const e = entryFor(r.robot);
    if (r.grid) drawGrid(e, r.grid);
    placeArrow(e.pose, r.pose);
    placeArrow(e.odometry, r.odometry);
    extendTrail(e.trail, r.pose);
    extendTrail(e.odometryTrail, r.odometry);
    if (r.particles) {
      const xy = r.particles;
      const pos = new Float32Array((xy.length / 2) * 3);
      for (let i = 0; i < xy.length / 2; i++) pos.set([xy[2 * i], xy[2 * i + 1], PARTICLE_LIFT], i * 3);
      e.particles.geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      e.particles.visible = true;
    } else {
      e.particles.visible = false;
    }
  }
  for (const robot of [...drawn.keys()]) if (!present.has(robot)) dropMap(robot);
  const label = $("map-toggle");
  if (label) label.hidden = drawn.size === 0;
  readout(msg.robots);
  root.visible = shown && !api.inLesson();
}

// The readout: for the robot mapping (the selected one when several map), how far off its pose estimate is, and the
// odometry alone, as the server measures against the truth it can see.
function readout(robots) {
  const hud = $("map-hud");
  if (!hud) return;
  const r = robots.find((x) => x.robot === api.selected()) ?? robots[0];
  hud.hidden = !r || r.error_m === undefined || !shown || api.inLesson();
  if (hud.hidden) return;
  const fmt = (v) => Number(v).toFixed(2);
  hud.replaceChildren();
  const title = document.createElement("div");
  title.textContent = t("map_hud_title", { robot: r.robot });
  title.className = "bad";
  const pose = document.createElement("div");
  pose.innerHTML = t("map_hud_pose", { err: `<b>${fmt(r.error_m)}</b>`, odo: `<b>${fmt(r.odometry_error_m)}</b>` });
  const mapped = document.createElement("div");
  mapped.className = "bad";
  mapped.textContent = t("map_hud_mapped", { m2: Math.round(r.mapped_m2), scans: r.scans });
  hud.append(title, pose, mapped);
}

function dropMap(robot) {
  const e = drawn.get(robot);
  if (!e) return;
  root.remove(e.group);
  e.sheet?.material.map?.dispose();
  drawn.delete(robot);
}

function clearMaps() {
  for (const robot of [...drawn.keys()]) dropMap(robot);
}

const forward = new THREE.Vector3();
const FROM_ABOVE = -0.8; // the camera's forward z below this: looking nearly straight down

export function mapFrame() {
  root.visible = shown && drawn.size > 0 && !api.inLesson();
  // Seen from above the map is drawn over shelves and crates (it is laid on the floor, which they hide from there);
  // from the side it stays under them, where it belongs.
  const over = api.camera.getWorldDirection(forward).z < FROM_ABOVE;
  for (const e of drawn.values()) if (e.sheet) e.sheet.material.depthTest = !over;
  if (drawn.size === 0) $("map-hud")?.setAttribute("hidden", "");
}

export function mapState() {
  return {
    shown,
    robots: [...drawn.entries()].map(([robot, e]) => ({
      robot,
      hasGrid: !!e.sheet,
      particles: e.particles.visible ? (e.particles.geometry.getAttribute("position")?.count ?? 0) : 0,
      pose: e.pose.visible ? [e.pose.position.x, e.pose.position.y, e.pose.rotation.z] : null,
      odometry: e.odometry.visible ? [e.odometry.position.x, e.odometry.position.y, e.odometry.rotation.z] : null,
      trail: e.trail.userData.points.length,
    })),
  };
}
