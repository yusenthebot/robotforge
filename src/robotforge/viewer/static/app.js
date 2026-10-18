// robotforge viewer: a thin three.js client. Physics runs in the Python server;
// this page draws the bodies it streams and sends user input back.
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { initSensors, sensorsFrame, sensorsScene, sensorsSelect, sensorsShow, sensorsState, sensorsUpdate } from "./sensors.js";
import { escapeHtml, t, tOr, translatePage } from "./i18n.js";
import { initLessons, lessonActive, lessonDriveSpec, lessonCamera, lessonFrame, lessonPlotAxes, lessonScene, lessonState, lessonTitle, lessonUpdate } from "./lesson.js";
import { studioEnvironment, surfaceMaterial } from "./look.js";
import { SURFACE_GEOM, leaveWater, makeWater, stillLevel, waterFrame, waterLook } from "./water.js";
import { clothGeometry, shapeCloth } from "./cloth.js";
import { buildRotors, rotorCommands, spinRotors } from "./rotors.js";

const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

translatePage();

// ---------------------------------------------------------------- renderer setup

const canvas = $("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene3 = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 500);
camera.up.set(0, 0, 1); // MuJoCo is Z-up
camera.position.set(1.6, -1.6, 1.0);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

// Most light comes from the studio environment (soft, reflected in every surface); the sun adds
// direction and shadows, and a faint sky/ground fill keeps shadowed sides from going black.
scene3.environment = studioEnvironment(renderer);
scene3.environmentIntensity = 0.85;
const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x3a3328, 0.35);
hemi.position.set(0, 0, 1);
scene3.add(hemi);
const sun = new THREE.DirectionalLight(0xfff6ec, 1.9);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.radius = 3;
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

// With panels open, the view's center slides into the part of the window they leave free, so what you look at
// is not under them: sideways for side panels, up or down for a strip of plots or a sheet along the bottom.
const viewShift = { x: 0, y: 0 };
const viewShiftTarget = { x: 0, y: 0 };

function applyViewShift() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  if (Math.abs(viewShift.x) < 0.5 && Math.abs(viewShift.y) < 0.5) camera.clearViewOffset();
  else camera.setViewOffset(w, h, viewShift.x, viewShift.y, w, h);
}

function updateViewShift() {
  if (viewShift.x === viewShiftTarget.x && viewShift.y === viewShiftTarget.y) return;
  const k = reduceMotion ? 1 : 0.2;
  for (const axis of ["x", "y"]) {
    viewShift[axis] += (viewShiftTarget[axis] - viewShift[axis]) * k;
    if (Math.abs(viewShiftTarget[axis] - viewShift[axis]) < 0.5) viewShift[axis] = viewShiftTarget[axis];
  }
  applyViewShift();
}

// A page opened hidden (a collapsed panel, a background pane, a minimized window) has no size yet: 0 / 0
// would make the aspect NaN and every framing after it NaN, and the view would stay black for good.
// Keep the last real shape instead, and frame the camera once the page can be seen.
let fitWhenSized = false;
const hasSize = () => window.innerWidth > 0 && window.innerHeight > 0;

function resize() {
  if (!hasSize()) return;
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  applyViewShift();
  if (fitWhenSized && sceneDesc) {
    fitWhenSized = false;
    frameCamera();
  }
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
let water = null; // the scene's water (water.js), or null

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

// The floor is drawn a little "behind" in depth, so whatever lies on it (a heightfield's flat parts, paths,
// start marks, a pond a few millimetres up) always wins instead of flickering against it.
const FLOOR_OFFSET = { polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 4 };
// Heightfields sit 2 mm under the floor in the physics, so their flat cells and the floor were nearly
// coplanar: at a low camera the two traded places pixel by pixel (streaky flicker). Drawn this much
// higher, a field is always in front inside its footprint; a foot sinking 2 mm into it is invisible.
const HFIELD_LIFT = 0.004; // m
let floorFrame = null; // { x, y, half, halfY, material } of the scene's floor, for lining up textures with it

function planeHalf(geom, extent) {
  const half = geom.size[0] > 0 ? geom.size[0] : Math.max(20, extent * 20); // infinite planes: big enough
  return [half, geom.size[1] > 0 ? geom.size[1] : half];
}

function planeObject(geom, extent, look) {
  const [half, halfY] = planeHalf(geom, extent);
  let mat;
  if (look && look.pattern !== "checker") {
    const tex = groundTexture(look.pattern);
    tex.repeat.set((2 * half) / look.tile, (2 * halfY) / look.tile);
    mat = new THREE.MeshStandardMaterial({ map: tex, color: rgbaColor(geom.rgba), roughness: 0.95, metalness: 0, ...FLOOR_OFFSET });
  } else {
    const tex = checkerTexture(geom.rgba);
    const cell = Math.max(0.25, extent / 4);
    tex.repeat.set((2 * half) / (2 * cell), (2 * halfY) / (2 * cell));
    mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95, metalness: 0, ...FLOOR_OFFSET });
  }
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2 * half, 2 * halfY), mat);
  mesh.receiveShadow = true;
  return mesh;
}

// ---------------------------------------------------------------- ground surfaces and terrain

// Grayscale surface patterns; the material color tints them, so one canvas serves every style.
const groundCanvases = new Map();

function groundCanvas(pattern) {
  if (groundCanvases.has(pattern)) return groundCanvases.get(pattern);
  const size = 256;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  if (pattern === "tiles") {
    g.fillStyle = "#ffffff";
    g.fillRect(0, 0, size, size);
    g.fillStyle = "#b8b8b8"; // grout
    for (const at of [0, size / 2]) {
      g.fillRect(at, 0, 3, size);
      g.fillRect(0, at, size, 3);
    }
  } else if (pattern === "planks") {
    const rows = 8;
    for (let r = 0; r < rows; r++) {
      const v = Math.round(225 + 30 * rand());
      g.fillStyle = `rgb(${v},${v},${v})`;
      g.fillRect(0, (r * size) / rows, size, size / rows);
      g.fillStyle = "#9a9a9a";
      g.fillRect(0, (r * size) / rows, size, 2);
      g.fillRect(((r % 2) + 0.5) * (size / 2) * rand(), (r * size) / rows, 2, size / rows); // butt joint
    }
  } else {
    // speckle: soft blotches plus finer mottling, like grass, soil or concrete seen from standing height.
    // Both layers wrap around (so tiles meet without a seam) and are smooth over several texels: per-texel
    // noise is finer than mipmapping can filter and shimmers when a low camera moves.
    const blotches = periodicNoise(16, size, rand);
    const mottle = periodicNoise(64, size, rand);
    const img = g.createImageData(size, size);
    for (let i = 0; i < size * size; i++) {
      const v = Math.round(228 + 27 * blotches[i] + 10 * (mottle[i] - 0.5));
      img.data.fill(Math.max(0, Math.min(255, v)), 4 * i, 4 * i + 3);
      img.data[4 * i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  }
  groundCanvases.set(pattern, c);
  return c;
}

// Value noise on a `cells` x `cells` grid that wraps around, sampled at `size` x `size` with smooth
// (smoothstep) interpolation: values in 0..1, seamless when tiled.
function periodicNoise(cells, size, rand) {
  const grid = Float32Array.from({ length: cells * cells }, () => rand());
  const at = (x, y) => grid[((y + cells) % cells) * cells + ((x + cells) % cells)];
  const smooth = (t) => t * t * (3 - 2 * t);
  const out = new Float32Array(size * size);
  for (let py = 0; py < size; py++) {
    const fy = (py / size) * cells;
    const y0 = Math.floor(fy);
    const ty = smooth(fy - y0);
    for (let px = 0; px < size; px++) {
      const fx = (px / size) * cells;
      const x0 = Math.floor(fx);
      const tx = smooth(fx - x0);
      const top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * tx;
      const bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * tx;
      out[py * size + px] = top + (bottom - top) * ty;
    }
  }
  return out;
}

function groundTexture(pattern) {
  const tex = new THREE.CanvasTexture(groundCanvas(pattern)); // its own texture, so disposing it is safe
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return tex;
}

// A MuJoCo heightfield as a displaced grid. Rows run from -y to +y (MuJoCo's order); PlaneGeometry
// lists vertices from +y down, hence the flipped row. Tops are tinted a little lighter than valleys.
// Scale a BoxGeometry's UVs so its texture repeats every `tile` meters on each face. three.js lists the
// faces +x, -x, +y, -y, +z, -z with 4 vertices each; u and v run along (z, y), (x, z) and (x, y).
function boxUvsInMeters(geometry, half, tile) {
  const uv = geometry.attributes.uv;
  const [x, y, z] = half.map((h) => (2 * h) / tile);
  const spans = [[z, y], [z, y], [x, z], [x, z], [x, y], [x, y]];
  for (let face = 0; face < 6; face++) {
    for (let k = 0; k < 4; k++) {
      const i = face * 4 + k;
      uv.setXY(i, uv.getX(i) * spans[face][0], uv.getY(i) * spans[face][1]);
    }
  }
  uv.needsUpdate = true;
}

function hfieldObject(geom, desc, floats, look) {
  const h = desc.hfields.find((x) => x.id === geom.hfield);
  if (!h) return null;
  const [rx, ry, top, base] = h.size;
  const g = new THREE.PlaneGeometry(2 * rx, 2 * ry, h.ncol - 1, h.nrow - 1);
  const pos = g.attributes.position;
  const colors = new Float32Array(pos.count * 3);
  for (let iy = 0; iy < h.nrow; iy++) {
    const row = h.nrow - 1 - iy;
    for (let ix = 0; ix < h.ncol; ix++) {
      const v = iy * h.ncol + ix;
      const t = floats[h.offset + row * h.ncol + ix];
      pos.setZ(v, t * top);
      colors.fill(1 + 0.3 * t, 3 * v, 3 * v + 3); // the rim (t = 0) matches the floor exactly
    }
  }
  g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  const tex = groundTexture(look ? look.pattern : "speckle");
  const tile = look ? look.tile : 0.6;
  tex.repeat.set((2 * rx) / tile, (2 * ry) / tile);
  const f = floorFrame;
  if (f && look && f.material === geom.material) {
    // Same surface as the floor: line the pattern up with the floor's, so the two meet without a seam.
    const shift = (a) => ((a % 1) + 1) % 1;
    tex.offset.set(shift((f.half - f.x + geom.pos[0] - rx) / tile), shift((f.halfY - f.y + geom.pos[1] - ry) / tile));
  }
  const mat = new THREE.MeshStandardMaterial({
    color: rgbaColor(geom.rgba),
    map: tex,
    vertexColors: true,
    roughness: 0.95,
    metalness: 0,
  });
  const mesh = new THREE.Mesh(g, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData.lift = HFIELD_LIFT;
  // Sides down to the base only where the edge stands above the floor; terrain that blends into the
  // floor at its rim would only show them as a dark seam.
  const edge = []; // the field's border samples, as heights
  for (let c = 0; c < h.ncol; c++) edge.push(floats[h.offset + c], floats[h.offset + (h.nrow - 1) * h.ncol + c]);
  for (let r = 0; r < h.nrow; r++) edge.push(floats[h.offset + r * h.ncol], floats[h.offset + r * h.ncol + h.ncol - 1]);
  if (geom.pos[2] + HFIELD_LIFT + Math.max(...edge) * top > 0.005) mesh.add(hfieldSkirt(h, floats, top, base, mat));
  return mesh;
}

// The sides of a heightfield, down to its base, for fields whose rim is not at floor level.
function hfieldSkirt(h, floats, top, base, mat) {
  const [rx, ry] = h.size;
  const verts = [];
  const at = (r, c) => [-rx + (2 * rx * c) / (h.ncol - 1), -ry + (2 * ry * r) / (h.nrow - 1), floats[h.offset + r * h.ncol + c] * top];
  const edge = (points) => {
    for (let i = 0; i + 1 < points.length; i++) {
      const [a, b] = [points[i], points[i + 1]];
      verts.push(...a, a[0], a[1], -base, ...b, ...b, a[0], a[1], -base, b[0], b[1], -base);
    }
  };
  const range = (n) => Array.from({ length: n }, (_, i) => i);
  edge(range(h.ncol).map((c) => at(0, c)));
  edge(range(h.ncol).map((c) => at(h.nrow - 1, h.ncol - 1 - c)));
  edge(range(h.nrow).map((r) => at(h.nrow - 1 - r, 0)));
  edge(range(h.nrow).map((r) => at(r, h.ncol - 1)));
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  g.computeVertexNormals();
  const skirt = new THREE.Mesh(
    g,
    new THREE.MeshStandardMaterial({
      color: mat.color.clone().multiplyScalar(0.7),
      roughness: 1,
      side: THREE.DoubleSide,
      polygonOffset: true, // behind the floor where it dips under it: only the part above the floor shows
      polygonOffsetFactor: 4,
      polygonOffsetUnits: 8,
    })
  );
  skirt.receiveShadow = true;
  return skirt;
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
    if (entry.colors !== undefined) {
      // A textured mesh: its texture's colors per vertex (the page draws no textures; see scene.py)
      g.setAttribute("color", new THREE.BufferAttribute(floats.subarray(entry.colors, entry.colors + 3 * n), 3));
      g.userData.vertexColors = true;
    }
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
  if (desc.meshes.length > 0 || (desc.hfields || []).length > 0) {
    const res = await fetch(`mesh/${desc.version}.bin`);
    if (!res.ok) return; // a newer scene is already on its way
    buffer = await res.arrayBuffer();
  }
  const meshGeometries = buildMeshGeometries(desc, buffer);
  const floats = new Float32Array(buffer);
  const grounds = desc.grounds || {};
  leaveWater(water, scene3);
  water = null;
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
  const floorGeom = desc.geoms.find((g) => g.type === "plane");
  floorFrame = null;
  if (floorGeom) {
    const [half, halfY] = planeHalf(floorGeom, extent);
    floorFrame = { x: floorGeom.pos[0], y: floorGeom.pos[1], half, halfY, material: floorGeom.material };
  }
  for (const geom of desc.geoms) {
    if (desc.water && geom.name === SURFACE_GEOM) continue; // drawn below, with its waves
    let obj;
    const look = grounds[geom.material];
    if (geom.type === "plane") {
      obj = planeObject(geom, extent, look);
    } else if (geom.type === "hfield") {
      obj = hfieldObject(geom, desc, floats, look);
      if (!obj) continue;
    } else {
      const g = geometryFor(geom, meshGeometries);
      if (!g) continue;
      const mat = look
        ? new THREE.MeshStandardMaterial({ color: rgbaColor(geom.rgba), roughness: 0.95, metalness: 0 })
        : surfaceMaterial(geom.material, rgbaColor(geom.rgba), geom.rgba[3]);
      if (g.userData.vertexColors) mat.vertexColors = true;
      if (look && geom.type === "box") {
        // Terrain blocks (stairs, ramps, stones) share the ground's surface, tiled in meters on every face:
        // one repeat for the whole box squeezed the pattern on thin faces (risers, sides) into shimmer.
        mat.map = groundTexture(look.pattern === "checker" ? "speckle" : look.pattern);
        boxUvsInMeters(g, geom.size, look.tile);
      }
      obj = new THREE.Mesh(g, mat);
      obj.castShadow = true;
      obj.receiveShadow = true;
    }
    obj.position.fromArray(geom.pos);
    obj.position.z += obj.userData.lift ?? 0;
    obj.quaternion.set(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]);
    obj.userData.body = geom.body;
    obj.userData.group = geom.group;
    obj.userData.isPlane = geom.type === "plane";
    obj.userData.geom = geom.id; // a segmentation camera names what it sees by its geoms (sensors.js)
    obj.visible = geom.group < desc.visibleGroupLimit || showCollision;
    bodyNodes[geom.body].add(obj);
  }
  if (desc.water) {
    water = makeWater(desc.water, waterLook(desc.water, floorGeom?.material), desc.stat.center, renderer);
    water.group.traverse((o) => {
      o.userData.body = 0;
    });
    bodyNodes[0].add(water.group);
  }
  flexMeshes = (desc.flexes || []).map(flexMesh);
  for (const flex of flexMeshes) worldRoot.add(flex.mesh);
  buildRotors(desc.rotors, bodyNodes); // aircraft: propellers the page turns (rotors.js)
  scene3.add(worldRoot);

  // Lights and (first time) camera are fitted once real poses arrive, see applyPoses.
  pendingFit = firstScene || wantFrame ? "camera" : "lights";
  firstScene = false;
  wantFrame = false;
  follow = null;
  tween = null;
  fitWhenPosed = refitRig = null; // asked for the scene this one replaces
  onScene(desc);
}

// A cloth (a MuJoCo flex) is a triangle mesh whose corners ride on bodies: one per vertex, or a point on one for a
// pinned edge. Its vertices follow those bodies' poses, which already stream every frame (see scene.py _flexes).
let flexMeshes = []; // { mesh, shape, points, bodies, local: Float32Array }
const CLOTH_THICKNESS = 0.006; // m, the radius of a cloth's points where they touch things (mechanisms.CLOTH_BALL)
const flexPoint = new THREE.Vector3();

function flexMesh(f) {
  const shape = clothGeometry(f.faces, f.bodies.length); // a smooth surface through its points (cloth.js)
  const material = new THREE.MeshStandardMaterial({
    color: rgbaColor(f.rgba),
    roughness: 0.95,
    metalness: 0,
    side: THREE.DoubleSide,
  });
  pullNearer(material, CLOTH_THICKNESS);
  const mesh = new THREE.Mesh(shape.geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.frustumCulled = false; // its bounds change every frame
  mesh.userData.flex = true;
  const points = new Float32Array(f.bodies.length * 3);
  return { mesh, shape, points, bodies: f.bodies, local: new Float32Array(f.local) };
}

// Draw a material's surfaces `by` metres nearer the camera than they are, in depth only: each vertex moves along its
// own line of sight, so it stays where it is on screen. A cloth is drawn through the middle of its points, which lie
// its thickness (6 mm) over what it rests on, and where it bends over a table's edge the table's rounded edge still
// showed through it in streaks; nearer by its thickness, it wins against what lies within it.
function pullNearer(material, by) {
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader.replace(
      "#include <project_vertex>",
      `#include <project_vertex>
      gl_Position = projectionMatrix * vec4(mvPosition.xyz * max(0.0, 1.0 - ${by.toFixed(4)} / length(mvPosition.xyz)), 1.0);`,
    );
  };
}

function updateFlexes() {
  for (const flex of flexMeshes) {
    for (let i = 0; i < flex.bodies.length; i++) {
      const node = bodyNodes[flex.bodies[i]];
      flexPoint.fromArray(flex.local, 3 * i).applyQuaternion(node.quaternion).add(node.position);
      flexPoint.toArray(flex.points, 3 * i);
    }
    shapeCloth(flex.shape, flex.points);
  }
}

// The body of the cloth's vertex nearest to where it was hit: grabbing a cloth drags that corner of its mesh.
function flexBodyAt(mesh, point) {
  const flex = flexMeshes.find((f) => f.mesh === mesh);
  if (!flex) return -1;
  const pos = mesh.geometry.attributes.position;
  let best = -1;
  let bestDist = Infinity;
  for (let i = 0; i < flex.bodies.length; i++) {
    const d = flexPoint.fromBufferAttribute(pos, i).distanceToSquared(point);
    if (d < bestDist && flex.bodies[i] > 0) {
      bestDist = d;
      best = flex.bodies[i];
    }
  }
  return best;
}

// The boxes of everything the scene shows but its floor, each in world space.
function contentBoxes() {
  const boxes = [];
  if (worldRoot) {
    worldRoot.updateMatrixWorld(true);
    worldRoot.traverse((o) => {
      if (o.isMesh && o.visible && !o.userData.isPlane) boxes.push(new THREE.Box3().setFromObject(o));
    });
  }
  return boxes;
}

function contentSphere() {
  const box = contentBoxes().reduce((all, b) => all.union(b), new THREE.Box3());
  if (box.isEmpty() && sceneDesc) {
    return new THREE.Sphere(new THREE.Vector3().fromArray(sceneDesc.stat.center), sceneDesc.stat.extent);
  }
  return box.getBoundingSphere(new THREE.Sphere());
}

function fitLights(sphere) {
  const c = sphere.center;
  const r = Math.max(sphere.radius, 0.2) * 1.5;
  // Fade the (often infinite) floor into the background instead of a hard horizon.
  scene3.fog = new THREE.Fog(theme.fog, r * 5, r * 18);
  sun.position.copy(c).add(new THREE.Vector3(0.6, -0.8, 1.6).multiplyScalar(r * 1.5));
  sun.target.position.copy(c);
  const cam = sun.shadow.camera;
  cam.left = cam.bottom = -r;
  cam.right = cam.top = r;
  cam.near = 0.01;
  cam.far = r * 6;
  cam.updateProjectionMatrix();
}

// Distance at which a sphere of radius r fills the view.
function fitDistance(r) {
  const vfov = THREE.MathUtils.degToRad(camera.fov);
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  return (r / Math.sin(Math.min(vfov, hfov) / 2)) * 1.05;
}

function frameCamera({ animate = false } = {}) {
  refitRig = null; // a view asked for now is not undone by a lesson's rig settling
  if (!hasSize()) {
    fitWhenSized = true;
    return;
  }
  const sphere = contentSphere();
  const d = fitDistance(Math.max(sphere.radius, 0.08));
  const eye = sphere.center.clone().add(new THREE.Vector3(1.0, -1.3, 0.8).normalize().multiplyScalar(d));
  const fit = fitContent(eye, sphere.center, fitDistance(0.08) / d);
  // A tight depth range keeps the depth buffer precise across the scene (wide ranges make distant
  // surfaces that nearly touch flicker); fog hides everything well before the far plane. Set for the
  // distance the camera ends at, which in a narrow free view is further than the sphere's.
  const reach = Math.max(d, fit.eye.distanceTo(fit.target));
  camera.near = Math.min(Math.max(reach / 400, 0.004), 0.05);
  camera.far = reach * 40;
  camera.updateProjectionMatrix();
  moveCamera(fit.eye, fit.target, animate);
}

// Come in from the distance where the scene's bounding sphere fills the view until the scene itself does (every
// part's box), centred in the free part of the window: seen from above a scene is flat, its corners are often empty
// and its near side reaches further down the screen than its far side up, and the sphere left the park's robots
// 16 px tall on a laptop, with empty grass all round. The look-at point moves over the ground, keeping its height.
// Where a panel leaves the free part narrower than the sphere (a tablet with the robots drawer open), go further out.
const CONTENT_MARGIN = 0.85; // share of the free half-width and half-height the scene may fill (room for a ring, a path)
const CONTENT_CLOSEST = 0.3; // never nearer than this share of the sphere's distance
const CONTENT_FURTHEST = 3; // never further than this many times the sphere's distance
function fitContent(eye, target, least) {
  const corners = [];
  const boxes = contentBoxes();
  if (ring.visible) boxes.push(new THREE.Box3().setFromObject(ring)); // the selected robot's ring is part of the picture
  for (const box of boxes) {
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) corners.push(new THREE.Vector3(x, y, z));
  }
  if (!corners.length || !hasSize()) return { eye, target };
  const r = freeRect();
  const halfW = (CONTENT_MARGIN * Math.max(80, r.right - r.left)) / 2; // px
  const halfH = (CONTENT_MARGIN * Math.max(80, r.bottom - r.top)) / 2;
  const probe = camera.clone();
  probe.clearViewOffset(); // the view offset only moves the picture's center into the free part
  probe.updateProjectionMatrix();
  const back = eye.clone().sub(target).normalize(); // from the look-at point to the eye
  const across = new THREE.Vector3(-back.y, back.x, 0).normalize(); // the screen's right, over the ground
  const ahead = new THREE.Vector3(-back.x, -back.y, 0).normalize(); // the screen's up, over the ground
  const rise = Math.max(0.2, back.z); // how much a step over the ground moves a point up the screen
  const tanV = Math.tan(THREE.MathUtils.degToRad(probe.fov) / 2);
  const [W, H] = [window.innerWidth, window.innerHeight];
  const start = eye.distanceTo(target);
  const closest = Math.max(CONTENT_CLOSEST, least) * start; // and no nearer than a small robot's own framing
  const furthest = CONTENT_FURTHEST * start;
  const center = target.clone();
  let dist = start;
  const p = new THREE.Vector3();
  // The scene's extent on the screen from (center, dist), -1 to 1 across the view; null when a part is behind.
  const extent = () => {
    probe.position.copy(center).addScaledVector(back, dist);
    probe.lookAt(center);
    probe.updateMatrixWorld(true);
    const e = [Infinity, -Infinity, Infinity, -Infinity];
    for (const c of corners) {
      p.copy(c).project(probe);
      if (p.z > 1 || p.z < -1) return null;
      [e[0], e[1], e[2], e[3]] = [Math.min(e[0], p.x), Math.max(e[1], p.x), Math.min(e[2], p.y), Math.max(e[3], p.y)];
    }
    return e;
  };
  const fill = ([x0, x1, y0, y1]) => Math.max(((x1 - x0) / 2) * (W / 2 / halfW), ((y1 - y0) / 2) * (H / 2 / halfH));
  for (let i = 0; i < 8; i++) {
    const e = extent();
    if (!e) {
      dist = Math.min(furthest, dist * 1.25);
      continue;
    }
    // Over the ground until the scene's middle on the screen is the picture's middle, then nearer or further.
    center.addScaledVector(across, ((e[0] + e[1]) / 2) * tanV * probe.aspect * dist);
    center.addScaledVector(ahead, (((e[2] + e[3]) / 2) * tanV * dist) / rise);
    dist = Math.min(furthest, Math.max(closest, dist * fill(e)));
  }
  for (let e = extent(); dist < furthest && (!e || fill(e) > 1); e = extent()) dist = Math.min(furthest, dist * 1.04);
  return { eye: center.clone().addScaledVector(back, dist), target: center };
}

// A lesson's camera, framed for the free part of the window and, if it asks, stepped back until its rig fits there.
let fitWhenPosed = null; // { eye, target } of a lesson camera that asked to fit before its scene's first poses
function aimLessonCamera(eye, target, fit) {
  const goal = new THREE.Vector3().fromArray(target);
  const away = fitToFree(new THREE.Vector3().fromArray(eye), goal);
  moveCamera(fit ? fitRig(away, goal) : away, goal, true);
}

// A rig fitted at its scene's first poses is fitted once more when it has settled, unless the view was moved by hand
// meanwhile: the PID triplets' arms hang at first and rise level within half a second, and the third then reached
// off a phone's screen (a reload, which fits the settled rig, kept it in).
const RIG_SETTLE_MS = 800;
let refitRig = null; // { eye, target, at } of the lesson camera to fit again
function refitSettledRig() {
  if (!refitRig || performance.now() < refitRig.at) return;
  const { eye, target } = refitRig;
  refitRig = null;
  if (lessonCamera()?.fit !== true) return; // the lesson has gone, or asks for no fit
  const kept = follow?.body; // a camera move drops the follow the lesson asked for; keep it
  aimLessonCamera(eye, target, true);
  if (kept !== undefined) startFollowBody(kept, { keepTween: true });
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
  if (flexMeshes.length) updateFlexes();
  if (pendingFit) {
    const sphere = contentSphere();
    fitLights(sphere);
    // Reloaded mid-lesson: aim at the lesson's rig, not the whole scene (its animated move may have been
    // cut short by the scene arriving again).
    const own = lessonCamera();
    if (pendingFit === "camera" && own) {
      const goal = new THREE.Vector3().fromArray(own.target);
      const eye = fitToFree(new THREE.Vector3().fromArray(own.eye), goal);
      moveCamera(own.fit ? fitRig(eye, goal) : eye, goal, false);
      if (own.fit) refitRig = { eye: own.eye, target: own.target, at: performance.now() + RIG_SETTLE_MS };
    }
    else if (pendingFit === "camera") frameCamera();
    else if (fitWhenPosed) {
      aimLessonCamera(fitWhenPosed.eye, fitWhenPosed.target, true);
      refitRig = { ...fitWhenPosed, at: performance.now() + RIG_SETTLE_MS };
    }
    fitWhenPosed = null;
    pendingFit = null;
    $("loading").hidden = true;
  }
}

// ---------------------------------------------------------------- camera moves

let tween = null; // { fromEye, fromTarget, toEye, toTarget, t0 }
const TWEEN_MS = 450;

function moveCamera(eye, target, animate) {
  follow = null;
  if (!animate || reduceMotion) {
    tween = null;
    camera.position.copy(eye);
    controls.target.copy(target);
    controls.update();
    return;
  }
  tween = { fromEye: camera.position.clone(), fromTarget: controls.target.clone(), toEye: eye, toTarget: target, t0: performance.now() };
}

function updateTween() {
  if (!tween) return;
  const k = Math.min(1, (performance.now() - tween.t0) / TWEEN_MS);
  const e = 1 - (1 - k) ** 3; // ease out
  camera.position.lerpVectors(tween.fromEye, tween.toEye, e);
  controls.target.lerpVectors(tween.fromTarget, tween.toTarget, e);
  if (k >= 1) tween = null;
}

// ---------------------------------------------------------------- theme

// Scene colors come from the CSS tokens, so the 3D view and the panels always agree (light or dark).
const theme = { fog: "#e2e7ec", amber: "#f2a900" };
const accentMaterials = [];

function readTheme() {
  const css = getComputedStyle(document.documentElement);
  const token = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
  const old = scene3.background;
  scene3.background = makeGradientTexture(token("--scene-top", "#f4f6f8"), token("--scene-bottom", "#d6dde5"));
  if (old && !old.userData.shared) old.dispose(); // not the sky over open water (water.js): the water mirrors it
  theme.fog = token("--scene-fog", theme.fog);
  theme.amber = token("--amber", theme.amber);
  if (scene3.fog) scene3.fog.color.set(theme.fog);
  for (const m of accentMaterials) m.color.set(theme.amber);
}

function accentMaterial(opts) {
  // Pulled toward the camera in depth so a ring lying on the floor never sinks into it at a distance.
  // Amber from the start: readTheme recolors the ones made so far, and a marker made after it stayed white.
  const m = new THREE.MeshBasicMaterial({
    color: theme.amber,
    transparent: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -4,
    polygonOffsetUnits: -8,
    ...opts,
  });
  accentMaterials.push(m);
  return m;
}

// ---------------------------------------------------------------- selection ring

// An amber ring on the ground under the selected robot; a notch on its rim points the way it drives.
const ring = new THREE.Group();
ring.visible = false;
ring.renderOrder = 2;
const ringLine = new THREE.Mesh(new THREE.RingGeometry(0.9, 1, 96), accentMaterial({ opacity: 0.95, side: THREE.DoubleSide }));
const ringDisc = new THREE.Mesh(new THREE.CircleGeometry(0.9, 96), accentMaterial({ opacity: 0.16 }));
const notchShape = new THREE.Shape([new THREE.Vector2(1.06, 0.09), new THREE.Vector2(1.22, 0), new THREE.Vector2(1.06, -0.09)]);
const ringNotch = new THREE.Mesh(new THREE.ShapeGeometry(notchShape), accentMaterial({ opacity: 0.95, side: THREE.DoubleSide }));
for (const m of [ringLine, ringDisc, ringNotch]) m.renderOrder = 2;
ring.add(ringDisc, ringLine, ringNotch);
scene3.add(ring);

let ringState = null; // { radius, center: Vector3 } for the current selection, measured on first use
let groundMeshes = []; // static scenery (floor, terrain, furniture) the ring can rest on
const groundRay = new THREE.Raycaster();
const DOWN = new THREE.Vector3(0, 0, -1);

// Height of the ground under (x, y): the robot's feet can sink a little into it, so measure the scenery.
function groundHeight(x, y, top, fallback) {
  groundRay.set(new THREE.Vector3(x, y, top), DOWN);
  const hit = groundRay.intersectObjects(groundMeshes, false).find((h) => h.object.visible);
  const ground = hit ? hit.point.z : fallback;
  const level = water ? stillLevel(water, x, y) : null; // the water takes no rays (water.js): on it, its level
  return level !== null && level > ground && level <= top ? level : ground;
}

const hasMesh = (id) => bodyNodes[id]?.children.some((o) => o.isMesh && o.visible);

function robotBox(name, { baseOnly = false } = {}) {
  const all = robotBodies.get(name);
  if (!all || !worldRoot) return null;
  const base = all.find(hasMesh); // the first body you can see; a robot's root may be a bare frame
  const ids = baseOnly && base !== undefined ? [base] : all;
  worldRoot.updateMatrixWorld();
  const box = new THREE.Box3();
  for (const id of ids) {
    bodyNodes[id]?.traverse((o) => {
      if (o.isMesh && o.visible) box.expandByObject(o);
    });
  }
  return box.isEmpty() ? null : box;
}

// The parts of a robot you can see, as meshes placed where it stands now; empty for a name that is not a robot.
function robotMeshes(name) {
  const meshes = [];
  if (!worldRoot) return meshes;
  worldRoot.updateMatrixWorld();
  for (const id of robotBodies.get(name) ?? []) {
    bodyNodes[id]?.traverse((o) => {
      if (o.isMesh && o.visible) meshes.push(o);
    });
  }
  return meshes;
}

function updateRing() {
  const robot = robotByName(selected);
  // A robot that drives is ringed whole; one bolted down (an arm) is ringed at its base, however far it reaches.
  const box = selected ? robotBox(selected, { baseOnly: !robot?.max_speed }) : null;
  // In a lesson the ring marks only a robot you drive; next to twins or rigs it would just be noise, and so it is
  // when the keys steer something else (a gripper's target: the drive spec says robot: false).
  if (!box || (lessonActive() && (!lessonDriveSpec() || lessonDriveSpec().robot === false))) {
    ring.visible = false;
    return;
  }
  const center = box.getCenter(new THREE.Vector3());
  const now = performance.now();
  if (!ringState) {
    const size = box.getSize(new THREE.Vector3());
    const pad = robot?.max_speed ? 1.05 : 1.6;
    ringState = { radius: Math.max(0.12, Math.hypot(size.x, size.y) * 0.5 * pad), center: center.clone(), t: now };
  }
  // Smooth out the swing of legs and arms, by time so a throttled tab does not leave the ring behind.
  ringState.center.lerp(center, 1 - Math.exp(-(now - ringState.t) / 80));
  ringState.t = now;
  const whole = robot?.max_speed ? box : robotBox(selected);
  const z = groundHeight(ringState.center.x, ringState.center.y, whole.max.z + 0.05, whole.min.z);
  ring.position.set(ringState.center.x, ringState.center.y, z + 0.006);
  ring.scale.setScalar(ringState.radius);
  ringNotch.visible = !!robot?.max_speed;
  const root = bodyNodes[robotBodies.get(selected)[0]];
  if (root) {
    const fwd = new THREE.Vector3(1, 0, 0).applyQuaternion(root.quaternion);
    ring.rotation.set(0, 0, Math.atan2(fwd.y, fwd.x));
  }
  ring.visible = true;
}

// ---------------------------------------------------------------- dragging bodies

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane();
let drag = null; // { body, local: Vector3 }
const dragLine = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineBasicMaterial({ color: theme.amber, depthTest: false, transparent: true })
);
accentMaterials.push(dragLine.material);
dragLine.renderOrder = 10;
dragLine.visible = false;
const dragDot = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 8), accentMaterial({ depthTest: false }));
dragDot.renderOrder = 11;
dragDot.visible = false;
scene3.add(dragLine, dragDot);

readTheme();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", readTheme);

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
    const body = h.object.userData.flex ? flexBodyAt(h.object, h.point) : h.object.userData.body;
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
    const owner = robotOfBody.get(hit.body);
    if (owner && owner !== selected) select(owner);
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

// Double-click the ground: the selected robot finds its way there round whatever is in the way (move with a
// destination) and shows its path. A robot or a loose object under the pointer is for dragging, not a place;
// so is a wall's side: only a face that looks up (the floor, a ramp, a table top) is.
canvas.addEventListener("dblclick", (e) => {
  const robot = driveTarget();
  if (!robot || lessonActive() || !worldRoot || !sceneDesc) return;
  setPointer(e);
  const hit = raycaster.intersectObjects(worldRoot.children, true).find((h) => h.object.visible);
  if (!hit || !hit.face) return;
  const body = hit.object.userData.body;
  if (body > 0 && sceneDesc.bodies[body]?.dynamic) return;
  const up = hit.face.normal.clone().transformDirection(hit.object.matrixWorld);
  if (up.z < 0.7) return;
  const to = [+hit.point.x.toFixed(3), +hit.point.y.toFixed(3)];
  request({ type: "go_to", robot: robot.name, to }).catch((err) => toast(err.message)); // no way: it says why
});

const dragTarget = new THREE.Vector3();
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

// ---------------------------------------------------------------- connection

let ws = null;
let retryMs = 500;
let latestTime = 0;
let bannerTimer = null;

function setConnected(ok) {
  clearTimeout(bannerTimer);
  if (ok) $("banner").hidden = true;
  else bannerTimer = setTimeout(() => ($("banner").hidden = false), 800); // a quick reload should not flash it
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
    setConnected(true);
  };
  ws.onclose = () => {
    setConnected(false);
    for (const p of pending.values()) p.reject(new Error(t("disconnected")));
    pending.clear();
    setTimeout(connect, retryMs);
    retryMs = Math.min(retryMs * 2, 5000);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") {
      applyPoses(ev.data);
      send({ type: "seen" }); // the server keeps only a few frames ahead of this (MAX_POSES_AHEAD in server.py)
      return;
    }
    const msg = JSON.parse(ev.data);
    if (msg.type === "scene") applyScene(msg);
    else if (msg.type === "state") applyState(msg);
    else if (msg.type === "reply") onReply(msg);
    else if (msg.type === "lesson") lessonUpdate(msg);
    else if (msg.type === "sensors") sensorsUpdate(msg);
    else if (msg.type === "error") toast(msg.message);
  };
}

// ---------------------------------------------------------------- robots and selection

let robots = []; // [{name, kind, model?, arms?, max_speed?}] from the server
let learnedWalkers = {}; // robot -> {policy, max_speed}: robots that can walk with a learned policy
let robotModes = {}; // robot -> the mode keys it offers (see modes.py)
let currentModes = {}; // robot -> the mode running now, as the server reports it
let worldInfo = { isWorld: false, floor: null, terrain: [], catalog: null };
let wantFrame = false; // frame the camera on the next scene (after picking a map or terrain)
let robotBodies = new Map(); // robot -> body ids, root first
let robotOfBody = new Map(); // body id -> robot
let selected = null; // robot name, or null when the world has no robots
let behaviors = [];
let worldName = "";

const robotByName = (name) => robots.find((r) => r.name === name) ?? null;

function onScene(desc) {
  robots = desc.robots || [];
  learnedWalkers = desc.learned || {};
  robotModes = desc.modes || {};
  worldInfo = {
    isWorld: !!desc.world,
    floor: desc.floor ?? null,
    terrain: desc.terrain || [],
    catalog: desc.catalog || worldInfo.catalog,
  };
  hideMarker();
  robotBodies = new Map();
  robotOfBody = new Map();
  for (const b of desc.bodies) {
    const cut = b.name.indexOf("/");
    const owner = cut > 0 ? b.name.slice(0, cut) : null;
    if (!owner || !robots.some((r) => r.name === owner)) continue;
    if (!robotBodies.has(owner)) robotBodies.set(owner, []);
    robotBodies.get(owner).push(b.id);
    robotOfBody.set(b.id, owner);
  }
  groundMeshes = [];
  worldRoot.traverse((o) => {
    const body = o.userData.body;
    if (o.isMesh && body !== undefined && !desc.bodies[body]?.dynamic && !robotOfBody.has(body)) groundMeshes.push(o);
  });
  if (!robotByName(selected)) selected = (robots.find((r) => r.max_speed) ?? robots[0])?.name ?? null;
  ringState = null;
  worldName = desc.world || worldName;
  renderTitle();
  renderRobotList();
  renderSelection();
  renderWorldPanel();
  lessonScene(desc);
  sensorsScene(desc);
  $("empty").hidden = !(desc.world && robots.length === 0);
}

function renderTitle() {
  const name = lessonTitle() ?? worldName;
  $("world").textContent = name;
  document.title = name ? `${name} · robotforge` : "robotforge";
}

function renderRobotList() {
  const list = $("robot-list");
  list.replaceChildren(
    ...robots.map((r) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "choice";
      b.dataset.robot = r.name;
      b.setAttribute("aria-pressed", String(r.name === selected));
      const name = document.createElement("span");
      name.className = "choice-title";
      name.textContent = r.name;
      const meta = document.createElement("span");
      meta.className = "choice-desc";
      meta.textContent = [t(`kind_${r.kind}`), r.model].filter(Boolean).join(" · ");
      b.append(name, meta);
      b.addEventListener("click", () => select(r.name, { focus: true }));
      return b;
    })
  );
  $("robot-section").hidden = robots.length === 0;
  const n = robots.length || (sceneDesc?.actuators.length ?? 0);
  $("drawer-label").textContent = robots.length ? t("robots") : t("joints");
  $("drawer-count").textContent = n ? String(n) : "";
}

function select(name, { focus = false } = {}) {
  if (name !== selected) {
    stopDriving();
    selected = name;
    ringState = null;
    follow = null;
  }
  for (const b of $("robot-list").children) b.setAttribute("aria-pressed", String(b.dataset.robot === selected));
  renderSelection();
  if (focus) focusRobot(name);
}

function focusRobot(name) {
  refitRig = null;
  const box = robotBox(name);
  if (!box || !hasSize()) return;
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const dir = camera.position.clone().sub(controls.target).normalize();
  const d = fitDistance(Math.max(sphere.radius, 0.15)) * 2.2; // room around it to see where it goes
  moveCamera(sphere.center.clone().add(clearSight(name, box, dir, d).multiplyScalar(d)), sphere.center, true);
}

// The way from a robot to the camera, turned around the vertical as little as it takes for nothing to stand in
// between: on a phone, whose narrow view puts the camera twice as far back, the park's tree hid the Go2 just picked.
// The view is clear when the robot's middle and the corners of its box (drawn in a little, to the robot's own
// outline) all see the camera: a ray from the middle alone passed the warehouse shelf hiding half the Go2.
const sightRay = new THREE.Raycaster();
const UP = new THREE.Vector3(0, 0, 1);
const SIGHT_TURNS = [0, 20, -20, 40, -40, 60, -60, 90, -90, 120, -120, 150, -150, 180]; // degrees
const SIGHT_INSET = 0.15; // share of the box each corner is drawn in by
function clearSight(name, box, dir, d) {
  const others = [];
  worldRoot.traverse((o) => {
    const body = o.userData.body;
    if (!o.isMesh || !o.visible || body === undefined || o.userData.isPlane) return;
    if (robotOfBody.get(body) !== name) others.push(o);
  });
  const center = box.getCenter(new THREE.Vector3());
  const inner = box.clone().expandByVector(box.getSize(new THREE.Vector3()).multiplyScalar(-SIGHT_INSET));
  const points = [center];
  for (const x of [inner.min.x, inner.max.x]) for (const y of [inner.min.y, inner.max.y]) for (const z of [inner.min.z, inner.max.z]) points.push(new THREE.Vector3(x, y, z));
  const eye = new THREE.Vector3();
  const toward = new THREE.Vector3();
  let best = dir; // walled in on every side: keep the view the learner had, or the least hidden one
  let fewest = points.length;
  for (const turn of SIGHT_TURNS) {
    const way = dir.clone().applyAxisAngle(UP, THREE.MathUtils.degToRad(turn));
    eye.copy(center).addScaledVector(way, d);
    let blocked = 0;
    for (const p of points) {
      toward.copy(eye).sub(p);
      sightRay.set(p, toward.clone().normalize());
      sightRay.far = toward.length();
      if (sightRay.intersectObjects(others, false).length && ++blocked >= fewest) break;
    }
    if (blocked === 0) return way;
    if (blocked < fewest) [best, fewest] = [way, blocked];
  }
  return best;
}

function renderSelection() {
  renderJoints();
  renderModes();
  renderDrive();
  sensorsSelect();
}

// ---------------------------------------------------------------- joint sliders

const sliderEls = new Map();
let activeSlider = null;

function humanize(label) {
  return label.replaceAll("_", " ").replace(/\s+/g, " ").trim();
}

function setFill(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const p = max > min ? ((Number(input.value) - min) / (max - min)) * 100 : 0;
  input.style.setProperty("--p", `${p}%`);
}

function sliderRow(a, label) {
  const row = document.createElement("label");
  row.className = "joint";
  const name = document.createElement("span");
  name.className = "joint-name";
  name.textContent = humanize(label);
  name.title = a.name;
  const value = document.createElement("span");
  value.className = "joint-value";
  value.textContent = "0.00";
  const input = document.createElement("input");
  input.type = "range";
  input.min = a.range[0];
  input.max = a.range[1];
  input.step = (a.range[1] - a.range[0]) / 500 || 0.001;
  input.value = Math.min(Math.max(0, a.range[0]), a.range[1]);
  input.setAttribute("aria-label", a.name);
  setFill(input);
  input.addEventListener("pointerdown", () => (activeSlider = input));
  input.addEventListener("pointerup", () => (activeSlider = null));
  input.addEventListener("input", () => {
    value.textContent = Number(input.value).toFixed(2);
    setFill(input);
    send({ type: "ctrl", id: a.id, value: Number(input.value) });
  });
  row.append(name, value, input);
  sliderEls.set(a.id, { input, value });
  return row;
}

function renderJoints() {
  const box = $("joints");
  box.replaceChildren();
  sliderEls.clear();
  const actuators = sceneDesc?.actuators ?? [];
  const split = (a) => {
    const cut = a.name.indexOf("/");
    return cut > 0 ? [a.name.slice(0, cut), a.name.slice(cut + 1)] : ["", a.name];
  };
  const mine = selected ? actuators.filter((a) => split(a)[0] === selected) : [];
  const loose = actuators.filter((a) => !robotByName(split(a)[0]));
  const title = $("detail-title");
  title.textContent = t("joints");
  if (selected) {
    const subject = document.createElement("span");
    subject.className = "subject";
    subject.textContent = ` · ${selected}`;
    title.append(subject);
  }
  $("joint-count").textContent = (selected ? mine.length : loose.length) || "";
  $("detail-section").hidden = !selected && loose.length === 0;
  const rows = selected ? mine : loose;
  for (const a of rows) box.append(sliderRow(a, split(a)[1]));
  if (selected && rows.length === 0) {
    const p = document.createElement("p");
    p.className = "note";
    p.textContent = t("no_joints");
    box.append(p);
  }
  if (selected && loose.length) {
    // motors that belong to no robot (e.g. a hand-written model next to a robot)
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `${t("other")} (${loose.length})`;
    details.append(summary, ...loose.map((a) => sliderRow(a, split(a)[1])));
    box.append(details);
  }
}

// ---------------------------------------------------------------- simulation state

let paused = false;
let rtf = 1;

function applyState(s) {
  paused = s.paused;
  rtf = s.rtf;
  rotorCommands(s.ctrl, paused);
  behaviors = s.behaviors || [];
  updatePaths(behaviors);
  const play = $("btn-play");
  play.classList.toggle("is-paused", paused);
  const tip = t(paused ? "play_tip" : "pause_tip");
  play.setAttribute("aria-label", tip);
  play.title = tip;
  for (const b of $("speed").children) b.setAttribute("aria-pressed", String(Number(b.dataset.speed) === s.speed));
  if (!worldName && s.source && !s.source.startsWith("lesson:")) {
    worldName = s.source.split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
    renderTitle();
  }
  $("world").title = s.source || "";
  const active = behaviors.find((b) => b.robot === selected);
  $("drive-speed").textContent = active && "speed" in active && Math.abs(active.speed) > 0.005 ? `${active.speed.toFixed(2)} m/s` : "";
  const modes = s.mode || {};
  if (JSON.stringify(modes) !== JSON.stringify(currentModes)) {
    currentModes = modes;
    renderModes();
    if (markerFor && currentModes[markerFor.robot] !== markerFor.mode) hideMarker();
  }
  s.ctrl.forEach((v, i) => {
    const el = sliderEls.get(i);
    if (el && el.input !== activeSlider) {
      el.input.value = v;
      el.value.textContent = v.toFixed(2);
      setFill(el.input);
    }
  });
}

// ---------------------------------------------------------------- keyboard driving

const DRIVE_KEYS = {
  w: "fwd", arrowup: "fwd", s: "back", arrowdown: "back", a: "left", arrowleft: "left", d: "right", arrowright: "right",
  e: "up", q: "down", // an underwater robot or an aircraft only: up and down
};
// Aircraft: A and D fly sideways (a rover can only turn), the arrow keys turn, E climbs and Q descends (and lands).
const FLY_KEYS = { ...DRIVE_KEYS, a: "slide_left", d: "slide_right", e: "up", q: "down" };
const isAircraft = (robot) => robot?.kind === "aircraft";
const RISES = new Set(["rov", "aircraft"]); // kinds that E and Q take up and down
// rad/s at full key; walkers turn at 1.0
const TURN_RATE = { rover: 2.0, balancer: 1.5, boat: 0.8, rov: 0.8, aircraft: 1.2 };
const HEAVE = 0.3; // m/s up or down for an underwater robot at full key
const FLY_CLIMB = 1.0; // m/s up or down for an aircraft at full key
const pressed = new Set();
let lastDrive = "";

function driveTarget() {
  const robot = robotByName(selected);
  return robot?.max_speed ? robot : null;
}

function renderDrive() {
  const robot = driveTarget();
  $("drive").hidden = !robot;
  $("drive-heave").hidden = robot?.kind !== "rov";
  $("fly-keys").hidden = !isAircraft(robot);
  const say = isAircraft(robot) ? "fly" : robot?.kind === "rov" ? "drive_rov" : "drive";
  if (robot) $("drive-text").innerHTML = t(say, { name: escapeHtml(robot.name) });
  lastDrive = "";
}

function driveCommand(robot) {
  const fwd = (pressed.has("fwd") ? 1 : 0) - (pressed.has("back") ? 1 : 0);
  const left = (pressed.has("left") ? 1 : 0) - (pressed.has("right") ? 1 : 0);
  const learned = learnedWalkers[robot.name];
  const top = (currentModes[robot.name] === "learned" ? learned?.max_speed : robot.max_speed) ?? 0.3;
  const cmd = { type: "teleop", robot: robot.name, speed: fwd * top * 0.6, turn: left * (TURN_RATE[robot.kind] ?? 1.0) };
  if (robot.kind === "rov") cmd.heave = ((pressed.has("up") ? 1 : 0) - (pressed.has("down") ? 1 : 0)) * HEAVE;
  if (isAircraft(robot)) {
    cmd.speed = fwd * top * 0.5;
    cmd.side = ((pressed.has("slide_left") ? 1 : 0) - (pressed.has("slide_right") ? 1 : 0)) * top * 0.5;
    cmd.climb = ((pressed.has("up") ? 1 : 0) - (pressed.has("down") ? 1 : 0)) * FLY_CLIMB;
  }
  return cmd;
}

function sendDrive() {
  const robot = driveTarget();
  for (const k of document.querySelectorAll("#drive kbd, #lesson-drive kbd")) k.classList.toggle("on", pressed.has((isAircraft(robot) && k.dataset.fly) || k.dataset.key));
  const lessonDrive = lessonDriveSpec();
  if (lessonDrive) {
    // A lesson with a robot to drive takes the keys; the lesson's own controller turns them into wheel speeds.
    let fwd = (pressed.has("fwd") ? 1 : 0) - (pressed.has("back") ? 1 : 0);
    let left = (pressed.has("left") ? 1 : 0) - (pressed.has("right") ? 1 : 0);
    if (lessonDrive.robot === false) [fwd, left] = screenToWorld(fwd, left);
    const cmd = { type: "lesson", action: "drive", speed: fwd * lessonDrive.max_speed * 0.7, turn: left * lessonDrive.turn * 0.7 };
    const key = `lesson:${cmd.speed}:${cmd.turn}`;
    if (key === lastDrive) return;
    lastDrive = key;
    send(cmd);
    return;
  }
  if (!robot) return;
  const cmd = driveCommand(robot);
  const key = `${cmd.robot}:${cmd.speed}:${cmd.turn}:${cmd.heave ?? ""}:${cmd.side ?? 0}:${cmd.climb ?? 0}`;
  if (key === lastDrive) return;
  lastDrive = key;
  send(cmd);
}

// Keys that steer a target over the ground (the drive spec says robot: false) go the way they point on screen: W
// away from the viewer, A to the left, whichever way the camera looks. The lesson takes x and y in the world (speed
// along x, turn along y); in the world's own axes A moved the target right from the imitation lesson's camera.
function screenToWorld(fwd, left) {
  const ahead = camera.getWorldDirection(new THREE.Vector3()).setZ(0);
  if (ahead.lengthSq() < 1e-6) ahead.set(0, 1, 0).applyQuaternion(camera.quaternion).setZ(0); // straight down: screen up
  ahead.normalize();
  const x = fwd * ahead.x - left * ahead.y; // left of ahead is ahead turned a quarter anticlockwise: (-y, x)
  const y = fwd * ahead.y + left * ahead.x;
  return [Number(x.toFixed(3)), Number(y.toFixed(3))];
}

// The camera turned while a key is held: the held key now points another way.
controls.addEventListener("change", () => {
  if (pressed.size && lessonDriveSpec()?.robot === false) sendDrive();
});

function stopDriving() {
  // Switching robots mid-drive must not leave the old one walking off on its own.
  const robot = driveTarget();
  if (robot && pressed.size) send({ type: "teleop", robot: robot.name, speed: 0, turn: 0 });
  pressed.clear();
  lastDrive = "";
}

// Follow the driven robot with the camera until the user orbits by hand.
let follow = null; // { body, last: Vector3, z: smoothed height, t: last update ms }
const FOLLOW_Z_SECONDS = 0.6; // height follows this slowly: up stairs and hills, not every stride
const FOLLOW_RECENTER = 1.0; // m: lessons aim a little beside the robot they follow; further off is an accident

function startFollow() {
  startFollowBody(sceneDesc?.bodies.find((b) => b.name.startsWith(`${selected}/`) && b.dynamic)?.id ?? null);
}

function startFollowBody(body, { keepTween = false } = {}) {
  if (body === null || !bodyNodes[body]) return;
  if (follow && follow.body === body) return;
  if (!keepTween) tween = null;
  const start = bodyNodes[body].position;
  follow = { body, last: start.clone(), z: start.z, t: performance.now() };
}

function updateFollow() {
  if (!follow || !bodyNodes[follow.body]) return;
  const now = bodyNodes[follow.body].position;
  const delta = now.clone().sub(follow.last);
  const t = performance.now();
  const alpha = 1 - Math.exp(-Math.min(t - follow.t, 200) / 1000 / FOLLOW_Z_SECONDS);
  const z = follow.z + (now.z - follow.z) * alpha;
  delta.z = z - follow.z; // across the ground at once; up and down slowly, so the gait does not bob the view
  follow.z = z;
  follow.t = t;
  camera.position.add(delta);
  controls.target.add(delta);
  // Mid-glide the tween sets the camera every frame: carry its ends along too, or a robot that moves (or is put
  // back on its start line) while the camera glides leaves the view that far behind for good.
  if (tween) for (const v of [tween.fromEye, tween.fromTarget, tween.toEye, tween.toTarget]) v.add(delta);
  follow.last.copy(now);
}

controls.addEventListener("start", () => {
  follow = null;
  tween = null;
  refitRig = null; // the learner's own view stays
});

function driveKey(e, down) {
  const lessonDrive = lessonDriveSpec();
  if (!driveTarget() && !lessonDrive) return false;
  const dir = (isAircraft(driveTarget()) && !lessonDrive ? FLY_KEYS : DRIVE_KEYS)[e.key.toLowerCase()];
  if (!dir) return false;
  if ((dir === "up" || dir === "down") && !RISES.has(driveTarget()?.kind)) return false;
  e.preventDefault();
  if (down) {
    if (!lessonDrive) startFollow();
    else if (lessonDrive.follow) startFollowBody(sceneDesc?.bodies.find((b) => b.name === lessonDrive.follow)?.id ?? null);
    pressed.add(dir);
  } else if (!pressed.delete(dir)) {
    return true; // a key that was not held here (typed into a field, say): nothing to stop
  }
  sendDrive();
  return true;
}

window.addEventListener("keyup", (e) => driveKey(e, false));
window.addEventListener("blur", () => {
  // Only a robot being driven is stopped. A stop sent for nothing counted as moving the selected robot, which
  // takes a quadruped off the stance it stands in before anyone moves it.
  if (!pressed.size) return;
  pressed.clear();
  sendDrive();
});

// ---------------------------------------------------------------- panels, buttons and keys

function storageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null; // storage may be unavailable (private mode, sandboxed iframe); the page still works
  }
}

function storageSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // see storageGet
  }
}

function setOpen(toggleId, panelId, open) {
  $(panelId).hidden = !open;
  $(toggleId).setAttribute("aria-expanded", String(open));
}

const narrow = () => window.innerWidth <= 700; // panels span the width; one at a time

function setDrawer(open) {
  setOpen("drawer-toggle", "drawer", open);
  if (open) setOpen("help-toggle", "help", false);
  if (open && narrow()) setOpen("world-toggle", "world-panel", false);
  storageSet("robotforge.drawer", open ? "1" : "0");
  shiftForDrawers();
}

function setWorldPanel(open) {
  setOpen("world-toggle", "world-panel", open);
  if (open) setOpen("help-toggle", "help", false);
  if (open && narrow()) setOpen("drawer-toggle", "drawer", false);
  storageSet("robotforge.world", open ? "1" : "0");
  shiftForDrawers();
}

// The part of the window no panel covers. Left panels and right columns bound it sideways; a panel spanning most
// of the width (a phone's sheet, plots across the top of what is left) bounds it from above or below.
function freeRect() {
  const W = window.innerWidth;
  const H = window.innerHeight;
  // Shown at all (offsetParent cannot tell: it is null for every fixed panel, shown or not).
  const box = (el) => (el && !el.hidden && el.getClientRects().length ? el.getBoundingClientRect() : null);
  let [left, right, top, bottom] = [0, W, 0, H];
  const across = (r) => {
    if (r.top + r.height / 2 > H / 2) bottom = Math.min(bottom, r.top);
    else top = Math.max(top, r.bottom);
  };
  for (const id of ["world-panel", "lesson-panel"]) {
    const r = box($(id));
    if (!r) continue;
    if (r.width > W * 0.7) across(r);
    else left = Math.max(left, r.right);
  }
  for (const id of ["drawer", "lesson-plots", "sensor-views"]) {
    const r = box($(id));
    if (!r) continue;
    if (r.width < W * 0.7 && r.left > left + 160) right = Math.min(right, r.left); // a column beside the view
    else across(r); // a strip over what is left
  }
  const bar = box(document.querySelector(".top"));
  if (bar) top = Math.max(top, bar.bottom);
  const dock = box(document.querySelector(".dock"));
  if (dock) bottom = Math.min(bottom, dock.top);
  return { left, right, top, bottom };
}

// Lessons frame their rig for a big window. Where the free part of the window is short (a split screen, a phone
// with the lesson sheet up), step back from the target so the rig still fits; a big window keeps the framing.
const FRAMED_FOR = 600; // px of free height a lesson camera is framed for
// Free width over window height a lesson camera is framed for (a rig's width on screen grows with the window's
// height). At 1280 x 800 between the lesson panel and the plots (0.69), the third PID twin reached under the plots.
const FRAMED_WIDE = 0.85;
function fitToFree(eye, target) {
  if (!hasSize()) return eye;
  const r = freeRect();
  const tall = Math.sqrt(FRAMED_FOR / Math.max(120, r.bottom - r.top));
  const wide = Math.sqrt((FRAMED_WIDE * window.innerHeight) / Math.max(120, r.right - r.left));
  const k = Math.min(1.6, Math.max(1, tall, wide));
  return target.clone().add(eye.clone().sub(target).multiplyScalar(k));
}

// A lesson camera that asks for it (`fit`) steps back until every robot of its rig is inside the free part of the
// window: on a phone the third PID twin's arm reached off the screen, framed for a wide window.
const RIG_MARGIN = 0.9; // share of the free half-width and half-height the rig may fill
function fitRig(eye, target) {
  if (!hasSize() || !worldRoot) return eye;
  const box = new THREE.Box3();
  worldRoot.updateMatrixWorld(true);
  worldRoot.traverse((o) => {
    if (o.isMesh && o.visible && robotOfBody.has(o.userData.body)) box.expandByObject(o);
  });
  if (box.isEmpty()) return eye;
  const corners = [];
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) corners.push(new THREE.Vector3(x, y, z));
  const r = freeRect();
  const halfW = (RIG_MARGIN * (r.right - r.left)) / 2;
  const halfH = (RIG_MARGIN * (r.bottom - r.top)) / 2;
  const probe = camera.clone();
  probe.clearViewOffset(); // the view offset only moves the picture's center into the free part
  probe.updateProjectionMatrix();
  const away = eye.clone().sub(target);
  let k = 1;
  for (let i = 0; i < 20; i++) {
    probe.position.copy(target).addScaledVector(away, k);
    probe.lookAt(target);
    probe.updateMatrixWorld(true);
    let worst = 0;
    for (const c of corners) {
      const p = c.clone().project(probe);
      worst = Math.max(worst, (Math.abs(p.x) * window.innerWidth) / 2 / halfW, (Math.abs(p.y) * window.innerHeight) / 2 / halfH);
    }
    if (worst <= 1 || k >= 2.5) break;
    k = Math.min(2.5, k * Math.min(1.2, Math.max(1.03, worst)));
  }
  return target.clone().addScaledVector(away, k);
}

// Slide the view's center into the free part of the window, so what you look at stays in the clear.
function shiftForDrawers() {
  const r = freeRect();
  const W = window.innerWidth;
  const H = window.innerHeight;
  const roomy = r.right - r.left > 80 && r.bottom - r.top > 80; // panels everywhere: no place to slide to
  viewShiftTarget.x = roomy ? W / 2 - (r.left + r.right) / 2 : 0;
  viewShiftTarget.y = roomy ? H / 2 - (r.top + r.bottom) / 2 : 0;
}
window.addEventListener("resize", shiftForDrawers);
// Panels change size with what they show (a quiz answered, the record unfolded): follow them.
const panelWatch = new ResizeObserver(() => shiftForDrawers());
for (const id of ["drawer", "world-panel", "lesson-panel", "lesson-plots", "sensor-views"]) panelWatch.observe($(id));

function setHelp(open) {
  setOpen("help-toggle", "help", open);
  if (open) {
    setOpen("drawer-toggle", "drawer", false);
    setOpen("world-toggle", "world-panel", false);
    shiftForDrawers();
  }
}

$("drawer-toggle").addEventListener("click", () => setDrawer($("drawer").hidden));
$("help-toggle").addEventListener("click", () => setHelp($("help").hidden));
$("world-toggle").addEventListener("click", () => setWorldPanel($("world-panel").hidden));
setDrawer(storageGet("robotforge.drawer") === "1");
const savedWorld = storageGet("robotforge.world"); // first visit on a big screen: show what you can pick
setWorldPanel(savedWorld === null ? window.innerWidth > 900 : savedWorld === "1");

let toastTimer = null;
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 5000);
}

// A one-time hint for first visits; it leaves on the first touch, or by itself.
function showCoach() {
  if (storageGet("robotforge.coached")) return;
  $("coach").hidden = false;
  setTimeout(dismissCoach, 9000);
}

function dismissCoach() {
  const el = $("coach");
  if (el.hidden || el.classList.contains("fade")) return;
  storageSet("robotforge.coached", "1");
  el.classList.add("fade");
  setTimeout(() => (el.hidden = true), 700);
}
showCoach();
// Any first touch counts, a button as much as the canvas: opening Learn left the hint floating over the panel.
document.addEventListener("pointerdown", dismissCoach, { capture: true });

const togglePlay = () => send({ type: "cmd", cmd: paused ? "play" : "pause" });
$("btn-play").addEventListener("click", togglePlay);
$("btn-reset").addEventListener("click", () => send({ type: "cmd", cmd: "reset" }));
$("btn-frame").addEventListener("click", () => frameCamera({ animate: true }));
$("btn-record").addEventListener("click", () => (recording ? stopRecording() : startRecording()));

// ---------------------------------------------------------------- recording

// The 3D view as a video a learner can share, recorded by the browser itself (MediaRecorder): no library.
const RECORD_MAX_S = 30;
const RECORD_TYPES = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm"];
let recording = null; // { recorder, chunks, type, t0, timer }

function startRecording() {
  const type = RECORD_TYPES.find((k) => window.MediaRecorder?.isTypeSupported?.(k));
  if (!type || !canvas.captureStream) {
    toast(t("record_unsupported"));
    return;
  }
  const recorder = new MediaRecorder(canvas.captureStream(30), { mimeType: type, videoBitsPerSecond: 8_000_000 });
  const chunks = [];
  recorder.ondataavailable = (e) => {
    if (e.data.size) chunks.push(e.data);
  };
  recorder.onstop = () => saveRecording(new Blob(chunks, { type: type.split(";")[0] }));
  recorder.start(1000);
  recording = { recorder, type, t0: performance.now(), timer: setInterval(tickRecording, 250) };
  $("btn-record").setAttribute("aria-pressed", "true");
  tickRecording();
}

function tickRecording() {
  if (!recording) return;
  const s = (performance.now() - recording.t0) / 1000;
  if (s >= RECORD_MAX_S) return stopRecording();
  $("record-label").textContent = `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

function stopRecording() {
  if (!recording) return;
  clearInterval(recording.timer);
  recording.recorder.stop(); // onstop saves what was recorded
  recording = null;
  $("btn-record").setAttribute("aria-pressed", "false");
  $("record-label").textContent = t("record");
}

function saveRecording(blob) {
  const d = new Date(); // local time, as the learner's clock shows it
  const two = (n) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${two(d.getMonth() + 1)}${two(d.getDate())}-${two(d.getHours())}${two(d.getMinutes())}${two(d.getSeconds())}`;
  const ext = blob.type.includes("mp4") ? "mp4" : "webm";
  const name = `robotforge-${(lessonTitle() ? "lesson" : worldName || "view").replace(/[^\w-]+/g, "_")}-${stamp}.${ext}`;
  if (window.__robotforge?.onRecording) return window.__robotforge.onRecording(blob, name); // scripted checks
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
  toast(t("record_saved", { name }));
}
for (const b of $("speed").children) {
  b.addEventListener("click", () => send({ type: "speed", value: Number(b.dataset.speed) }));
}
$("show-collision").addEventListener("change", (e) => {
  showCollision = e.target.checked;
  ringState = null;
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
  if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
  dismissCoach();
  if (driveKey(e, true)) return;
  if (e.target instanceof HTMLButtonElement && (e.key === " " || e.key === "Enter")) return; // let buttons click
  if (e.code === "Space") {
    e.preventDefault();
    togglePlay();
  } else if (e.key === "r" || e.key === "R") {
    send({ type: "cmd", cmd: "reset" });
  } else if (e.key === ".") {
    send({ type: "cmd", cmd: "step" });
  } else if (e.key === "f" || e.key === "F") {
    frameCamera({ animate: true });
  } else if (e.key === "?") {
    setHelp($("help").hidden);
  } else if (e.key === "Escape") {
    setHelp(false);
    setDrawer(false);
    setWorldPanel(false);
  }
});

// ---------------------------------------------------------------- world panel: maps, terrain, modes

let requestSeq = 0;
const pending = new Map(); // request id -> { resolve, reject }

// Ask the server for something slow (a new map, a control mode); resolves with its reply.
function request(msg) {
  return new Promise((resolve, reject) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      reject(new Error(t("disconnected")));
      return;
    }
    const id = ++requestSeq;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ ...msg, id }));
  });
}

function onReply(msg) {
  const waiting = pending.get(msg.id);
  if (!waiting) return;
  pending.delete(msg.id);
  if (msg.ok) waiting.resolve(msg.data);
  else waiting.reject(new Error(msg.message));
}

let busy = false;

async function runChoice(button, msg, { frame = false, done } = {}) {
  if (busy) return;
  busy = true;
  wantFrame = frame;
  button.classList.add("busy");
  setPanelBusy(true);
  try {
    const data = await request(msg); // not inside done?.(...): that skips its arguments when done is unset
    done?.(data);
  } catch (err) {
    wantFrame = false;
    toast(err.message);
  } finally {
    busy = false;
    button.classList.remove("busy");
    setPanelBusy(false);
  }
}

function setPanelBusy(on) {
  $("world-panel").setAttribute("aria-busy", String(on));
  for (const b of $("world-panel").querySelectorAll(".choice, .swatch")) b.disabled = on || b.dataset.off === "1";
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function choice({ title, desc, pressed, off = false, onClick }) {
  const b = el("button", "choice");
  b.type = "button";
  b.setAttribute("aria-pressed", String(!!pressed));
  b.dataset.off = off ? "1" : "0";
  b.disabled = off || busy;
  const spinner = el("span", "spinner");
  spinner.setAttribute("aria-hidden", "true");
  b.append(el("span", "choice-title", title), spinner, el("span", "choice-desc", desc));
  b.addEventListener("click", () => onClick(b));
  return b;
}

function selectTab(name) {
  for (const tab of $("world-panel").querySelectorAll("[role=tab]")) tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
  for (const body of $("world-panel").querySelectorAll(".tab-body")) body.hidden = body.dataset.body !== name;
  storageSet("robotforge.tab", name);
}

for (const tab of $("world-panel").querySelectorAll("[role=tab]")) tab.addEventListener("click", () => selectTab(tab.dataset.tab));
selectTab(storageGet("robotforge.tab") || "maps");

let lastMap = null; // the map picked here; a world does not remember which preset it came from

function renderWorldPanel() {
  renderMaps();
  renderGround();
  renderTerrain();
  renderModes();
}

function renderMaps() {
  const cat = worldInfo.catalog;
  if (!cat) return;
  const names = ["empty", ...cat.scenes.map((sc) => sc.name)];
  $("map-list").replaceChildren(
    ...names.map((name) => {
      const scene = cat.scenes.find((sc) => sc.name === name);
      return choice({
        title: tOr(`map_${name}`, scene?.title ?? name),
        desc: tOr(`map_${name}_desc`, scene?.summary ?? ""),
        pressed: name === lastMap,
        onClick: (b) =>
          runChoice(b, { type: "world", action: "scene", preset: name }, { frame: true, done: () => {
            lastMap = name;
            renderMaps();
          } }),
      });
    })
  );
}

function renderGround() {
  const cat = worldInfo.catalog;
  if (!cat) return;
  $("ground-list").replaceChildren(
    ...Object.entries(cat.grounds).map(([name, rgb]) => {
      const b = el("button", "swatch");
      b.type = "button";
      b.setAttribute("aria-pressed", String(worldInfo.floor === name));
      b.dataset.off = worldInfo.isWorld ? "0" : "1";
      b.disabled = !worldInfo.isWorld || busy;
      const dot = el("i");
      dot.style.setProperty("--c", `rgb(${rgb.map((c) => Math.round(c * 255)).join(",")})`);
      b.append(dot, el("span", "", tOr(`ground_${name}`, name)));
      b.addEventListener("click", () => runChoice(b, { type: "world", action: "floor", style: name }));
      return b;
    })
  );
}

function renderTerrain() {
  const cat = worldInfo.catalog;
  if (!cat) return;
  const placed = worldInfo.terrain;
  const now = placed.length === 0 ? "flat" : placed.length === 1 ? placed[0] : null;
  $("terrain-list").replaceChildren(
    ...cat.terrains.map((type) =>
      choice({
        title: tOr(`terrain_${type}`, type),
        desc: tOr(`terrain_${type}_desc`, ""),
        pressed: worldInfo.isWorld && type === now,
        off: !worldInfo.isWorld,
        onClick: (b) => runChoice(b, { type: "world", action: "terrain", terrain: type, robot: selected }, { frame: true }),
      })
    )
  );
  $("terrain-note").hidden = worldInfo.isWorld;
}

const RETARGETS = new Set(["reach_ik", "reach_cartesian", "reach_mpc", "goto_mpc"]); // a new random target each time

function renderModes() {
  const chips = $("mode-robots");
  chips.replaceChildren(
    ...robots.map((r) => {
      const b = el("button", "", r.name);
      b.type = "button";
      b.setAttribute("aria-pressed", String(r.name === selected));
      b.addEventListener("click", () => select(r.name, { focus: true }));
      return b;
    })
  );
  chips.hidden = robots.length < 2;
  if (!selected) {
    $("mode-lead").textContent = t("modes_none");
    $("mode-list").replaceChildren();
    return;
  }
  $("mode-lead").textContent = t("modes_lead", { name: selected });
  const robot = robotByName(selected);
  const now = currentModes[selected] ?? (robot?.max_speed ? "drive" : "manual");
  $("mode-list").replaceChildren(
    ...(robotModes[selected] || []).map((key) =>
      choice({
        title: tOr(`mode_${key}`, key),
        desc: [tOr(`mode_${key}_desc`, ""), RETARGETS.has(key) ? t("again") : ""].filter(Boolean).join(" "),
        pressed: key === now,
        onClick: (b) => runChoice(b, { type: "mode", robot: selected, mode: key }, { done: onModeSet }),
      })
    )
  );
}

function onModeSet(data) {
  currentModes = { ...currentModes, [data.robot]: data.mode };
  renderModes();
  if (data.marker) showMarker(data.robot, data.mode, data.marker);
  else hideMarker();
  if (data.follows && data.robot === selected) startFollow();
}

// ---------------------------------------------------------------- target marker

// Where a reaching arm or a rover driving to a goal is headed: an amber dot, or a pad on the ground.
const markerDot = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 12), accentMaterial({ opacity: 0.95 }));
const markerPin = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineBasicMaterial({ color: theme.amber, transparent: true, opacity: 0.8 })
);
accentMaterials.push(markerPin.material);
markerPin.visible = false;
scene3.add(markerPin);
const markerPad = new THREE.Mesh(new THREE.RingGeometry(0.55, 1, 48), accentMaterial({ opacity: 0.95, side: THREE.DoubleSide }));
markerDot.visible = markerPad.visible = false;
markerDot.renderOrder = markerPad.renderOrder = 3;
scene3.add(markerDot, markerPad);
let markerFor = null; // { robot, mode } the marker belongs to

function showMarker(robot, mode, point) {
  markerFor = { robot, mode };
  if (point.length === 3) {
    markerDot.position.fromArray(point);
    markerDot.scale.setScalar(0.025);
    const floor = groundHeight(point[0], point[1], point[2], 0);
    const pin = markerPin.geometry.attributes.position;
    pin.setXYZ(0, point[0], point[1], point[2]);
    pin.setXYZ(1, point[0], point[1], floor);
    pin.needsUpdate = true;
    markerPin.geometry.computeBoundingSphere();
    markerDot.visible = markerPin.visible = true;
    markerPad.visible = false;
  } else {
    const [x, y] = point;
    markerPad.position.set(x, y, groundHeight(x, y, 5, 0) + 0.008);
    markerPad.scale.setScalar(0.12);
    markerPad.visible = true;
    markerDot.visible = markerPin.visible = false;
  }
}

function hideMarker() {
  markerFor = null;
  markerDot.visible = markerPad.visible = markerPin.visible = false;
}

// ---------------------------------------------------------------- planned paths

// A robot sent somewhere (move with a destination) shows the way still ahead of it: a thin amber line on the
// ground and a ring where it ends. The server sends the path (x, y and the ground height at each corner) with
// every state message while the robot follows it, and stops once it is there, so the line goes with it.
const PATH_WIDTH = 0.035; // m, seen from nearby
// CSS px it keeps on screen from further away, across the line of sight: with the whole obstacle course in view
// (15 to 20 m off, looking down at 38 degrees) a 3.5 cm line was 1 to 2 px wide and hard to make out.
const PATH_MIN_PX = 3;
const PATH_LIFT = 0.012; // m above the ground: it lies on the floor without flickering into it
const PATH_STEP = 0.15; // m between the points that follow a hill or a ramp along a straight stretch
const pathMaterial = accentMaterial({ opacity: 0.9, side: THREE.DoubleSide });
const pathEndGeometry = new THREE.RingGeometry(0.055, 0.085, 40);
const paths = new Map(); // robot -> { line, end, key }

function updatePaths(list) {
  const live = new Set();
  for (const b of list) {
    if (!b.navigating || !Array.isArray(b.path) || b.path.length < 2) continue;
    live.add(b.navigating);
    let shape = paths.get(b.navigating);
    if (!shape) {
      shape = {
        line: new THREE.Mesh(new THREE.BufferGeometry(), pathMaterial),
        end: new THREE.Mesh(pathEndGeometry, pathMaterial),
        key: "",
      };
      shape.line.renderOrder = shape.end.renderOrder = 3;
      scene3.add(shape.line, shape.end);
      paths.set(b.navigating, shape);
    }
    const key = JSON.stringify(b.path);
    if (key === shape.key) continue;
    shape.key = key;
    shape.points = groundPoints(b.path);
    ribbon(shape.line.geometry, shape.points);
    shape.end.position.copy(shape.points[shape.points.length - 1]);
    shape.end.scale.setScalar(pathWidth(shape.end.position) / PATH_WIDTH);
  }
  for (const [robot, shape] of paths) {
    if (live.has(robot)) continue;
    scene3.remove(shape.line, shape.end);
    shape.line.geometry.dispose();
    paths.delete(robot);
  }
}

// The path's corners, with points in between that sit on the scenery under them (a hill, a ramp). The ray starts
// a little above the height the server gives, so a table the path runs under is not taken for the ground.
function groundPoints(path) {
  const out = [];
  for (let i = 0; i < path.length; i++) {
    const [x, y, z] = path[i];
    out.push(new THREE.Vector3(x, y, groundHeight(x, y, z + 0.3, z) + PATH_LIFT));
    if (i === path.length - 1) break;
    const [x2, y2, z2] = path[i + 1];
    const n = Math.floor(Math.hypot(x2 - x, y2 - y) / PATH_STEP);
    for (let k = 1; k < n; k++) {
      const f = k / n;
      const px = x + (x2 - x) * f;
      const py = y + (y2 - y) * f;
      const pz = z + (z2 - z) * f;
      out.push(new THREE.Vector3(px, py, groundHeight(px, py, pz + 0.3, pz) + PATH_LIFT));
    }
  }
  return out;
}

// Each frame: the further a stretch of path is from the camera, the wider it is drawn, at least PATH_MIN_PX across
// on screen. Rebuilt only once the camera has moved by a few percent of its distance to the path.
function updatePathWidths() {
  if (!paths.size || !hasSize()) return;
  for (const shape of paths.values()) {
    const middle = shape.line.geometry.boundingSphere?.center;
    if (!middle || !shape.points) continue;
    const away = camera.position.distanceTo(middle);
    if (shape.seenFrom && shape.seenFrom.distanceTo(camera.position) < 0.03 * away) continue;
    shape.seenFrom = camera.position.clone();
    ribbon(shape.line.geometry, shape.points);
    const end = shape.points[shape.points.length - 1];
    shape.end.scale.setScalar(pathWidth(end) / PATH_WIDTH);
  }
}

// Width (m) of the path at a point: PATH_WIDTH, or wider from far away (a flat line seen at a slant foreshortens).
function pathWidth(p) {
  if (!hasSize()) return PATH_WIDTH;
  const away = Math.max(camera.position.distanceTo(p), 1e-6);
  const perPx = (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / window.innerHeight; // m per px, 1 m off
  const across = Math.max((camera.position.z - p.z) / away, 0.3);
  return Math.max(PATH_WIDTH, (away * perPx * PATH_MIN_PX) / across);
}

// A flat strip along the points (pathWidth wide at each), mitred at the corners.
function ribbon(geometry, points) {
  const pos = [];
  const index = [];
  for (let i = 0; i < points.length; i++) {
    const before = points[Math.max(i - 1, 0)];
    const after = points[Math.min(i + 1, points.length - 1)];
    const dir = new THREE.Vector2(after.x - before.x, after.y - before.y).normalize();
    let side = new THREE.Vector2(-dir.y, dir.x);
    if (i > 0 && i < points.length - 1) {
      const into = new THREE.Vector2(points[i].x - before.x, points[i].y - before.y).normalize();
      const miter = Math.max(0.35, Math.abs(side.dot(new THREE.Vector2(-into.y, into.x))));
      side = side.multiplyScalar(1 / miter);
    }
    const w = pathWidth(points[i]) / 2;
    const p = points[i];
    pos.push(p.x + side.x * w, p.y + side.y * w, p.z, p.x - side.x * w, p.y - side.y * w, p.z);
    if (i > 0) {
      const a = 2 * (i - 1);
      index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
  }
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geometry.setIndex(index);
  geometry.computeBoundingSphere();
}

// ---------------------------------------------------------------- render loop

let frames = 0;
let fps = 0;
let fpsT0 = performance.now();

function updateClock() {
  const clock = $("clock");
  clock.textContent = paused ? t("paused") : `${latestTime.toFixed(1)} s`;
  clock.classList.toggle("paused", paused);
  clock.title = t("clock_tip", { rtf: rtf.toFixed(2), fps });
}

// Last line of defence: a camera that went NaN draws nothing, forever. Frame the scene again instead.
function healCamera() {
  const p = camera.position;
  const q = controls.target;
  if ([p.x, p.y, p.z, q.x, q.y, q.z].every(Number.isFinite) || !sceneDesc || !hasSize()) return;
  console.warn("robotforge: the camera position was not a number; framing the scene again");
  tween = null;
  follow = null;
  camera.position.set(1.6, -1.6, 1);
  controls.target.set(0, 0, 0);
  frameCamera();
}

function loop() {
  requestAnimationFrame(loop);
  healCamera();
  updateViewShift();
  refitSettledRig();
  updateTween();
  updateFollow();
  controls.update();
  updateRing();
  updatePathWidths();
  spinRotors();
  refreshDragVisual();
  lessonFrame();
  if (water) waterFrame(water, latestTime, camera, scene3);
  sensorsFrame();
  renderer.render(scene3, camera);
  frames++;
  const now = performance.now();
  if (now - fpsT0 > 500) {
    fps = Math.round((frames * 1000) / (now - fpsT0));
    updateClock();
    frames = 0;
    fpsT0 = now;
  }
}

initLessons({
  scene: scene3,
  request,
  send,
  toast,
  bodyNode: (id) => bodyNodes[id],
  robotTop: (name) => robotBox(name)?.max.z ?? null, // the highest point of a robot as it stands now (m)
  robotMeshes: (name) => robotMeshes(name), // the parts of a robot you can see, where it stands now
  freeRect: () => freeRect(), // the part of the window no panel covers, in CSS px
  color: (token) => new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue(token).trim() || "#888"),
  moveCamera: (eye, target, fit = false) => {
    refitRig = null; // the lesson's new camera replaces the one still to be fitted again
    // A rig is measured where its robots stand: a scene whose first poses have not come yet has every body at the
    // origin, and the fit framed that instead (on a phone the third PID twin and the Panda ended off the screen).
    if (fit && pendingFit) fitWhenPosed = { eye, target };
    else aimLessonCamera(eye, target, fit);
  },
  closePanels: () => {
    setHelp(false);
    setDrawer(false);
    setWorldPanel(false);
  },
  wantFrame: () => {
    wantFrame = true;
  },
  shift: () => shiftForDrawers(),
  retitle: () => renderTitle(),
  // Keep the camera with a lesson's robot as it walks (null stops); the lesson's own camera move finishes first.
  followBody: (body) => {
    if (body === null) follow = null;
    else {
      const from = follow && follow.body !== body ? bodyNodes[follow.body] : null;
      if (from && bodyNodes[body]) {
        // Switching robots: glide over by the distance between them, keeping the same view of the new one.
        const delta = bodyNodes[body].position.clone().sub(from.position).setZ(0);
        moveCamera(camera.position.clone().add(delta), controls.target.clone().add(delta), true);
        follow = null;
      } else if (!follow && bodyNodes[body]) {
        // Starting to follow a robot well away from where the camera looks (a page reloaded while it was
        // mid-walk): bring it to the middle first, or the camera would keep that offset for as long as it follows.
        const aimEye = tween ? tween.toEye : camera.position;
        const aimTarget = tween ? tween.toTarget : controls.target;
        const off = bodyNodes[body].position.clone().sub(aimTarget).setZ(0);
        if (off.length() > FOLLOW_RECENTER) moveCamera(aimEye.clone().add(off), aimTarget.clone().add(off), true);
      }
      startFollowBody(body, { keepTween: true });
    }
    return body === null || follow?.body === body; // false while the scene's bodies are still arriving
  },
  renderer,
  camera,
  canvas,
  rayAt: (clientX, clientY) => {
    setPointer({ clientX, clientY });
    return raycaster;
  },
});

initSensors({
  scene: scene3,
  renderer,
  camera,
  bodyNode: (id) => bodyNodes[id],
  world: () => worldRoot,
  desc: () => sceneDesc,
  selected: () => selected,
  select: (robot) => select(robot),
  inLesson: () => lessonActive(),
  color: (token) => new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue(token).trim() || "#888"),
  shift: () => shiftForDrawers(),
  storageGet,
  storageSet,
});

connect();
loop();

// Expose a tiny hook for automated checks (headless verification reads this).
window.__robotforge = {
  get scene() { return sceneDesc; },
  get time() { return latestTime; },
  bodyPosition: (id) => bodyNodes[id]?.position.toArray(),
  pick: (clientX, clientY) => pickBody({ clientX, clientY })?.body ?? null,
  cameraTarget: () => controls.target.toArray(),
  // Place the camera for scripted screenshots: eye and target in world meters.
  lookAt: (eye, target) => {
    moveCamera(new THREE.Vector3().fromArray(eye), new THREE.Vector3().fromArray(target), false);
  },
  // Render now and read the frame back (RGBA, bottom row first), for scripted image checks.
  pixels: () => {
    controls.update();
    renderer.render(scene3, camera);
    const gl = renderer.getContext();
    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const data = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
    return { w, h, data };
  },
  three: () => ({ scene: scene3, camera, renderer, sun }), // for experiments in scripted checks
  // Flicker check: slide the camera `steps` times by `step` m along `dir` and count pixels whose brightness
  // flips there and back (A, B, A) by more than `threshold`. Steady motion shifts edges one way; depth
  // fighting and texture shimmer flip back. Pause the simulation first so only the camera moves.
  flicker: (eye, target, dir, { step = 0.005, steps = 16, threshold = 5 } = {}) => {
    const shot = (k) => {
      window.__robotforge.lookAt(eye.map((v, i) => v + dir[i] * step * k), target.map((v, i) => v + dir[i] * step * k));
      const { w, h, data } = window.__robotforge.pixels();
      const lum = new Float32Array(w * h);
      for (let i = 0; i < w * h; i++) lum[i] = 0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2];
      return lum;
    };
    const frames = Array.from({ length: steps }, (_, k) => shot(k));
    let flips = 0;
    for (let k = 1; k + 1 < steps; k++) {
      const [a, b, c] = [frames[k - 1], frames[k], frames[k + 1]];
      for (let i = 0; i < b.length; i++) {
        const d1 = b[i] - a[i];
        const d2 = c[i] - b[i];
        if (Math.abs(d1) > threshold && Math.abs(d2) > threshold && Math.sign(d1) !== Math.sign(d2) && Math.abs(c[i] - a[i]) < threshold / 2) flips++;
      }
    }
    return { flips, perMegapixel: Math.round((flips / (frames[0].length * (steps - 2))) * 1e6) };
  },
  following: () => (follow ? follow.body : null),
  tweening: () => tween !== null,
  freeRect: () => freeRect(), // the part of the window no panel covers; the view centres there
  selected: () => selected,
  lesson: () => lessonState(),
  plotAxes: () => lessonPlotAxes(), // each live plot's y range, to check a still robot's wobble is not magnified
  busy: () => ({ busy, pending: pending.size }),
  ring: () => (ring.visible ? { center: ring.position.toArray(), radius: ringState?.radius ?? null } : null),
  marker: () => (markerDot.visible ? markerDot.position.toArray() : markerPad.visible ? markerPad.position.toArray() : null),
  select: (name) => select(name, { focus: true }),
  // CSS-pixel position of a world point, for scripted pointer tests.
  screenOfPoint: (xyz) => {
    camera.updateMatrixWorld();
    const v = new THREE.Vector3().fromArray(xyz).project(camera);
    const rect = canvas.getBoundingClientRect();
    return [rect.left + ((v.x + 1) / 2) * rect.width, rect.top + ((1 - v.y) / 2) * rect.height];
  },
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
  fps: () => fps, // frames drawn a second, counted over the last half second
  sensors: () => sensorsState(), // what the sensor drawings and views show (sensors.js)
  showSensor: (name, on = true) => sensorsShow(name, on), // as switching it in the robots drawer
};
