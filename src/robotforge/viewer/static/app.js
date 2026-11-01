// robotforge viewer: a thin three.js client. Physics runs in the Python server;
// this page draws the bodies it streams and sends user input back.
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { flowState, flowUpdate, initFlow } from "./flowview.js";
import { initGamepad, padAxes, padConnected, padFrame, padMode, padProfile, padState, release as padRelease } from "./gamepad.js";
import { initMap, mapFrame, mapState, mapUpdate } from "./mapview.js";
import { initTraining, trainingState, trainingUpdate } from "./training.js";
import { initSensors, sensorsFrame, sensorsScene, sensorsSelect, sensorsShow, sensorsState, sensorsUpdate } from "./sensors.js";
import { escapeHtml, t, tOr, translatePage } from "./i18n.js";
import { initLessons, lessonActive, lessonDriveSpec, lessonCamera, lessonFrame, lessonPlotAxes, lessonScene, lessonState, lessonTitle, lessonUpdate } from "./lesson.js";
import { orientWoodGrain, surfaceMaterial } from "./look.js";
import { createLighting } from "./lighting.js";
import { groundMaterial, groundPreview, groundStyle, prepareGroundTextures, prepareGrassTextures, prepareRockTextures } from "./ground-look.js";
import { SURFACE_GEOM, leaveWater, makeWater, stillLevel, waterFrame, waterLook } from "./water.js";
import { clothGeometry, shapeCloth } from "./cloth.js";
import { buildRotors, rotorCommands, spinRotors } from "./rotors.js";
import { canMove, moveAxes, moveSpeed } from "./padmap.js";
import { createPlacementGesture, initCreation, isObjectGroup } from "./creation.js";
import { placementCoordinates } from "./placement-coordinate.js";
import { createPlacementPreview } from "./placement-preview.js";
import { createArrangementShelf } from "./arrangements.js";
import { createScreenDrive } from "./screen-drive.js";
import { cameraView, cameraArc, arcEye } from "./camera-views.js";
import { createCameraBookmarks } from "./camera-bookmarks.js";
import { initRobotSurface } from "./robot-surface.js";

const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

translatePage();

// ---------------------------------------------------------------- renderer setup

const canvas = $("view");
let screenDrive = null;
let cameraBookmarks = null;
const creator = initCreation({ start: beginCreation, submit: placeEntity, cancel: cancelCreation,
  change: changePlacementOptions, coordinateChange: changePlacementOptions });
const creationGesture = createPlacementGesture();
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene3 = new THREE.Scene();
const placementPreview = createPlacementPreview(scene3);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 500);
camera.up.set(0, 0, 1); // MuJoCo is Z-up
camera.position.set(1.6, -1.6, 1.0);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

const lighting = createLighting(scene3, renderer);
const { sun } = lighting;
canvas.addEventListener("webglcontextrestored", () => lighting.restore());

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
const renderSize = new THREE.Vector2();
let ratioMedia = null, watchedRatio = null;

function deviceRatio() {
  const ratio = window.devicePixelRatio;
  return Number.isFinite(ratio) && ratio > 0 ? ratio : 1;
}

function resize() {
  const rawRatio = deviceRatio();
  if (rawRatio !== watchedRatio) {
    ratioMedia?.removeEventListener("change", resize);
    ratioMedia = window.matchMedia(`(resolution: ${rawRatio}dppx)`);
    ratioMedia.addEventListener("change", resize);
    watchedRatio = rawRatio;
  }
  if (!hasSize()) return;
  const w = window.innerWidth;
  const h = window.innerHeight;
  const ratio = Math.min(rawRatio, 2);
  renderer.getSize(renderSize);
  const sizeChanged = renderSize.x !== w || renderSize.y !== h;
  if (sizeChanged || renderer.getPixelRatio() !== ratio) renderer.setDrawingBufferSize(w, h, ratio);
  // Display density does not change CSS coordinates, camera framing or placement anchors.
  if (sizeChanged || camera.aspect !== w / h) {
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    applyViewShift();
  }
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
let posedVersion = 0;
let renderedVersion = 0; // submitted by the normal main-view render, not a presentation/GPU-completion signal
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
    mat = groundLook(geom, look);
    Object.assign(mat, FLOOR_OFFSET);
    groundMapping(mat, (2 * half) / look.tile, (2 * halfY) / look.tile);
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

let sceneWoodTile = null;
let sceneGrassTile = null;
let sceneRockTile = null;
function groundLook(geom, look) {
  return groundMaterial(groundStyle(geom.material, look), rgbaColor(geom.rgba),
    renderer.capabilities.getMaxAnisotropy(), sceneWoodTile, sceneGrassTile, sceneRockTile);
}

function groundMapping(mat, x, y, offset = [0, 0]) {
  for (const tex of new Set([mat.map, mat.bumpMap, mat.roughnessMap])) {
    tex.repeat.set(x, y);
    tex.offset.set(...offset);
  }
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
  const mat = groundLook(geom, look);
  mat.vertexColors = true;
  const tile = look ? look.tile : 0.6;
  let offset = [0, 0];
  const f = floorFrame;
  if (f && look && f.material === geom.material) {
    // Same surface as the floor: line the pattern up with the floor's, so the two meet without a seam.
    const shift = (a) => ((a % 1) + 1) % 1;
    offset = [shift((f.half - f.x + geom.pos[0] - rx) / tile), shift((f.halfY - f.y + geom.pos[1] - ry) / tile)];
  }
  groundMapping(mat, (2 * rx) / tile, (2 * ry) / tile, offset);
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
      // Retain approximate vertex colors for legacy or unsupported image bindings.
      g.setAttribute("color", new THREE.BufferAttribute(floats.subarray(entry.colors, entry.colors + 3 * n), 3));
      g.userData.vertexColors = true;
    }
    if (entry.uv !== undefined) {
      g.setAttribute("uv", new THREE.BufferAttribute(floats.subarray(entry.uv, entry.uv + 2 * n), 2));
    }
    out.set(entry.id, g);
  }
  return out;
}

function disposeWorld() {
  releaseCreationPreview();
  sceneRockTile = null;
  if (!worldRoot) return;
  const textures = new Set();
  worldRoot.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      for (const tex of [o.material.map, o.material.bumpMap, o.material.roughnessMap]) if (tex) textures.add(tex);
      o.material.dispose();
    }
  });
  for (const tex of textures) tex.dispose();
  scene3.remove(worldRoot);
  lighting.bind(null);
  worldRoot = null;
}

let sceneRequest = 0;
let loadingScene = false;
let pendingRobotInfo = null;
let layoutCatalog = null, layoutConfirmation = null, layoutMutating = false, layoutLoading = false;
let layoutRequest = 0, layoutError = "";
let arrangementShelf = null;

async function applyScene(desc) {
  cameraBookmarks?.invalidate();
  arrangementShelf?.invalidate();
  cancelInspection();
  screenDrive?.clear(false); // the server already replaced the old scene; never stop its new controllers
  const request = ++sceneRequest;
  loadingScene = true;
  releaseCreationPreview();
  follow = null;
  if (driveCameraIntent === "follow") driveCameraIntent = "auto";
  renderScreenDrive();
  posedVersion = 0;
  renderedVersion = 0;
  renderLayouts();
  renderRobotStart();
  $("loading").hidden = false;
  $("loading").querySelector(".spinner").hidden = false;
  $("loading").querySelector("span[data-i18n]").textContent = t("loading_scene");
  $("scene-retry").hidden = true;
  $("scene-retry").disabled = false;
  updateEmpty();
  if (pendingRobotInfo?.version <= desc.version) pendingRobotInfo = null;
  try {
    await installScene(desc, request);
  } catch (err) {
    if (request !== sceneRequest) return;
    loadingScene = true;
    console.warn("robotforge: scene load failed", err);
    $("loading").querySelector(".spinner").hidden = true;
    $("loading").querySelector("span[data-i18n]").textContent = t("scene_load_failed");
    $("scene-retry").hidden = false;
  }
}

async function installScene(desc, request) {
  const textureLoad = desc.textures?.length ? Promise.all([
    import("./model-textures.js"),
    fetch(`texture/${desc.version}.bin`).then(async (res) => {
      if (!res.ok) throw new Error(`texture request returned ${res.status}`);
      return res.arrayBuffer();
    }),
  ]) : Promise.resolve(null);
  let buffer = new ArrayBuffer(0);
  const meshLoad = desc.meshes.length > 0 || (desc.hfields || []).length > 0
    ? fetch(`mesh/${desc.version}.bin`).then(async (res) => {
      if (!res.ok) throw new Error(`mesh request returned ${res.status}`);
      return res.arrayBuffer();
    }) : Promise.resolve(buffer);
  const [meshBuffer, textureData, preparedWood, preparedGrass, preparedRock] = await Promise.all([
    meshLoad, textureLoad, prepareGroundTextures(desc), prepareGrassTextures(desc), prepareRockTextures(desc),
  ]);
  buffer = meshBuffer;
  if (request !== sceneRequest) return; // another scene or connection superseded this mesh fetch
  const textureFor = textureData ? textureData[0].modelTextures(desc.textures, textureData[1], renderer.capabilities.getMaxAnisotropy()) : () => null;
  // Validate all bindings before releasing the current world, so a failed load remains retryable.
  for (const geom of desc.geoms) if (geom.texture) textureFor(geom.texture);
  const meshGeometries = buildMeshGeometries(desc, buffer);
  const floats = new Float32Array(buffer);
  const grounds = desc.grounds || {};
  leaveWater(water, scene3);
  water = null;
  disposeWorld();
  sceneWoodTile = preparedWood;
  sceneGrassTile = preparedGrass;
  sceneRockTile = preparedRock;
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
        ? groundLook(geom, look)
        : surfaceMaterial(geom.material, rgbaColor(geom.rgba), geom.rgba[3], desc.surfaces,
          desc.native_materials?.[String(geom.material_id)]);
      orientWoodGrain(mat, g);
      const texture = textureFor(geom.texture);
      if (texture) {
        mat.map = texture;
        if (texture.userData.modelAlpha) { mat.transparent = true; mat.depthWrite = false; }
      }
      mat.vertexColors = !texture && (geom.vertex_colors ?? g.userData.vertexColors ?? false);
      if (look && geom.type === "box") {
        // Terrain blocks (stairs, ramps, stones) share the ground's surface, tiled in meters on every face:
        // one repeat for the whole box squeezed the pattern on thin faces (risers, sides) into shimmer.
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
  lighting.bind(worldRoot, bodyNodes, desc);

  // Lights and (first time) camera are fitted once real poses arrive, see applyPoses.
  pendingFit = firstScene || wantFrame ? "camera" : "lights";
  firstScene = false;
  wantFrame = false;
  follow = null;
  if (driveCameraIntent === "follow") driveCameraIntent = "auto";
  tween = null;
  fitWhenPosed = refitRig = null; // asked for the scene this one replaces
  onScene(desc);
  loadingScene = false;
  applyRobotInfo();
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

let aboveFog = null;
let fogRadius = 0.2;

function updateFogRange() {
  if (!aboveFog) return;
  // Editing can shrink the scene without moving the user's camera. Keep its target clear.
  // Water temporarily replaces scene3.fog; only adjust the stored above-water fog.
  const near = Math.max(fogRadius * 7.5, camera.position.distanceTo(controls.target) + fogRadius * 2);
  aboveFog.near = near;
  aboveFog.far = Math.max(fogRadius * 27, near + fogRadius * 19.5);
}

function fitFog(sphere) {
  fogRadius = Math.max(sphere.radius, 0.2);
  aboveFog = new THREE.Fog(theme.fog);
  scene3.fog = aboveFog;
  updateFogRange();
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
  if (!sceneDesc || loadingScene) return; // still acknowledge frames while a newer mesh is being fetched
  const f = new Float32Array(buffer);
  const nb = sceneDesc.nbody;
  if (f.length !== 1 + 7 * nb) return; // frame from a different model version
  const firstPoses = posedVersion !== sceneDesc.version;
  const resetLights = f[0] < latestTime;
  // A new model starts at zero too; its late first pose must keep the matching committed Undo.
  if (resetLights && !firstPoses) { lastAddition = null; updateUndo(); }
  latestTime = f[0];
  const qOff = 1 + 3 * nb;
  for (let b = 0; b < nb; b++) {
    const node = bodyNodes[b];
    node.position.set(f[1 + 3 * b], f[2 + 3 * b], f[3 + 3 * b]);
    node.quaternion.set(f[qOff + 4 * b + 1], f[qOff + 4 * b + 2], f[qOff + 4 * b + 3], f[qOff + 4 * b]);
  }
  if (flexMeshes.length) updateFlexes();
  lighting.update(resetLights);
  if (pendingFit) {
    const sphere = contentSphere();
    fitFog(sphere);
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
  }
  posedVersion = sceneDesc.version;
  if (firstPoses) renderLayouts();
  resolveCreatedRobot();
}

function sceneReadyToDraw() {
  return sceneDesc && !loadingScene && posedVersion === sceneDesc.version;
}

// ---------------------------------------------------------------- camera moves

let tween = null; // { fromEye, fromTarget, toEye, toTarget, t0 }
let originalCameraView = null, currentCameraView = null;
let inspection = null;
const viewButtons = [...$("view-panel").querySelectorAll("[data-camera-view]")];
const TWEEN_MS = 450;

function moveCamera(eye, target, animate, orbit = false) {
  follow = null;
  renderCameraControls();
  if (!animate || reduceMotion) {
    tween = null;
    camera.position.copy(eye);
    controls.target.copy(target);
    controls.update();
    return;
  }
  tween = { fromEye: camera.position.clone(), fromTarget: controls.target.clone(), toEye: eye, toTarget: target, t0: performance.now() };
  if (orbit) tween.arc = cameraArc(tween.fromEye.toArray(), tween.fromTarget.toArray(), eye.toArray(), target.toArray());
}

function updateTween() {
  if (!tween) return;
  const k = Math.min(1, (performance.now() - tween.t0) / TWEEN_MS);
  const e = 1 - (1 - k) ** 3; // ease out
  controls.target.lerpVectors(tween.fromTarget, tween.toTarget, e);
  if (tween.arc) camera.position.fromArray(arcEye(tween.arc, controls.target.toArray(), e));
  else camera.position.lerpVectors(tween.fromEye, tween.toEye, e);
  if (k >= 1) tween = null;
}

function setViewPanel(open) {
  if (open) cancelInspection();
  else cameraBookmarks?.cancel();
  setOpen("view-toggle", "view-panel", open);
  if (open) {
    cameraBookmarks?.open();
    stopDriving(); endDrag();
    creationGesture.invalidate();
    if (creator.active()) clearCreationPoint();
    (viewButtons.find((button) => button.getAttribute("aria-pressed") === "true") || viewButtons[0]).focus();
  }
  shiftForDrawers();
}

function resetCameraViews() {
  cameraBookmarks?.invalidate();
  cancelInspection();
  originalCameraView = currentCameraView = null;
  setViewPanel(false);
}

function chooseCameraView(name) {
  cancelInspection();
  if ($("view-toggle").disabled || !hasSize()) return;
  if (name === "back" && !originalCameraView) return;
  // Drain residual orbit damping before saving a view or starting a deliberate arc.
  const damping = controls.enableDamping;
  controls.enableDamping = false; controls.update(); controls.enableDamping = damping;
  const target = name === "back" ? originalCameraView.target.clone() : controls.target.clone();
  const eye = name === "back" ? originalCameraView.eye.toArray() : cameraView(name, camera.position.toArray(), target.toArray());
  if (!eye) return;
  if (name !== "back" && !originalCameraView) originalCameraView = { eye: camera.position.clone(), target: controls.target.clone() };
  freeDriveCamera(); endDrag();
  creationGesture.invalidate();
  if (creator.active()) clearCreationPoint();
  creationClickUntil = performance.now() + (reduceMotion ? 60 : TWEEN_MS + 60);
  currentCameraView = name === "back" ? null : name;
  if (name === "back") originalCameraView = null;
  setViewPanel(false);
  moveCamera(new THREE.Vector3().fromArray(eye), target, true, true);
  renderCameraControls();
  canvas.focus();
}

$("view-toggle").addEventListener("click", () => setViewPanel($("view-panel").hidden));
for (const button of viewButtons) {
  button.addEventListener("click", () => chooseCameraView(button.dataset.cameraView));
}
$("view-back").addEventListener("click", () => chooseCameraView("back"));

function bookmarkContext() {
  const ready = !!(sceneReadyToDraw() && hasSize() && renderedVersion === sceneDesc.version
    && stateVersion === sceneDesc.version && Number.isInteger(stateGeneration)
    && ws?.readyState === WebSocket.OPEN && !busy && !lessonActive() && !document.hidden
    && !creator.active() && !selectingObjects && !inspection);
  return { scope: sceneDesc?.view_scope ?? null, version: sceneDesc?.version ?? null, generation: stateGeneration,
    ready, canSave: ready && !tween && !follow };
}

function captureCameraBookmark() {
  return { eye: camera.position.toArray(), target: controls.target.toArray() };
}

function restoreCameraBookmark(view) {
  const damping = controls.enableDamping;
  controls.enableDamping = false; controls.update(); controls.enableDamping = damping;
  if (!originalCameraView) originalCameraView = { eye: camera.position.clone(), target: controls.target.clone() };
  cancelInspection(); freeDriveCamera(); stopDriving(); endDrag();
  creationGesture.invalidate();
  creationClickUntil = performance.now() + (reduceMotion ? 60 : TWEEN_MS + 60);
  currentCameraView = null;
  setViewPanel(false);
  moveCamera(new THREE.Vector3().fromArray(view.eye), new THREE.Vector3().fromArray(view.target), true, true);
  renderCameraControls();
  canvas.focus();
}

function cancelInspection() {
  if (!inspection) return;
  const previous = inspection;
  inspection = null;
  creationGesture.invalidate();
  $("inspection-bar").hidden = true;
  canvas.classList.remove("inspecting");
  if (creator.session() === previous.session) {
    $("placement-bar").hidden = previous.placementHidden;
    $("object-edit-bar").hidden = previous.objectHidden;
  }
  shiftForDrawers(); renderDrive();
}

$("view-inspect").addEventListener("click", () => {
  cancelInspection();
  if ($("view-inspect").disabled || !hasSize()) return;
  stopDriving(); endDrag(); freeDriveCamera();
  const damping = controls.enableDamping;
  controls.enableDamping = false; controls.update(); controls.enableDamping = damping;
  creationGesture.invalidate();
  if (creator.active()) clearCreationPoint();
  inspection = { version: sceneDesc.version, generation: stateGeneration, session: creator.session(),
    eye: camera.position.clone(), target: controls.target.clone(),
    placementHidden: $("placement-bar").hidden, objectHidden: $("object-edit-bar").hidden };
  $("placement-bar").hidden = $("object-edit-bar").hidden = true;
  setViewPanel(false); setDrawer(false); setWorldPanel(false); setHelp(false);
  $("inspection-bar").hidden = false;
  canvas.classList.remove("hover-body"); canvas.classList.add("inspecting");
  shiftForDrawers(); renderDrive(); canvas.focus();
});
$("inspection-cancel").addEventListener("click", () => { cancelInspection(); canvas.focus(); });

function inspectSurface(point) {
  if (!inspection || !worldRoot) return;
  setPointer(point);
  const inDepth = (point) => {
    const depth = -point.clone().applyMatrix4(camera.matrixWorldInverse).z;
    return depth >= camera.near && depth <= camera.far;
  };
  let hit = raycaster.intersectObjects(worldRoot.children, true).find(({ object, point, face }) => {
    if (!inDepth(point)) return false;
    if (!object.isMesh) return false;
    for (let node = object; node; node = node.parent) if (!node.visible) return false;
    const material = Array.isArray(object.material) ? object.material[face?.materialIndex ?? 0] : object.material;
    return material?.visible && material.opacity > 0;
  });
  // The animated water deliberately has no triangle raycast. Use its bounded still
  // level, as paths/rings do, rather than silently selecting the bottom through it.
  if (water) {
    const point = raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -water.water.level), new THREE.Vector3());
    if (point && inDepth(point) && stillLevel(water, point.x, point.y) !== null
        && (!hit || camera.position.distanceTo(point) < hit.distance)) hit = { point };
  }
  if (!hit) { toast(t("view_inspect_miss")); return; }
  const direction = camera.position.clone().sub(controls.target);
  const distance = camera.position.distanceTo(hit.point);
  if (!Number.isFinite(distance) || distance <= 0 || direction.lengthSq() < 1e-10) return;
  const previous = inspection;
  if (!originalCameraView) originalCameraView = { eye: previous.eye, target: previous.target };
  const eye = hit.point.clone().addScaledVector(direction.normalize(), distance);
  cancelInspection();
  refitRig = null; currentCameraView = null;
  creationClickUntil = performance.now() + (reduceMotion ? 350 : TWEEN_MS + 350);
  moveCamera(eye, hit.point.clone(), true, true);
  renderCameraControls(); canvas.focus();
}

document.addEventListener("pointerdown", (event) => {
  if (inspection && event.target !== canvas && !$("inspection-bar").contains(event.target)) cancelInspection();
  if ($("view-panel").hidden || $("view-panel").contains(event.target) || $("view-toggle").contains(event.target)) return;
  setViewPanel(false);
  // A tap that dismisses the view picker must not also place a prop or push a robot.
  if (event.target === canvas) {
    creationGesture.invalidate(); creationClickUntil = performance.now() + 350;
    event.preventDefault(); event.stopPropagation();
  }
}, { capture: true });

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
  const box = selected ? robotBox(selected, { baseOnly: !canMove(robot) }) : null;
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
    const pad = canMove(robot) ? 1.05 : 1.6;
    ringState = { radius: Math.max(0.12, Math.hypot(size.x, size.y) * 0.5 * pad), center: center.clone(), t: now };
  }
  // Smooth out the swing of legs and arms, by time so a throttled tab does not leave the ring behind.
  ringState.center.lerp(center, 1 - Math.exp(-(now - ringState.t) / 80));
  ringState.t = now;
  const whole = canMove(robot) ? box : robotBox(selected);
  const z = groundHeight(ringState.center.x, ringState.center.y, whole.max.z + 0.05, whole.min.z);
  ring.position.set(ringState.center.x, ringState.center.y, z + 0.006);
  ring.scale.setScalar(ringState.radius);
  ringNotch.visible = !!canMove(robot);
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
  if (!worldRoot || !sceneReadyToDraw() || renderedVersion !== sceneDesc.version) return null;
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
    creationGesture.down(e, (inspection || creator.active() || selectingObjects) && !busy && !tween?.arc && performance.now() >= creationClickUntil
      ? { session: creator.session(), version: sceneDesc?.version, generation: stateGeneration, inspection } : null);
    if (inspection || creator.active() || selectingObjects) {
      if (creator.active()) clearCreationPoint();
      return; // OrbitControls owns camera motion; only a qualified release will place.
    }
    if (e.button !== 0) return;
    if (tween?.arc) return;
    if (performance.now() < creationClickUntil) return;
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
  creationGesture.move(e);
  if (inspection || creator.active() || selectingObjects) {
    if (inspection) return;
    if (creator.active() && !creationGesture.active() && e.pointerType !== "touch") creationPoint(e);
    return;
  }
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

function finishCreationGesture(e) {
  const context = creationGesture.up(e, e.type === "pointercancel");
  if (!context) return;
  const epoch = creationGesture.epoch();
  const rect = canvas.getBoundingClientRect();
  if (e.clientX < rect.left || e.clientX >= rect.right || e.clientY < rect.top || e.clientY >= rect.bottom) return;
  const point = { clientX: e.clientX, clientY: e.clientY };
  // OrbitControls installs its release listener dynamically. Let it drain before a scene edit,
  // and recheck ownership in case the session changed during this pointer event.
  setTimeout(() => {
    if (context.inspection) {
      if (inspection === context.inspection && !creationGesture.active() && creationGesture.epoch() === epoch
          && !busy && sceneReadyToDraw() && renderedVersion === context.version && sceneDesc?.version === context.version
          && stateGeneration === context.generation && creator.session() === context.session
          && ws?.readyState === WebSocket.OPEN && performance.now() >= creationClickUntil) inspectSurface(point);
      return;
    }
    if (!$("view-panel").hidden || creationGesture.active() || creationGesture.epoch() !== epoch || !(creator.active() || selectingObjects) || busy || creator.session() !== context.session
        || sceneDesc?.version !== context.version || stateGeneration !== context.generation
        || performance.now() < creationClickUntil) return;
    if (selectingObjects) pickObjectForEdit(point);
    else { creationPoint(point); creator.place(); }
  }, 0);
}
window.addEventListener("pointerup", finishCreationGesture, { capture: true });
window.addEventListener("pointercancel", finishCreationGesture, { capture: true });
canvas.addEventListener("lostpointercapture", () => {
  if (creationGesture.active()) creationGesture.invalidate();
});
window.addEventListener("blur", () => {
  cancelInspection();
  creationGesture.reset();
  if (creator.active()) clearCreationPoint();
});

// Double-click the ground: the selected robot finds its way there round whatever is in the way (move with a
// destination) and shows its path. A robot or a loose object under the pointer is for dragging, not a place;
// so is a wall's side: only a face that looks up (the floor, a ramp, a table top) is.
canvas.addEventListener("dblclick", (e) => {
  if (inspection || creator.active() || selectingObjects || performance.now() < creationClickUntil) return;
  if (!sceneReadyToDraw() || renderedVersion !== sceneDesc.version) return;
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
    pendingRobotInfo = null;
    setConnected(true);
  };
  ws.onclose = () => {
    cameraBookmarks?.invalidate();
    screenDrive?.clear(false); // the server's bounded hold expires even when no stop can cross this socket
    follow = null;
    if (driveCameraIntent === "follow") driveCameraIntent = "auto";
    sceneRequest++;
    loadingScene = true;
    renderedVersion = 0;
    renderScreenDrive();
    renderRobotStart();
    setConnected(false);
    arrangementShelf?.invalidate({ clear: true });
    for (const p of pending.values()) p.reject(new Error(t("disconnected")));
    pending.clear();
    layoutCatalog = layoutConfirmation = null;
    renderLayouts();
    creationSelection?.reject(new Error(t("disconnected")));
    lastAddition = lastObjectEdit = lastRobotEdit = lastTerrainEdit = null; updateUndo();
    cancelCreation(true);
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
    else if (msg.type === "robot_info") receiveRobotInfo(msg);
    else if (msg.type === "state") applyState(msg);
    else if (msg.type === "reply") onReply(msg);
    else if (msg.type === "progress") pending.get(msg.id)?.progress?.(msg);
    else if (msg.type === "lesson") lessonUpdate(msg);
    else if (msg.type === "sensors") sensorsUpdate(msg);
    else if (msg.type === "map") mapUpdate(msg);
    else if (msg.type === "training") trainingUpdate(msg);
    else if (msg.type === "flow") flowUpdate(msg);
    else if (msg.type === "error") toast(msg.message);
  };
}

// Reconnect on explicit retry so the server sends its latest scene, not a possibly expired mesh version.
$("scene-retry").addEventListener("click", () => {
  $("scene-retry").disabled = true;
  if (ws?.readyState === WebSocket.OPEN) ws.close();
});

// ---------------------------------------------------------------- robots and selection

let robots = []; // Robot descriptions with shared capabilities and legacy fields from the server
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
  // Camera history belongs to this rendered model, never another map, layout or lesson.
  resetCameraViews();
  if (!layoutMutating && layoutCatalog?.version !== desc.version) {
    layoutCatalog = null;
    layoutConfirmation = null;
    if (!$("world-panel").hidden) refreshLayouts();
  }
  const pendingSurface = surfaceReselect && busy && desc.version === surfaceReselect.sourceVersion + 1;
  clearObjectSelection(pendingSurface);
  clearTerrainHighlight();
  if (lastTerrainEdit?.version !== desc.version) lastTerrainEdit = null;
  if (lastObjectEdit?.version !== desc.version) lastObjectEdit = null;
  if (lastRobotEdit?.version !== desc.version) lastRobotEdit = null;
  if (lastAddition?.version !== desc.version) lastAddition = null;
  if (creator.active() && !busy) cancelCreation(true);
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
  if (!robotByName(selected)) {
    selected = (robots.find(canMove) ?? robots[0])?.name ?? null;
    driveCameraIntent = "auto";
  }
  ringState = null;
  worldName = desc.world || worldName;
  renderTitle();
  renderRobotList();
  renderSelection();
  renderWorldPanel();
  lessonScene(desc);
  sensorsScene(desc);
  updateEmpty();
  updateUndo();
}

function renderTitle() {
  const name = lessonTitle() ?? worldName;
  $("world").textContent = name;
  document.title = name ? `${name} · robotforge` : "robotforge";
}

function renderRobotList() {
  renderEditableObjects();
  creator.render(worldInfo.catalog?.robot_templates || [], worldInfo.isWorld && !sceneDesc?.lesson, busy, worldInfo.catalog?.object_presets || [], worldInfo.catalog?.robot_models || [], worldInfo.catalog?.terrain_presets || []);
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
  $("drawer-label").textContent = worldInfo.isWorld || robots.length ? t("robots") : t("joints");
  $("drawer-count").textContent = n ? String(n) : "";
  renderRobotStart();
}

function select(name, { focus = false } = {}) {
  if (name !== selected) {
    stopDriving();
    selected = name;
    ringState = null;
    follow = null;
    driveCameraIntent = "auto";
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
  renderRobotStart();
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

function applyRobotInfo() {
  if (!pendingRobotInfo || loadingScene || pendingRobotInfo.version !== sceneDesc?.version) return;
  robots = pendingRobotInfo.robots;
  learnedWalkers = pendingRobotInfo.learned;
  robotModes = pendingRobotInfo.modes;
  Object.assign(sceneDesc, { robots, learned: learnedWalkers, modes: robotModes, task_contracts: pendingRobotInfo.task_contracts });
  pendingRobotInfo = null;
  renderRobotList();
  renderModes();
  renderDrive();
}

function receiveRobotInfo(info) {
  if (!pendingRobotInfo || info.version >= pendingRobotInfo.version) {
    pendingRobotInfo = info;
    applyRobotInfo();
  }
}

let stateGeneration = null;
let stateVersion = null;

function applyState(s) {
  if (Number.isInteger(s.version)) stateVersion = s.version;
  if (Number.isInteger(s.generation)) {
    const changed = stateGeneration !== s.generation;
    if (changed) { screenDrive?.clear(false); arrangementShelf?.invalidate(); }
    if (stateGeneration !== null && stateGeneration !== s.generation && !busy) {
      lastAddition = lastObjectEdit = lastRobotEdit = lastTerrainEdit = null; cancelCreation(true);
    }
    stateGeneration = s.generation; // retain changes during our own edit, then reconcile against its reply
    if (changed) cameraBookmarks?.invalidate();
    if (changed && !layoutMutating) {
      layoutConfirmation = null;
      if (layoutCatalog?.generation !== stateGeneration) {
        ++layoutRequest; layoutLoading = false;
        layoutCatalog = null;
        if (!$("world-panel").hidden) refreshLayouts();
      }
      renderLayouts();
    }
    if (creator.active() && !busy && placementGeneration === null) placementGeneration = s.generation;
    resolveCreatedRobot();
  }
  if (paused !== s.paused) arrangementShelf?.invalidate();
  paused = s.paused;
  if (selectingObjectGroup) renderObjectSelectionControls();
  arrangementShelf?.render();
  renderScreenDrive();
  renderRobotStart();
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
  renderDemo(s.recording, s.replaying);
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
const isAircraft = (robot) => moveAxes(robot).includes("climb");
const canRise = (robot) => moveAxes(robot).some((axis) => ["heave", "climb"].includes(axis));
// rad/s at full key; walkers turn at 1.0
const TURN_RATE = { rover: 2.0, balancer: 1.5, boat: 0.8, rov: 0.8, aircraft: 1.2 };
const HEAVE = 0.3; // m/s up or down for an underwater robot at full key
const FLY_CLIMB = 1.0; // m/s up or down for an aircraft at full key
const pressed = new Set();
let lastDrive = "";

function driveTarget() {
  if (inspection || creator.active() || selectingObjects) return null;
  const robot = robotByName(selected);
  return canMove(robot) ? robot : null;
}

function renderDrive() {
  const robot = driveTarget();
  const jog = padConnected() && (padProfile() === "arm" || (padProfile() === "both" && padMode() === "arm"));
  const shown = robotByName(selected);
  $("drive").hidden = !robot && !jog;
  $("drive-wasd").hidden = jog;
  $("drive-heave").hidden = !moveAxes(robot).includes("heave");
  $("fly-keys").hidden = !isAircraft(robot) || jog;
  $("drive-pad").hidden = !(padConnected() && padProfile());
  const say = jog ? "drive_arm" : isAircraft(robot) ? "fly" : moveAxes(robot).includes("heave") ? "drive_rov" : "drive";
  const named = robot || shown;
  if (named) $("drive-text").innerHTML = t(say, { name: escapeHtml(named.name) });
  lastDrive = "";
  renderScreenDrive();
}

const coarsePointer = window.matchMedia("(any-pointer: coarse)");
let screenDrivePreference = storageGet("robotforge.screen-drive");
function renderScreenDrive() {
  renderCameraControls();
  if (!screenDrive) return;
  const robot = driveTarget();
  const wanted = screenDrivePreference === null ? coarsePointer.matches : screenDrivePreference === "1";
  $("show-screen-drive").checked = wanted;
  const editingPanel = narrow() && (!$("world-panel").hidden || !$("drawer").hidden);
  const show = Boolean(wanted && robot && !editingPanel && !lessonActive() && !loadingScene);
  const ready = Boolean(ws?.readyState === WebSocket.OPEN && !paused && !busy && !document.hidden
    && renderedVersion === sceneDesc?.version && (stateVersion === null || stateVersion === sceneDesc.version));
  $("screen-drive-target").textContent = robot?.name ?? "";
  $("screen-drive-hint").textContent = t(paused ? "screen_drive_play" : driveCameraIntent === "free" ? "screen_drive_free" : "screen_drive_hold");
  screenDrive.update({ target: `${selected}:${sceneDesc?.version}:${stateGeneration}`, show, ready,
    rise: canRise(robot), slide: moveAxes(robot).includes("slide") });
}

function setScreenDriving(on) {
  screenDrivePreference = on ? "1" : "0";
  storageSet("robotforge.screen-drive", screenDrivePreference);
  renderScreenDrive();
}
$("show-screen-drive").addEventListener("change", (event) => setScreenDriving(event.target.checked));
$("screen-drive-close").addEventListener("click", () => setScreenDriving(false));
coarsePointer.addEventListener("change", renderScreenDrive);

// What the keys ask for, -1..1 on each axis; the gamepad's stick (gamepad.js) asks for the same things in between.
function keyAxes() {
  const n = (plus, minus) => (pressed.has(plus) ? 1 : 0) - (pressed.has(minus) ? 1 : 0);
  return { fwd: n("fwd", "back"), left: n("left", "right"), slide: n("slide_left", "slide_right"), up: n("up", "down") };
}

// The pad where its stick asks for something on an axis, else the keys.
function driveAxes() {
  const keys = keyAxes();
  const screen = screenDrive?.axes();
  if (screen) for (const key of Object.keys(keys)) keys[key] = Math.max(-1, Math.min(1, keys[key] + screen[key]));
  const pad = padAxes();
  if (!pad) return keys;
  return { fwd: pad.fwd || keys.fwd, left: pad.left || keys.left, slide: pad.slide || keys.slide, up: pad.up || keys.up };
}

function driveCommand(robot) {
  const { fwd, left, slide, up } = driveAxes();
  const learned = learnedWalkers[robot.name];
  const top = (currentModes[robot.name] === "learned" ? learned?.max_speed : moveSpeed(robot)) ?? 0.3;
  const cmd = { type: "teleop", robot: robot.name, speed: fwd * top * 0.6, turn: left * (TURN_RATE[robot.kind] ?? 1.0),
    version: sceneDesc.version, generation: stateGeneration };
  if (moveAxes(robot).includes("heave")) cmd.heave = up * HEAVE;
  if (isAircraft(robot)) {
    cmd.speed = fwd * top * 0.5;
    cmd.side = slide * top * 0.5;
    cmd.climb = up * FLY_CLIMB;
  }
  return cmd;
}

function sendDrive(refresh = false) {
  const robot = driveTarget();
  for (const k of document.querySelectorAll("#drive kbd, #lesson-drive kbd")) k.classList.toggle("on", pressed.has((isAircraft(robot) && k.dataset.fly) || k.dataset.key));
  const lessonDrive = lessonDriveSpec();
  if (lessonDrive) {
    // A lesson with a robot to drive takes the keys; the lesson's own controller turns them into wheel speeds.
    let { fwd, left } = driveAxes();
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
  if (screenDrive?.held()) cmd.hold_ms = 750;
  const key = `${cmd.robot}:${cmd.speed}:${cmd.turn}:${cmd.heave ?? ""}:${cmd.side ?? 0}:${cmd.climb ?? 0}:${cmd.hold_ms ?? ""}`;
  if (key === lastDrive && !refresh) return;
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
  updateFogRange();
  if (pressed.size && lessonDriveSpec()?.robot === false) sendDrive();
});

function stopDriving() {
  // Switching robots mid-drive must not leave the old one walking off on its own.
  const robot = robotByName(selected);
  if (canMove(robot) && (pressed.size || padAxes() || screenDrive?.held())) {
    const command = driveCommand(robot);
    for (const key of ["speed", "turn", "side", "climb", "heave"]) if (key in command) command[key] = 0;
    send(command);
  }
  screenDrive?.clear(false);
  pressed.clear();
  padRelease(false); // the pad's arm, if it was jogging one, is told to hold
  lastDrive = "";
}

// Follow the driven robot with the camera until the user orbits by hand.
let follow = null; // { body, last: Vector3, z: smoothed height, t: last update ms }
let driveCameraIntent = "auto"; // auto on first drive; explicit free view survives input renewals
const FOLLOW_Z_SECONDS = 0.6; // height follows this slowly: up stairs and hills, not every stride
const FOLLOW_RECENTER = 1.0; // m: lessons aim a little beside the robot they follow; further off is an accident

function startFollow() {
  if (!lessonActive() && driveCameraIntent === "free") return;
  startFollowBody(sceneDesc?.bodies.find((b) => b.name.startsWith(`${selected}/`) && b.dynamic)?.id ?? null);
  if (!lessonActive() && follow) driveCameraIntent = "follow";
  renderCameraControls();
}

function freeDriveCamera() {
  follow = null;
  tween = null;
  refitRig = null;
  currentCameraView = null;
  if (!lessonActive()) driveCameraIntent = "free";
  renderScreenDrive();
}

function renderCameraControls() {
  cameraBookmarks?.update();
  const disabled = !sceneReadyToDraw() || renderedVersion !== sceneDesc?.version || busy || ws?.readyState !== WebSocket.OPEN;
  if (inspection && (disabled || inspection.version !== sceneDesc?.version || inspection.generation !== stateGeneration)) cancelInspection();
  $("view-toggle").disabled = disabled;
  $("view-inspect").disabled = disabled;
  $("view-back").disabled = disabled || !originalCameraView;
  for (const entry of viewButtons) {
    entry.disabled = disabled;
    entry.setAttribute("aria-pressed", String(currentCameraView === entry.dataset.cameraView));
  }
  const button = $("camera-view"), robot = robotByName(selected);
  button.hidden = !robot || lessonActive() || creator.active() || selectingObjects;
  button.disabled = disabled;
  const movable = canMove(robot), active = !!follow && movable;
  if (movable) button.setAttribute("aria-pressed", String(active));
  else button.removeAttribute("aria-pressed");
  button.setAttribute("aria-label", t(movable ? "camera_follow_robot" : "camera_focus_robot", { name: robot?.name ?? "" }));
  button.title = button.getAttribute("aria-label");
  $("camera-view-label").textContent = t(movable ? active ? "camera_following" : "camera_follow" : "camera_focus");
}

$("camera-view").addEventListener("click", () => {
  cancelInspection();
  if ($("camera-view").hidden || $("camera-view").disabled) return;
  if (follow && canMove(robotByName(selected))) { freeDriveCamera(); return; }
  focusRobot(selected);
  if (canMove(robotByName(selected))) {
    driveCameraIntent = "follow";
    startFollowBody(sceneDesc?.bodies.find((b) => b.name.startsWith(`${selected}/`) && b.dynamic)?.id ?? null, { keepTween: true });
  }
  renderScreenDrive();
});

function startFollowBody(body, { keepTween = false } = {}) {
  if (body === null || !bodyNodes[body]) return;
  if (follow && follow.body === body) return;
  currentCameraView = null;
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
  freeDriveCamera();
  tween = null;
  refitRig = null; // the learner's own view stays
});

function driveKey(e, down) {
  const lessonDrive = lessonDriveSpec();
  if (!driveTarget() && !lessonDrive) return false;
  const dir = (isAircraft(driveTarget()) && !lessonDrive ? FLY_KEYS : DRIVE_KEYS)[e.key.toLowerCase()];
  if (!dir) return false;
  if ((dir === "up" || dir === "down") && !canRise(driveTarget())) return false;
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
document.addEventListener("visibilitychange", () => { if (document.hidden) stopDriving(); });
window.addEventListener("pagehide", stopDriving);

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
  if (panelId === "world-panel" && !open) arrangementShelf?.invalidate();
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
  updateEmpty();
  renderScreenDrive();
}

function setWorldPanel(open) {
  setOpen("world-toggle", "world-panel", open);
  if (open) renderGround();
  if (open) { refreshLayouts(); arrangementShelf?.opened(); }
  else layoutConfirmation = null;
  if (open) setOpen("help-toggle", "help", false);
  if (open && narrow()) setOpen("drawer-toggle", "drawer", false);
  storageSet("robotforge.world", open ? "1" : "0");
  shiftForDrawers();
  updateEmpty();
  renderScreenDrive();
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
  for (const id of ["drawer", "lesson-plots", "sensor-views", "view-panel"]) {
    const r = box($(id));
    if (!r) continue;
    if (r.width < W * 0.7 && r.left > left + 160) right = Math.min(right, r.left); // a column beside the view
    else across(r); // a strip over what is left
  }
  const bar = box(document.querySelector(".top"));
  if (bar) top = Math.max(top, bar.bottom);
  const dock = box(document.querySelector(".dock"));
  if (dock) bottom = Math.min(bottom, dock.top);
  for (const id of ["object-edit-bar", "placement-bar", "inspection-bar"]) {
    const edit = box($(id));
    if (edit) bottom = Math.min(bottom, edit.top);
  }
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
window.addEventListener("resize", renderScreenDrive);
// Panels change size with what they show (a quiz answered, the record unfolded): follow them.
const panelWatch = new ResizeObserver(() => shiftForDrawers());
for (const id of ["drawer", "world-panel", "lesson-panel", "lesson-plots", "sensor-views", "object-edit-bar", "placement-bar", "inspection-bar", "view-panel"]) panelWatch.observe($(id));

function setHelp(open) {
  setOpen("help-toggle", "help", open);
  if (open) {
    setOpen("drawer-toggle", "drawer", false);
    setOpen("world-toggle", "world-panel", false);
    shiftForDrawers();
    renderScreenDrive();
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

const togglePlay = () => { arrangementShelf?.invalidate(); if (!paused) stopDriving(); send({ type: "cmd", cmd: paused ? "play" : "pause" }); };
$("btn-play").addEventListener("click", togglePlay);
function resetWorld() {
  stopDriving();
  cancelCreation(true);
  lastAddition = lastObjectEdit = lastRobotEdit = lastTerrainEdit = null; updateUndo();
  send({ type: "cmd", cmd: "reset" });
}
$("btn-reset").addEventListener("click", resetWorld);
function viewWholeScene() { cancelInspection(); freeDriveCamera(); frameCamera({ animate: true }); }
$("btn-frame").addEventListener("click", viewWholeScene);
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
  lighting.bind(worldRoot, bodyNodes, sceneDesc);
  lighting.update(true);
});
// Reuse the existing environment and shadow map; this is a local view preference.
function selectLighting(name) {
  if (!lighting.select(name)) return;
  for (const value of ["studio", "daylight"]) $("lighting-" + value).setAttribute("aria-pressed", String(value === name));
  storageSet("robotforge.lighting", name);
}
for (const name of ["studio", "daylight"]) $("lighting-" + name).addEventListener("click", () => selectLighting(name));
selectLighting(storageGet("robotforge.lighting") === "daylight" ? "daylight" : "studio");

$("show-shadows").addEventListener("change", (e) => {
  sun.castShadow = e.target.checked;
});

window.addEventListener("keydown", (e) => {
  if (inspection) {
    if (e.key === "Escape") { cancelInspection(); canvas.focus(); e.preventDefault(); }
    return; // Inspection owns input until a qualified surface tap or explicit cancel.
  }
  if (!$("view-panel").hidden) {
    if (e.key === "Escape") {
      if (!cameraBookmarks?.cancel()) { setViewPanel(false); $("view-toggle").focus(); }
      e.preventDefault();
    }
    return; // The picker owns keyboard focus; selecting a view cannot start driving.
  }
  if (e.key === "Escape" && layoutConfirmation) {
    cancelLayoutConfirmation(); e.preventDefault(); return;
  }
  if (e.defaultPrevented) return;
  if (selectingObjects && e.key === "Escape") { finishObjectSelection(); e.preventDefault(); return; }
  if (creator.active() && e.key === "Escape") { cancelCreation(); canvas.focus(); e.preventDefault(); return; }
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement
      || e.target instanceof HTMLTextAreaElement || e.target.isContentEditable || e.metaKey || e.ctrlKey || e.altKey) return;
  if (selectingObjects) return;
  if (creator.active()) {
    if (e.repeat && e.key === "Enter") { e.preventDefault(); return; }
    if (e.key === "Enter" && !(e.target instanceof HTMLButtonElement)) {
      creator.place(); e.preventDefault();
    }
    return; // Tab and focused buttons retain their native keyboard behavior.
  }
  dismissCoach();
  if (driveKey(e, true)) return;
  if (e.target instanceof HTMLButtonElement && (e.key === " " || e.key === "Enter")) return; // let buttons click
  if (e.code === "Space") {
    e.preventDefault();
    togglePlay();
  } else if (e.key === "r" || e.key === "R") {
    resetWorld();
  } else if (e.key === ".") {
    send({ type: "cmd", cmd: "step" });
  } else if (e.key === "f" || e.key === "F") {
    viewWholeScene();
  } else if (e.key === "?") {
    setHelp($("help").hidden);
  } else if (e.key === "Escape") {
    const back = !$("world-panel").hidden ? $("world-toggle") : !$("drawer").hidden ? $("drawer-toggle") : $("help-toggle");
    setHelp(false);
    setDrawer(false);
    setWorldPanel(false);
    back.focus();
  }
});

// ---------------------------------------------------------------- world panel: maps, terrain, modes

let requestSeq = 0;
const pending = new Map(); // request id -> { resolve, reject }

// Ask the server for something slow (a new map, a control mode); resolves with its reply.
function request(msg, progress) {
  return new Promise((resolve, reject) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      reject(new Error(t("disconnected")));
      return;
    }
    const id = ++requestSeq;
    pending.set(id, { resolve, reject, progress });
    ws.send(JSON.stringify({ ...msg, id }));
  });
}

function onReply(msg) {
  const waiting = pending.get(msg.id);
  if (!waiting) return;
  pending.delete(msg.id);
  if (msg.ok) waiting.resolve(msg.data);
  else {
    const localized = msg.code?.startsWith("layout_") || msg.code?.startsWith("robot_start_") || msg.code?.startsWith("arrangement_");
    const error = new Error(localized ? tOr(msg.code, msg.message) : msg.message);
    error.code = msg.code;
    waiting.reject(error);
  }
}

let busy = false;

async function runChoice(button, msg, { frame = false, done, progress, failed } = {}) {
  if (busy) return;
  const changingMap = msg.type === "world" && msg.action === "scene";
  if (changingMap) {
    arrangementShelf?.invalidate();
    // Its reply can precede the new scene/state. Never re-enable the old world's layout actions.
    ++layoutRequest; layoutLoading = false;
    layoutCatalog = layoutConfirmation = null;
  }
  busy = true;
  wantFrame = frame;
  button.classList.add("busy");
  setPanelBusy(true);
  try {
    const data = await request(msg, progress); // not inside done?.(...): that skips its arguments when done is unset
    await done?.(data);
  } catch (err) {
    wantFrame = false;
    failed?.(err);
    toast(err.message);
  } finally {
    busy = false;
    button.classList.remove("busy");
    setPanelBusy(false);
    if (changingMap && !layoutLoading && !$("world-panel").hidden) refreshLayouts();
  }
}

function setPanelBusy(on) {
  if (on) stopDriving();
  if (!on && selectingObjects && (placementVersion !== sceneDesc?.version || placementGeneration !== stateGeneration)) {
    clearObjectSelection(); updateEmpty(); renderDrive();
  }
  $("world-panel").setAttribute("aria-busy", String(on));
  $("drawer").setAttribute("aria-busy", String(on));
  for (const b of document.querySelectorAll("#world-panel .choice, #world-panel .swatch, #world-panel .object-card, #build-add-robot, #drawer .choice, #drawer .create-card")) b.disabled = on || b.dataset.off === "1";
  updateUndo();
  renderEditableObjects();
  renderLayouts();
  arrangementShelf?.render();
  renderScreenDrive();
  renderRobotStart();
  renderObjectSelectionControls();
  if (!on) {
    const surface = surfaceReselect;
    surfaceReselect = null;
    if (surface && surface.version === sceneDesc?.version && surface.version === stateVersion
        && surface.generation === stateGeneration && sceneReadyToDraw()
        && renderedVersion === surface.version && posedVersion === surface.version) {
      const entries = surface.names.map((name) => editableObjects().find((o) => o.name === name));
      if (entries.every((entry) => entry && Object.hasOwn(entry, "surface_material"))) {
        if (surface.group) {
          beginObjectGroupSelection();
          for (const entry of entries) toggleObjectGroupMember(entry);
        } else beginObjectSelection(entries[0]);
        showObjectSurface(true);
      }
    }
  }
}

// ------------------------------------------------------------ pick a local robot, click to place

let placementVersion = 0, placementGeneration = null, placementWorldSha = null, creationClickUntil = 0, creationSelection = null;
let lastAddition = null, creationSurface = null, creationAnchor = null;
const supportRay = new THREE.Raycaster();
let objectSelection = null, selectingObjects = false, objectHighlight = null, lastObjectEdit = null, lastRobotEdit = null, lastTerrainEdit = null;
let selectingObjectGroup = false, objectGroupEpoch = 0;
const objectGroupSelection = new Map(), objectGroupHighlights = new Map();
let surfaceReselect = null;
let terrainHighlight = null, terrainListKey = null;
const creationMarker = new THREE.Mesh(new THREE.RingGeometry(0.08, 0.1, 32), accentMaterial({ opacity: 0.85, side: THREE.DoubleSide }));
creationMarker.name = "placement-point-marker";
const headingShape = new THREE.Shape();
headingShape.moveTo(0.19, 0); headingShape.lineTo(0.1, 0.045); headingShape.lineTo(0.1, -0.045); headingShape.closePath();
const creationHeading = new THREE.Mesh(new THREE.ShapeGeometry(headingShape), creationMarker.material);
creationHeading.visible = false;
creationMarker.add(creationHeading);
creationMarker.visible = false;
scene3.add(creationMarker);

function updateEmpty() {
  const hasContent = water || flexMeshes.length || sceneDesc?.geoms.some((g) =>
    g.type !== "plane" && g.group < sceneDesc.visibleGroupLimit && g.rgba[3] > 0);
  $("empty").hidden = !(sceneDesc?.world && robots.length === 0 && !hasContent) || creator.active() || selectingObjects
    || !$("loading").hidden || !$("world-panel").hidden || !$("drawer").hidden || lessonActive();
}

$("empty-add").addEventListener("click", () => { setDrawer(true); creator.open(); });

function beginCreation() {
  arrangementShelf?.invalidate();
  placementPreview.clear();
  clearObjectSelection();
  clearTerrainHighlight();
  if (creator.editing()?.kind === "terrain-edit") {
    terrainHighlight = new THREE.Box3Helper(new THREE.Box3(), theme.amber);
    terrainHighlight.name = "terrain-edit-highlight"; terrainHighlight.material.depthTest = false;
    scene3.add(terrainHighlight); updateTerrainHighlight();
  }
  stopDriving(); endDrag();
  freeDriveCamera();
  creationGesture.invalidate();
  placementVersion = sceneDesc.version;
  placementGeneration = stateGeneration;
  placementWorldSha = creator.editing()?.world_sha256 || sceneDesc.object_edit?.world_sha256 || null;
  setDrawer(false); setWorldPanel(false); setHelp(false);
  controls.enabled = true;
  canvas.classList.add("placing"); canvas.focus();
  // The orbit aim stays visible as panels move the projection; the window centre does not.
  camera.updateMatrixWorld();
  const aim = controls.target.clone().project(camera);
  creationPoint({ clientX: (aim.x + 1) * window.innerWidth / 2, clientY: (1 - aim.y) * window.innerHeight / 2 });
  updateEmpty();
  updateUndo();
  renderScreenDrive();
}

function cancelCreation(force = false) {
  arrangementShelf?.invalidate();
  clearObjectSelection();
  clearTerrainHighlight();
  if (busy && !force) return;
  if (force) creationSelection?.reject(new Error(t("create_scene_changed")));
  creationGesture.invalidate();
  creator.clear(); creationMarker.visible = false;
  creationSurface = creationAnchor = null; placementPreview.clear();
  shiftForDrawers();
  canvas.classList.remove("placing"); controls.enabled = true;
  updateEmpty();
  renderDrive();
  updateUndo();
}

function setCreationPoint(point, status = null) {
  creator.point(point, status); creationSurface = point; creationMarker.visible = !!point;
  if (point) creationMarker.position.set(point[0], point[1], point[2] + 0.008);
  updateCreationPreview();
}

function clearCreationPoint(status = null) {
  creationAnchor = null; setCreationPoint(null, status);
}

function resolveCreationAnchor() {
  const anchor = creationAnchor;
  if (!anchor || anchor.session !== creator.session() || anchor.version !== sceneDesc?.version
      || anchor.generation !== stateGeneration) return { point: null, status: null };
  const point = placementCoordinates(anchor.point, creator.spacing());
  if (!point) return { point: null, status: "placement_coordinate_invalid" };
  anchor.object.updateWorldMatrix(true, false);
  if (anchor.matrix.some((v, i) => v !== anchor.object.matrixWorld.elements[i])) {
    creationAnchor = null;
    return { point: null, status: "placement_support_changed" };
  }
  if (creator.spacing() === "free") return { point, status: null };
  if (anchor.up < 0.9999 || anchor.dynamic || robotOfBody.has(anchor.body) || anchor.object.userData.flex) {
    return { point: null, status: "placement_snap_surface" };
  }
  // Recheck the snapped XY at the same thin height band: never jump off a table to the floor,
  // onto another stair height, or through a different visible surface.
  supportRay.set(new THREE.Vector3(point[0], point[1], anchor.point[2] + 0.002), new THREE.Vector3(0, 0, -1));
  supportRay.near = 0; supportRay.far = 0.004;
  const excluded = creator.editing()?.geoms || [];
  const hit = worldRoot && supportRay.intersectObjects(worldRoot.children, true).find((h) =>
    h.object.visible && !excluded.includes(h.object.userData.geom));
  const up = hit?.face?.normal.clone().transformDirection(hit.object.matrixWorld);
  if (!hit || hit.object !== anchor.object || up?.z < 0.9999 || !up || Math.abs(hit.point.z - anchor.point[2]) > 0.001) {
    return { point: null, status: "placement_snap_outside" };
  }
  return { point: placementCoordinates([point[0], point[1], hit.point.z], creator.spacing()), status: null };
}

function changePlacementOptions() {
  creationGesture.invalidate();
  const { point, status } = resolveCreationAnchor();
  setCreationPoint(point, status);
}

function creationPoint(e) {
  if (!$("view-panel").hidden || tween?.arc || performance.now() < creationClickUntil
      || !sceneReadyToDraw() || renderedVersion !== sceneDesc.version) {
    clearCreationPoint(); return;
  }
  setPointer(e);
  const excluded = creator.editing()?.geoms || [];
  const hit = worldRoot && raycaster.intersectObjects(worldRoot.children, true).find((h) => h.object.visible && !excluded.includes(h.object.userData.geom));
  const body = hit?.object.userData.body;
  const up = hit?.face?.normal.clone().transformDirection(hit.object.matrixWorld);
  const dynamicSurface = body > 0 && sceneDesc.bodies[body]?.dynamic;
  const propPlacement = creator.preview() !== null;
  const valid = hit && up?.z >= 0.7 && (creator.placingTerrain()
    ? sceneDesc.geoms.find((g) => g.id === hit.object.userData.geom)?.type === "plane" && Math.abs(hit.point.z) <= 0.001
    : propPlacement ? !robotOfBody.has(body) : !dynamicSurface);
  if (!valid) { clearCreationPoint(); return; }
  hit.object.updateWorldMatrix(true, false);
  creationAnchor = { point: hit.point.toArray(), object: hit.object, body, dynamic: !!dynamicSurface,
    up: up.z, matrix: hit.object.matrixWorld.elements.slice(), session: creator.session(),
    version: sceneDesc.version, generation: stateGeneration };
  const { point, status } = resolveCreationAnchor();
  setCreationPoint(point, status);
}

function updateCreationPreview() {
  if (loadingScene) { placementPreview.clear(); creationHeading.visible = false; return; }
  placementPreview.update(creator.preview(), creationSurface);
  const heading = creator.heading();
  creationHeading.visible = heading !== null;
  creationMarker.rotation.z = THREE.MathUtils.degToRad(heading ?? 0);
}

function releaseCreationPreview() {
  // Detach borrowers before their world's buffers are disposed, while leaving the
  // active request/session intact so its new posed and rendered scene can complete.
  placementPreview.clear();
  creator.releasePreview();
  clearCreationPoint();
}

function editableRobotStart() {
  const info = sceneDesc?.robot_edit;
  return sceneDesc?.world && !sceneDesc?.lesson && info?.version === sceneDesc.version
    ? info.robots.find((entry) => entry.name === selected) : null;
}

function renderRobotStart() {
  renderEditableTerrains();
  robotSurface.render();
  const button = $("robot-start");
  button.hidden = !editableRobotStart();
  button.disabled = busy || !sceneReadyToDraw() || renderedVersion !== sceneDesc?.version
    || stateVersion !== sceneDesc?.version || !Number.isInteger(stateGeneration);
}

$("robot-start").addEventListener("click", () => {
  const entry = editableRobotStart();
  if (!$("robot-start").disabled && entry) {
    creator.repositionRobot({ ...entry, world_sha256: sceneDesc.robot_edit.world_sha256 });
  }
});

const robotSurface = initRobotSurface({
  current: () => {
    const info = sceneDesc?.robot_surface;
    const entry = info && sceneDesc?.world && info.version === sceneDesc.version && !sceneDesc.lesson
      ? info.robots.find((r) => r.name === selected) : null;
    return entry && ws?.readyState === WebSocket.OPEN
      ? { ...entry, version: info.version, generation: stateGeneration, world_sha256: info.world_sha256 } : null;
  },
  ready: () => !busy && !creator.active() && !selectingObjects && !creationGesture.active()
    && sceneReadyToDraw() && renderedVersion === sceneDesc?.version && stateVersion === sceneDesc?.version
    && Number.isInteger(stateGeneration),
  apply: async (entry, targets, surface, button) => {
    stopDriving(); endDrag();
    const batch = Array.isArray(targets);
    let acknowledgement = null;
    await runChoice(button, { type: "world", action: "set_robot_surface", name: entry.name,
      ...(batch ? { bodies: targets } : { body: targets }),
      surface_material: surface, version: entry.version, generation: stateGeneration,
      world_sha256: entry.world_sha256 }, {
      done: async (data) => {
        if (data.unchanged) {
          if (stateVersion !== data.version || stateGeneration !== data.generation) {
            lastAddition = lastObjectEdit = lastRobotEdit = lastTerrainEdit = null;
            throw new Error(t("create_scene_changed"));
          }
          acknowledgement = { ...data, world_sha256: sceneDesc.robot_surface?.world_sha256 };
          toast(t("surface_unchanged")); return;
        }
        const receipt = data.undo_surface ? { ...data.undo_surface, kind: "robot_surface" } : null;
        lastObjectEdit = receipt; lastAddition = lastRobotEdit = lastTerrainEdit = null;
        try { await waitForCreation(data, true); }
        catch (error) {
          lastObjectEdit = stateVersion === data.version && stateGeneration === data.generation ? receipt : null;
          updateUndo(); throw error;
        }
        acknowledgement = { ...data, world_sha256: sceneDesc.robot_surface?.world_sha256 };
        updateUndo(); toast(batch ? t("robot_surface_saved", { name: entry.name, count: targets.length })
          : t("surface_saved", { name: `${entry.name} / ${targets}` }));
      },
    });
    return acknowledgement;
  },
});
let robotSurfaceHighlight = null;
function updateRobotSurfaceHighlight() {
  const geoms = robotSurface.highlight();
  if (!geoms.length || !worldRoot) {
    if (robotSurfaceHighlight) {
      scene3.remove(robotSurfaceHighlight); robotSurfaceHighlight.geometry.dispose();
      robotSurfaceHighlight.material.dispose(); robotSurfaceHighlight = null;
    }
    return;
  }
  if (!robotSurfaceHighlight) {
    robotSurfaceHighlight = new THREE.Box3Helper(new THREE.Box3(), theme.amber);
    robotSurfaceHighlight.name = "robot-part-highlight"; robotSurfaceHighlight.material.depthTest = false;
    scene3.add(robotSurfaceHighlight);
  }
  worldRoot.updateMatrixWorld(); robotSurfaceHighlight.box.makeEmpty();
  worldRoot.traverse((o) => { if (o.isMesh && o.visible && geoms.includes(o.userData.geom)) robotSurfaceHighlight.box.expandByObject(o); });
  robotSurfaceHighlight.visible = !robotSurfaceHighlight.box.isEmpty();
}

function editableTerrains() {
  const info = sceneDesc?.terrain_edit;
  return sceneDesc?.world && !sceneDesc?.lesson && info?.version === sceneDesc.version ? info.terrains : [];
}

function renderEditableTerrains() {
  const entries = editableTerrains();
  $("edit-terrain-section").hidden = !entries.length;
  const key = `${sceneDesc?.version}:${entries.map((entry) => entry.name).join(",")}`;
  if (terrainListKey !== key) {
    terrainListKey = key;
    $("edit-terrain-list").replaceChildren(...entries.map((entry) => {
      const button = document.createElement("button"); button.type = "button"; button.className = "pill";
      button.textContent = t("move_named_terrain", { name: entry.name }); button.dataset.editTerrain = entry.name;
      button.addEventListener("click", () => {
        if (!button.disabled) creator.editTerrain({ ...entry, world_sha256: sceneDesc.terrain_edit.world_sha256 });
      });
      return button;
    }));
  }
  const disabled = busy || !sceneReadyToDraw() || renderedVersion !== sceneDesc?.version
    || stateVersion !== sceneDesc?.version || !Number.isInteger(stateGeneration);
  for (const button of $("edit-terrain-list").children) button.disabled = disabled;
}

function clearTerrainHighlight() {
  if (!terrainHighlight) return;
  scene3.remove(terrainHighlight); terrainHighlight.geometry.dispose(); terrainHighlight.material.dispose(); terrainHighlight = null;
}

function updateTerrainHighlight() {
  const entry = creator.editing();
  if (!terrainHighlight || entry?.kind !== "terrain-edit" || !worldRoot) return;
  worldRoot.updateMatrixWorld(); terrainHighlight.box.makeEmpty();
  worldRoot.traverse((o) => { if (o.isMesh && o.visible && entry.geoms.includes(o.userData.geom)) terrainHighlight.box.expandByObject(o); });
  terrainHighlight.visible = !terrainHighlight.box.isEmpty();
}

function editableObjects() {
  return sceneDesc?.world && !sceneDesc?.lesson && sceneDesc?.object_edit?.version === sceneDesc.version
    ? sceneDesc.object_edit.objects : [];
}

function renderEditableObjects() {
  const entries = editableObjects();
  $("edit-object-section").hidden = !entries.length;
  $("edit-objects").disabled = busy;
  $("select-object-group").disabled = busy || entries.length < 2;
  $("select-object-group").setAttribute("aria-pressed", String(selectingObjectGroup));
  $("edit-object-list").replaceChildren(...entries.map((entry) => {
    const b = document.createElement("button"); b.type = "button"; b.className = "pill";
    b.textContent = entry.name; b.dataset.editObject = entry.name; b.disabled = busy;
    if (selectingObjectGroup) b.setAttribute("aria-pressed", String(objectGroupSelection.has(entry.name)));
    b.addEventListener("click", () => selectingObjectGroup ? toggleObjectGroupMember(entry) : beginObjectSelection(entry));
    return b;
  }));
}

function clearObjectSelection(preserveSurface = false) {
  if (!preserveSurface) surfaceReselect = null;
  selectingObjects = false; objectSelection = null;
  selectingObjectGroup = false; objectGroupEpoch++; objectGroupSelection.clear();
  for (const highlight of objectGroupHighlights.values()) {
    scene3.remove(highlight); highlight.geometry.dispose(); highlight.material.dispose();
  }
  objectGroupHighlights.clear();
  $("object-edit-bar").hidden = true;
  showObjectSurface(false);
  creationGesture.invalidate();
  if (objectHighlight) {
    scene3.remove(objectHighlight); objectHighlight.geometry.dispose(); objectHighlight.material.dispose(); objectHighlight = null;
  }
  renderEditableObjects();
}

function objectSelectionCurrent() {
  return sceneReadyToDraw() && renderedVersion === sceneDesc?.version && posedVersion === sceneDesc?.version
    && stateVersion === sceneDesc?.version && placementVersion === sceneDesc?.version
    && Number.isInteger(stateGeneration) && placementGeneration === stateGeneration;
}

function renderObjectSelectionControls() {
  const group = selectingObjectGroup;
  for (const id of ["object-focus", "object-move", "object-copy", "object-remove"]) {
    $(id).hidden = group;
    $(id).disabled = busy || !objectSelection;
  }
  const surfaceMembers = group ? [...objectGroupSelection.values()] : objectSelection ? [objectSelection] : [];
  const eligible = surfaceMembers.length > 0 && surfaceMembers.every((entry) => Object.hasOwn(entry, "surface_material"));
  const surfaceReady = eligible && (!group || (paused && surfaceMembers.length >= 2 && objectSelectionCurrent()));
  $("object-surface").hidden = !eligible;
  for (const id of ["object-surface", "object-surface-material"]) $(id).disabled = busy || !surfaceReady;
  $("object-surface-apply").disabled = busy || !surfaceReady || !["", "plastic", "wood", "steel"].includes($("object-surface-material").value);
  if (!eligible) showObjectSurface(false);
  $("object-edit-done").disabled = busy;
  for (const id of ["object-group-copy", "object-group-move"]) {
    $(id).hidden = !group;
    $(id).disabled = busy || !paused || objectGroupSelection.size < 2 || !objectSelectionCurrent();
  }
  if (group) {
    $("object-edit-title").textContent = t("prop_group_count", { count: objectGroupSelection.size });
    $("object-edit-hint").textContent = t(surfaceMembers.length && !eligible ? "surface_group_unsupported"
      : paused ? "prop_group_hint" : "prop_group_pause_hint");
  } else $("object-edit-hint").textContent = t("edit_prop_hint");
  arrangementShelf?.render();
}

function beginObjectGroupSelection() {
  if (busy || editableObjects().length < 2) return;
  beginObjectSelection();
  if (!selectingObjects) return;
  selectingObjectGroup = true; objectGroupEpoch++;
  renderEditableObjects(); renderObjectSelectionControls();
}

function toggleObjectGroupMember(entry) {
  if (busy || !selectingObjectGroup || !objectSelectionCurrent()) return;
  surfaceReselect = null;
  creationGesture.invalidate(); objectGroupEpoch++;
  if (objectGroupSelection.has(entry.name)) {
    objectGroupSelection.delete(entry.name);
    const highlight = objectGroupHighlights.get(entry.name);
    if (highlight) { scene3.remove(highlight); highlight.geometry.dispose(); highlight.material.dispose(); }
    objectGroupHighlights.delete(entry.name);
  } else {
    if (objectGroupSelection.size >= 16) { toast(t("prop_group_limit")); return; }
    objectGroupSelection.set(entry.name, entry);
    const highlight = new THREE.Box3Helper(new THREE.Box3(), theme.amber);
    highlight.name = "object-group-member-highlight"; highlight.material.depthTest = false;
    scene3.add(highlight); objectGroupHighlights.set(entry.name, highlight);
  }
  syncObjectSurfaceChoice();
  updateObjectHighlight(); renderEditableObjects(); renderObjectSelectionControls();
}

function beginObjectSelection(entry = null) {
  if (busy || !editableObjects().length || !sceneReadyToDraw() || !Number.isInteger(stateGeneration)) return;
  cancelCreation(); stopDriving(); endDrag();
  freeDriveCamera();
  placementVersion = sceneDesc.version; placementGeneration = stateGeneration;
  selectingObjects = true; objectSelection = entry;
  creationClickUntil = 0; // An explicit edit-mode choice starts a new deliberate interaction.
  setWorldPanel(false); setDrawer(false); setHelp(false); canvas.focus();
  $("object-edit-title").textContent = entry?.name || t("select_prop");
  $("object-edit-bar").hidden = false;
  $("object-surface").hidden = !entry || !Object.hasOwn(entry, "surface_material");
  syncObjectSurfaceChoice();
  showObjectSurface(false);
  for (const id of ["object-surface", "object-surface-apply", "object-surface-material"]) $(id).disabled = !entry;
  $("object-focus").disabled = $("object-move").disabled = $("object-copy").disabled = $("object-remove").disabled = !entry;
  if (entry) {
    objectHighlight = new THREE.Box3Helper(new THREE.Box3(), theme.amber);
    objectHighlight.name = "object-edit-highlight"; objectHighlight.material.depthTest = false;
    scene3.add(objectHighlight); updateObjectHighlight();
  }
  renderObjectSelectionControls();
  updateEmpty(); updateUndo(); renderDrive();
}

function updateObjectHighlight() {
  if (!worldRoot || (!objectHighlight && !objectGroupHighlights.size)) return;
  worldRoot.updateMatrixWorld();
  const selections = selectingObjectGroup ? [...objectGroupSelection.values()] : objectSelection ? [objectSelection] : [];
  const byGeom = new Map();
  for (const entry of selections) {
    const highlight = selectingObjectGroup ? objectGroupHighlights.get(entry.name) : objectHighlight;
    if (!highlight) continue;
    highlight.box.makeEmpty();
    for (const geom of entry.geoms) byGeom.set(geom, highlight);
  }
  worldRoot.traverse((node) => {
    const highlight = byGeom.get(node.userData.geom);
    if (highlight && node.isMesh && node.visible) highlight.box.expandByObject(node);
  });
  for (const highlight of new Set(byGeom.values())) highlight.visible = !highlight.box.isEmpty();
}

function focusObject() {
  if (busy || !objectSelection || !hasSize() || placementVersion !== sceneDesc?.version
      || placementGeneration !== stateGeneration || renderedVersion !== sceneDesc.version) return;
  updateObjectHighlight();
  if (!objectHighlight?.visible) return;
  // Use current rendered geometry, including a free prop's motion and every part of a fixed prop.
  const sphere = objectHighlight.box.getBoundingSphere(new THREE.Sphere());
  const r = freeRect(), width = r.right - r.left, height = r.bottom - r.top;
  if (width <= 80 || height <= 80) return;
  const direction = camera.position.clone().sub(controls.target);
  if (direction.lengthSq() < 1e-10) direction.set(1, -1.3, 0.8);
  direction.normalize();
  const d = fitDistance(Math.max(sphere.radius, 0.04)) * Math.max(window.innerWidth / width, window.innerHeight / height) * 1.15;
  refitRig = null;
  camera.near = Math.min(camera.near, Math.max(d / 400, 0.001));
  camera.far = Math.max(camera.far, d * 40);
  camera.updateProjectionMatrix();
  moveCamera(sphere.center.clone().addScaledVector(direction, d), sphere.center, true);
  canvas.focus();
}

function pickObjectForEdit(point) {
  setPointer(point);
  const hit = worldRoot && raycaster.intersectObjects(worldRoot.children, true).find((h) => h.object.visible);
  const entry = hit && editableObjects().find((o) => o.geoms.includes(hit.object.userData.geom));
  if (entry) {
    if (selectingObjectGroup) toggleObjectGroupMember(entry);
    else beginObjectSelection(entry);
  }
}

$("edit-objects").addEventListener("click", () => beginObjectSelection());
$("select-object-group").addEventListener("click", beginObjectGroupSelection);
async function beginObjectGroup(mode) {
  if (busy || !selectingObjectGroup || !paused || creationGesture.active()
      || objectGroupSelection.size < 2 || !objectSelectionCurrent()) return;
  const names = [...objectGroupSelection.keys()];
  const guard = { version: placementVersion, generation: placementGeneration, world_sha256: sceneDesc.object_edit.world_sha256 };
  creationGesture.invalidate();
  const epoch = objectGroupEpoch, pointerEpoch = creationGesture.epoch();
  stopDriving(); endDrag();
  await runChoice($(`object-group-${mode}`), { type: "world", action: "prepare_object_group", names, ...guard,
    ...(mode === "move" ? { mode } : {}) }, {
    done: (data) => {
      if (!selectingObjectGroup || objectGroupEpoch !== epoch || creationGesture.epoch() !== pointerEpoch
          || creationGesture.active() || !paused || !objectSelectionCurrent()
          || data.version !== guard.version || data.generation !== guard.generation
          || (mode === "move" ? data.mode !== "move" : data.mode !== undefined && data.mode !== "copy")
          || data.world_sha256 !== guard.world_sha256 || sceneDesc.object_edit?.world_sha256 !== guard.world_sha256
          || !/^[0-9a-f]{32}$/.test(data.token || "") || !/^[0-9a-f]{64}$/.test(data.pose_sha256 || "")
          || !Array.isArray(data.anchor) || data.anchor.length !== 3 || !data.anchor.every(Number.isFinite)
          || !Array.isArray(data.members) || data.members.length !== names.length
          || new Set(data.members.map((member) => member.name)).size !== names.length
          || data.members.some((member) => !names.includes(member.name) || !Array.isArray(member.geoms)
            || !member.geoms.length || !member.geoms.every(Number.isInteger))) throw new Error(t("create_scene_changed"));
      // Freeze the rendered parts in anchor coordinates, including every table leg and bin wall.
      // Their buffers are borrowed read-only and must never be disposed by the ghost.
      worldRoot.updateMatrixWorld(true);
      const previewParts = [], found = new Set();
      const sceneGeoms = new Map(sceneDesc.geoms.map((geom) => [geom.id, geom]));
      const visibleMembers = data.members.map((member) => member.geoms.filter((id) => {
        const geom = sceneGeoms.get(id);
        if (!geom) throw new Error(t("create_scene_changed"));
        // Covered collision primitives remain owned by the prop but have no drawn surface.
        // Include the complete visual meshes, even when debug collision display is enabled.
        return geom.group < sceneDesc.visibleGroupLimit && geom.rgba[3] > 0;
      }));
      if (visibleMembers.some((geoms) => !geoms.length)) throw new Error(t("create_scene_changed"));
      const geoms = new Set(visibleMembers.flat());
      const origin = new THREE.Matrix4().makeTranslation(...data.anchor.map((v) => -v));
      worldRoot.traverse((node) => {
        if (!node.isMesh || !node.visible || !geoms.has(node.userData.geom)) return;
        previewParts.push({ geometry: node.geometry, matrix: origin.clone().multiply(node.matrixWorld).toArray() });
        found.add(node.userData.geom);
      });
      if (!previewParts.length || [...geoms].some((geom) => !found.has(geom))) throw new Error(t("create_scene_changed"));
      creator.objectGroup({ ...data, previewParts }, mode);
    },
  });
}
$("object-group-copy").addEventListener("click", () => beginObjectGroup("copy"));
$("object-group-move").addEventListener("click", () => beginObjectGroup("move"));
function finishObjectSelection() {
  if (busy) return;
  clearObjectSelection(); updateUndo(); updateEmpty(); renderDrive(); canvas.focus();
}
$("object-edit-done").addEventListener("click", finishObjectSelection);
$("object-focus").addEventListener("click", focusObject);
function showObjectSurface(open) {
  $("object-surface-controls").hidden = !open;
  $("object-surface").setAttribute("aria-expanded", String(open));
}
function syncObjectSurfaceChoice() {
  const entries = selectingObjectGroup ? [...objectGroupSelection.values()] : objectSelection ? [objectSelection] : [];
  const finishes = new Set(entries.map((entry) => entry.surface_material || ""));
  $("object-surface-material").value = finishes.size > 1 ? "mixed" : [...finishes][0] || "";
}
$("object-surface-material").addEventListener("change", renderObjectSelectionControls);
$("object-surface").addEventListener("click", () => {
  if (!$("object-surface").disabled && !$("object-surface").hidden) {
    showObjectSurface($("object-surface-controls").hidden);
  }
});
$("object-surface-apply").addEventListener("click", async () => {
  const group = selectingObjectGroup;
  const entries = group ? [...objectGroupSelection.values()] : objectSelection ? [objectSelection] : [];
  const finish = $("object-surface-material").value;
  if (busy || !entries.length || creationGesture.active() || !entries.every((entry) => Object.hasOwn(entry, "surface_material"))
      || !["", "plastic", "wood", "steel"].includes(finish)
      || (group && (!paused || entries.length < 2 || !objectSelectionCurrent()))) return;
  const names = entries.map((entry) => entry.name), epoch = objectGroupEpoch;
  const guard = { version: placementVersion, generation: placementGeneration, world_sha256: sceneDesc.object_edit.world_sha256 };
  const intent = { names, group, sourceVersion: guard.version };
  surfaceReselect = intent;
  stopDriving(); endDrag();
  await runChoice($("object-surface-apply"), { type: "world", action: group ? "set_object_surfaces" : "set_object_surface",
    ...(group ? { names } : { name: names[0] }), surface_material: finish || null, ...guard }, {
    done: async (data) => {
      if (group ? !validGroupObjects(data.objects, names.length, names) : data.object !== names[0]) {
        surfaceReselect = null; throw new Error(t("create_scene_changed"));
      }
      if (data.unchanged) {
        surfaceReselect = null;
        if (ws?.readyState !== WebSocket.OPEN || stateVersion !== data.version || stateGeneration !== data.generation
            || data.version !== guard.version || data.generation !== guard.generation) {
          lastAddition = lastObjectEdit = lastRobotEdit = lastTerrainEdit = null;
          clearObjectSelection(); throw new Error(t("create_scene_changed"));
        }
        if (!selectingObjects || objectGroupEpoch !== epoch || !objectSelectionCurrent()) {
          clearObjectSelection(); updateUndo(); throw new Error(t("create_scene_changed"));
        }
        toast(t("surface_unchanged")); return;
      }
      const receipt = group ? groupEditUndo(data, "surface-group") : data.undo_edit || null;
      if (!receipt) { surfaceReselect = null; throw new Error(t("create_scene_changed")); }
      lastObjectEdit = null;
      lastAddition = lastRobotEdit = lastTerrainEdit = null;
      try { await waitForCreation(data, true); }
      catch (error) {
        lastObjectEdit = ws?.readyState === WebSocket.OPEN && surfaceReselect === intent
          && stateVersion === data.version && stateGeneration === data.generation ? receipt : null;
        clearObjectSelection(); updateUndo(); throw error;
      }
      if (stateVersion !== data.version || stateGeneration !== data.generation) {
        lastObjectEdit = null; clearObjectSelection(); updateUndo(); throw new Error(t("create_scene_changed"));
      }
      lastObjectEdit = receipt;
      const owned = surfaceReselect === intent;
      clearObjectSelection(owned);
      if (owned) Object.assign(intent, { version: data.version, generation: data.generation });
      updateUndo(); toast(t(group ? "surface_group_saved" : "surface_saved", { count: names.length, name: names[0] }));
    },
  });
});
$("object-move").addEventListener("click", () => { if (!busy && objectSelection) creator.edit(objectSelection); });
$("object-copy").addEventListener("click", () => {
  if (!busy && objectSelection) creator.duplicate({ ...objectSelection, world_sha256: sceneDesc.object_edit.world_sha256 });
});

async function removeSelectedObject() {
  if (busy || !objectSelection || creationGesture.active()) return;
  const entry = objectSelection;
  stopDriving(); endDrag();
  await runChoice($("object-remove"), { type: "world", action: "remove_object", name: entry.name,
    version: placementVersion, generation: placementGeneration, world_sha256: sceneDesc.object_edit.world_sha256 }, {
    done: async (data) => {
      await waitForCreation(data, true); lastObjectEdit = data.undo_edit; lastAddition = lastRobotEdit = lastTerrainEdit = null;
      clearObjectSelection(); updateUndo(); renderDrive(); toast(t("prop_removed", { name: data.removed })); canvas.focus();
    },
  });
}
$("object-remove").addEventListener("click", removeSelectedObject);
$("undo-edit").addEventListener("click", async () => {
  if (busy || (!lastObjectEdit && !lastRobotEdit && !lastTerrainEdit)) return;
  stopDriving(); endDrag();
  const robotStart = !!lastRobotEdit, terrainEdit = !!lastTerrainEdit;
  const selectedReceipt = lastTerrainEdit || lastRobotEdit || lastObjectEdit;
  const robotFinish = selectedReceipt.kind === "robot_surface";
  const groupMove = selectedReceipt.kind === "group-move";
  const groupSurface = selectedReceipt.kind === "surface-group";
  const arrangementInsert = selectedReceipt.kind === "arrangement-insert";
  const { kind: _kind, objects: groupObjects, ...receipt } = selectedReceipt;
  await runChoice($("undo-edit"), { type: "world", action: terrainEdit ? "undo_terrain_edit" : robotStart ? "undo_robot_start" : robotFinish ? "undo_robot_surface" : "undo_object_edit", ...receipt }, {
    done: async (data) => {
      if ((groupMove || groupSurface) && ((groupMove && data.moved !== true) || !validGroupObjects(data.objects, groupObjects?.length, groupObjects))) {
        throw new Error(t("create_scene_changed"));
      }
      if (arrangementInsert && !validGroupObjects(data.removed, groupObjects?.length, groupObjects)) {
        throw new Error(t("create_scene_changed"));
      }
      await waitForCreation(data, true); lastObjectEdit = lastRobotEdit = lastTerrainEdit = null; updateUndo();
      toast(t(groupSurface ? "surface_group_undone" : groupMove ? "prop_group_move_undone" : Array.isArray(data.removed) ? "prop_group_removed" : terrainEdit ? "terrain_edit_undone" : robotStart ? "robot_start_undone" : robotFinish ? "robot_surface_undone" : data.removed ? "prop_removed" : "prop_edit_undone", { name: data.robot || data.object || data.removed, count: groupMove || groupSurface ? data.objects.length : data.removed?.length })); canvas.focus();
    },
  });
});

function objectInScene(name) {
  return sceneDesc?.bodies.some((b) => b.name === name)
    || sceneDesc?.geoms.some((g) => g.name === name || g.name?.startsWith(`${name}/`));
}

function arrangementObjectsDisplayed(names) {
  const drawn = new Set();
  worldRoot?.traverse((node) => { if (node.isMesh && node.visible) drawn.add(node.userData.geom); });
  const visible = new Set(sceneDesc.geoms.filter((geom) => geom.group < sceneDesc.visibleGroupLimit && geom.rgba[3] > 0).map((geom) => geom.id));
  return names.every((name) => {
    const entry = sceneDesc.object_edit?.objects.find((object) => object.name === name);
    const geoms = entry?.geoms.filter((id) => visible.has(id));
    return geoms?.length > 0 && geoms.every((id) => drawn.has(id));
  });
}

function resolveCreatedRobot() {
  const wanted = creationSelection;
  if (!wanted || posedVersion !== sceneDesc?.version) return;
  if (stateVersion !== wanted.version) {
    if (stateVersion > wanted.version) wanted.reject(new Error(t("create_scene_changed")));
    return;
  }
  if (Number.isInteger(wanted.generation) && stateGeneration !== wanted.generation) {
    if (stateGeneration > wanted.generation) wanted.reject(new Error(t("create_scene_changed")));
    return;
  }
  if (sceneDesc.version < wanted.version) return;
  const removed = Array.isArray(wanted.removed) ? wanted.removed : wanted.removed ? [wanted.removed] : [];
  if (sceneDesc.version !== wanted.version || (wanted.robot && !robotByName(wanted.robot))
      || (wanted.object && !objectInScene(wanted.object))
      || (wanted.objects && (!Array.isArray(wanted.objects) || wanted.objects.some((name) => !objectInScene(name))))
      || (wanted.arrangement && !arrangementObjectsDisplayed(wanted.objects))
      || (wanted.terrain && !objectInScene(wanted.terrain))
      || removed.some((name) => robotByName(name) || objectInScene(name))) {
    wanted.reject(new Error(t("create_scene_changed"))); return;
  }
  if (wanted.rendered && renderedVersion !== wanted.version) return;
  if (wanted.robot) select(wanted.robot, { focus: wanted.focus });
  wanted.resolve();
}

function waitForCreation(data, rendered = false, { focus = true } = {}) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      const error = new Error(t("create_scene_changed"));
      error.displayTimeout = true;
      creationSelection?.reject(error);
    }, 15000);
    const finish = (callback, error) => { clearTimeout(timer); creationSelection = null; callback(error); };
    creationSelection = { ...data, rendered, focus, resolve: () => finish(resolve), reject: (error) => finish(reject, error) };
    resolveCreatedRobot();
  });
}

function validGroupObjects(objects, count, expected = null) {
  return Array.isArray(objects) && objects.length === count
    && objects.every((name) => typeof name === "string" && name.length > 0)
    && new Set(objects).size === count && (!expected || objects.every((name) => expected.includes(name)));
}

function groupEditUndo(data, kind) {
  const receipt = data.undo_edit;
  return receipt?.version === data.version && receipt.generation === data.generation
    && /^[0-9a-f]{32}$/.test(receipt.token || "") ? { ...receipt, kind, objects: data.objects.slice() } : null;
}

async function placeEntity(item, pos) {
  if (busy || creationGesture.active() || !item || !pos) return;
  if (creator.spacing() !== "free") {
    const resolved = resolveCreationAnchor();
    if (!resolved.point || resolved.point.some((v, i) => v !== pos[i])) {
      clearCreationPoint(resolved.status || "placement_support_changed"); return;
    }
  }
  pos = pos.slice();
  const repeating = creator.repeating();
  if (isObjectGroup(item) && (!paused || !objectSelectionCurrent()
      || item.version !== placementVersion || item.generation !== placementGeneration
      || item.world_sha256 !== sceneDesc.object_edit?.world_sha256)) { toast(t("create_scene_changed")); return; }
  if (repeating && !Number.isInteger(placementGeneration)) { toast(t("loading_scene")); return; }
  if ((item.kind === "terrain" || item.kind === "terrain-edit") && (!Number.isInteger(placementGeneration) || !placementWorldSha)) { toast(t("loading_scene")); return; }
  const placementSession = creator.session();
  creationGesture.invalidate();
  creator.busy(true);
  const payload = item.kind === "robot" ? { action: "add_robot", template: item.name }
    : item.kind === "model" ? { action: "add_model", source: item.source, model: item.name }
    : item.kind === "start" ? { action: "edit_robot_start", name: item.name, world_sha256: item.world_sha256, yaw: item.yaw }
    : item.kind === "edit" ? { action: "edit_object", name: item.name, world_sha256: sceneDesc.object_edit.world_sha256, yaw: item.yaw }
      : item.kind === "copy" ? { action: "duplicate_object", name: item.name, world_sha256: item.world_sha256, yaw: item.yaw }
    : item.kind === "arrangement-insert" ? { action: "arrangement_insert", token: item.token, world_sha256: item.world_sha256, yaw: item.yaw }
    : isObjectGroup(item) ? { action: item.kind === "group-move" ? "move_object_group" : "duplicate_object_group", token: item.token, world_sha256: item.world_sha256, yaw: item.yaw }
    : item.kind === "terrain-edit" ? { action: "edit_terrain", name: item.name, yaw_delta: item.yaw_delta, world_sha256: item.world_sha256 }
    : item.kind === "terrain" ? { action: "add_terrain", preset: item.name, yaw: item.yaw, world_sha256: placementWorldSha }
      : { action: "add_object", preset: item.name, size: item.scale, yaw: item.yaw,
        ...(item.material ? { material: item.material } : {}) };
  await runChoice($("place-here"), { type: "world", ...payload, pos, version: placementVersion, generation: placementGeneration ?? undefined }, {
    frame: !repeating && (item.kind === "robot" || item.kind === "model"),
    progress: (event) => { if (placementSession === creator.session()) creator.progress(event.stage, event.fraction); },
    failed: (error) => {
      if (placementSession !== creator.session()) return;
      if (item.kind === "arrangement-insert" && error.code?.startsWith("arrangement_")
          && !["arrangement_occupied", "arrangement_target", "arrangement_busy"].includes(error.code)) {
        cancelCreation(true); arrangementShelf?.invalidate({ clear: true });
        selectTab("objects", false, false);
        if ($("world-panel").hidden) setWorldPanel(true);
        else arrangementShelf?.opened();
      }
    },
    done: async (data) => {
      if (placementSession !== creator.session()) return;
      if (isObjectGroup(item) && (!validGroupObjects(data.objects, item.members.length,
          item.kind === "group-move" ? item.members.map((member) => member.name) : null)
          || !Number.isInteger(data.version) || !Number.isInteger(data.generation)
          || (item.kind === "group-move" && !data.unchanged && data.moved !== true)
          || (item.kind === "arrangement-insert" && (data.version <= item.version || data.generation <= item.generation || data.paused !== true
            || !groupEditUndo(data, item.kind))))) {
        lastAddition = lastObjectEdit = lastRobotEdit = lastTerrainEdit = null;
        cancelCreation(true); throw new Error(t("create_scene_changed"));
      }
      if (data.unchanged) {
        if (placementSession !== creator.session()) return;
        if (stateVersion !== data.version || stateGeneration !== data.generation) {
          lastAddition = lastObjectEdit = lastRobotEdit = lastTerrainEdit = null;
          cancelCreation(true); throw new Error(t("create_scene_changed"));
        }
        cancelCreation(true); toast(t(isObjectGroup(item) ? "prop_group_unchanged" : "terrain_unchanged")); return;
      }
      if (item.kind === "terrain-edit") lastTerrainEdit = data.undo_terrain_edit || null;
      // A committed terrain edit remains undoable even if its first displayed geometry times out.
      if (item.kind === "terrain") lastAddition = data.undo || null;
      if (repeating && !Number.isInteger(data.generation)) {
        lastAddition = lastObjectEdit = lastRobotEdit = lastTerrainEdit = null;
        cancelCreation(true); throw new Error(t("create_scene_changed"));
      }
      if (item.kind === "model") creator.progress("display");
      try {
        await waitForCreation(item.kind === "arrangement-insert" ? { ...data, arrangement: true } : data, isObjectGroup(item) || item.kind === "terrain-edit" || item.kind === "terrain" || item.kind === "model" || repeating || item.kind === "edit" || item.kind === "copy" || item.kind === "start", { focus: !repeating });
      } catch (error) {
        if (isObjectGroup(item)) {
          lastObjectEdit = item.kind !== "arrangement-insert" && error.displayTimeout && stateVersion === data.version && stateGeneration === data.generation
            ? groupEditUndo(data, item.kind) : null;
          lastAddition = lastRobotEdit = lastTerrainEdit = null;
          cancelCreation(true);
        }
        if (item.kind === "terrain-edit") {
          lastTerrainEdit = stateVersion === data.version && stateGeneration === data.generation ? data.undo_terrain_edit || null : null;
          cancelCreation(true);
        }
        if (item.kind === "terrain") {
          lastAddition = stateVersion === data.version && stateGeneration === data.generation ? data.undo || null : null;
          cancelCreation(true);
        }
        if (repeating && placementSession === creator.session()) {
          lastAddition = stateVersion === data.version && stateGeneration === data.generation ? data.undo || null : null;
          lastObjectEdit = lastRobotEdit = lastTerrainEdit = null;
          cancelCreation(true);
        }
        // A committed model stays undoable if only display completion timed out.
        if (item.kind === "model" && placementSession === creator.session()
            && stateVersion === data.version && stateGeneration === data.generation) {
          lastAddition = data.undo || null;
          cancelCreation(true);
        }
        throw error;
      }
      if (placementSession !== creator.session()) return;
      if (repeating && (loadingScene || stateVersion !== data.version || stateGeneration !== data.generation
          || sceneDesc?.version !== data.version || posedVersion !== data.version || renderedVersion !== data.version)) {
        lastAddition = stateVersion === data.version && stateGeneration === data.generation ? data.undo || null : null;
        lastObjectEdit = lastRobotEdit = lastTerrainEdit = null;
        cancelCreation(true); throw new Error(t("create_scene_changed"));
      }
      if (item.kind === "terrain" && repeating) {
        const info = sceneDesc?.terrain_edit;
        const current = Number.isInteger(data.generation)
          && stateVersion === data.version && stateGeneration === data.generation;
        // Scene extras are collected separately: never carry a digest from another displayed world.
        if (!current || loadingScene || sceneDesc?.version !== data.version || posedVersion !== data.version
            || renderedVersion !== data.version || info?.version !== data.version
            || !/^[0-9a-f]{64}$/.test(info?.world_sha256 ?? "")) {
          lastAddition = current ? data.undo || null : null;
          cancelCreation(true); throw new Error(t("create_scene_changed"));
        }
        placementWorldSha = info.world_sha256;
      }
      lastAddition = data.undo || null;
      lastObjectEdit = isObjectGroup(item) ? groupEditUndo(data, item.kind) : data.undo_edit || null;
      lastRobotEdit = data.undo_robot_start || null;
      lastTerrainEdit = data.undo_terrain_edit || null;
      creationClickUntil = performance.now() + 500;
      if (repeating) {
        if (item.kind === "copy" && (!editableObjects().some((o) => o.name === item.name) || !sceneDesc.object_edit.world_sha256)) {
          cancelCreation(true); throw new Error(t("create_scene_changed"));
        }
        creationGesture.invalidate();
        placementVersion = data.version; placementGeneration = data.generation ?? stateGeneration;
        creator.next(item.kind === "copy" ? { world_sha256: sceneDesc.object_edit.world_sha256 } : {});
        creationSurface = creationAnchor = null; creationMarker.visible = false; placementPreview.clear();
      } else cancelCreation(true);
      toast(t(item.kind === "arrangement-insert" ? "arrangement_inserted" : isObjectGroup(item) ? item.kind === "group-move" ? "prop_group_moved" : "prop_group_copied" : item.kind === "terrain-edit" ? "terrain_edited" : item.kind === "terrain" ? "terrain_added" : item.kind === "start" ? "robot_start_saved" : item.kind === "edit" ? "prop_edited" : item.kind === "copy" ? "prop_copied" : "robot_added", { name: data.robot || data.object || data.terrain, count: data.objects?.length }));
      canvas.focus();
    },
  });
  creator.busy(false);
  if (sceneDesc?.version !== placementVersion
      || (placementGeneration !== null && stateGeneration !== placementGeneration)) cancelCreation(true);
}

function updateUndo() {
  $("undo-add").hidden = !lastAddition || creator.active() || selectingObjects || !!sceneDesc?.lesson;
  $("undo-edit").hidden = (!lastObjectEdit && !lastRobotEdit && !lastTerrainEdit) || creator.active() || selectingObjects || !!sceneDesc?.lesson;
  $("undo-edit").textContent = t(lastTerrainEdit ? "undo_terrain_edit" : lastRobotEdit ? "undo_robot_start" : lastObjectEdit?.kind === "robot_surface" ? "undo_robot_surface" : lastObjectEdit?.kind === "surface-group" ? "undo_surface_group" : lastObjectEdit?.kind === "arrangement-insert" ? "arrangement_undo" : isObjectGroup(lastObjectEdit) ? lastObjectEdit.kind === "group-move" ? "undo_prop_group_move" : "undo_prop_group" : "undo_object_edit");
  $("undo-edit").disabled = busy;
  $("undo-add").disabled = busy;
  $("undo-add").title = lastAddition?.name || "";
}

$("undo-add").addEventListener("click", async () => {
  if (busy || !lastAddition) return;
  stopDriving(); endDrag();
  await runChoice($("undo-add"), { type: "world", action: "undo_add", token: lastAddition.token, version: lastAddition.version }, {
    done: async (data) => {
      await waitForCreation(data, data.paused === true);
      lastAddition = lastObjectEdit = lastRobotEdit = lastTerrainEdit = null; updateUndo();
      toast(t(data.paused === true ? "terrain_undone" : "addition_undone", { name: data.removed })); canvas.focus();
    },
  });
});

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

arrangementShelf = createArrangementShelf({
  context: () => ({ version: sceneDesc?.version, generation: stateGeneration,
    world_sha256: sceneDesc?.object_edit?.world_sha256, session: creator.session(), selectionEpoch: objectGroupEpoch, pointerEpoch: creationGesture.epoch(),
    supported: !!sceneDesc?.world && !sceneDesc?.lesson, connected: ws?.readyState === WebSocket.OPEN,
    ready: !!sceneDesc?.world && !sceneDesc?.lesson && sceneReadyToDraw()
      && stateVersion === sceneDesc.version && posedVersion === sceneDesc.version && renderedVersion === sceneDesc.version
      && Number.isInteger(stateGeneration) && !creationGesture.active(), paused, busy }),
  selection: () => ({ group: selectingObjectGroup, names: [...objectGroupSelection.keys()],
    ready: selectingObjectGroup && paused && objectGroupSelection.size >= 2 && objectGroupSelection.size <= 16
      && objectSelectionCurrent() && [...objectGroupSelection.values()].every((member) =>
        ["box", "sphere", "cylinder", "table", "bin"].includes(member.shape)) }),
  request, notice: toast, prepare: (data) => creator.arrangement(data),
});

const worldTabs = [...$("world-panel").querySelectorAll("[role=tab]")];
function selectTab(name, focus = false, load = true) {
  if (!worldTabs.some((tab) => tab.dataset.tab === name)) name = "maps";
  for (const tab of worldTabs) {
    const selected = tab.dataset.tab === name;
    tab.setAttribute("aria-selected", String(selected)); tab.tabIndex = selected ? 0 : -1;
    if (selected && focus) tab.focus();
  }
  for (const body of $("world-panel").querySelectorAll(".tab-body")) body.hidden = body.dataset.body !== name;
  storageSet("robotforge.tab", name);
  if (name === "terrain") renderGround();
  if (name === "objects" && load) arrangementShelf?.opened();
  if (name !== "objects") arrangementShelf?.invalidate();
}

for (const tab of worldTabs) {
  tab.addEventListener("click", () => selectTab(tab.dataset.tab));
  tab.addEventListener("keydown", (e) => {
    const index = worldTabs.indexOf(tab);
    const to = { ArrowRight: (index + 1) % worldTabs.length, ArrowLeft: (index + worldTabs.length - 1) % worldTabs.length,
      Home: 0, End: worldTabs.length - 1 }[e.key];
    if (to === undefined) return;
    e.preventDefault(); e.stopPropagation(); selectTab(worldTabs[to].dataset.tab, true);
  });
}
$("build-add-robot").addEventListener("click", () => { setWorldPanel(false); setDrawer(true); creator.open(); });
$("build-pick-map").addEventListener("click", () => selectTab("maps", true));
selectTab(storageGet("robotforge.tab") || "maps", false, false);

let lastMap = null; // the map picked here; a world does not remember which preset it came from

function layoutReady() {
  return layoutCatalog?.available && layoutCatalog.version === sceneDesc?.version
    && layoutCatalog.generation === stateGeneration && !sceneDesc?.lesson && sceneReadyToDraw()
    && renderedVersion === sceneDesc.version
    && ws?.readyState === WebSocket.OPEN;
}

function layoutGuard() {
  return { version: layoutCatalog.version, generation: layoutCatalog.generation, world_sha256: layoutCatalog.world_sha256 };
}

function renderLayouts() {
  const ready = layoutReady(), blocked = busy || layoutLoading || layoutMutating;
  $("layout-name").disabled = !ready || blocked;
  $("layout-save").disabled = !ready || blocked || layoutCatalog.entries.length >= layoutCatalog.limit;
  $("layout-refresh").disabled = blocked || !sceneDesc;
  $("layout-status").textContent = layoutMutating ? t("layout_working") : layoutLoading ? t("layout_loading")
    : layoutError || (!ready ? t(layoutCatalog?.reason || "layout_refresh_hint")
      : layoutCatalog.entries.length >= layoutCatalog.limit ? t("layout_full")
      : layoutCatalog.entries.length ? t("layout_count", { count: layoutCatalog.entries.length, limit: layoutCatalog.limit }) : t("layout_empty"));
  $("layout-list").replaceChildren(...(layoutCatalog?.entries || []).map((entry) => {
    const row = el("div", "layout-row"); row.dataset.layout = entry.id;
    row.append(el("span", "", entry.name));
    for (const action of ["restore", "remove"]) {
      const button = el("button", "pill", t(`layout_${action}`));
      button.type = "button"; button.dataset.layoutAction = action; button.disabled = !ready || blocked;
      button.setAttribute("aria-label", `${t(`layout_${action}`)} ${entry.name}`);
      button.addEventListener("click", () => {
        layoutConfirmation = { action, entry, ...layoutGuard() }; renderLayouts(); $("layout-cancel").focus();
      });
      row.append(button);
    }
    return row;
  }));
  $("layout-confirm").hidden = !layoutConfirmation;
  if (layoutConfirmation) {
    $("layout-confirm-text").textContent = t(`layout_${layoutConfirmation.action}_confirm`, { name: layoutConfirmation.entry.name });
    $("layout-apply").textContent = t(`layout_${layoutConfirmation.action}`);
    $("layout-apply").disabled = blocked || !ready || layoutConfirmation.version !== layoutCatalog.version
      || layoutConfirmation.generation !== layoutCatalog.generation || layoutConfirmation.world_sha256 !== layoutCatalog.world_sha256;
    $("layout-cancel").disabled = blocked;
  }
}

async function refreshLayouts() {
  if (layoutMutating || !sceneDesc || ws?.readyState !== WebSocket.OPEN) return;
  const serial = ++layoutRequest;
  layoutLoading = true; layoutError = ""; layoutConfirmation = null; renderLayouts();
  try {
    const data = await request({ type: "world", action: "layout_list" });
    if (serial !== layoutRequest) return;
    layoutCatalog = data.version === sceneDesc?.version && data.generation === stateGeneration ? data : null;
  } catch (error) {
    if (serial === layoutRequest) { layoutCatalog = null; layoutError = error.message; }
  } finally {
    if (serial === layoutRequest) { layoutLoading = false; renderLayouts(); }
  }
}

function cancelLayoutConfirmation() {
  const identity = layoutConfirmation?.entry.id;
  layoutConfirmation = null; renderLayouts();
  $("layout-list").querySelector(`[data-layout="${identity}"] button`)?.focus();
}

async function mutateLayout(button, action, payload) {
  if (!layoutReady() || busy || layoutMutating) return;
  ++layoutRequest; layoutLoading = false; layoutError = ""; layoutMutating = true;
  await runChoice(button, { type: "world", action: `layout_${action}`, ...payload }, {
    frame: action === "restore", done: (data) => {
      layoutCatalog = data; layoutConfirmation = null;
      if (action === "save") $("layout-name").value = "";
      if (action === "restore") lastMap = null;
      toast(t(action === "save" ? "layout_saved" : action === "restore" ? "layout_restored" : "layout_removed", { name: data.name }));
    },
  });
  layoutMutating = false;
  if (layoutCatalog && (layoutCatalog.version !== sceneDesc?.version || layoutCatalog.generation !== stateGeneration)) {
    layoutCatalog = null; layoutConfirmation = null;
    if (!$("world-panel").hidden) refreshLayouts();
  }
  renderLayouts();
}

$("layout-refresh").addEventListener("click", refreshLayouts);
$("layout-save-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (layoutReady()) mutateLayout($("layout-save"), "save", { ...layoutGuard(), name: $("layout-name").value });
});
$("layout-cancel").addEventListener("click", cancelLayoutConfirmation);
$("layout-apply").addEventListener("click", () => {
  const pending = layoutConfirmation;
  if (!pending || $("layout-apply").disabled) return;
  mutateLayout($("layout-apply"), pending.action, { version: pending.version, generation: pending.generation,
    world_sha256: pending.world_sha256, layout_id: pending.entry.id, sha256: pending.entry.sha256 });
});

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
  const preview = !$("world-panel").hidden && !$("world-panel").querySelector('[data-body="terrain"]').hidden;
  $("ground-list").replaceChildren(
    ...Object.entries(cat.grounds).map(([name, rgb]) => {
      const b = el("button", "swatch");
      b.type = "button";
      b.setAttribute("aria-pressed", String(worldInfo.floor === name));
      b.dataset.off = worldInfo.isWorld ? "0" : "1";
      b.disabled = !worldInfo.isWorld || busy;
      b.dataset.ground = name;
      b.append(groundPreview(name, rgb, preview), el("span", "", tOr(`ground_${name}`, name)));
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
  const now = currentModes[selected] ?? (canMove(robot) ? "drive" : "manual");
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

function renderDemo(recording, replaying) {
  const chip = $("demo-chip");
  const rec = recording?.[0];
  const play = replaying?.[0];
  chip.hidden = !rec && !play;
  if (!rec && !play) return;
  chip.classList.toggle("rec", Boolean(rec));
  $("demo-text").textContent = rec
    ? `${t("demo_recording", { name: rec.name })} ${rec.seconds.toFixed(0)} s`
    : t("demo_replaying", { name: play.name });
}

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
  const ready = sceneReadyToDraw();
  updateViewShift();
  if (ready) {
    healCamera();
    refitSettledRig();
    updateTween();
    updateFollow();
  }
  controls.update();
  if (!$("view-panel").hidden) cameraBookmarks?.update();
  if (ready) {
    updateRing();
    updatePathWidths();
    spinRotors();
    refreshDragVisual();
    updateObjectHighlight();
    updateTerrainHighlight();
    updateRobotSurfaceHighlight();
    lessonFrame();
    if (water) waterFrame(water, latestTime, camera, scene3);
    sensorsFrame();
  }
  mapFrame();
  padFrame();
  // Before poses, fog/light/camera variants are still provisional. Drawing them compiles work twice.
  // Keep the initial background and UI alive, then draw the complete posed scene once it is available.
  if (sceneDesc && (!ready || !hasSize())) return;
  renderer.render(scene3, camera);
  if (ready && renderedVersion !== sceneDesc.version && !renderer.getContext().isContextLost()) {
    renderedVersion = sceneDesc.version;
    $("loading").hidden = true;
    renderLayouts();
    renderScreenDrive();
    renderRobotStart();
    updateEmpty();
    resolveCreatedRobot();
  }
  frames++;
  const now = performance.now();
  if (now - fpsT0 > 500) {
    fps = Math.round((frames * 1000) / (now - fpsT0));
    updateClock();
    frames = 0;
    fpsT0 = now;
  }
}

canvas.addEventListener("webglcontextlost", () => {
  arrangementShelf?.invalidate();
  if (creator.preview()?.members) cancelCreation(true);
  renderedVersion = 0;
  renderLayouts();
  renderRobotStart();
  $("loading").hidden = false;
  updateEmpty();
});

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
    else {
      aimLessonCamera(eye, target, fit);
      // The lesson-open reply can follow first poses and replace their scheduled settling fit.
      if (fit) refitRig = { eye, target, at: performance.now() + RIG_SETTLE_MS };
    }
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
  sensorLighting: lighting.sensorLighting,
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

initMap({
  scene: scene3,
  camera,
  selected: () => selected,
  color: (token) => new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue(token).trim() || "#888"),
  inLesson: () => lessonActive(),
  storageGet,
  storageSet,
});

initTraining({ request, toast });
initFlow({ inLesson: () => lessonActive(), storageGet, storageSet });

screenDrive = createScreenDrive($("screen-drive"), (refresh) => {
  if (screenDrive.held() && (loadingScene || renderedVersion !== sceneDesc?.version || paused
    || ws?.readyState !== WebSocket.OPEN || busy || document.hidden || !driveTarget())) {
    screenDrive.clear(false); renderScreenDrive(); return;
  }
  if (screenDrive.held() && driveTarget()) startFollow();
  sendDrive(refresh);
});

initGamepad({
  send,
  toast,
  robot: () => inspection || creator.active() || selectingObjects ? null : robotByName(selected),
  screenToWorld,
  changed: renderDrive,
  onDrive: () => {
    if (padAxes() && driveTarget()) startFollow();
    sendDrive();
  },
});

cameraBookmarks = createCameraBookmarks({ element: $("camera-bookmarks"), context: bookmarkContext,
  capture: captureCameraBookmark, restore: restoreCameraBookmark, t });

connect();
loop();

// Expose a tiny hook for automated checks (headless verification reads this).
window.__robotforge = {
  get scene() { return sceneDesc; },
  get time() { return latestTime; },
  get renderedVersion() { return renderedVersion; },
  bodyPosition: (id) => bodyNodes[id]?.position.toArray(),
  pick: (clientX, clientY) => pickBody({ clientX, clientY })?.body ?? null,
  cameraTarget: () => controls.target.toArray(),
  // Place the camera for scripted screenshots: eye and target in world meters.
  lookAt: (eye, target) => {
    moveCamera(new THREE.Vector3().fromArray(eye), new THREE.Vector3().fromArray(target), false);
  },
  // Render now and read the frame back (RGBA, bottom row first), for scripted image checks.
  pixels: () => {
    if (sceneReadyToDraw() && hasSize()) {
      controls.update();
      renderer.render(scene3, camera);
    }
    const gl = renderer.getContext();
    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const data = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
    return { w, h, data };
  },
  three: () => ({ scene: scene3, camera, renderer, sun, lighting }), // for experiments in scripted checks
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
  pad: () => padState(), // the gamepad: connected, profile, what it asks for (gamepad.js)
  flow: () => flowState(), // the behavior tree a script runs (flowview.js)
  map: () => mapState(), // the maps robots build: grid, particles, poses (mapview.js)
  training: () => trainingState(), // the run the Training panel shows (training.js)
  showSensor: (name, on = true) => sensorsShow(name, on), // as switching it in the robots drawer
};
