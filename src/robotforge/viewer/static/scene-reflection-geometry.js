// Bind/edit-time geometry qualification, never a per-frame scene traversal.
import * as THREE from "three";

const finite3 = (v) => (Array.isArray(v) || ArrayBuffer.isView(v)) && v.length === 3 && [...v].every(Number.isFinite);
const validBox = (b) => b && finite3(b.min) && finite3(b.max) && b.min.every((x, i) => x <= b.max[i]);
const result = (scope, status, reason, details = {}) => ({ scope, status, reason, ...details });
const tuple = (v) => [v.x, v.y, v.z];

// Pass the same complete robotOfBody map used by the viewer, even for fixed-base robots.
// excludedObjects may contain a whole animated/auxiliary subtree root.
export function classifyCaptureMesh(mesh, { desc, robotOfBody, excludedObjects = new Set() }) {
  const no = (reason) => result("capture-ownership", "excluded", reason);
  if (!mesh?.isMesh) return no("not-mesh");
  for (let node = mesh; node; node = node.parent) {
    if (!node.visible) return no("hidden");
    if (excludedObjects.has(node)) return no("excluded-subtree");
    for (const flag of ["flex", "rotor", "water", "overlay", "helper"]) {
      if (node.userData?.[flag]) return no(flag);
    }
  }
  if (mesh.isInstancedMesh || mesh.isSkinnedMesh || Object.keys(mesh.geometry?.morphAttributes || {}).length) return no("deforming-or-instanced");
  if (!(robotOfBody instanceof Map)) return no("unknown-robot-ownership");
  const bodyId = mesh.userData?.body;
  const body = desc?.bodies?.find((b) => b.id === bodyId);
  const geom = desc?.geoms?.find((g) => g.id === mesh.userData?.geom);
  if (!body || !geom || geom.body !== bodyId) return no("unknown-geometry-ownership");
  if (robotOfBody.has(bodyId)) return no("robot");
  if ((desc.rotors || []).some((r) => r.body === bodyId)) return no("rotor");
  if (body.dynamic !== false) return no(body.dynamic === true ? "dynamic" : "unknown-body-motion");
  if (body.mocap === true || (Number.isInteger(body.mocap_id) && body.mocap_id >= 0)) return no("mocap");
  if (!Number.isFinite(desc.visibleGroupLimit) || !Number.isInteger(geom.group)
      || geom.group >= desc.visibleGroupLimit || geom.group < 0) return no("collision-or-unknown-group");
  const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  if (!materials.length || materials.some((m) => !m || !m.visible || !Number.isFinite(m.opacity)
      || m.opacity < 1 || m.transparent || m.alphaTest > 0 || m.alphaHash || m.transmission > 0)) return no("non-opaque");
  if (materials.some((m) => m.isShaderMaterial)) return no("unknown-shader-opacity");
  return result("capture-ownership", "eligible", "visible-fixed-opaque", { body: bodyId, geom: geom.id });
}

// bounds contains world AABBs of all finite receivers, including moving objects.
// Infinite planes and overlays must be omitted by the caller. Camera is deliberately absent.
export function fitReflectionProbe(center, bounds) {
  const no = (reason) => result("probe-bounds", "invalid", reason);
  if (!finite3(center)) return no("nonfinite-anchor");
  if (!Array.isArray(bounds) || !bounds.length) return no("empty-receivers");
  if (bounds.some((b) => !validBox(b))) return no("invalid-receiver-bounds");
  const half = [0, 0, 0];
  for (const box of bounds) for (let i = 0; i < 3; i++) half[i] = Math.max(half[i], Math.abs(box.min[i] - center[i]), Math.abs(box.max[i] - center[i]));
  if (half.every((x) => x === 0)) return no("degenerate-receivers");
  const min = half.map((x, i) => center[i] - x - 0.10);
  const max = half.map((x, i) => center[i] + x + 0.10);
  const far = Math.hypot(...max.map((x, i) => x - min[i])) + 1;
  if (!Number.isFinite(far) || min.some((x, i) => x >= max[i])) return no("unrepresentable-bounds");
  return result("probe-bounds", "valid", "symmetric-receiver-bounds", { center: [...center], min, max, near: 0.01, far, margin: 0.10 });
}

// CPU reference for the shader's box-proxy parallax approximation, not a scene ray trace.
export function projectReflectionRay(point, direction, probe) {
  const no = (status, reason) => result("box-projection", status, reason);
  if (!finite3(point) || !finite3(direction) || !finite3(probe?.center) || !validBox(probe)
      || probe.min.some((x, i) => x >= probe.max[i])) return no("invalid", "nonfinite-or-degenerate-input");
  if (point.some((x, i) => x < probe.min[i] || x > probe.max[i])) return no("outside", "receiver-outside-box");
  const length = Math.hypot(...direction);
  if (!Number.isFinite(length) || length === 0) return no("invalid", "zero-or-invalid-direction");
  const ray = direction.map((x) => x / length);
  let distance = Infinity;
  for (let i = 0; i < 3; i++) {
    if (ray[i] === 0) continue;
    distance = Math.min(distance, ((ray[i] > 0 ? probe.max : probe.min)[i] - point[i]) / ray[i]);
  }
  const hit = point.map((x, i) => x + ray[i] * distance);
  const projected = hit.map((x, i) => x - probe.center[i]);
  if (!finite3(hit) || Math.hypot(...projected) === 0) return no("invalid", "invalid-exit");
  return result("box-projection", "valid", "box-proxy-exit", { hit, direction: projected, distance });
}

function signature(mesh) {
  const geometry = mesh.geometry;
  const p = geometry?.attributes?.position;
  const index = geometry?.index;
  // InterleavedBufferAttribute forwards needsUpdate to its backing data buffer.
  return [geometry, p, p?.version, p?.data, p?.data?.version, p?.count, index, index?.version, index?.count,
    geometry?.drawRange?.start, geometry?.drawRange?.count, ...mesh.matrixWorld.elements];
}

// Explicit invalidate(mesh) is required after raw array writes without needsUpdate.
// Normals/UVs do not enter solid containment. Cache lifetime belongs to the controller.
export function createSolidGeometryCache() {
  let snapshots = new WeakMap();
  return {
    snapshot(mesh) {
      const key = signature(mesh);
      const previous = snapshots.get(mesh);
      if (previous && key.every((value, i) => value === previous.key[i])) return previous.value;
      const value = worldBounds(mesh);
      snapshots.set(mesh, { key, value });
      return value;
    },
    invalidate(mesh) { snapshots.delete(mesh); },
    clear() { snapshots = new WeakMap(); },
  };
}

function worldBounds(mesh) {
  const geometry = mesh.geometry, position = geometry?.attributes?.position;
  if (!position || position.itemSize !== 3 || !position.count) return { reason: "missing-triangles" };
  if (mesh.isInstancedMesh || mesh.isSkinnedMesh || Object.keys(geometry.morphAttributes || {}).length) return { reason: "unsupported-deformation" };
  if (!mesh.matrixWorld.elements.every(Number.isFinite) || Math.abs(mesh.matrixWorld.determinant()) === 0) return { reason: "invalid-transform" };
  // Recomputed on geometry/version changes only; this cheap broad phase avoids allocating
  // triangles or constructing topology for distant scenery. Caller owns needsUpdate.
  geometry.computeBoundingBox();
  const box = geometry.boundingBox.clone().applyMatrix4(mesh.matrixWorld);
  if (!finite3(tuple(box.min)) || !finite3(tuple(box.max))) return { reason: "nonfinite-triangle" };
  return { box, triangles: null };
}

function worldTriangles(mesh) {
  const geometry = mesh.geometry, position = geometry?.attributes?.position;
  const no = (reason) => ({ reason, triangles: [], box: null });
  if (!position || position.itemSize !== 3 || !position.count) return no("missing-triangles");
  if (mesh.isInstancedMesh || mesh.isSkinnedMesh || Object.keys(geometry.morphAttributes || {}).length) return no("unsupported-deformation");
  if (!mesh.matrixWorld.elements.every(Number.isFinite) || Math.abs(mesh.matrixWorld.determinant()) === 0) return no("invalid-transform");
  const index = geometry.index;
  const count = index ? index.count : position.count;
  if (count % 3 || geometry.drawRange.start !== 0 || geometry.drawRange.count < count) return no("partial-or-invalid-triangles");
  const vertices = [], box = new THREE.Box3();
  for (let i = 0; i < position.count; i++) {
    const v = new THREE.Vector3().fromBufferAttribute(position, i).applyMatrix4(mesh.matrixWorld);
    if (!finite3(tuple(v))) return no("nonfinite-triangle");
    vertices.push(v); box.expandByPoint(v);
  }
  const triangles = [];
  let degenerate = 0;
  for (let i = 0; i < count; i += 3) {
    const ids = [0, 1, 2].map((j) => index ? index.getX(i + j) : i + j);
    if (ids.some((id) => !Number.isInteger(id) || !vertices[id])) return no("invalid-index");
    const t = new THREE.Triangle(...ids.map((id) => vertices[id]));
    if (t.getArea() === 0) { degenerate++; continue; }
    triangles.push(t);
  }
  if (!triangles.length) return no("degenerate-triangles");
  return { triangles, box, degenerate, topology: null };
}

// Weld only for edge incidence; point/triangle distances and solid angles use original
// transformed vertices. Tolerance accounts for float32 seams, and is disclosed in receipts.
function topology(snapshot) {
  if (snapshot.topology) return snapshot.topology;
  const extent = snapshot.box.getSize(new THREE.Vector3()).length();
  const tolerance = Math.max(1e-10, extent * 1e-7);
  const cells = new Map(), vertices = [], edges = new Map(), parent = snapshot.triangles.map((_, i) => i);
  const root = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  const vertexId = (v) => {
    const cell = tuple(v).map((x, i) => Math.floor((x - snapshot.box.min.getComponent(i)) / tolerance));
    for (let x = -1; x <= 1; x++) for (let y = -1; y <= 1; y++) for (let z = -1; z <= 1; z++) {
      for (const id of cells.get(`${cell[0] + x},${cell[1] + y},${cell[2] + z}`) || []) {
        if (vertices[id].distanceTo(v) <= tolerance) return id;
      }
    }
    const id = vertices.length, key = cell.join(",");
    vertices.push(v);
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key).push(id);
    return id;
  };
  let reason = null;
  snapshot.triangles.forEach((t, i) => {
    const ids = [t.a, t.b, t.c].map(vertexId);
    if (new Set(ids).size !== 3) reason = "weld-collapsed-triangle";
    for (let j = 0; j < 3; j++) {
      const a = ids[j], b = ids[(j + 1) % 3], key = a < b ? `${a},${b}` : `${b},${a}`;
      if (!edges.has(key)) edges.set(key, []);
      const entries = edges.get(key);
      if (entries.length) parent[root(i)] = root(entries[0].triangle);
      entries.push({ triangle: i, sign: a < b ? 1 : -1 });
    }
  });
  const invalid = new Set();
  for (const entries of edges.values()) {
    if (entries.length !== 2 || entries[0].sign === entries[1].sign) {
      for (const e of entries) invalid.add(root(e.triangle));
    }
  }
  const components = new Map();
  snapshot.triangles.forEach((t, i) => {
    const id = root(i);
    if (!components.has(id)) components.set(id, { triangles: [], box: new THREE.Box3(), closed: !invalid.has(id) });
    const c = components.get(id);
    c.triangles.push(t); c.box.expandByPoint(t.a).expandByPoint(t.b).expandByPoint(t.c);
  });
  snapshot.topology = { components: [...components.values()], tolerance, reason };
  return snapshot.topology;
}

function solidAngle(t, p) {
  const a = t.a.clone().sub(p), b = t.b.clone().sub(p), c = t.c.clone().sub(p);
  const al = a.length(), bl = b.length(), cl = c.length();
  return 2 * Math.atan2(a.dot(b.clone().cross(c)), al * bl * cl + a.dot(b) * cl + b.dot(c) * al + c.dot(a) * bl);
}

function qualifyMesh(anchor, mesh, near, cache) {
  const no = (status, reason, details = {}) => result("anchor-solid", status, reason, { geom: mesh.userData?.geom ?? null, ...details });
  const snapshot = cache.snapshot(mesh);
  if (snapshot.reason) return no("unknown", snapshot.reason);
  const point = new THREE.Vector3().fromArray(anchor);
  const clipBox = new THREE.Box3(point.clone().addScalar(-near), point.clone().addScalar(near));
  // AABB overlap is only a broad phase. An assembly's empty space remains eligible.
  if (!mesh.userData?.isPlane && !snapshot.box.intersectsBox(clipBox)) return no("clear", "outside-mesh-bounds");
  if (!snapshot.triangles) Object.assign(snapshot, worldTriangles(mesh));
  if (snapshot.reason) return no("unknown", snapshot.reason);
  if (mesh.userData?.isPlane) {
    // Declared planes are infinite surfaces; validate coplanarity and transformed normal.
    const first = snapshot.triangles[0], normal = first.getNormal(new THREE.Vector3());
    const distance = normal.dot(point.clone().sub(first.a));
    const tolerance = Math.max(1e-10, snapshot.box.getSize(new THREE.Vector3()).length() * 1e-7);
    if (snapshot.triangles.some((t) => [t.a, t.b, t.c].some((v) => Math.abs(normal.dot(v.clone().sub(first.a))) > tolerance))) return no("unknown", "nonplanar-declared-plane");
    const support = near * (Math.abs(normal.x) + Math.abs(normal.y) + Math.abs(normal.z));
    return Math.abs(distance) <= support + tolerance ? no("blocked", "near-clip-intersects-plane", { distance }) : no("clear", "plane-clearance", { distance });
  }
  for (const triangle of snapshot.triangles) {
    if (clipBox.intersectsTriangle(triangle)) return no("blocked", "near-clip-intersects-triangle");
  }
  const structure = topology(snapshot);
  const details = { weldTolerance: structure.tolerance, components: structure.components.length, degenerateTriangles: snapshot.degenerate };
  if (structure.reason) return no("unknown", structure.reason, details);
  let unknown = false;
  const inside = [];
  for (const component of structure.components) {
    if (!component.box.containsPoint(point)) continue;
    if (!component.closed) { unknown = true; continue; }
    const winding = component.triangles.reduce((sum, t) => sum + solidAngle(t, point), 0) / (4 * Math.PI);
    if (!Number.isFinite(winding) || Math.abs(winding - Math.round(winding)) > 1e-5 || Math.abs(winding) > 1.00001) { unknown = true; continue; }
    if (Math.abs(winding) > 0.5) inside.push(winding);
  }
  if (inside.some((x) => x > 0) && inside.some((x) => x < 0)) return no("unknown", "nested-or-overlapping-opposed-shells", { ...details, winding: inside });
  if (inside.length) return no("blocked", "inside-closed-component", { ...details, winding: inside });
  return unknown ? no("unknown", "open-nonmanifold-or-ambiguous-component", details) : no("clear", "outside-closed-components", details);
}

// Caller supplies only meshes already eligible for capture, after updateMatrixWorld(true).
// Closed components use winding independent of normals. Open geometry whose AABB contains
// the point is unknown. Near clipping uses the union of the six cubemap near planes: the
// axis-aligned cube of half-width near. Geometry touching or inside it is blocked.
export function qualifyProbeAnchor(anchor, meshes, { near = 0.01, cache = createSolidGeometryCache() } = {}) {
  if (!finite3(anchor) || !Number.isFinite(near) || near <= 0 || !Array.isArray(meshes)) return result("probe-anchor", "invalid", "invalid-anchor-or-near");
  const receipts = meshes.map((mesh) => qualifyMesh(anchor, mesh, near, cache));
  const blocked = receipts.find((r) => r.status === "blocked"), unknown = receipts.find((r) => r.status === "unknown");
  return result("probe-anchor", blocked ? "blocked" : unknown ? "unknown" : "clear", blocked?.reason ?? unknown?.reason ?? "fixed-opaque-solids-clear", {
    anchor: [...anchor], near, receipts, method: "transformed-triangles-closed-component-winding-and-near-cube",
  });
}
