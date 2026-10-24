// Shared direct/reflected illumination and one world-fitted shadow map. No camera-dependent lighting or extra pass.
import * as THREE from "three";
import { KEY_POSITION, fitShadowBounds } from "./lighting-fit.js";
import { studioEnvironment } from "./look.js";

export function createLighting(scene, renderer) {
  scene.environment = studioEnvironment(renderer);
  scene.environmentIntensity = 0.85;
  const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x3a3328, 0.35);
  hemi.position.set(0, 0, 1);
  const sun = new THREE.DirectionalLight(0xfff6ec, 1.9);
  sun.castShadow = true;
  // Reuse rigid-world depth until a caster or its framing changes. Sensor pictures
  // keep their existing global autoUpdate gate and cannot consume this dirty flag.
  sun.shadow.autoUpdate = false;
  sun.shadow.needsUpdate = true;
  sun.position.fromArray(KEY_POSITION);
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.up.set(0, 0, 1);
  scene.add(hemi, sun, sun.target);

  const direction = new THREE.Vector3().fromArray(KEY_POSITION).normalize();
  const basis = new THREE.Matrix4().lookAt(direction, new THREE.Vector3(), new THREE.Vector3(0, 0, 1));
  const inverse = basis.clone().invert();
  const point = new THREE.Vector3(), projected = new THREE.Vector3(), matrix = new THREE.Matrix4();
  const staticBounds = new THREE.Box3(), bounds = new THREE.Box3();
  let groups = [], fixed = [], root = null, floor = null, fitted = null, fixedReady = false;

  function include(box, transform, into) {
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
      point.set(x, y, z).applyMatrix4(transform);
      projected.copy(point).applyMatrix4(inverse);
      into.expandByPoint(projected);
      // Include receivers below airborne/tabletop casters, without widening the light's XY footprint.
      if (floor?.alongLight > 0.001) {
        const distance = floor.plane.distanceToPoint(point) / floor.alongLight;
        if (distance > 0) into.min.z = Math.min(into.min.z, projected.z - distance);
      }
    }
  }

  function bind(world, nodes = [], desc = {}) {
    sun.shadow.needsUpdate = true;
    // These worlds also animate below body transforms; retain their existing redraws.
    sun.shadow.autoUpdate = !!(desc.rotors?.length || desc.flexes?.length);
    root = world; groups = []; fixed = []; floor = null; fitted = null; fixedReady = false;
    if (!world) return;
    world.updateMatrixWorld(true);
    const byBody = new Map();
    const inverseBody = new THREE.Matrix4(), local = new THREE.Box3();
    world.traverse((mesh) => {
      if (!mesh.isMesh || !mesh.castShadow || !mesh.visible) return;
      if (mesh.userData.flex) { groups.push({ node: mesh, flex: true }); return; }
      const id = mesh.userData.body;
      const node = nodes[id];
      if (!node) return;
      if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
      inverseBody.copy(node.matrixWorld).invert();
      matrix.multiplyMatrices(inverseBody, mesh.matrixWorld);
      local.copy(mesh.geometry.boundingBox).applyMatrix4(matrix);
      let group = byBody.get(id);
      if (!group) {
        group = { node, box: new THREE.Box3(), dynamic: !!desc.bodies?.[id]?.dynamic, transform: new THREE.Matrix4() };
        byBody.set(id, group);
      }
      group.box.union(local);
    });
    // Visual rotor animation changes descendants without a new physics pose. Enclose the entire swept disc.
    for (const rotor of desc.rotors || []) {
      const group = byBody.get(rotor.body);
      if (!group) continue;
      point.fromArray(rotor.center);
      local.set(point, point).expandByScalar(rotor.radius + 0.01);
      group.box.union(local);
    }
    for (const group of byBody.values()) (group.dynamic ? groups : fixed).push(group);
    const plane = desc.geoms?.find((g) => g.type === "plane");
    if (plane) {
      const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(new THREE.Quaternion(plane.quat[1], plane.quat[2], plane.quat[3], plane.quat[0]));
      const localPlane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, new THREE.Vector3().fromArray(plane.pos));
      floor = { local: localPlane, plane: new THREE.Plane(), node: nodes[plane.body], transform: null, alongLight: 0 };
    }
  }

  function update(force = false) {
    if (!root) return;
    root.updateMatrixWorld(true);
    if (floor?.node && (!floor.transform || !floor.transform.equals(floor.node.matrixWorld))) {
      floor.plane.copy(floor.local).applyMatrix4(floor.node.matrixWorld);
      floor.alongLight = floor.plane.normal.dot(direction);
      floor.transform = floor.node.matrixWorld.clone();
      fixedReady = false;
    }
    // Imported mocap bodies can move despite having no DOFs. Compare cached transforms rather than treating
    // the streamed dynamic flag as a guarantee that a body can never move.
    if (fixed.some((g) => !g.transform.equals(g.node.matrixWorld))) fixedReady = false;
    const changedFixed = !fixedReady;
    if (!fixedReady) {
      staticBounds.makeEmpty();
      for (const g of fixed) {
        include(g.box, g.node.matrixWorld, staticBounds);
        g.transform.copy(g.node.matrixWorld);
      }
      fixedReady = true;
    }
    if (!force && !changedFixed && fitted && groups.every((g) => !g.flex && g.transform.equals(g.node.matrixWorld))) return;
    sun.shadow.needsUpdate = true; // A caster can move without changing the fitted bounds.
    bounds.copy(staticBounds);
    for (const g of groups) {
      if (g.flex) {
        g.node.geometry.computeBoundingBox();
        include(g.node.geometry.boundingBox, g.node.matrixWorld, bounds);
      } else {
        include(g.box, g.node.matrixWorld, bounds);
        g.transform.copy(g.node.matrixWorld);
      }
    }
    const next = fitShadowBounds({ min: bounds.min.toArray(), max: bounds.max.toArray() }, force ? null : fitted);
    if (!next || next === fitted) return;
    fitted = next;
    sun.position.set(next.x, next.y, next.z).applyMatrix4(basis);
    sun.target.position.set(next.x, next.y, next.z - 1).applyMatrix4(basis);
    const camera = sun.shadow.camera;
    camera.left = -next.width / 2; camera.right = next.width / 2;
    camera.bottom = -next.height / 2; camera.top = next.height / 2;
    camera.near = next.near; camera.far = next.far;
    camera.updateProjectionMatrix();
    sun.shadow.normalBias = next.normalBias;
    sun.shadow.bias = next.bias;
  }

  function restore() {
    // The studio is GPU-generated; unlike image textures it cannot re-upload CPU pixels.
    const previous = scene.environment;
    scene.environment = studioEnvironment(renderer);
    previous?.dispose();
    sun.shadow.needsUpdate = true;
  }

  return { sun, bind, update, restore, state: () => fitted };
}
