// One bounded, fixed-scene specular probe. Physics, authored surfaces and diffuse IBL
// remain owned by the original viewer. All generated GPU objects have one owner here.
import * as THREE from "three";
import { attachSceneReflectionMaterial } from "./look.js";
import { classifyCaptureMesh, fitReflectionProbe, createSolidGeometryCache, qualifyProbeAnchor } from "./scene-reflection-geometry.js";

const SIZE = 128;
const now = () => performance.now();
const materialsOf = (mesh) => Array.isArray(mesh.material) ? mesh.material : [mesh.material];
const visible = (mesh) => {
  for (let o = mesh; o; o = o.parent) if (!o.visible) return false;
  return true;
};

export function createSceneReflections({ scene, renderer, camera, lighting }) {
  const born = now(), cache = createSolidGeometryCache();
  let enabled = true, lost = false, disposed = false, capturing = false;
  let rendererRecoveryRequired = false, shaderRecoveryRequired = false;
  let world = null, desc = null, ownership = null, excluded = new Set();
  let meshes = [], fixed = [], detach = [], active = null, building = null;
  let generation = 0, revision = 0, dirty = false, posedReady = false, sceneStarted = born;
  let status = "unbound", reason = "no-world", qualification = null, probe = null;
  let firstFrameSubmitted = null, firstInteractive = null, fullReady = null, buildDuration = null, captureDraws = 0;
  let captures = 0, allocated = 0, released = 0, exclusions = [], receivers = [];
  let preset = lighting.preset();
  const uniforms = {
    rfSceneEnvironment: { value: null },
    rfProbeCenter: { value: new THREE.Vector3() },
    rfProbeMin: { value: new THREE.Vector3() },
    rfProbeMax: { value: new THREE.Vector3() },
  };

  function release(target) { if (target) { target.dispose(); released++; } }
  function dropActive() {
    uniforms.rfSceneEnvironment.value = null;
    release(active); active = null; revision++;
  }
  function fail(error) {
    generation++; dirty = false; fullReady = null; status = rendererRecoveryRequired ? "fatal" : "disabled"; reason = String(error?.message ?? error);
    dropActive();
  }
  const context = {
    isMainPass: (r, s, c) => enabled && !lost && !capturing && !!active && !materialFailure()
      && preset === lighting.preset() && r === renderer && s === scene && c === camera && r.getRenderTarget() === null,
    revision: () => revision,
    uniforms: () => uniforms,
    failed: (error) => {
      // Three may cache the safe fallback under the attempted local program key.
      // Do not retry that variant on this page, even if the world reuses a material.
      shaderRecoveryRequired = true;
      fail(`unsupported-shader-reload-required: ${error?.message ?? error}`);
    },
  };
  function materialFailure() { return status === "disabled"; }

  function invalidate(cause = "fixed-scene-edit") {
    if (disposed || !world || rendererRecoveryRequired || shaderRecoveryRequired) return;
    generation++; revision++; dirty = true; status = enabled ? "dirty" : "off"; reason = cause;
    fullReady = null;
    // No old-world/preset reflection is presented as current while rebuilding.
    dropActive();
  }

  function unbind() {
    generation++; dirty = false; posedReady = false;
    dropActive();
    for (const undo of detach) undo();
    detach = []; meshes = []; fixed = []; receivers = []; exclusions = [];
    cache.clear(); world = null; desc = null; ownership = null; probe = null; qualification = null;
    if (!rendererRecoveryRequired && !shaderRecoveryRequired) { status = "unbound"; reason = "no-world"; }
  }

  function bind(root, description, robotOfBody, options = {}) {
    unbind();
    if (!root || disposed || rendererRecoveryRequired || shaderRecoveryRequired) return;
    world = root; desc = description; ownership = robotOfBody;
    excluded = options.excludedObjects ?? new Set();
    sceneStarted = now(); firstFrameSubmitted = null; firstInteractive = null; fullReady = null; buildDuration = null;
    root.traverse((o) => { if (o.isMesh) meshes.push(o); });
    const seen = new Set();
    receivers = [];
    for (const mesh of meshes) {
      const body = desc.bodies?.find((b) => b.id === mesh.userData.body);
      if (body?.dynamic === false && !ownership?.has(body.id)) fixed.push({ mesh, transform: null });
      for (const material of materialsOf(mesh)) {
        if (seen.has(material)) continue;
        seen.add(material);
        const undo = attachSceneReflectionMaterial(material, context);
        receivers.push({ material: material?.uuid ?? null,
          status: undo ? "eligible" : "excluded",
          reason: undo ? "standard-or-physical" : material?.envMap ? "explicit-env-map" : "unsupported-material" });
        if (undo) detach.push(undo);
      }
    }
    preset = lighting.preset();
    invalidate("world-bind");
  }

  function posed() {
    if (!world || disposed || rendererRecoveryRequired || shaderRecoveryRequired) return;
    // Called after the existing pose/shadow update, which already updates matrices.
    // No scene traversal, topology reconstruction or dynamic-body comparison per frame.
    let changed = false;
    for (const item of fixed) {
      if (item.transform && !item.transform.equals(item.mesh.matrixWorld)) changed = true;
      if (!item.transform) item.transform = item.mesh.matrixWorld.clone();
      else item.transform.copy(item.mesh.matrixWorld);
    }
    if (changed) invalidate("fixed-transform");
    posedReady = true;
  }

  function prepare() {
    world.updateMatrixWorld(true);
    const eligible = [], bounds = [], byGeom = new Map(desc.geoms.map((g) => [g.id, g]));
    exclusions = [];
    for (const mesh of meshes) {
      const receipt = classifyCaptureMesh(mesh, { desc, robotOfBody: ownership, excludedObjects: excluded });
      exclusions.push({ ...receipt, object: mesh.uuid, geom: mesh.userData.geom ?? null });
      if (receipt.status === "eligible") eligible.push(mesh);
      const geom = byGeom.get(mesh.userData.geom);
      if (!geom || geom.type === "plane" || !visible(mesh)) continue;
      let auxiliary = false;
      for (let o = mesh; o; o = o.parent) {
        if (excluded.has(o) || o.userData.flex || o.userData.rotor || o.userData.water || o.userData.overlay || o.userData.helper) auxiliary = true;
      }
      if (auxiliary) continue;
      mesh.geometry.computeBoundingBox();
      if (!mesh.geometry.boundingBox) return { error: "missing-receiver-bounds" };
      const box = mesh.geometry.boundingBox.clone().applyMatrix4(mesh.matrixWorld);
      bounds.push({ min: box.min.toArray(), max: box.max.toArray() });
    }
    probe = fitReflectionProbe(desc.stat?.center, bounds);
    if (probe.status !== "valid") return { error: probe.reason };
    if (!eligible.length) return { error: "no-eligible-fixed-scenery" };
    qualification = qualifyProbeAnchor(probe.center, eligible, { near: probe.near, cache });
    if (qualification.status !== "clear") return { error: `anchor-${qualification.status}: ${qualification.reason}` };
    return { eligible };
  }

  function capture(eligible, expectedGeneration) {
    const start = now(), hidden = [], shadows = [];
    const before = {
      target: renderer.getRenderTarget(), face: renderer.getActiveCubeFace(), mip: renderer.getActiveMipmapLevel(),
      viewport: renderer.getViewport(new THREE.Vector4()), scissor: renderer.getScissor(new THREE.Vector4()),
      scissorTest: renderer.getScissorTest(), clear: renderer.getClearColor(new THREE.Color()), alpha: renderer.getClearAlpha(),
      autoClear: renderer.autoClear, autoClearColor: renderer.autoClearColor,
      autoClearDepth: renderer.autoClearDepth, autoClearStencil: renderer.autoClearStencil,
      toneMapping: renderer.toneMapping, exposure: renderer.toneMappingExposure, outputColorSpace: renderer.outputColorSpace,
      xr: renderer.xr.enabled, shadowAuto: renderer.shadowMap.autoUpdate, shadowNeeds: renderer.shadowMap.needsUpdate,
      background: scene.background, backgroundIntensity: scene.backgroundIntensity,
      backgroundBlurriness: scene.backgroundBlurriness, backgroundRotation: scene.backgroundRotation.clone(), fog: scene.fog,
    };
    let generator = null, target = null, renderFailure = null, committed = false;
    const guards = [];
    const originalRender = renderer.render, originalDraw = renderer.renderBufferDirect;
    const recordFailure = (error) => { renderFailure ??= error; };
    function guard(object, name) {
      const original = object[name];
      if (typeof original !== "function") return;
      const replacement = function(...args) {
        try { return original.apply(this, args); }
        catch (error) { recordFailure(error); }
      };
      object[name] = replacement;
      guards.push([object, name, original, replacement]);
    }
    const guardedDraw = function(...args) {
      if (renderFailure || shaderRecoveryRequired) return;
      // This public draw boundary does not push Three's render-state/list stacks.
      // Catch here so its caller completes the normal render() epilogue and pops them.
      try { return originalDraw.apply(this, args); }
      catch (error) { recordFailure(error); }
    };
    const guardedRender = function(...args) {
      if (renderFailure || shaderRecoveryRequired) return;
      captureDraws++;
      try { return originalRender.apply(this, args); }
      catch (error) {
        // An escape outside the guarded callbacks/draw boundary may leave private
        // Three render stacks dirty. Public state restoration cannot certify recovery.
        rendererRecoveryRequired = true;
        recordFailure(new Error(`renderer-reload-required: ${error?.message ?? error}`));
      }
    };
    captureDraws = 0; capturing = true; revision++;
    try {
      const allowed = new Set(eligible);
      scene.traverse((o) => {
        if ((o.isMesh || o.isLine || o.isPoints || o.isSprite) && !allowed.has(o)) {
          hidden.push([o, o.visible]); o.visible = false;
        }
      });
      for (const mesh of eligible) { shadows.push([mesh, mesh.receiveShadow]); mesh.receiveShadow = false; }
      renderer.shadowMap.autoUpdate = false; renderer.shadowMap.needsUpdate = false;
      renderer.xr.enabled = false;
      renderer.toneMapping = THREE.NoToneMapping; renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
      scene.background = lighting.studio(); scene.backgroundIntensity = 1; scene.backgroundBlurriness = 0;
      scene.backgroundRotation.copy(scene.environmentRotation); scene.fog = null;
      generator = new THREE.PMREMGenerator(renderer);
      // Vendored r178 does not expose the output target until all passes succeed.
      // Observe this instance's allocation so even pre-render failures retain an owner.
      const allocate = generator._allocateTargets;
      if (typeof allocate !== "function") throw new Error("unsupported-pmrem-target-ownership");
      generator._allocateTargets = function() {
        target = allocate.call(this); building = target; allocated++; return target;
      };
      // Keep exceptions inside the callback boundary, before Three can unwind its
      // render() without popping the private render-state/list stacks. Always invoke
      // each hook with its original receiver/arguments and restore by identity.
      guard(scene, "onBeforeRender"); guard(scene, "onAfterRender");
      const guardedMaterials = new Set();
      for (const mesh of eligible) {
        guard(mesh, "onBeforeRender"); guard(mesh, "onAfterRender");
        for (const material of materialsOf(mesh)) {
          if (guardedMaterials.has(material)) continue;
          guardedMaterials.add(material);
          guard(material, "onBeforeRender");
          // Compile/cache-key hooks execute inside guardedDraw. Do not replace them:
          // Material's default cache key reads onBeforeCompile.toString().
        }
      }
      // After a failure, skip subsequent draws but let PMREM dispose its temporary
      // background mesh. An unguarded render escape is fatal, never a successful retry.
      renderer.renderBufferDirect = guardedDraw;
      renderer.render = guardedRender;
      target = generator.fromScene(scene, 0, probe.near, probe.far,
        { size: SIZE, position: new THREE.Vector3().fromArray(probe.center) });
      if (renderFailure) throw renderFailure;
      if (lost || renderer.getContext().isContextLost()) throw new Error("context-lost-during-capture");
      if (generation !== expectedGeneration || disposed || !enabled) return;
      if (target.width !== 384 || target.height !== 512) throw new Error("unsupported-pmrem-atlas-size");
      release(active); active = target; building = null; committed = true;
      uniforms.rfSceneEnvironment.value = target.texture;
      uniforms.rfProbeCenter.value.fromArray(probe.center);
      uniforms.rfProbeMin.value.fromArray(probe.min); uniforms.rfProbeMax.value.fromArray(probe.max);
      dirty = false; status = "built"; reason = "awaiting-main-submission";
      captures++;
    } catch (error) {
      if (rendererRecoveryRequired || generation === expectedGeneration) fail(`capture-failed: ${error?.message ?? error}`);
    } finally {
      if (renderer.render === guardedRender) renderer.render = originalRender;
      if (renderer.renderBufferDirect === guardedDraw) renderer.renderBufferDirect = originalDraw;
      for (const [object, name, original, replacement] of guards.reverse())
        if (object[name] === replacement) object[name] = original;
      for (const [mesh, value] of shadows) mesh.receiveShadow = value;
      for (const [object, value] of hidden) object.visible = value;
      scene.background = before.background; scene.fog = before.fog;
      scene.backgroundIntensity = before.backgroundIntensity; scene.backgroundBlurriness = before.backgroundBlurriness;
      scene.backgroundRotation.copy(before.backgroundRotation);
      renderer.toneMapping = before.toneMapping; renderer.toneMappingExposure = before.exposure;
      renderer.outputColorSpace = before.outputColorSpace; renderer.xr.enabled = before.xr;
      renderer.autoClear = before.autoClear; renderer.autoClearColor = before.autoClearColor;
      renderer.autoClearDepth = before.autoClearDepth; renderer.autoClearStencil = before.autoClearStencil;
      renderer.shadowMap.autoUpdate = before.shadowAuto; renderer.shadowMap.needsUpdate = before.shadowNeeds;
      renderer.setClearColor(before.clear, before.alpha);
      renderer.setRenderTarget(before.target, before.face, before.mip);
      renderer.setViewport(before.viewport); renderer.setScissor(before.scissor); renderer.setScissorTest(before.scissorTest);
      generator?.dispose();
      if (!committed) release(target);
      building = null; capturing = false; revision++;
      buildDuration = now() - start;
    }
  }

  function afterMainRender() {
    if (!world || !posedReady || disposed || lost || rendererRecoveryRequired || shaderRecoveryRequired || renderer.getContext().isContextLost()) return;
    if (firstFrameSubmitted === null) firstFrameSubmitted = now() - sceneStarted;
    if (status === "built") {
      status = "ready"; reason = "fixed-scene-specular"; fullReady = now() - sceneStarted;
    }
    if (preset !== lighting.preset()) { preset = lighting.preset(); invalidate("lighting-preset"); }
    if (!enabled || !dirty || capturing || renderer.getRenderTarget() !== null || !lighting.shadowReady()) {
      if (firstInteractive === null && !rendererRecoveryRequired) firstInteractive = now() - sceneStarted;
      return;
    }
    const expected = generation;
    status = "qualifying";
    const start = now();
    try {
      const prepared = prepare();
      if (generation !== expected) return;
      if (prepared.error) { fail(prepared.error); return; }
      capture(prepared.eligible, expected);
    } catch (error) { if (generation === expected) fail(`qualification-failed: ${error?.message ?? error}`); }
    finally {
      buildDuration = now() - start;
      if (firstInteractive === null && !rendererRecoveryRequired) firstInteractive = now() - sceneStarted;
    }
  }

  function setEnabled(value) {
    if (disposed || rendererRecoveryRequired || shaderRecoveryRequired || enabled === !!value) return;
    enabled = !!value; revision++;
    if (enabled) invalidate("enabled");
    else { generation++; dropActive(); status = "off"; reason = "explicit-off"; fullReady = null; }
  }
  function contextLost() { lost = true; invalidate("context-lost"); }
  function contextRestored() { lost = false; invalidate("context-restored"); }
  function dispose() { if (disposed) return; unbind(); disposed = true; status = "disposed"; }
  function state() {
    const targetSummary = (target) => target ? {
      width: target.width, height: target.height, depthBuffer: target.depthBuffer,
      texture: { uuid: target.texture.uuid, name: target.texture.name, type: target.texture.type,
        format: target.texture.format, colorSpace: target.texture.colorSpace, mapping: target.texture.mapping },
    } : null;
    return {
      candidate: "scene-specular-v1", enabled, status, reason, generation, revision, capturing, rendererRecoveryRequired, shaderRecoveryRequired,
      worldVersion: desc?.version ?? null, preset, probe, qualification, exclusions, receivers,
      size: SIZE, atlas: [384, 512], nominalTargetBytes: 2359296,
      activeTargets: active ? 1 : 0, buildingTargets: building ? 1 : 0, allocatedTargets: allocated, releasedTargets: released,
      activeTarget: targetSummary(active), buildingTarget: targetSummary(building),
      activeUniformMatches: uniforms.rfSceneEnvironment.value === (active ? active.texture : null),
      captures, captureDraws, timing: { sceneStarted, firstFrameSubmitted, firstInteractive, fullReady, buildDuration, sinceController: now() - born },
      approximation: "fixed-opaque-only; box-projected; no-baked-direct-shadows; studio-diffuse",
    };
  }
  return { bind, unbind, posed, afterMainRender, invalidate, setEnabled, contextLost, contextRestored, state, dispose };
}
