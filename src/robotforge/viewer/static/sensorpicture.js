// What a robot's camera sees, drawn in the browser with three.js from the camera's pose, so the picture works
// where the server has no OpenGL: rgb tone mapped as the view is, depth packed in 16 bits and colored like the
// server's depth pictures, segmentation in the server's palette. The picture is read back asynchronously into a
// 2D canvas in the sensor views (sensors.js), so the page never waits on the GPU for it.

import * as THREE from "three";
import { depthPicture, spanOf, steadySpan, thingColor, thingName, unpackDepth } from "./sensorkit.js";

const PIP_EVERY_MS = { rgb: 66, depth: 100, segmentation: 100, stereo: 120, event: 66, fisheye: 120 }; // the picture is drawn again this often
const EVENT_LEVELS = 4; // events one pixel gives between two pictures, at most (sensors/optics.py)
const EVENT_FLOOR = 8; // levels of 255 added before the logarithm
const CUBE_SIZE = 384; // px: each face of the fisheye's cube
const PIP_MAX = [640, 480]; // px: a bigger camera is drawn at this size in the browser

// The camera's pose now: its body's pose in this frame, times its place on the body (MuJoCo's cameras and three.js's
// both look along their -z with +y up, so the quaternion carries over as it is).
function placeCamera(node, e, cam, shift = 0) {
  const [w, x, y, z] = e.local_quat;
  cam.position.fromArray(e.local_pos).applyQuaternion(node.quaternion).add(node.position);
  cam.quaternion.copy(node.quaternion).multiply(new THREE.Quaternion(x, y, z, w));
  if (shift) cam.position.addScaledVector(new THREE.Vector3(1, 0, 0).applyQuaternion(cam.quaternion), shift); // its right
}

// What kind of picture a camera's entry asks for: a fisheye lens is its own, and so is a stereo pair, an event camera.
const kindOf = (e) => (e.lens === "fisheye" ? "fisheye" : e.mode);

// A fisheye lens (r = f theta, the equidistant one real ones come close to) looked up in a cube map of the scene
// drawn from the camera's place; `look` turns a ray of the lens into the world's.
const FISHEYE_VERTEX = "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }";
const FISHEYE_FRAGMENT = `
  uniform samplerCube cube;
  uniform mat3 look;
  uniform vec2 size;
  uniform float perRadian;
  uniform float reach;
  varying vec2 vUv;
  void main() {
    vec2 d = (vUv - 0.5) * size;
    float theta = length(d) / perRadian;
    if (theta > reach) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
    float phi = atan(d.y, d.x);
    vec3 ray = look * vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), -cos(theta));
    gl_FragColor = vec4(textureCube(cube, ray).rgb, 1.0);
  }`;

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

export const picture = {
  api: null, // the page's handles (sensors.js's initSensors)
  onPicture: null, // called with the camera's entry after each picture (the legend under it)
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
  lumas: new Map(), // event cameras: the last frame each drew, as brightness
  left: null, // a stereo pair: the left picture, while the right one is drawn
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
    this.api.world()?.traverse((o) => {
      if (o.isMesh) this.meshes.push(o);
    });
    return this.meshes;
  },
  flatMaterial(mesh) {
    if (mesh.userData.thing === undefined) {
      const desc = this.api.desc();
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
  draw(e, canvas, pass = 0) {
    const now = performance.now();
    const kind = kindOf(e);
    if (pass === 0 && (this.busy || now - this.last < PIP_EVERY_MS[kind] || document.hidden)) return;
    const node = this.api.bodyNode(e.body);
    if (!node) return;
    if (pass === 0) this.last = now;
    const t0 = performance.now();
    const w = Math.min(e.size[0], PIP_MAX[0]);
    const h = Math.round((w * e.size[1]) / e.size[0]);
    this.setSize(w, Math.min(h, PIP_MAX[1]));
    const cam = this.cam;
    cam.fov = kind === "fisheye" ? 90 : e.fov_deg;
    cam.aspect = kind === "fisheye" ? 1 : e.size[0] / e.size[1];
    cam.near = 0.01;
    cam.far = e.mode === "rgb" || e.mode === "event" || e.mode === "stereo" ? 200 : e.range_m * 1.02;
    cam.updateProjectionMatrix();
    // A stereo pair is drawn twice, from a lens each side of where its camera sits (the right one second).
    placeCamera(node, e, cam, kind === "stereo" ? (pass === 0 ? -1 : 1) * (e.baseline_m ?? 0.12) / 2 : 0);
    cam.updateMatrixWorld();
    const renderer = this.api.renderer;
    const scene = this.api.scene;
    const world = this.api.world();
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
      if (kind === "fisheye") {
        scene.fog = null;
        this.fisheyeInto(e, cam, renderer, scene);
        const tone = this.toneQuad();
        tone.material.uniforms.light.value = this.light.texture;
        renderer.setRenderTarget(this.target);
        renderer.render(tone.scene, tone.camera);
      } else if (e.mode === "rgb" || e.mode === "event" || e.mode === "stereo") {
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
      if (kind === "stereo" && pass === 0) {
        this.left = this.bytes.slice(); // the left lens's picture, kept while the right one is drawn
        this.draw(e, canvas, 1);
        return;
      }
      this.renders++;
      this.develop(e, canvas, pass);
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
  develop(e, canvas, pass = 0) {
    const [W, H] = this.size;
    const across = kindOf(e) === "stereo" ? 2 * W : W; // a stereo pair side by side
    if (canvas.width !== across || canvas.height !== H) {
      canvas.width = across;
      canvas.height = H;
    }
    // Another camera picked builds a new canvas the same size: drawing on with the old one's context left it blank.
    if (this.ctx?.canvas !== canvas) this.ctx = canvas.getContext("2d");
    if (!this.image || this.image.width !== across || this.image.height !== H) this.image = this.ctx.createImageData(across, H);
    const rows = this.image.data;
    const src = this.bytes;
    if (kindOf(e) === "stereo") {
      for (let y = 0; y < H; y++) {
        const from = (H - 1 - y) * W * 4;
        rows.set(this.left.subarray(from, from + W * 4), y * across * 4);
        rows.set(src.subarray(from, from + W * 4), (y * across + W) * 4);
      }
    } else if (e.mode === "event") {
      this.events(e, rows, src, W, H);
    } else if (e.mode === "depth") {
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
    this.onPicture?.(e);
  },
  // An event camera's picture: this frame against the last one this camera drew, in log brightness. A pixel that
  // brightened by more than the contrast threshold is an ON event (red), one that darkened an OFF event (blue); the rest
  // is gray. Paused, the frames are the same and nothing fires, as a real one's would not.
  events(e, rows, src, W, H) {
    const threshold = e.threshold ?? 0.15;
    const luma = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = ((H - 1 - y) * W + x) * 4;
        luma[y * W + x] = 0.299 * src[i] + 0.587 * src[i + 1] + 0.114 * src[i + 2];
      }
    }
    const before = this.lumas.get(e.name);
    this.lumas.set(e.name, luma);
    let on = 0;
    let off = 0;
    for (let k = 0; k < W * H; k++) {
      let color = [122, 122, 122];
      if (before && before.length === luma.length) {
        const change = Math.log(luma[k] + EVENT_FLOOR) - Math.log(before[k] + EVENT_FLOOR);
        const count = Math.min(EVENT_LEVELS, Math.floor(Math.abs(change) / threshold));
        if (count > 0) {
          color = change > 0 ? [232, 64, 56] : [56, 112, 236];
          if (change > 0) on += count;
          else off += count;
        }
      }
      rows[k * 4] = color[0];
      rows[k * 4 + 1] = color[1];
      rows[k * 4 + 2] = color[2];
      rows[k * 4 + 3] = 255;
    }
    this.eventCount = { on, off };
  },
  // The scene seen from the lens's place in all directions, then the lens's pixels looked up in it (see FISHEYE_*).
  fisheyeInto(e, cam, renderer, scene) {
    if (!this.cubeTarget) {
      this.cubeTarget = new THREE.WebGLCubeRenderTarget(CUBE_SIZE, { type: THREE.HalfFloatType });
      this.cubeCamera = new THREE.CubeCamera(0.01, 200, this.cubeTarget);
      this.fisheye = new THREE.ShaderMaterial({
        uniforms: { cube: { value: this.cubeTarget.texture }, look: { value: new THREE.Matrix3() }, size: { value: new THREE.Vector2() }, perRadian: { value: 1 }, reach: { value: 1 } },
        vertexShader: FISHEYE_VERTEX,
        fragmentShader: FISHEYE_FRAGMENT,
        depthTest: false,
        depthWrite: false,
      });
      const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.fisheye);
      quad.frustumCulled = false;
      this.fisheyeScene = new THREE.Scene();
      this.fisheyeScene.add(quad);
      this.fisheyeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    }
    this.cubeCamera.position.copy(cam.position);
    this.cubeCamera.updateMatrixWorld();
    this.cubeCamera.update(renderer, scene);
    const [W, H] = this.size;
    const half = (e.fov_deg * Math.PI) / 360;
    const u = this.fisheye.uniforms;
    u.look.value.setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(cam.quaternion));
    u.size.value.set(W, H);
    u.perRadian.value = Math.min(W, H) / 2 / half;
    u.reach.value = half;
    renderer.setRenderTarget(this.light);
    renderer.render(this.fisheyeScene, this.fisheyeCamera);
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
