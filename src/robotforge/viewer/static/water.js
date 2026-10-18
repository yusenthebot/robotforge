// Water: the surface the physics feels (sim/water.py sends its waves in the scene), drawn with the same waves on
// the simulation's clock, see-through when you look down into it and mirror-like at a slant, and a hazy
// blue-green world once the camera dips under it.
import * as THREE from "three";

export const SURFACE_GEOM = "water:surface"; // MuJoCo's own see-through sheet, for its renderer: hidden here
const SPACING = 0.1; // m between the vertices of the wavy surface
const MAX_SEGMENTS = 360; // per side
const NEAR = 24; // m from the scene's middle that open water is wavy; beyond, a flat sheet runs to the horizon
const FADE = 4; // m over which the waves die out toward that edge
const HORIZON = 5000; // m, half the width of the open sea: past the floor's own edge, which fog hides

// Colors are what the water tints things seen through it; `clear` is how much of the bottom shows looking straight
// down; under water `see` is how far (m) things stay visible.
const LOOKS = {
  pool: { color: 0x48c2d6, deep: 0x1b7f9c, clear: 0.66, fog: 0x3e9fb5, see: 16 },
  lake: { color: 0x34635a, deep: 0x15322c, clear: 0.34, fog: 0x2f5c53, see: 6 },
  sea: { color: 0x22607c, deep: 0x0c2c3f, clear: 0.3, fog: 0x245d76, see: 9 },
};
const SUN = [0.6, -0.8, 1.6]; // the way to the sun, as the viewer's own sun light

// What water mirrors: an open sky, pale at the horizon and deeper blue overhead, with the sun in it (its glints on
// the waves). The studio light boxes that make robots look good made the water look like a bathroom floor.
let skyMap = null;
function sky(renderer) {
  if (skyMap) return skyMap;
  const env = new THREE.Scene();
  const dome = new THREE.SphereGeometry(50, 48, 24);
  const colors = [];
  const zenith = new THREE.Color(0x5d93c9);
  const horizon = new THREE.Color(0xdfe8ef);
  const below = new THREE.Color(0x3a5260);
  const pos = dome.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const up = pos.getY(i) / 50; // the sphere is built y-up: its y is our z here
    const c = up >= 0 ? horizon.clone().lerp(zenith, Math.pow(up, 0.6)) : horizon.clone().lerp(below, Math.min(1, -up * 4));
    colors.push(c.r, c.g, c.b);
  }
  dome.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  dome.rotateX(Math.PI / 2); // y-up to z-up
  env.add(new THREE.Mesh(dome, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })));
  const sun = new THREE.Mesh(
    new THREE.SphereGeometry(2.2, 16, 8),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(14, 12.5, 10.5), toneMapped: false })
  );
  sun.position.copy(new THREE.Vector3(...SUN).normalize().multiplyScalar(40));
  env.add(sun);
  const pmrem = new THREE.PMREMGenerator(renderer);
  skyMap = pmrem.fromScene(env, 0.0).texture;
  pmrem.dispose();
  env.traverse((o) => {
    o.geometry?.dispose();
    o.material?.dispose();
  });
  return skyMap;
}

export function waterLook(water, floorMaterial) {
  if (floorMaterial === "ground_tile") return "pool";
  return water.density > 1010 ? "sea" : "lake";
}

// Height of the surface at (x, y) at time t, as the physics has it (null where there is no water).
export function surfaceAt(water, x, y, t) {
  const a = water.area;
  if (a && (x < a[0] || x > a[2] || y < a[1] || y > a[3])) return null;
  let h = water.level;
  for (const [amp, kx, ky, omega, phase] of water.waves) h += amp * Math.sin(kx * x + ky * y - omega * t + phase);
  return h;
}

function material(water, look, fade, renderer) {
  const waves = water.waves.slice(0, 3);
  while (waves.length < 3) waves.push([0, 0, 0, 0, 0]);
  const uniforms = {
    uTime: { value: 0 },
    uWaves: { value: waves.map(([amp, kx, ky, omega]) => new THREE.Vector4(amp, kx, ky, omega)) },
    uPhase: { value: new THREE.Vector3(...waves.map((w) => w[4])) },
    uFade: { value: new THREE.Vector4(...fade) },
    uClear: { value: look.clear },
    uDeep: { value: new THREE.Color(look.deep) },
    uUnder: { value: new THREE.Color(look.fog) },
  };
  const mat = new THREE.MeshStandardMaterial({
    color: look.color,
    roughness: 0.05,
    metalness: 0.0,
    transparent: true,
    opacity: 1 - look.clear,
    side: THREE.DoubleSide,
    depthWrite: false,
    envMap: renderer ? sky(renderer) : null,
    envMapIntensity: 1.0,
  });
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    const common = `#include <common>
uniform float uTime;
uniform vec4 uWaves[3];
uniform vec3 uPhase;
uniform vec4 uFade;`;
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `${common}\nvarying float vDepthView;`)
      .replace(
        "#include <beginnormal_vertex>",
        `vec2 wp = (modelMatrix * vec4(position, 1.0)).xy;
vec2 edge = uFade.zw - abs(wp - uFade.xy);
float calm = clamp(min(edge.x, edge.y) / ${FADE.toFixed(1)}, 0.0, 1.0);
float waveH = 0.0;
vec2 slope = vec2(0.0);
for (int i = 0; i < 3; i++) {
  vec4 w = uWaves[i];
  float th = w.y * wp.x + w.z * wp.y - w.w * uTime + uPhase[i];
  waveH += w.x * sin(th);
  slope += w.x * cos(th) * w.yz;
}
waveH *= calm;
slope *= calm;
vec3 objectNormal = normalize(vec3(-slope, 1.0));
#ifdef USE_TANGENT
vec3 objectTangent = vec3(1.0, 0.0, 0.0);
#endif`
      )
      .replace("#include <begin_vertex>", "vec3 transformed = vec3(position.xy, position.z + waveH);");
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
uniform float uClear;
uniform vec3 uDeep;
uniform vec3 uUnder;`
      )
      .replace(
        "#include <normal_fragment_maps>",
        `#include <normal_fragment_maps>
{
  // Fresnel: looking straight down most light comes up from the bottom; at a slant the surface mirrors the sky.
  float facing = clamp(abs(dot(normal, normalize(vViewPosition))), 0.0, 1.0);
  float mirror = pow(1.0 - facing, 3.0);
  if (gl_FrontFacing) {
    diffuseColor.rgb = mix(diffuseColor.rgb, uDeep, 0.45 * (1.0 - facing));
    diffuseColor.a = mix(1.0 - uClear, 0.97, mirror);
  } else {
    // From under water the surface is a bright, wobbly ceiling (and a mirror beyond the critical angle).
    diffuseColor.rgb = mix(uUnder * 1.6, vec3(0.85, 0.95, 1.0), 0.35 * facing);
    diffuseColor.a = mix(0.55, 0.92, mirror);
  }
}`
      );
  };
  mat.customProgramCacheKey = () => "rf-water";
  return { mat, uniforms };
}

// The water of a scene: a wavy sheet over the water's area (or round the scene's middle, with a flat sea beyond),
// in the world body's node so paths and the selection ring rest on it.
export function makeWater(water, look, center, renderer) {
  const style = LOOKS[look] ?? LOOKS.lake;
  const group = new THREE.Group();
  group.name = "water";
  let x0, y0, x1, y1;
  const open = !water.area;
  if (open) {
    [x0, y0, x1, y1] = [center[0] - NEAR, center[1] - NEAR, center[0] + NEAR, center[1] + NEAR];
  } else {
    [x0, y0, x1, y1] = water.area;
  }
  const w = x1 - x0;
  const h = y1 - y0;
  // Bounded water has no edge to calm down to: its shore hides the rim.
  const fade = open ? [(x0 + x1) / 2, (y0 + y1) / 2, w / 2, h / 2] : [0, 0, 1e6, 1e6];
  const { mat, uniforms } = material(water, style, fade, renderer);
  const sx = Math.min(MAX_SEGMENTS, Math.ceil(w / SPACING));
  const sy = Math.min(MAX_SEGMENTS, Math.ceil(h / SPACING));
  const sheet = new THREE.Mesh(new THREE.PlaneGeometry(w, h, sx, sy), mat);
  sheet.position.set((x0 + x1) / 2, (y0 + y1) / 2, water.level);
  group.add(sheet);
  if (open) {
    // Four flat strips round the wavy middle, out to the horizon (the fog hides where they end): north and south
    // the whole width, east and west between them.
    const reach = HORIZON - NEAR;
    const strips = [
      [center[0], y1 + reach / 2, 2 * HORIZON, reach],
      [center[0], y0 - reach / 2, 2 * HORIZON, reach],
      [x1 + reach / 2, (y0 + y1) / 2, reach, h],
      [x0 - reach / 2, (y0 + y1) / 2, reach, h],
    ];
    for (const [cx, cy, sw, sh] of strips) {
      const strip = new THREE.Mesh(new THREE.PlaneGeometry(sw, sh, 4, 4), mat);
      strip.position.set(cx, cy, water.level);
      group.add(strip);
    }
  }
  group.traverse((o) => {
    if (!o.isMesh) return;
    o.receiveShadow = true;
    o.castShadow = false;
    o.renderOrder = 1;
    o.userData.isPlane = true; // not part of what the camera frames, nor something to grab
    o.userData.water = true;
  });
  const under = new THREE.Fog(style.fog, 0.05, style.see);
  return { group, uniforms, water, style, under, above: null, isUnder: false, background: null };
}

// Each frame: the waves at the simulation's time, and the look under water when the camera is below the surface.
export function waterFrame(state, time, camera, scene) {
  state.uniforms.uTime.value = time;
  const p = camera.position;
  const top = surfaceAt(state.water, p.x, p.y, time);
  const under = top !== null && p.z < top - 0.01;
  if (under) {
    if (scene.fog !== state.under) {
      state.above = scene.fog; // the scene's own fog, put back on surfacing (it may have been refitted meanwhile)
      state.background = scene.background;
      scene.fog = state.under;
      scene.background = new THREE.Color(state.style.fog).multiplyScalar(0.85);
    }
  } else if (state.isUnder) {
    scene.fog = state.above;
    if (state.background) scene.background = state.background;
  }
  state.isUnder = under;
}

// Put the scene's own fog and background back (a new scene arrives while the camera is under water).
export function leaveWater(state, scene) {
  if (state?.isUnder) {
    scene.fog = state.above;
    if (state.background) scene.background = state.background;
  }
}
