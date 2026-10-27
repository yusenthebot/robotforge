// How things look: a studio light environment that every surface reflects, and the surface
// presets behind the `rf_*` material names the robot and prop builders give their parts.
import * as THREE from "three";
import { KEY_POSITION } from "./lighting-fit.js";

// A soft photo studio, built from glowing panels and prefiltered once into an environment map.
// MuJoCo is Z-up, so "overhead" is +Z here.
export function studioEnvironment(renderer) {
  const env = new THREE.Scene();
  const room = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(0.42, 0.44, 0.47), side: THREE.BackSide })
  );
  room.scale.set(40, 40, 24);
  room.position.set(0, 0, 9);
  env.add(room);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshBasicMaterial({ color: new THREE.Color(0.18, 0.18, 0.19) }));
  floor.position.z = -2.9;
  env.add(floor);
  const panel = (w, h, pos, strength, tint = [1, 1, 1]) => {
    const color = new THREE.Color(tint[0] * strength, tint[1] * strength, tint[2] * strength);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }));
    mesh.position.set(...pos);
    mesh.up.set(0, 0, 1);
    mesh.lookAt(0.001, 0.001, 0);
    env.add(mesh);
  };
  panel(12, 7, KEY_POSITION, 9, [1, 0.97, 0.92]); // key softbox, front left, warm
  panel(10, 10, [1, 1, 16], 3.2); // overhead fill
  panel(7, 4, [10, 6, 6], 5, [0.9, 0.95, 1]); // rim light, back right, cool
  panel(16, 2.5, [0, 14, 4], 1.6); // a long strip behind, for a clean edge highlight
  const pmrem = new THREE.PMREMGenerator(renderer);
  const texture = pmrem.fromScene(env, 0.04).texture;
  pmrem.dispose();
  env.traverse((o) => {
    o.geometry?.dispose();
    o.material?.dispose();
  });
  return texture;
}

// Surface presets. The geom's own color tints each one.
const PRESETS = {
  rf_shell: { physical: true, roughness: 0.34, metalness: 0, clearcoat: 0.55, clearcoatRoughness: 0.28 },
  rf_dark: { roughness: 0.55, metalness: 0.12 },
  rf_metal: { roughness: 0.32, metalness: 1 },
  // Parts built from aluminium keep their color: anodised metal with a light seal coat, lit a little more by the
  // studio (a metal takes its brightness from reflections, and at plain strength a blue part turned navy).
  rf_aluminum: { physical: true, roughness: 0.3, metalness: 0.72, clearcoat: 0.2, clearcoatRoughness: 0.3, env: 1.45 },
  // Carbon fibre: a woven layer under a glossy clear coat (the weave is drawn in the shader, see addWeave).
  rf_carbon: { physical: true, roughness: 0.42, metalness: 0.1, clearcoat: 1, clearcoatRoughness: 0.06, weave: true },
  rf_rubber: { roughness: 0.92, metalness: 0 },
  rf_glass: { physical: true, roughness: 0.06, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.04 },
  rf_led: { roughness: 0.3, metalness: 0, glow: 2.2 },
  rf_wood: { roughness: 0.58, metalness: 0, texture: "wood", textureStrength: 0.65 },
  rf_bark: { roughness: 0.96, metalness: 0, texture: "bark", textureStrength: 0.65 },
  rf_foliage: { physical: true, roughness: 0.78, metalness: 0, sheen: 0.35 },
  rf_stone: { roughness: 0.9, metalness: 0, texture: "stone", textureStrength: 0.45 },
  rf_fabric: { physical: true, roughness: 0.95, metalness: 0, sheen: 0.6, texture: "cloth" },
  rf_ceramic: { physical: true, roughness: 0.18, metalness: 0, clearcoat: 0.5, clearcoatRoughness: 0.15 },
  rf_paint: { roughness: 0.6, metalness: 0 },
};
const DEFAULT = { roughness: 0.5, metalness: 0.05 };

// Imported models (Menagerie, URDFs) name their materials freely; read the common words.
const GUESSES = [
  [/metal|steel|alumin|chrome|silver|joint/i, "rf_metal"],
  [/rubber|tire|tyre|foot|feet|sole/i, "rf_rubber"],
  [/glass|lens|visor|camera/i, "rf_glass"],
  [/black|dark|carbon/i, "rf_dark"],
  [/white|shell|body|plastic|cover/i, "rf_shell"],
  [/wood/i, "rf_wood"],
];

function namedPreset(materialName) {
  if (!materialName) return null;
  // Robots put into a world get their name as a prefix: "dog/rf_shell".
  const base = materialName.slice(materialName.lastIndexOf("/") + 1);
  return Object.hasOwn(PRESETS, base) ? PRESETS[base] : null;
}

export function presetFor(materialName) {
  const named = namedPreset(materialName);
  if (named) return named;
  if (!materialName) return DEFAULT;
  for (const [pattern, preset] of GUESSES) if (pattern.test(materialName)) return PRESETS[preset];
  return DEFAULT;
}

// A material the scene description carries (`surfaces`, from the asset engine) is turned into the same shape as a
// preset above, so a plugin's new material reaches the browser without a change here.
function presetFromSurface(t) {
  const texture = t.texture && t.texture !== "none" ? t.texture : null;
  return {
    physical: (t.clearcoat ?? 0) > 0 || (t.sheen ?? 0) > 0,
    roughness: t.roughness,
    metalness: t.metalness,
    clearcoat: t.clearcoat ?? 0,
    clearcoatRoughness: t.clearcoatRoughness ?? 0,
    sheen: t.sheen ?? 0,
    glow: t.glow ?? 0,
    env: t.env ?? 1,
    texture,
    textureScale: t.textureScale ?? 1,
    textureStrength: t.textureStrength ?? 1,
  };
}

function nativePreset(values) {
  if (!values) return null;
  // A partial author definition gets a neutral counterpart, never guessed coat/grain/metal.
  const p = { roughness: 0.5, metalness: 0 };
  let found = false;
  for (const field of ["roughness", "metalness"]) {
    const value = values[field];
    if (Number.isFinite(value) && value >= 0 && value <= 1) { p[field] = value; found = true; }
  }
  return found ? p : null;
}

export function surfaceMaterial(materialName, color, opacity, surfaces, native) {
  const surface = surfaces && Object.hasOwn(surfaces, materialName) ? surfaces[materialName] : null;
  const p = surface ? presetFromSurface(surface) : namedPreset(materialName) ?? nativePreset(native) ?? presetFor(materialName);
  const transparent = opacity < 1;
  const common = {
    color,
    roughness: p.roughness,
    metalness: p.metalness,
    transparent,
    opacity,
    depthWrite: !transparent,
    envMapIntensity: p.env ?? 1,
  };
  const mat = p.physical
    ? new THREE.MeshPhysicalMaterial({
        ...common,
        clearcoat: p.clearcoat ?? 0,
        clearcoatRoughness: p.clearcoatRoughness ?? 0,
        sheen: p.sheen ?? 0,
        sheenColor: color,
      })
    : new THREE.MeshStandardMaterial(common);
  if (p.glow) {
    mat.emissive = color.clone();
    mat.emissiveIntensity = p.glow;
  }
  if (p.weave || p.texture === "weave" || p.texture === "cloth") addWeave(mat, p.textureScale, p.textureStrength, p.texture === "cloth");
  else if (p.texture) addGrain(mat, p.texture, p.textureScale, p.textureStrength);
  addEnvironmentScale(mat, p.env ?? 1);
  return mat;
}

function addEnvironmentScale(mat, scale) {
  if (!Number.isFinite(scale) || scale < 0) throw new Error("Surface env must be finite and non-negative");
  if (scale === 1) return;
  // Three replaces a material's intensity with scene.environmentIntensity for implicit maps.
  // Multiply inside IBL instead, retaining scene lighting, sensor overrides and context restoration.
  const chunk = THREE.ShaderChunk.envmap_physical_pars_fragment;
  if ((chunk.match(/\benvMapIntensity\b/g) || []).length !== 2) throw new Error("Unsupported environment shader chunk");
  const compile = mat.onBeforeCompile, key = mat.customProgramCacheKey();
  mat.onBeforeCompile = (shader, renderer) => {
    compile.call(mat, shader, renderer);
    shader.uniforms.rfEnvMultiplier = {
      // Explicit maps already use the material's own intensity; avoid applying it twice.
      get value() { return mat.envMap ? 1 : mat.envMapIntensity; },
    };
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>\nuniform float rfEnvMultiplier;")
      .replace("#include <envmap_physical_pars_fragment>",
        chunk.replace(/\benvMapIntensity\b/g, "(envMapIntensity * rfEnvMultiplier)"));
  };
  mat.customProgramCacheKey = () => `${key}-rf-env-scale-v1`;
}

// MuJoCo may rotate mesh vertices into a principal-axis frame. Keep fibres along the
// visible board's length, rather than assuming that its local X is still the long axis.
export function orientWoodGrain(material, geometry) {
  const axis = material.userData.woodGrainAxis;
  if (!axis) return;
  if (!geometry.boundingBox) geometry.computeBoundingBox();
  const { min, max } = geometry.boundingBox;
  const lengths = [max.x - min.x, max.y - min.y, max.z - min.z];
  const longest = lengths.indexOf(Math.max(...lengths));
  axis.set(0, 0, 0).setComponent(longest, 1);
}

// Procedural surface grain, drawn from the mesh's own coordinates like the weave: no UVs and no image. Each kind
// gives a small height field; its slope tilts the normal (a bump) and a little of it darkens and roughens the
// valleys. Where one feature is smaller than a pixel the effect fades out instead of shimmering.
//   brushed: fine streaks along the part's z axis.   speckle: orange peel or micro-blasting.
//   rubber: coarser grain.                           knurl: a diamond cross-hatch round the z axis.
const GRAIN_PITCH = { brushed: 0.00045, speckle: 0.0012, rubber: 0.0007, knurl: 0.0016, wood: 0.012, bark: 0.018, stone: 0.025 }; // m, one feature
const GRAIN_BUMP = { brushed: 0.16, speckle: 0.5, rubber: 0.9, knurl: 1.4, wood: 0.18, bark: 0.65, stone: 0.45 }; // normal tilt at strength 1
const GRAIN_HEIGHT = {
  brushed: "vec3 grain = rfNoiseSample(vec3(q.x, q.y, q.z * 0.03)) * 0.7 + rfNoiseSample(vec3(q.x * 2.3, q.y * 2.3, q.z * 0.07)) * 0.3;",
  speckle: "vec3 grain = rfNoiseSample(q) * 0.65 + rfNoiseSample(q * 2.7) * 0.35;",
  rubber: "vec3 grain = rfNoiseSample(q) * 0.5 + rfNoiseSample(q * 2.1) * 0.3 + rfNoiseSample(q * 4.4) * 0.2;",
  wood: "vec3 along = rfWoodAxis * dot(q, rfWoodAxis), across = q - along; vec3 grain = vec3(0.15, 0.0, 0.0) + 0.6 * rfNoiseSample(along * 0.025 + across * 0.75) + 0.25 * rfNoiseSample(along * 0.1 + across * 2.8);",
  bark: "vec3 grain = rfNoiseSample(vec3(q.xy, q.z * 0.08)) * 0.7 + rfNoiseSample(q * 2.0) * 0.3;",
  stone: "vec3 grain = rfNoiseSample(q) * 0.8 + rfNoiseSample(q * 3.0) * 0.2;",
  knurl:
    "float ang = atan(vGrainPos.y, vGrainPos.x) * length(vGrainPos.xy) / pitch;" +
    "float zz = vGrainPos.z / pitch; float h = 0.5 + 0.5 * sin(ang + zz) * sin(ang - zz);" +
    "vec3 grain = rfFilterSample(h, 0.5, max(fwidth(2.0 * ang), fwidth(2.0 * zz)) / 6.28318);",
};

// Height derivatives and position derivatives must keep the same units. Normalizing each position
// derivative independently makes bump strength shrink as DPR rises. Rescale all three together to
// avoid tiny determinants, preserving the actual surface gradient and the double-sided convention.
export const METRIC_BUMP = `
vec3 rfMetricBump(vec3 pos, vec3 n, vec2 dH, float faceDir) {
  vec3 sx = dFdx(pos), sy = dFdy(pos);
  float spacing = max(length(sx), length(sy));
  if (spacing < 1e-20) return n;
  sx /= spacing; sy /= spacing; dH /= spacing;
  vec3 r1 = cross(sy, n), r2 = cross(n, sx);
  float det = dot(sx, r1) * faceDir;
  if (abs(det) < 1e-6) return n;
  vec3 grad = sign(det) * (dH.x * r1 + dH.y * r2);
  return normalize(abs(det) * n - grad);
}`;

function addGrain(mat, kind, scale = 1, strength = 1) {
  if (!GRAIN_PITCH[kind] || strength === 0) return;
  const pitch = (GRAIN_PITCH[kind] * scale).toFixed(6);
  const bump = (GRAIN_BUMP[kind] * strength).toFixed(4);
  const shade = (0.22 * strength).toFixed(4);
  if (kind === "wood") mat.userData.woodGrainAxis = new THREE.Vector3(1, 0, 0);
  mat.onBeforeCompile = (shader) => {
    if (kind === "wood") shader.uniforms.rfWoodAxis = { value: mat.userData.woodGrainAxis };
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vGrainPos;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvGrainPos = position;");
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vGrainPos;
${kind === "wood" ? "uniform vec3 rfWoodAxis;" : ""}
uint rfPcg(uint v) { uint state = v * 747796405u + 2891336453u; uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u; return (word >> 22u) ^ word; }
float rfHash(vec3 p) { uvec3 q = uvec3(ivec3(floor(p + 0.5)) + 32768); return float(rfPcg(q.x + rfPcg(q.y + rfPcg(q.z)))) / 4294967295.0; }
float rfNoise(vec3 x) {
  vec3 i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(rfHash(i), rfHash(i + vec3(1, 0, 0)), f.x), mix(rfHash(i + vec3(0, 1, 0)), rfHash(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(rfHash(i + vec3(0, 0, 1)), rfHash(i + vec3(1, 0, 1)), f.x), mix(rfHash(i + vec3(0, 1, 1)), rfHash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
// Each octave has its own footprint. Carry the unfiltered height's first derivatives separately:
// differentiating a height that already contains fwidth would ask for undefined second derivatives.
vec3 rfFilterSample(float h, float mean, float footprint) {
  float fade = 1.0 - smoothstep(0.2, 0.5, footprint);
  return vec3(mix(mean, h, fade), vec2(dFdx(h), dFdy(h)) * fade);
}
vec3 rfNoiseSample(vec3 q) {
  vec3 width = fwidth(q);
  // Value-noise features span about two lattice cells; sine phases below measure full cycles.
  return rfFilterSample(rfNoise(q), 0.5, 0.5 * max(max(width.x, width.y), width.z));
}
vec3 rfWaveSample(float phase) {
  return rfFilterSample(sin(phase), 0.0, fwidth(phase) / 6.28318);
}
${METRIC_BUMP}
float grainH;
vec2 grainSlope;`
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
{
  const float pitch = ${pitch};
  vec3 q = vGrainPos / pitch;
  ${GRAIN_HEIGHT[kind]}
  grainH = grain.x;
  grainSlope = grain.yz;
  // Keep the filtered mean tone at a distance instead of brightening back to the base color.
  ${kind === "wood"
    ? `diffuseColor.rgb *= 1.0 - ${shade} * 0.425 + ${(0.55 * strength).toFixed(4)} * (grainH - 0.575);`
    : `diffuseColor.rgb *= 1.0 - ${shade} * (1.0 - grainH);`}
}`
      )
      .replace(
        "#include <roughnessmap_fragment>",
        `#include <roughnessmap_fragment>
roughnessFactor = clamp(roughnessFactor + (0.5 - grainH) * ${(0.3 * strength).toFixed(4)}, 0.04, 1.0);`
      )
      .replace(
        "#include <normal_fragment_maps>",
        `#include <normal_fragment_maps>
// Height is 8% of a feature pitch at unit bump, independent of drawing resolution.
normal = rfMetricBump(-vViewPosition, normal, grainSlope * ${pitch} * ${bump} * 0.08, faceDirection);`
      );
  };
  mat.customProgramCacheKey = () => `rf-grain-metric-${kind}-${pitch}-${bump}-${shade}${kind === "wood" ? "-fibres-v3" : ""}`;
}

// A 2x2 twill drawn from the mesh's own coordinates: no UVs and no texture, so it costs a few
// shader instructions. Each face takes the plane its normal points out of. Where one tow is
// smaller than a pixel the weave fades to its average instead of shimmering.
const WEAVE_PITCH = 0.004; // m, the width of one tow (a 12K carbon twill, the coarse weave robot plates show)

function addWeave(mat, scale = 1, strength = 1, cloth = false) {
  if (strength === 0) return;
  const pitch = ((cloth ? 0.0015 : WEAVE_PITCH) * scale).toFixed(6);
  mat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vWeavePos;\nvarying vec3 vWeaveNormal;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvWeavePos = position;\nvWeaveNormal = normal;");
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vWeavePos;
varying vec3 vWeaveNormal;
float weaveTow;
float weaveFade;`
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
{
  vec3 an = abs(vWeaveNormal);
  vec2 plane = an.x > an.y && an.x > an.z ? vWeavePos.yz : (an.y > an.z ? vWeavePos.xz : vWeavePos.xy);
  vec2 p = plane / ${pitch};
  vec2 cell = floor(p);
  vec2 f = fract(p);
  bool alongX = ${cloth ? "mod(cell.x + cell.y, 2.0) < 1.0" : "mod(cell.x + cell.y, 4.0) < 2.0"};
  float across = alongX ? f.y : f.x;
  float ridge = sin(3.14159265 * across); // a tow is a flat ridge with a dark gap to the next one
  weaveFade = clamp((0.4 - max(fwidth(p.x), fwidth(p.y))) / 0.23, 0.0, 1.0); // full at 6 px a tow, gone at 2.5
  weaveTow = alongX ? 1.0 : 0.0;
  diffuseColor.rgb *= mix(1.0, mix(${cloth ? "0.98, mix(0.88, 1.0, ridge)" : "0.86, mix(0.68, 1.04, ridge)"}, weaveFade), ${strength.toFixed(4)});
}`
      )
      .replace(
        "#include <roughnessmap_fragment>",
        `#include <roughnessmap_fragment>
roughnessFactor = clamp(roughnessFactor * mix(1.0, weaveTow > 0.5 ? 0.82 : 1.18, weaveFade * ${(strength * (cloth ? 0.25 : 1)).toFixed(4)}), 0.04, 1.0);`
      );
  };
  mat.customProgramCacheKey = () => `rf-weave-${pitch}-${strength}-${cloth}`;
}
