// How things look: a studio light environment that every surface reflects, and the surface
// presets behind the `rf_*` material names the robot and prop builders give their parts.
import * as THREE from "three";

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
  panel(12, 7, [-7, -9, 11], 9, [1, 0.97, 0.92]); // key softbox, front left, warm
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
  // Parts built from aluminium keep their color: anodised, half metal so the color stays bright, a light seal coat.
  rf_aluminum: { physical: true, roughness: 0.3, metalness: 0.72, clearcoat: 0.2, clearcoatRoughness: 0.3 },
  // Carbon fibre: a woven layer under a glossy clear coat (the weave is drawn in the shader, see addWeave).
  rf_carbon: { physical: true, roughness: 0.42, metalness: 0.1, clearcoat: 1, clearcoatRoughness: 0.06, weave: true },
  rf_rubber: { roughness: 0.92, metalness: 0 },
  rf_glass: { physical: true, roughness: 0.06, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.04 },
  rf_led: { roughness: 0.3, metalness: 0, glow: 2.2 },
  rf_wood: { roughness: 0.72, metalness: 0 },
  rf_bark: { roughness: 0.96, metalness: 0 },
  rf_foliage: { physical: true, roughness: 0.78, metalness: 0, sheen: 0.35 },
  rf_stone: { roughness: 0.9, metalness: 0 },
  rf_fabric: { physical: true, roughness: 0.95, metalness: 0, sheen: 0.6 },
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

export function presetFor(materialName) {
  if (!materialName) return DEFAULT;
  // Robots put into a world get their name as a prefix: "dog/rf_shell".
  const base = materialName.slice(materialName.lastIndexOf("/") + 1);
  if (PRESETS[base]) return PRESETS[base];
  for (const [pattern, preset] of GUESSES) if (pattern.test(materialName)) return PRESETS[preset];
  return DEFAULT;
}

export function surfaceMaterial(materialName, color, opacity) {
  const p = presetFor(materialName);
  const transparent = opacity < 1;
  const common = {
    color,
    roughness: p.roughness,
    metalness: p.metalness,
    transparent,
    opacity,
    depthWrite: !transparent,
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
  if (p.weave) addWeave(mat);
  return mat;
}

// A 2x2 twill drawn from the mesh's own coordinates: no UVs and no texture, so it costs a few
// shader instructions. Each face takes the plane its normal points out of. Where one tow is
// smaller than a pixel the weave fades to its average instead of shimmering.
const WEAVE_PITCH = 0.004; // m, the width of one tow (a 12K carbon twill, the coarse weave robot plates show)

function addWeave(mat) {
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
  vec2 p = plane / ${WEAVE_PITCH.toFixed(5)};
  vec2 cell = floor(p);
  vec2 f = fract(p);
  bool alongX = mod(cell.x + cell.y, 4.0) < 2.0; // the tow on top steps one cell per row: a twill
  float across = alongX ? f.y : f.x;
  float ridge = sin(3.14159265 * across); // a tow is a flat ridge with a dark gap to the next one
  weaveFade = clamp((0.4 - max(fwidth(p.x), fwidth(p.y))) / 0.23, 0.0, 1.0); // full at 6 px a tow, gone at 2.5
  weaveTow = alongX ? 1.0 : 0.0;
  diffuseColor.rgb *= mix(0.86, mix(0.68, 1.04, ridge), weaveFade);
}`
      )
      .replace(
        "#include <roughnessmap_fragment>",
        `#include <roughnessmap_fragment>
roughnessFactor *= mix(1.0, weaveTow > 0.5 ? 0.82 : 1.18, weaveFade); // tows at right angles catch light differently`
      );
  };
  mat.customProgramCacheKey = () => "rf-weave";
}
