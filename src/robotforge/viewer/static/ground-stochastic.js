// Four translated samples per logical lookup, blended with a quintic partition.
// Natural rock only: 24 logical fetches replace the metric ground's 6.
import * as THREE from "three";

const KEY = "square-quintic-rock-v1";
const fail = (message) => { throw new Error(`Stochastic rock: ${message}`); };
const once = (source, anchor, replacement) => {
  if (source.split(anchor).length !== 2) fail(`expected exactly one shader anchor: ${anchor}`);
  return source.replace(anchor, replacement);
};
const SAMPLE = `
vec2 rfRockDx, rfRockDy;
uint rfRockPcg(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
vec2 rfRockOffset(ivec2 id) {
  uvec2 p = uvec2(id + ivec2(32768));
  uint seed = rfRockPcg(p.x + rfRockPcg(p.y));
  return vec2(rfRockPcg(seed), rfRockPcg(seed + 1013904223u)) / 4294967296.0;
}
vec4 rfRockSample(sampler2D tex, vec2 uv, vec2 dx, vec2 dy) {
  ivec2 cell = ivec2(floor(uv));
  vec2 f = fract(uv);
  vec2 w = f * f * f * (10.0 - 15.0 * f + 6.0 * f * f);
  vec4 a = textureGrad(tex, uv + rfRockOffset(cell), dx, dy);
  vec4 b = textureGrad(tex, uv + rfRockOffset(cell + ivec2(1, 0)), dx, dy);
  vec4 c = textureGrad(tex, uv + rfRockOffset(cell + ivec2(0, 1)), dx, dy);
  vec4 d = textureGrad(tex, uv + rfRockOffset(cell + ivec2(1, 1)), dx, dy);
  return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}
`;

const wrapped = new WeakSet();

// Called once when constructing a natural rock material. The existing metric
// bump hook runs first; every other ground keeps that original hook and key.
export function stochasticRock(material) {
  if (wrapped.has(material)) return material;
  const mapAnchor = "texture2D( map, vMapUv )", roughAnchor = "texture2D( roughnessMap, vRoughnessMapUv )";
  const mapChunk = once(THREE.ShaderChunk.map_fragment, mapAnchor, "rfRockSample(map, vMapUv, rfRockDx, rfRockDy)");
  const roughChunk = once(THREE.ShaderChunk.roughnessmap_fragment, roughAnchor, "rfRockSample(roughnessMap, vRoughnessMapUv, rfRockDx, rfRockDy)");
  function patch(source) {
    source = once(source, "#include <common>", `#include <common>\n${SAMPLE}`);
    source = once(source, "void main() {", "void main() {\n  rfRockDx = dFdx(vBumpMapUv);\n  rfRockDy = dFdy(vBumpMapUv);");
    source = once(source, "#include <map_fragment>", mapChunk);
    source = once(source, "#include <roughnessmap_fragment>", roughChunk);
    for (const uv of ["uv + vec2(texel, 0.0)", "uv - vec2(texel, 0.0)", "uv + vec2(0.0, texel)", "uv - vec2(0.0, texel)"]) {
      source = once(source, `texture2D(bumpMap, ${uv})`, `rfRockSample(bumpMap, ${uv}, rfRockDx, rfRockDy)`);
    }
    source = once(source, "return vec2(dot(gradient, dFdx(uv)), dot(gradient, dFdy(uv)));",
      "return vec2(dot(gradient, rfRockDx), dot(gradient, rfRockDy));");
    return source;
  }
  const hook = material.onBeforeCompile, key = material.customProgramCacheKey;
  material.onBeforeCompile = function(shader, renderer) {
    hook.call(this, shader, renderer);
    shader.fragmentShader = patch(shader.fragmentShader);
  };
  material.customProgramCacheKey = function() { return key.call(this) + ":" + KEY; };
  wrapped.add(material);
  return material;
}
