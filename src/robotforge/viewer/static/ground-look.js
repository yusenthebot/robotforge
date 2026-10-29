// Small, deterministic surface tiles. Color and height/roughness are generated once per style,
// on demand; each mesh owns its GPU textures so changing a world can release all of them.
import * as THREE from "three";
import { METRIC_BUMP } from "./look.js";

const SIZE = 256;
const PROFILES = {
  grass: { roughness: 0.98, bump: 0.0025 },
  wood: { roughness: 0.48, bump: 0.0008 },
  concrete: { roughness: 0.88, bump: 0.0012 },
  dirt: { roughness: 0.98, bump: 0.003 },
  sand: { roughness: 0.94, bump: 0.0015 },
  rock: { roughness: 0.9, bump: 0.003 },
  snow: { roughness: 0.88, bump: 0.002 },
  tile: { roughness: 0.3, bump: 0.0006 },
  checker: { roughness: 0.95, bump: 0 },
};
const tiles = new Map();

// Old scene descriptions need only pattern/tile. Keep those valid; known ground names refine the look.
export function groundStyle(materialName, look) {
  const name = (materialName || "").split("/").pop().replace(/^ground_/, "");
  if (Object.hasOwn(PROFILES, name)) return name;
  return ({ planks: "wood", tiles: "tile", checker: "checker" })[look?.pattern] || "concrete";
}

// Periodic value noise: filtered features and matching edges prevent seams and distant shimmer.
function noise(cells, rand, rows = cells) {
  const grid = Float32Array.from({ length: cells * rows }, rand);
  const at = (x, y) => grid[(y % rows) * cells + (x % cells)];
  const smooth = (t) => t * t * (3 - 2 * t);
  const out = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    const fy = y / SIZE * rows, iy = Math.floor(fy), ty = smooth(fy - iy);
    for (let x = 0; x < SIZE; x++) {
      const fx = x / SIZE * cells, ix = Math.floor(fx), tx = smooth(fx - ix);
      const a = at(ix, iy) * (1 - tx) + at(ix + 1, iy) * tx;
      const b = at(ix, iy + 1) * (1 - tx) + at(ix + 1, iy + 1) * tx;
      out[y * SIZE + x] = a * (1 - ty) + b * ty;
    }
  }
  return out;
}

const fract = (v) => v - Math.floor(v);
const edge = (v) => Math.min(fract(v), 1 - fract(v));
const byte = (v) => Math.round(Math.max(0, Math.min(1, v)) * 255);

// Average a thin joint over a source texel instead of snapping it to a wide dark pixel.
function stripeCoverage(distance, halfWidth, footprint) {
  return Math.max(0, Math.min(distance + footprint, halfWidth) - Math.max(distance - footprint, -halfWidth)) / (2 * footprint);
}

function surfaceTile(style) {
  if (!Object.hasOwn(PROFILES, style)) style = "concrete";
  if (tiles.has(style)) return tiles.get(style);
  let seed = 7;
  const rand = () => ((seed = seed * 16807 % 2147483647) / 2147483647);
  const broad = noise(8, rand), fine = noise(48, rand);
  const boards = Array.from({ length: 8 }, rand);
  const figure = style === "wood" ? noise(3, rand, 24) : null;
  const fibres = style === "wood" ? noise(6, rand, 96) : null;
  // Centre only concrete's colour fields on their actual samples. Preserve the old
  // decoded-linear average as the new neutral base; relief and roughness stay separate.
  let concreteColor = null;
  if (style === "concrete") {
    const histogram = new Uint32Array(256);
    let broadMean = 0, fineMean = 0;
    for (let i = 0; i < broad.length; i++) {
      broadMean += broad[i]; fineMean += fine[i];
      histogram[byte(0.92 + 0.06 * broad[i] + 0.02 * fine[i])]++;
    }
    let linearMean = 0;
    for (let value = 0; value < histogram.length; value++) {
      const c = new THREE.Color(value / 255, value / 255, value / 255).convertSRGBToLinear();
      linearMean += c.r * histogram[value];
    }
    linearMean /= broad.length;
    const base = new THREE.Color(linearMean, linearMean, linearMean).convertLinearToSRGB().r;
    concreteColor = { base, broadMean: broadMean / broad.length, fineMean: fineMean / fine.length };
  }
  const make = () => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = SIZE;
    return canvas;
  };
  const color = make(), detail = make();
  const ctx = color.getContext("2d"), dctx = detail.getContext("2d");
  const rgb = ctx.createImageData(SIZE, SIZE), data = dctx.createImageData(SIZE, SIZE);
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    const i = y * SIZE + x, u = x / SIZE, v = y / SIZE;
    const b = broad[i], f = fine[i];
    let shade = 0.92 + 0.06 * b + 0.02 * f, height = 0.4 * b + 0.6 * f;
    let roughness = PROFILES[style].roughness;
    if (style === "wood") {
      const row = Math.floor(v * 8);
      // Long, irregular fibres and broader figure share the object wood's 0.575 mean.
      // Shift each board independently; the fields themselves wrap at the tile boundary.
      const sx = (x + Math.floor(boards[row] * SIZE)) % SIZE, sample = y * SIZE + sx;
      const grain = 0.15 + 0.6 * figure[sample] + 0.25 * fibres[sample];
      const joint = Math.max(stripeCoverage(edge(v * 8), 1 / 150, 4 / SIZE),
        stripeCoverage(edge(u - boards[row]), 1 / 1200, 0.5 / SIZE)); // 2 mm across the 1.2 m tile
      height = 0.62 + 0.18 * (grain - 0.575) + 0.025 * (f - 0.5);
      shade = 0.875 + boards[row] * 0.075 + 0.18 * (grain - 0.575) + 0.01 * (f - 0.5);
      roughness += 0.05 + 0.06 * (0.575 - grain);
      height += (0.18 - height) * joint;
      shade += (0.68 - shade) * joint;
      roughness += (0.8 - roughness) * joint;
    } else if (style === "concrete") {
      shade = concreteColor.base + 0.02 * (b - concreteColor.broadMean) + 0.04 * (f - concreteColor.fineMean);
    } else if (style === "tile") {
      const grout = Math.min(edge(u * 2), edge(v * 2)) < 0.018;
      shade = grout ? 0.72 : 0.97 + b * 0.025;
      height = grout ? 0.15 : 0.8;
      roughness = grout ? 0.96 : 0.28 + 0.06 * f;
    } else if (style === "grass") {
      const blades = Math.sin((v * 54 + u * 16 + b) * Math.PI * 2);
      shade = 0.84 + 0.11 * b + 0.035 * f + 0.015 * blades;
      height = 0.45 * b + 0.4 * f + 0.075 * blades;
    } else if (style === "sand") {
      const ripple = Math.sin((v * 12 + u * 2 + b * 0.35) * Math.PI * 2);
      height = 0.5 + ripple * 0.16 + f * 0.12;
      shade = 0.94 + ripple * 0.018 + f * 0.035;
    } else if (style === "rock") {
      height = b * 0.8 + f * 0.2;
      shade = 0.83 + 0.12 * b + 0.045 * f;
    } else if (style === "snow") {
      height = b * 0.85 + f * 0.15;
      shade = 0.96 + 0.035 * b;
    } else if (style === "dirt") {
      shade = 0.86 + 0.09 * b + 0.045 * f;
    } else if (style === "checker") {
      shade = (Math.floor(u * 2) + Math.floor(v * 2)) % 2 ? 0.94 : 1;
    }
    roughness = Math.min(1, roughness + (0.5 - f) * 0.035);
    rgb.data.fill(byte(shade), i * 4, i * 4 + 3);
    rgb.data[i * 4 + 3] = 255;
    data.data[i * 4] = byte(height); // bump reads red; roughness reads green
    data.data[i * 4 + 1] = byte(roughness);
    data.data[i * 4 + 2] = 0;
    data.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(rgb, 0, 0);
  dctx.putImageData(data, 0, 0);
  // Shared Sources let three.js reuse one GPU allocation per channel, even when meshes need different UVs.
  const result = { color, detail, colorSource: new THREE.Source(color), detailSource: new THREE.Source(detail) };
  tiles.set(style, result);
  return result;
}

export function groundMaterial(style, color, anisotropy, preparedWood = null, preparedGrass = null) {
  const tile = (style === "wood" && preparedWood) || (style === "grass" && preparedGrass) || surfaceTile(style);
  const map = new THREE.CanvasTexture(tile.color);
  map.source = tile.colorSource;
  map.colorSpace = THREE.SRGBColorSpace;
  const detail = new THREE.CanvasTexture(tile.detail);
  detail.source = tile.detailSource;
  for (const tex of [map, detail]) {
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.anisotropy = anisotropy;
    tex.needsUpdate = true;
  }
  const mat = new THREE.MeshStandardMaterial({
    color, map, roughness: 1, roughnessMap: detail, bumpMap: detail,
    bumpScale: (PROFILES[style] || PROFILES.concrete).bump, metalness: 0,
  });
  mat.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>\n${METRIC_BUMP}`)
      .replace("#include <bumpmap_pars_fragment>", `#include <bumpmap_pars_fragment>
#ifdef USE_BUMPMAP
vec2 rfTileSlope() {
  // Differentiate the stored height over one source texel, then project its UV gradient into
  // screen space. A one-screen-pixel forward difference changes the physical slope with DPR.
  const float texel = 1.0 / ${SIZE.toFixed(1)};
  vec2 uv = vBumpMapUv;
  vec2 gradient = vec2(
    texture2D(bumpMap, uv + vec2(texel, 0.0)).x - texture2D(bumpMap, uv - vec2(texel, 0.0)).x,
    texture2D(bumpMap, uv + vec2(0.0, texel)).x - texture2D(bumpMap, uv - vec2(0.0, texel)).x
  ) * (bumpScale / (2.0 * texel));
  return vec2(dot(gradient, dFdx(uv)), dot(gradient, dFdy(uv)));
}
#endif`)
      .replace("#include <normal_fragment_maps>", THREE.ShaderChunk.normal_fragment_maps.replace(
        "perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection )",
        "rfMetricBump( - vViewPosition, normal, rfTileSlope(), faceDirection )"));
  };
  mat.userData.groundSource = style === "wood" && preparedWood ? "natural-oak"
    : style === "grass" && preparedGrass ? "natural-leafy-grass" : "procedural";
  mat.customProgramCacheKey = () => "rf-ground-metric-v1";
  return mat;
}

// Use the same tiles in the chooser, without extra network requests or GPU textures.
export function groundPreview(style, rgb, detailed = true) {
  const canvas = document.createElement("canvas");
  canvas.width = 96;
  canvas.height = 64;
  canvas.setAttribute("aria-hidden", "true");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = `rgb(${rgb.map((c) => Math.round(c * 255)).join(",")})`;
  ctx.fillRect(0, 0, 96, 64);
  if (detailed) {
    ctx.globalCompositeOperation = "multiply";
    const natural = style === "wood" ? naturalWood : style === "grass" ? naturalGrass : null;
    ctx.drawImage((natural || surfaceTile(style)).color, 0, 0, 96, 64);
    const pending = !natural && (style === "wood" ? loadWood() : style === "grass" ? loadGrass() : null);
    if (pending) pending.then((tile) => {
      if (!tile) return;
      ctx.globalCompositeOperation = "source-over";
      ctx.fillRect(0, 0, 96, 64);
      ctx.globalCompositeOperation = "multiply";
      ctx.drawImage(tile.color, 0, 0, 96, 64);
    });
  }
  return canvas;
}

// Optional local images are prepared before installation. Late work fills only the
// CPU cache; an installed scene keeps the tile it selected, including its fallback.
let naturalWood = null, woodPending = null, woodWarned = false;

function woodFallback(reason) {
  if (!woodWarned) console.warn("robotforge: optional wood texture unavailable; using procedural ground", reason);
  woodWarned = true;
  return null;
}

function loadWood() {
  if (naturalWood) return Promise.resolve(naturalWood);
  if (!woodPending) {
    woodPending = import("./ground-wood.js")
      .then((module) => module.woodTiles())
      .then(({ color, detail }) => {
        naturalWood = { color, detail, colorSource: new THREE.Source(color), detailSource: new THREE.Source(detail) };
        return naturalWood;
      })
      .catch((error) => woodFallback(error?.name || "load"))
      .finally(() => { woodPending = null; });
  }
  return woodPending;
}

export async function prepareGroundTextures(desc) {
  const usesWood = desc.geoms.some((geom) => {
    const look = desc.grounds?.[geom.material];
    return look && !(geom.type === "plane" && look.pattern === "checker") && groundStyle(geom.material, look) === "wood";
  });
  if (!usesWood) return null;
  let timer;
  try {
    return await Promise.race([loadWood(), new Promise((resolve) => {
      timer = setTimeout(() => resolve(woodFallback("timeout")), 1000);
    })]);
  } finally {
    clearTimeout(timer);
  }
}

// Keep the wood preparation API and cache independent. A mixed scene prepares both
// concurrently; a late grass decode can help later scenes without changing this one.
let naturalGrass = null, grassPending = null, grassWarned = false;

function grassFallback(reason) {
  if (!grassWarned) console.warn("robotforge: optional grass texture unavailable; using procedural ground", reason);
  grassWarned = true;
  return null;
}

function loadGrass() {
  if (naturalGrass) return Promise.resolve(naturalGrass);
  if (!grassPending) {
    grassPending = import("./ground-grass.js")
      .then((module) => module.grassTiles())
      .then(({ color, detail }) => {
        naturalGrass = { color, detail, colorSource: new THREE.Source(color), detailSource: new THREE.Source(detail) };
        return naturalGrass;
      })
      .catch((error) => grassFallback(error?.name || "load"))
      .finally(() => { grassPending = null; });
  }
  return grassPending;
}

export async function prepareGrassTextures(desc) {
  const usesGrass = desc.geoms.some((geom) => {
    const look = desc.grounds?.[geom.material];
    return look && !(geom.type === "plane" && look.pattern === "checker") && groundStyle(geom.material, look) === "grass";
  });
  if (!usesGrass) return null;
  let timer;
  try {
    return await Promise.race([loadGrass(), new Promise((resolve) => {
      timer = setTimeout(() => resolve(grassFallback("timeout")), 1000);
    })]);
  } finally {
    clearTimeout(timer);
  }
}
