// Small, deterministic surface tiles. Color and height/roughness are generated once per style,
// on demand; each mesh owns its GPU textures so changing a world can release all of them.
import * as THREE from "three";

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
function noise(cells, rand) {
  const grid = Float32Array.from({ length: cells * cells }, rand);
  const at = (x, y) => grid[(y % cells) * cells + (x % cells)];
  const smooth = (t) => t * t * (3 - 2 * t);
  const out = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    const fy = y / SIZE * cells, iy = Math.floor(fy), ty = smooth(fy - iy);
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

function surfaceTile(style) {
  if (!Object.hasOwn(PROFILES, style)) style = "concrete";
  if (tiles.has(style)) return tiles.get(style);
  let seed = 7;
  const rand = () => ((seed = seed * 16807 % 2147483647) / 2147483647);
  const broad = noise(8, rand), fine = noise(48, rand);
  const boards = Array.from({ length: 8 }, rand);
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
      const grain = Math.sin((v * 64 + 0.45 * b + 0.18 * Math.sin(u * Math.PI * 2)) * Math.PI * 2);
      const joint = edge(v * 8) < 0.025 || edge(u - boards[row]) < 0.004;
      height = joint ? 0.05 : 0.64 + 0.09 * grain + 0.08 * f;
      shade = joint ? 0.72 : 0.87 + boards[row] * 0.09 + grain * 0.025 + f * 0.025;
      roughness += joint ? 0.36 : 0.1 * (1 - f);
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

export function groundMaterial(style, color, anisotropy) {
  const tile = surfaceTile(style);
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
  return new THREE.MeshStandardMaterial({
    color, map, roughness: 1, roughnessMap: detail, bumpMap: detail,
    bumpScale: (PROFILES[style] || PROFILES.concrete).bump, metalness: 0,
  });
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
    ctx.drawImage(surfaceTile(style).color, 0, 0, 96, 64);
  }
  return canvas;
}
