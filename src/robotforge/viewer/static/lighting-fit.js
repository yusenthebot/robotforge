// One dominant source for direct light and the studio's brightest reflection panel.
export const KEY_POSITION = Object.freeze([-7, -9, 11]);

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const spanStep = (x) => 2 ** (Math.ceil(Math.log2(Math.max(x, 0.25)) * 8) / 8);

// Bounds are in a fixed light basis, independent of the viewing camera. Padding and a guard band keep small
// movements from refitting the map; texel snapping keeps translation from sliding static shadow edges.
export function fitShadowBounds(bounds, previous = null, mapSize = 2048) {
  const { min, max } = bounds;
  if (![...min, ...max].every(Number.isFinite) || min.some((v, i) => v > max[i])) return null;
  if (!Number.isInteger(mapSize) || mapSize < 16) throw new Error("shadow map size must be an integer of at least 16");
  if (previous && min.every((v, i) => v >= previous.guard.min[i])
      && max.every((v, i) => v <= previous.guard.max[i])) return previous;
  const extent = max.map((v, i) => v - min[i]);
  const padding = Math.max(0.025, Math.max(extent[0], extent[1]) * 0.12);
  const width = spanStep(extent[0] + 2 * padding), height = spanStep(extent[1] + 2 * padding);
  const dx = width / mapSize, dy = height / mapSize, texel = Math.max(dx, dy);
  const x = Math.round((min[0] + max[0]) / (2 * dx)) * dx;
  const y = Math.round((min[1] + max[1]) / (2 * dy)) * dy;
  const zPad = Math.max(0.01, extent[2] * 0.1);
  const z = Math.ceil((max[2] + zPad) / texel) * texel;
  const near = Math.max(0.0001, zPad * 0.25), far = z - min[2] + zPad;
  const guard = {
    min: [x - width / 2 + padding / 4, y - height / 2 + padding / 4, z - far + zPad / 4],
    max: [x + width / 2 - padding / 4, y + height / 2 - padding / 4, z - near - zPad / 4],
  };
  return { x, y, z, width, height, near, far, texel, guard,
    normalBias: clamp(texel * 0.35, 0.00015, 0.004),
    bias: -Math.min(texel * 0.15, 0.001) / (far - near) };
}
