// Loaded only for native model pictures. Pixel storage is shared; repeats belong to each material.
import * as THREE from "three";

const MAX_PIXELS = 512 * 512;
const MAX_IMAGES = 64;

export function modelTextures(entries, buffer, anisotropy) {
  if (entries.length > MAX_IMAGES || buffer.byteLength > 4 * MAX_PIXELS) throw new Error("model texture budget exceeded");
  const images = new Map(), maps = new Map();
  let pixels = 0;
  for (const entry of entries) {
    const { id, width, height, offset, length } = entry;
    if (![id, width, height, offset, length].every(Number.isSafeInteger) || id < 0 || images.has(id) ||
        width < 1 || height < 1 || Math.max(width, height) > 512 || offset < 0 ||
        length !== width * height * 4 || offset + length > buffer.byteLength) throw new Error("invalid model texture data");
    pixels += width * height;
    if (pixels > MAX_PIXELS) throw new Error("model texture pixel budget exceeded");
    images.set(id, { source: new THREE.Source({ data: new Uint8Array(buffer, offset, length), width, height }), alpha: entry.alpha });
  }
  return (binding) => {
    if (!binding) return null;
    const image = images.get(binding.id), repeat = binding.repeat;
    if (!image || !Array.isArray(repeat) || repeat.length !== 2 || !repeat.every(Number.isFinite)) {
      throw new Error("invalid model texture binding");
    }
    const key = JSON.stringify([binding.id, repeat]);
    if (!maps.has(key)) {
      const texture = new THREE.DataTexture();
      texture.source = image.source;
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.flipY = false; // MuJoCo's row zero, like the legacy vertex sampling, is at V=0.
      texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
      texture.repeat.fromArray(repeat);
      texture.generateMipmaps = true;
      texture.minFilter = THREE.LinearMipmapLinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.anisotropy = Math.max(1, Math.min(4, anisotropy));
      texture.userData.modelAlpha = image.alpha;
      texture.needsUpdate = true;
      maps.set(key, texture);
    }
    return maps.get(key);
  };
}
