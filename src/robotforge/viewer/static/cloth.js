// A cloth (a MuJoCo flex) drawn as a smooth surface through its points.
//
// Its points are 4 cm or more apart (a finer cloth costs real-time speed), and drawn as flat triangles between them
// the cloth cut under the edge of a table it hung over: the straight side from the last point on the top to the
// first one hanging passed 7 mm inside the edge, and the table top showed through all round. Each triangle is drawn
// as CLOTH_SPLIT x CLOTH_SPLIT smaller ones on a Phong tessellation of it (Boubekeur and Alexa, 2008): every point
// in it is pulled toward the planes the cloth is tangent to at the triangle's corners, so a fold rounds outward
// over an edge as a cloth does, and a flat stretch stays flat. Cheap: a few thousand points a frame, all in JS.
import * as THREE from "three";

const CLOTH_SPLIT = 3; // each triangle of the cloth is drawn as this squared
const SHAPE = 1.0; // how far toward the tangent planes (0: flat triangles); all the way rounds a fold over a table's edge best

// The drawn surface's points: the cloth's own first (so index i < bodies.length is point i), then the points inside
// each side of a triangle (shared by the two triangles on it), then those inside each triangle.
export function clothGeometry(faces, count) {
  const n = CLOTH_SPLIT;
  const sides = new Map(); // lo * count + hi -> index of its first inner point, counted from lo
  const sideList = []; // lo, hi, first
  const inner = []; // a, b, c, i, j (lattice place), index
  let next = count;
  const side = (x, y) => {
    const lo = Math.min(x, y);
    const hi = Math.max(x, y);
    let first = sides.get(lo * count + hi);
    if (first === undefined) {
      first = next;
      next += n - 1;
      sides.set(lo * count + hi, first);
      sideList.push(lo, hi, first);
    }
    return (step) => first + (x === lo ? step : n - step) - 1; // `step` of n from x toward y
  };
  const index = [];
  for (let f = 0; f < faces.length; f += 3) {
    const [a, b, c] = [faces[f], faces[f + 1], faces[f + 2]];
    const ab = side(a, b);
    const ac = side(a, c);
    const bc = side(b, c);
    const at = new Map(); // lattice place i, j (i toward b, j toward c) -> point
    const point = (i, j) => {
      if (i === 0 && j === 0) return a;
      if (i === n) return b;
      if (j === n) return c;
      if (j === 0) return ab(i);
      if (i === 0) return ac(j);
      if (i + j === n) return bc(j);
      const key = i * (n + 1) + j;
      if (!at.has(key)) {
        at.set(key, next);
        inner.push(a, b, c, i, j, next++);
      }
      return at.get(key);
    };
    for (let i = 0; i < n; i++) {
      for (let j = 0; i + j < n; j++) {
        index.push(point(i, j), point(i + 1, j), point(i, j + 1));
        if (i + j < n - 1) index.push(point(i + 1, j), point(i + 1, j + 1), point(i, j + 1));
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(next * 3), 3));
  geometry.setIndex(index);
  return { geometry, faces: Int32Array.from(faces), count, sides: Int32Array.from(sideList), inner: Int32Array.from(inner) };
}

// Lay the drawn surface through the cloth's points `p` (count x 3, world).
export function shapeCloth(shape, p) {
  const { geometry, faces, count, sides, inner } = shape;
  const n = CLOTH_SPLIT;
  const normals = new Float32Array(count * 3); // at the cloth's points: its faces' normals, weighted by area
  for (let f = 0; f < faces.length; f += 3) {
    const a = 3 * faces[f];
    const b = 3 * faces[f + 1];
    const c = 3 * faces[f + 2];
    const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
    const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const k of [a, b, c]) {
      normals[k] += nx;
      normals[k + 1] += ny;
      normals[k + 2] += nz;
    }
  }
  for (let k = 0; k < normals.length; k += 3) {
    const len = Math.hypot(normals[k], normals[k + 1], normals[k + 2]) || 1;
    normals[k] /= len;
    normals[k + 1] /= len;
    normals[k + 2] /= len;
  }
  const out = geometry.attributes.position.array;
  out.set(p.subarray(0, count * 3));
  const flat = [0, 0, 0];
  const pulled = [0, 0, 0];
  // The point with weights w at corners v: on the flat triangle, moved toward each corner's tangent plane.
  const blend = (at, v, w) => {
    flat[0] = flat[1] = flat[2] = 0;
    for (let k = 0; k < v.length; k++) {
      flat[0] += w[k] * p[3 * v[k]];
      flat[1] += w[k] * p[3 * v[k] + 1];
      flat[2] += w[k] * p[3 * v[k] + 2];
    }
    pulled[0] = pulled[1] = pulled[2] = 0;
    for (let k = 0; k < v.length; k++) {
      const q = 3 * v[k];
      const d = (flat[0] - p[q]) * normals[q] + (flat[1] - p[q + 1]) * normals[q + 1] + (flat[2] - p[q + 2]) * normals[q + 2];
      pulled[0] += w[k] * (flat[0] - d * normals[q]);
      pulled[1] += w[k] * (flat[1] - d * normals[q + 1]);
      pulled[2] += w[k] * (flat[2] - d * normals[q + 2]);
    }
    for (let k = 0; k < 3; k++) out[3 * at + k] = (1 - SHAPE) * flat[k] + SHAPE * pulled[k];
  };
  for (let s = 0; s < sides.length; s += 3) {
    for (let step = 1; step < n; step++) {
      const t = step / n;
      blend(sides[s + 2] + step - 1, [sides[s], sides[s + 1]], [1 - t, t]);
    }
  }
  for (let s = 0; s < inner.length; s += 6) {
    const i = inner[s + 3] / n;
    const j = inner[s + 4] / n;
    blend(inner[s + 5], [inner[s], inner[s + 1], inner[s + 2]], [1 - i - j, i, j]);
  }
  geometry.attributes.position.needsUpdate = true;
  geometry.computeVertexNormals();
}
