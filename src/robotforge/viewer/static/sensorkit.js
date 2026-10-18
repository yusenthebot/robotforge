// What drawing sensors needs that is plain arithmetic: colors, the depth picture, frustum corners, which rays to
// draw. No three.js and no page here, so Node runs these as they are (tests/test_viewer_sensors.py).

// The fields of the server's "sensors" message (viewer/server.py) that sensors.js reads, for every sensor and by
// type. A test checks the server sends each of them, so the two cannot drift apart unnoticed.
export const READS = {
  all: ["name", "robot", "type", "added", "uid", "body", "local_pos", "local_quat"],
  lidar: ["origin", "points", "range_m"],
  lidar3d: ["origin", "points", "range_m", "channels", "quat"],
  camera: ["mode", "fov_deg", "size", "range_m"],
  imu: ["accel_m_s2", "gyro_rad_s", "orientation_deg"],
  force_torque: ["force_n", "torque_nm"],
  encoders: ["rad"],
  touch: ["force_n", "total_n", "cells_local", "cell_half", "cell_shape", "pad_quat"],
  rangefinder: ["direction", "distance_m", "range_m"],
};
// Fields read when they are there: a wrist's payload_kg, or a standing leg's supported_kg instead.
export const READS_WHEN_SENT = { force_torque: ["payload_kg", "supported_kg"] };

// Where the page keeps a viewer's switch for one sensor: the world, the sensor's name and its uid (the same while
// the sensor stays on its robot, new when it is added again), so an earlier world's choice never hides it.
export function choiceKey(world, entry) {
  return `robotforge.sensor.${world ?? ""}|${entry.name}|${entry.uid ?? ""}`;
}

// Google's Turbo colormap as polynomials in t, the same as sensors/camera.py: 0 dark blue, 1 dark red.
const TURBO = [
  [0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
  [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
  [0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
];

export function turbo(t) {
  return TURBO.map((c) => {
    let v = 0;
    for (let i = c.length - 1; i >= 0; i--) v = v * t + c[i];
    return Math.min(1, Math.max(0, v));
  });
}

// Near is red and far is blue across low..high, as the server's depth pictures and lidar colors are. Turbo's two
// ends are nearly black, so the scale runs from 0.12 to 0.95 of it: black is left for "nothing in range".
export function nearColor(d, low, high) {
  const near = 1 - Math.min(1, Math.max(0, (d - low) / Math.max(high - low, 1e-6)));
  return turbo(0.12 + 0.83 * near);
}

// A depth camera's picture: `depth` holds meters row by row (NaN where nothing is in range), `out` gets RGBA bytes,
// top row first; `flip` reads `depth` bottom row first, as a GPU gives it back, `width` pixels a row. The same colors
// as read_sensors(image=true) gives an agent (camera.py's depth_picture), with the span given rather than taken
// from this one frame, so a live view does not flicker as things come and go. Colors come from a table of 1024
// steps: Turbo per pixel cost 28 ms a second at ten pictures a second.
const STEPS = 1024;
export function depthPicture(depth, low, high, out, { width = depth.length, flip = false } = {}) {
  const table = new Uint8Array(3 * STEPS);
  for (let k = 0; k < STEPS; k++) {
    const [r, g, b] = turbo(0.12 + 0.83 * (1 - k / (STEPS - 1)));
    table[3 * k] = Math.trunc(r * 255);
    table[3 * k + 1] = Math.trunc(g * 255);
    table[3 * k + 2] = Math.trunc(b * 255);
  }
  const scale = (STEPS - 1) / Math.max(high - low, 1e-6);
  const rows = depth.length / width;
  for (let y = 0; y < rows; y++) {
    const from = (flip ? rows - 1 - y : y) * width;
    for (let x = 0; x < width; x++) {
      const d = depth[from + x];
      const o = 4 * (y * width + x);
      if (d === d) {
        const k = Math.round(Math.min(STEPS - 1, Math.max(0, (d - low) * scale)));
        out[o] = table[3 * k];
        out[o + 1] = table[3 * k + 1];
        out[o + 2] = table[3 * k + 2];
      } else {
        out[o] = out[o + 1] = out[o + 2] = 0; // NaN: nothing within range
      }
      out[o + 3] = 255;
    }
  }
  return out;
}

// The nearest and farthest distance in a picture or a scan, or null when nothing is in range.
export function spanOf(values) {
  let low = Infinity;
  let high = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < low) low = v; // NaN compares false both ways: skipped
    if (v > high) high = v;
  }
  return Number.isFinite(low) ? [low, high] : null;
}

// The span between two quantiles of the finite values (0.05 and 0.9: the nearest and farthest few left out), or
// null. A lidar's points crowd near the robot (the floor rings, the walls close by) while a few reach across the
// hall: colored from the nearest to the farthest, nearly all of them came out the same red.
export function quantileSpan(values, low = 0.05, high = 0.9) {
  const finite = Float32Array.from(values).filter(Number.isFinite).sort();
  if (!finite.length) return null;
  const at = (q) => finite[Math.min(finite.length - 1, Math.max(0, Math.round(q * (finite.length - 1))))];
  return [at(low), Math.max(at(high), at(low) + 1e-3)];
}

// A color scale's span over time: it widens at once to take in what is seen, and narrows slowly (by `ease` of
// the way each time), so colors do not jump while the robot walks.
export function steadySpan(prev, seen, ease = 0.08) {
  if (!seen) return prev;
  if (!prev) return seen.slice();
  const low = seen[0] < prev[0] ? seen[0] : prev[0] + (seen[0] - prev[0]) * ease;
  const high = seen[1] > prev[1] ? seen[1] : prev[1] + (seen[1] - prev[1]) * ease;
  return [low, high];
}

// Depth packed in a picture's red and green bytes as 16 bits of the camera's range (the browser reads pictures
// back as bytes); blue is 255 where something was drawn, 0 for empty sky. See sensors.js's depth pass.
export function unpackDepth(bytes, range, out) {
  for (let i = 0; i < out.length; i++) {
    const o = 4 * i;
    out[i] = bytes[o + 2] < 128 ? NaN : ((bytes[o] * 256 + bytes[o + 1]) / 65535) * range;
  }
  return out;
}

// CRC-32 (IEEE), as Python's zlib.crc32: the segmentation colors are keyed by it.
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(text) {
  const bytes = new TextEncoder().encode(text);
  let c = 0xffffffff;
  for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function hsv(h, s, v) {
  const i = Math.trunc(h * 6) % 6;
  const f = h * 6 - Math.trunc(h * 6);
  const [p, q, t] = [v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)];
  const [r, g, b] = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i];
  return [Math.trunc(r * 255), Math.trunc(g * 255), Math.trunc(b * 255)];
}

// One flat color per thing a segmentation camera tells apart, as bytes: the same as camera.py's, so the browser's
// picture and the one an agent gets agree.
export function thingColor(name) {
  const code = crc32(name);
  return hsv((code % 360) / 360, 0.55 + (Math.floor(code / 512) % 40) / 100, 0.75 + (Math.floor(code / 131072) % 25) / 100);
}

// What a geom belongs to as the world names it (camera.py's thing_name): the robot or object its root body is,
// or for the world's own shapes their name ("floor" when they have none). `bodies` and `geoms` as the scene
// message gives them.
export function thingName(bodies, geom) {
  let body = geom.body;
  if (body === 0) return (geom.name || "floor").split("/")[0];
  while (bodies[body] && bodies[body].parent !== 0) body = bodies[body].parent;
  return (bodies[body]?.name || `body${body}`).split("/")[0];
}

// The far corners of a camera's view `length` meters out, in the camera's own frame (x right, y up, looking along
// -z, as MuJoCo's and three.js's cameras both are), for drawing its frustum. fov is the vertical field of view.
export function frustumCorners(fovDeg, aspect, length) {
  const h = Math.tan((fovDeg * Math.PI) / 360) * length;
  const w = h * aspect;
  return [[-w, -h, -length], [w, -h, -length], [w, h, -length], [-w, h, -length]];
}

// Which of `count` hit points get a ray drawn to them: every one up to `most`, else an even spread of `most`.
export function rayPicks(count, most) {
  if (count <= most) return Array.from({ length: count }, (_, i) => i);
  return Array.from({ length: most }, (_, i) => Math.floor((i * count) / most));
}

// Which of a 3D scan's hit points get a ray drawn: in each of `bins` slices of bearing round the sensor, the point
// whose ray is nearest the sensor's own horizontal, so the rays read as one fan sweeping out to what it hits. An
// even pick across a 16-ring scan was mostly short rays into the floor, faint and crossing each other. `flat` is the
// points as [x, y, z, ...] (the first `count`), `origin` the sensor's place and `quat` its world orientation
// [w, x, y, z].
export function fanPicks(flat, origin, quat, bins, count = flat.length / 3) {
  const [w, qx, qy, qz] = quat;
  const best = new Array(bins).fill(-1);
  const lowest = new Array(bins).fill(Infinity);
  for (let i = 0; i < count; i++) {
    const dx = flat[3 * i] - origin[0];
    const dy = flat[3 * i + 1] - origin[1];
    const dz = flat[3 * i + 2] - origin[2];
    // into the sensor's frame: rotate by the inverse quaternion, v' = v + 2 q' x (q' x v + w v), with q' = -q
    const tx = -qy * dz + qz * dy + w * dx;
    const ty = -qz * dx + qx * dz + w * dy;
    const tz = -qx * dy + qy * dx + w * dz;
    const x = dx + 2 * (-qy * tz + qz * ty);
    const y = dy + 2 * (-qz * tx + qx * tz);
    const z = dz + 2 * (-qx * ty + qy * tx);
    const slope = Math.abs(z) / Math.max(Math.hypot(x, y), 1e-9);
    const turn = (Math.atan2(y, x) / (2 * Math.PI) + 1) % 1;
    const k = Math.min(bins - 1, Math.floor(turn * bins));
    if (slope < lowest[k]) {
      lowest[k] = slope;
      best[k] = i;
    }
  }
  return best.filter((i) => i >= 0);
}

// A point pulled `back` meters toward the sensor, so its dot sits in front of the surface it lies on instead of
// half inside it.
export function pullBack(origin, point, back) {
  const d = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]];
  const len = Math.hypot(d[0], d[1], d[2]);
  if (len <= back) return point.slice();
  const k = (len - back) / len;
  return [origin[0] + d[0] * k, origin[1] + d[1] * k, origin[2] + d[2] * k];
}

// Arrow length for a force: `reach` meters for the largest force seen lately, the rest in proportion, so a light
// hand's newtons and a dog's foot both read.
export function arrowLength(force, largest, reach = 0.22) {
  if (!(largest > 1e-6)) return 0;
  return Math.min(1, Math.abs(force) / largest) * reach;
}

// An encoder's joints in groups a plot can show at once (at most `most` lines): by the name's first word when that
// splits them well (a Go2's FL, FR, RL and RR legs), else in runs. Each joint keeps a short name for the legend.
export function jointGroups(names, most = 6) {
  const short = (n) => n.replace(/_joint$/, "");
  const byPrefix = new Map();
  for (const n of names) {
    const cut = n.indexOf("_");
    const key = cut > 0 ? n.slice(0, cut) : "";
    if (!byPrefix.has(key)) byPrefix.set(key, []);
    byPrefix.get(key).push(n);
  }
  const groups = [...byPrefix];
  const split = names.length > most && groups.length >= 2 && groups.length <= 8 && groups.every(([key, list]) => key && list.length <= most);
  if (split) return groups.map(([key, list]) => ({ label: key, joints: list.map((n) => ({ name: n, short: short(n.slice(key.length + 1)) })) }));
  const out = [];
  for (let i = 0; i < names.length; i += most) {
    const list = names.slice(i, i + most);
    const label = names.length > most ? `${i + 1}-${i + list.length}` : "";
    out.push({ label, joints: list.map((n) => ({ name: n, short: short(n) })) });
  }
  return out;
}
