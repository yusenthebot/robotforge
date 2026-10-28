// World-axis perspective views. The pole tolerance matches OrbitControls.makeSafe;
// the tiny -Y component keeps +Y at the top of the screen without changing Z-up.
const POLE = 1e-6;
const directions = { top: [0, -Math.sin(POLE), Math.cos(POLE)], front: [0, -1, 0], side: [1, 0, 0] };

function orbit(eye, target) {
  const [x, y, z] = eye.map((v, i) => v - target[i]);
  const radius = Math.hypot(x, y, z);
  return { radius, theta: Math.atan2(x, -y), phi: Math.acos(Math.max(-1, Math.min(1, z / radius))) };
}

export function cameraView(name, eye, target) {
  const direction = directions[name];
  if (!direction) return null;
  const radius = Math.hypot(...eye.map((v, i) => v - target[i]));
  if (!Number.isFinite(radius) || radius <= 0) return null;
  return target.map((v, i) => v + radius * direction[i]);
}

export function cameraArc(fromEye, fromTarget, toEye, toTarget) {
  const from = orbit(fromEye, fromTarget), to = orbit(toEye, toTarget);
  return { from, to, turn: Math.atan2(Math.sin(to.theta - from.theta), Math.cos(to.theta - from.theta)) };
}

export function arcEye(arc, target, fraction) {
  const radius = arc.from.radius + (arc.to.radius - arc.from.radius) * fraction;
  const theta = arc.from.theta + arc.turn * fraction;
  const phi = arc.from.phi + (arc.to.phi - arc.from.phi) * fraction;
  return [radius * Math.sin(phi) * Math.sin(theta), -radius * Math.sin(phi) * Math.cos(theta), radius * Math.cos(phi)]
    .map((v, i) => v + target[i]);
}
