// Propellers that turn: the physics welds each propeller to its aircraft (simulating it spinning would need a
// time step a hundred times smaller), so the page turns it, as fast as its rotor's command says. The scene lists
// the rotors (body, actuator, axis, which way, full throttle, radius); the state brings the commands four times a
// second, and each frame eases every propeller toward its speed and turns it.
import * as THREE from "three";

const FULL_SPEED = 18; // rad/s drawn at full throttle: fast, yet a two-bladed propeller still reads as turning (at 26 a
// 30 fps recording showed it a quarter turn apart from frame to frame, and the eye lost which way it went)
const SPIN_UP = 0.18; // s, how quickly a propeller's drawn speed follows its command
const BLUR_OPACITY = 0.16; // a faint disc where the blades sweep, at full speed

let rotors = []; // { pivot, base, actuator, spin, max, speed, target, angle, blur }
let last = performance.now();

// Move every geom of each rotor's body into a pivot that turns about the rotor's own axis.
export function buildRotors(list, bodyNodes) {
  rotors = [];
  for (const r of list || []) {
    const node = bodyNodes[r.body];
    if (!node) continue;
    const pivot = new THREE.Group();
    pivot.position.fromArray(r.center);
    pivot.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3().fromArray(r.axis).normalize());
    node.add(pivot);
    node.updateMatrixWorld(true);
    for (const child of [...node.children]) if (child !== pivot) pivot.attach(child);
    const blur = new THREE.Mesh(
      new THREE.CircleGeometry(r.radius, 48),
      new THREE.MeshBasicMaterial({ color: 0x20242b, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }),
    );
    blur.position.z = 0.003;
    blur.renderOrder = 2;
    blur.userData.group = 2; // drawn like the robot's own visual geoms (see the collision toggle)
    blur.userData.body = r.body;
    blur.raycast = () => {}; // never picked: dragging grabs the drone, not the air
    pivot.add(blur);
    rotors.push({ pivot, base: pivot.quaternion.clone(), actuator: r.actuator, spin: r.spin || 1, max: r.max || 1, speed: 0, target: 0, angle: 0, blur });
  }
}

// From the state message: rotor speed grows with the square root of thrust, so the drawn speed does too.
export function rotorCommands(ctrl, paused) {
  for (const r of rotors) {
    const share = Math.max(0, Math.min(1, (ctrl?.[r.actuator] ?? 0) / r.max));
    r.target = paused ? 0 : r.spin * FULL_SPEED * Math.sqrt(share);
    r.paused = paused;
  }
}

const turn = new THREE.Quaternion();
const zAxis = new THREE.Vector3(0, 0, 1);

export function spinRotors() {
  const now = performance.now();
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  const ease = 1 - Math.exp(-dt / SPIN_UP);
  for (const r of rotors) {
    if (r.paused) continue; // paused: frozen where they were, like everything else
    r.speed += (r.target - r.speed) * ease;
    r.angle = (r.angle + r.speed * dt) % (2 * Math.PI);
    r.pivot.quaternion.copy(r.base).multiply(turn.setFromAxisAngle(zAxis, r.angle));
    r.blur.material.opacity = (BLUR_OPACITY * Math.abs(r.speed)) / FULL_SPEED;
    r.blur.visible = r.blur.material.opacity > 0.01;
  }
}
