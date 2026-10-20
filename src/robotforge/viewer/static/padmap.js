// The gamepad's mapping, as pure functions (no page, no imports): the standard layout in, one robot command out.
//
// A gamepad that reports the "standard" mapping has these buttons and axes (the Gamepad API's own layout):
//   axes 0 and 1: the left stick (x to the right, y down), axes 2 and 3: the right stick
//   buttons 0 A, 1 B, 2 X, 3 Y, 4 and 5 the bumpers, 6 and 7 the triggers (analog), 12 to 15 the d-pad (up, down, left, right)
//
// What the sticks do depends on what the selected robot is (profileFor):
//   drive  a robot that walks, drives or swims: the left stick goes forward and back and turns (the right stick's x turns
//          too), the d-pad does the same at half strength
//   fly    an aircraft: left stick forward and sideways, right stick x turns and y climbs
//   rov    an underwater robot: left stick forward and turn, right stick y heaves up and down
//   arm    an arm: the left stick moves the tool over the floor (up on the stick is away from the camera), the right stick
//          y raises and lowers it; A or the right trigger closes the gripper, B or the left trigger opens it
//   both   a robot that goes and has an arm (a dog carrying one): drive as above; Y switches to the arm's keys and back

export const DEADZONE = 0.15; // of a stick's travel, where a worn stick still rests
export const ARM_SPEED = 0.15; // m/s the tool moves at full stick
export const BUTTON = { a: 0, b: 1, x: 2, y: 3, lb: 4, rb: 5, lt: 6, rt: 7, up: 12, down: 13, left: 14, right: 15 };
const PULL = 0.5; // how far a trigger is pulled to count as pressed

// A stick axis with its dead zone taken out and the rest stretched back to the full range.
export function shape(value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return 0;
  const a = Math.abs(v);
  if (a < DEADZONE) return 0;
  return Math.sign(v) * Math.min(1, (a - DEADZONE) / (1 - DEADZONE));
}

const level = (pad, i) => (pad.buttons?.[i] ? Number(pad.buttons[i].value ?? (pad.buttons[i].pressed ? 1 : 0)) : 0);
const held = (pad, i) => Boolean(pad.buttons?.[i]?.pressed) || level(pad, i) > PULL;

// The pad as plain numbers: sticks shaped, buttons held or not. Up and forward are positive, left is positive.
export function read(pad) {
  const axis = (i) => shape(pad.axes?.[i] ?? 0);
  const dpad = (plus, minus) => (held(pad, plus) ? 1 : 0) - (held(pad, minus) ? 1 : 0);
  return {
    lx: -axis(0), // left of centre is positive
    ly: -axis(1), // up is positive
    rx: -axis(2),
    ry: -axis(3),
    dx: -dpad(BUTTON.right, BUTTON.left),
    dy: dpad(BUTTON.up, BUTTON.down),
    a: held(pad, BUTTON.a),
    b: held(pad, BUTTON.b),
    x: held(pad, BUTTON.x),
    y: held(pad, BUTTON.y),
    lb: held(pad, BUTTON.lb),
    rb: held(pad, BUTTON.rb),
    lt: level(pad, BUTTON.lt),
    rt: level(pad, BUTTON.rt),
  };
}

// Which set of controls a robot gets: from what the viewer knows of it (kind, max_speed for the ones that go, arms).
export function profileFor(robot) {
  if (!robot) return null;
  const goes = Boolean(robot.max_speed);
  const arm = robot.kind === "arm" || (robot.arms?.length ?? 0) > 0;
  if (robot.kind === "aircraft") return "fly";
  if (robot.kind === "rov") return "rov";
  if (goes && arm) return "both";
  if (goes) return "drive";
  if (arm) return "arm";
  return null;
}

const clamp = (v) => Math.max(-1, Math.min(1, v));
const q = (v) => Math.round(v * 20) / 20; // steps of 0.05: a stick's jitter is not a new command

// What the pad asks for, as numbers in -1..1: drive (fwd, left, slide, up) and arm (x, y, z in the view's frame:
// x to the right on screen, y away from the camera, z up) and the gripper ("open", "close" or null).
export function commandFor(profile, p, mode = "drive") {
  const none = { drive: { fwd: 0, left: 0, slide: 0, up: 0 }, arm: { x: 0, y: 0, z: 0 }, gripper: null };
  if (!profile) return none;
  const arm = profile === "arm" || (profile === "both" && mode === "arm");
  if (arm) {
    const close = p.a || p.rt > PULL;
    const open = p.b || p.lt > PULL;
    return {
      ...none,
      arm: { x: q(clamp(-p.lx + (p.dx ? -p.dx * 0.5 : 0))), y: q(clamp(p.ly + p.dy * 0.5)), z: q(p.ry) },
      gripper: close && !open ? "close" : open && !close ? "open" : null,
    };
  }
  const fwd = clamp(p.ly + p.dy * 0.5);
  if (profile === "fly") {
    return { ...none, drive: { fwd: q(fwd), left: q(clamp(p.rx)), slide: q(clamp(p.lx)), up: q(clamp(p.ry + p.rt - p.lt)) } };
  }
  if (profile === "rov") {
    return { ...none, drive: { fwd: q(fwd), left: q(clamp(p.lx + p.rx + p.dx * 0.5)), slide: 0, up: q(clamp(p.ry + p.rt - p.lt)) } };
  }
  return { ...none, drive: { fwd: q(fwd), left: q(clamp(p.lx + p.rx + p.dx * 0.5)), slide: 0, up: 0 } };
}

// Whether the pad is asking for anything at all (a stick off centre, a button down).
export function active(command) {
  const d = command.drive;
  const a = command.arm;
  return Boolean(d.fwd || d.left || d.slide || d.up || a.x || a.y || a.z || command.gripper);
}
