// A gamepad as a second way to drive, besides the keyboard: the Gamepad API (no library), polled once a frame.
//
// What the sticks do depends on the selected robot (padmap.js has the table): a robot that walks or drives goes
// where the left stick points and turns with the right; an aircraft and an underwater robot add climbing and heaving;
// an arm's tool is jogged over the floor with the left stick and up and down with the right, A closing the gripper and
// B opening it; a robot that goes and carries an arm switches between the two with Y. The drive reaches the robot as
// the same `teleop` message the keys send (app.js asks padAxes() for the stick's share), the arm as `teleop_arm`.
//
// Nothing is sent while the sticks rest, a stick let go sends one zero, and a pad that goes away, a tab that is hidden
// and a robot that is deselected each stop what they were driving.

import { t } from "./i18n.js";
import { ARM_SPEED, active, commandFor, profileFor, read } from "./padmap.js";

const RESEND_MS = 250; // a jogged arm holds still when it is not told again within half a second (the server's own limit)

let api = null;
let known = null; // the pad last seen: { id, index }
let mode = "drive"; // "drive" or "arm", for a robot that has both (Y switches)
let lastY = false;
let lastGrip = null;
let lastDrive = "";
let drive = { fwd: 0, left: 0, slide: 0, up: 0 }; // what the stick asks of the robot that goes, -1..1
let armSent = { key: "", at: 0, robot: null };
let seen = { robot: null, profile: null, mode: "drive" };
let last = { drive: null, arm: null, gripper: null };

export function initGamepad(context) {
  api = context;
  window.addEventListener("gamepadconnected", (e) => connected(e.gamepad));
  window.addEventListener("gamepaddisconnected", (e) => {
    if (known && e.gamepad.index === known.index) disconnected();
  });
  window.addEventListener("blur", () => release());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) release();
  });
}

function connected(pad) {
  if (known && known.index === pad.index) return;
  known = { id: pad.id, index: pad.index };
  api.toast(t("pad_connected"));
  api.changed();
}

function disconnected() {
  known = null;
  release();
  api.toast(t("pad_disconnected"));
  api.changed();
}

function current() {
  const list = navigator.getGamepads ? navigator.getGamepads() : [];
  for (const pad of list) if (pad && pad.connected !== false) return pad;
  return null;
}

// What the pad asks of a robot that goes: { fwd, left, slide, up } in -1..1, or null while it rests.
export function padAxes() {
  const any = drive.fwd || drive.left || drive.slide || drive.up;
  return any ? drive : null;
}

export const padConnected = () => known !== null;
export const padMode = () => mode;
export const padProfile = () => seen.profile;

// The pad as scripted checks read it.
export function padState() {
  return { connected: known !== null, id: known?.id ?? null, profile: seen.profile, mode, drive: { ...drive }, last: { ...last } };
}

// Stop what the pad was driving, as far as the page knows (the caller sends the stop for the robot it drove).
export function release(notify = true) {
  const was = padAxes() !== null || armSent.key !== "";
  drive = { fwd: 0, left: 0, slide: 0, up: 0 };
  lastDrive = "";
  if (armSent.key !== "" && armSent.robot) api.send({ type: "teleop_arm", robot: armSent.robot, velocity: [0, 0, 0] });
  armSent = { key: "", at: 0, robot: null };
  if (was && notify) api.onDrive();
}

// Once a frame: read the pad, send what changed.
export function padFrame() {
  const pad = current();
  if (!pad) {
    if (known) disconnected();
    return;
  }
  if (!known) connected(pad); // a pad plugged in before the page loaded appears only once a button is pressed
  const robot = api.robot();
  const profile = profileFor(robot);
  const p = read(pad);
  if (robot?.name !== seen.robot || profile !== seen.profile) {
    seen = { robot: robot?.name ?? null, profile, mode };
    if (profile !== "both") mode = profile === "arm" ? "arm" : "drive";
    api.changed();
  }
  if (p.y && !lastY && profile === "both") {
    mode = mode === "drive" ? "arm" : "drive";
    release();
    api.changed();
  }
  lastY = p.y;
  if (!robot || !profile) {
    if (padAxes() || armSent.key) release();
    return;
  }
  const cmd = commandFor(profile, p, mode);
  const key = JSON.stringify(cmd.drive);
  if (key !== lastDrive) {
    lastDrive = key;
    drive = cmd.drive;
    last.drive = cmd.drive;
    api.onDrive();
  }
  armFrame(robot, cmd);
}

function armFrame(robot, cmd) {
  const { x, y, z } = cmd.arm;
  const key = `${robot.name}:${x}:${y}:${z}`;
  const now = performance.now();
  const moving = x || y || z;
  const grip = cmd.gripper;
  const message = { type: "teleop_arm", robot: robot.name };
  let send = false;
  if (moving && (key !== armSent.key || now - armSent.at > RESEND_MS)) {
    const [wx, wy] = api.screenToWorld(y, -x); // x to the right on screen, y away from the camera
    message.velocity = [wx * ARM_SPEED, wy * ARM_SPEED, z * ARM_SPEED];
    send = true;
  } else if (!moving && armSent.key.split(":").slice(1).some((v) => v !== "0")) {
    message.velocity = [0, 0, 0]; // the stick came back to centre: hold
    send = true;
  }
  if (grip && grip !== lastGrip) {
    message.gripper = grip;
    message.velocity = message.velocity ?? [0, 0, 0];
    send = true;
  }
  lastGrip = grip;
  if (!send) return;
  api.send(message);
  last.arm = message;
  if (grip) last.gripper = grip;
  armSent = { key: moving ? key : `${robot.name}:0:0:0`, at: now, robot: robot.name };
}
