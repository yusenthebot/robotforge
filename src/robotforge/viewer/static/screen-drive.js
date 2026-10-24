// Independent pointer/key holds. The app combines these with keyboard/gamepad axes.
export function createScreenDrive(root, changed) {
  const buttons = [...root.querySelectorAll("[data-direction]")];
  const holds = new Map();
  let timer = null, identity = null, enabled = false;

  function paint() {
    for (const button of buttons) button.setAttribute("aria-pressed", String([...holds.values()].some((h) => h.button === button)));
    if (holds.size && timer === null) timer = setInterval(() => {
      // Browsers may defer lostpointercapture until the next pointer event.
      // Never renew a hold whose button no longer owns its pointer.
      let lost = false;
      for (const [token, hold] of holds) if (typeof token === "number" && !hold.button.hasPointerCapture(token)) {
        holds.delete(token); lost = true;
      }
      if (lost) paint();
      changed(!lost);
    }, 250);
    if (!holds.size && timer !== null) { clearInterval(timer); timer = null; }
  }

  function press(token, button) {
    if (!enabled || root.hidden || button.hidden || button.disabled || holds.has(token)) return false;
    holds.set(token, { button, direction: button.dataset.direction });
    paint(); changed(false); return true;
  }

  function release(token) {
    if (!holds.delete(token)) return;
    paint(); changed(false);
  }

  function clear(notify = true) {
    const previous = [...holds];
    holds.clear(); paint();
    for (const [token, { button }] of previous) {
      if (typeof token === "number" && button.hasPointerCapture(token)) button.releasePointerCapture(token);
    }
    if (previous.length && notify) changed(false);
  }

  for (const button of buttons) {
    button.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault(); event.stopPropagation();
      if (press(event.pointerId, button)) button.setPointerCapture(event.pointerId);
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
      button.addEventListener(type, (event) => { event.stopPropagation(); release(event.pointerId); });
    }
    button.addEventListener("keydown", (event) => {
      if (event.key !== " " && event.key !== "Enter") return;
      event.preventDefault(); event.stopPropagation();
      if (!event.repeat) press(`key:${event.key}`, button);
    });
    button.addEventListener("keyup", (event) => {
      if (event.key !== " " && event.key !== "Enter") return;
      event.preventDefault(); event.stopPropagation(); release(`key:${event.key}`);
    });
    button.addEventListener("blur", () => {
      for (const [token, hold] of holds) if (typeof token === "string" && hold.button === button) release(token);
    });
    button.addEventListener("click", (event) => event.preventDefault());
    button.addEventListener("contextmenu", (event) => event.preventDefault());
  }
  window.addEventListener("blur", () => clear());
  document.addEventListener("visibilitychange", () => { if (document.hidden) clear(); });

  return {
    held: () => holds.size > 0,
    clear,
    axes() {
      const directions = new Set([...holds.values()].map((hold) => hold.direction));
      const n = (plus, minus) => Number(directions.has(plus)) - Number(directions.has(minus));
      return { fwd: n("fwd", "back"), left: n("left", "right"), slide: n("slide_left", "slide_right"), up: n("up", "down") };
    },
    update({ target, show, ready, rise, slide }) {
      if (identity !== target || !show || !ready) clear();
      identity = target; enabled = show && ready;
      root.hidden = !show;
      for (const button of buttons) {
        button.hidden = ["up", "down"].includes(button.dataset.direction) ? !rise : button.dataset.direction.startsWith("slide_") && !slide;
        button.disabled = !enabled;
      }
      root.querySelector(".screen-drive-extra").hidden = !rise && !slide;
    },
  };
}
