// Small live line plots for the sensor views, drawn the way the lessons' plots are (lesson.js): the same margins,
// grid, tick labels and color tokens, and axes that widen at once but narrow slowly. No imports, so Node can test
// the arithmetic (tests/test_viewer_sensors.py).

// The roundest step near span / count; a finer one when that leaves a single line, which gives no scale.
export function niceTicks(lo, hi, count) {
  const span = hi - lo;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const steps = [0.5, 1, 2, 5, 10].map((m) => m * mag);
  let k = steps.findIndex((s) => s >= step0);
  if (k < 0) k = steps.length - 1;
  for (; k >= 0; k--) {
    const step = steps[k];
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(Math.abs(v) < step / 1e6 ? 0 : v);
    if (out.length >= 2 || k === 0) return out;
  }
  return [];
}

// As many decimals as the tick spacing needs (0.02 apart reads 0.02 and 0.04, not 0.0 and 0.0), and never "-0".
export function formatTick(v, ticks = []) {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : Math.abs(v);
  const decimals = step > 0 ? Math.min(3, Math.max(0, -Math.floor(Math.log10(step) + 1e-9))) : 0;
  const text = v.toFixed(decimals);
  return Number(text) === 0 ? "0" : text;
}

// The y range to draw: the data's, at least `span` tall (a sensor's noise at rest is not drawn as a swing), with a
// margin; grown at once and shrunk slowly from `prev`, so the axes do not jump around while you watch.
export function plotRange(prev, lo, hi, span = 0) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) [lo, hi] = [-1, 1];
  if (hi - lo < 1e-9) [lo, hi] = [lo - 0.5 * Math.max(span, 1e-3), hi + 0.5 * Math.max(span, 1e-3)];
  const short = span - (hi - lo);
  if (short > 0) [lo, hi] = [lo - short / 2, hi + short / 2];
  const pad = (hi - lo) * 0.12;
  const target = [lo - pad, hi + pad];
  if (!prev) return target;
  return [Math.min(target[0], prev[0] + (target[0] - prev[0]) * 0.05), Math.max(target[1], prev[1] + (target[1] - prev[1]) * 0.05)];
}

// Draw `series` ({values: [[t, v], ...], color, width}) over the last `window` seconds ending at `end` on a 2D
// context `w` x `h` CSS px. `token(name)` reads a CSS custom property; `range` is from plotRange.
export function drawLines(g, w, h, { series, end, window, range, token }) {
  const start = end - window;
  const x = (t) => 36 + ((t - start) / window) * (w - 44);
  const y = (v) => 8 + (1 - (v - range[0]) / (range[1] - range[0])) * (h - 22);
  g.clearRect(0, 0, w, h);
  g.font = `10px ${token("--mono") || "monospace"}`;
  g.fillStyle = token("--muted");
  g.strokeStyle = token("--line");
  g.lineWidth = 1;
  const ticks = niceTicks(range[0], range[1], h < 80 ? 2 : 3); // a compact plot has room for two labels
  for (const v of ticks) {
    g.beginPath();
    g.moveTo(36, Math.round(y(v)) + 0.5);
    g.lineTo(w - 8, Math.round(y(v)) + 0.5);
    g.stroke();
    g.fillText(formatTick(v, ticks), 2, y(v) + 3);
  }
  g.save();
  g.beginPath();
  g.rect(36, 0, w - 44, h); // a line that runs past the window's start is cut at the axis, not drawn over its labels
  g.clip();
  for (const s of series) {
    g.strokeStyle = token(s.color);
    g.lineWidth = s.width ?? 1.4;
    g.lineJoin = "round";
    g.beginPath();
    let started = false;
    for (const [t, v] of s.values) {
      if (t < start - 1 || !Number.isFinite(v)) continue;
      if (!started) g.moveTo(x(t), y(v));
      else g.lineTo(x(t), y(v));
      started = true;
    }
    g.stroke();
  }
  g.restore();
}
