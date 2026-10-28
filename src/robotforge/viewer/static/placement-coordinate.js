// World-origin XY spacing. Z remains the hit support height, never an object's centre.
export function placementCoordinates(point, spacing = "free") {
  if (!Array.isArray(point) || point.length !== 3
      || !point.every((v) => Number.isFinite(v) && Math.abs(v) <= 100)) return null;
  if (spacing === "free") return point.map((v) => +v.toFixed(4));
  const step = spacing === "0.1" ? 0.1 : spacing === "0.25" ? 0.25 : null;
  if (step === null) return null;
  const snap = (value) => {
    const units = Math.abs(value) / step;
    // Symmetric half-away-from-zero ties; absorb only floating-point representation error.
    const nearest = Math.floor(units + 0.5 + 8 * Number.EPSILON * Math.max(1, units));
    const result = +(Math.sign(value) * nearest * step).toFixed(4);
    return Object.is(result, -0) ? 0 : result;
  };
  return [snap(point[0]), snap(point[1]), +point[2].toFixed(4)];
}
