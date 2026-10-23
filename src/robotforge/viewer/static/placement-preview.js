import * as THREE from "three";

// A temporary size envelope, outside worldRoot so it never becomes a placement/raycast surface.
export function createPlacementPreview(scene) {
  const material = new THREE.MeshBasicMaterial({ color: 0xe4a65b, wireframe: true, transparent: true, opacity: 0.65 });
  let mesh = null, key = null;
  function clear() {
    if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); mesh = null; }
    key = null;
  }
  function update(spec, point) {
    if (!spec || !point) { if (mesh) mesh.visible = false; return; }
    const nextKey = JSON.stringify([spec.shape, spec.size]);
    if (key !== nextKey) {
      clear(); key = nextKey;
      const s = spec.size;
      const geometry = spec.shape === "sphere" ? new THREE.SphereGeometry(s[0] / 2, 12, 8)
        : spec.shape === "cylinder" ? new THREE.CylinderGeometry(s[0] / 2, s[0] / 2, s[1], 16)
          : new THREE.BoxGeometry(...s);
      if (spec.shape === "cylinder") geometry.rotateX(Math.PI / 2);
      mesh = new THREE.Mesh(geometry, material);
      mesh.name = "placement-size-preview";
      scene.add(mesh);
    }
    mesh.visible = true;
    mesh.rotation.z = spec.yaw * Math.PI / 180;
    mesh.position.set(point[0], point[1], point[2] + spec.size.at(-1) / 2 + 0.002);
  }
  return { update, clear };
}
