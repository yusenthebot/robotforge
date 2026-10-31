import * as THREE from "three";

// A temporary size envelope, outside worldRoot so it never becomes a placement/raycast surface.
export function createPlacementPreview(scene) {
  const material = new THREE.MeshBasicMaterial({ color: 0xe4a65b, wireframe: true, transparent: true, opacity: 0.65 });
  let mesh = null, key = null;
  function clear() {
    if (mesh) {
      scene.remove(mesh);
      // Group ghosts borrow source buffers. Only primitive preview buffers belong to us.
      mesh.traverse((node) => { if (node.userData.previewOwnGeometry) node.geometry.dispose(); });
      mesh = null;
    }
    key = null;
  }
  function update(spec, point) {
    if (!spec || !point) { if (mesh) mesh.visible = false; return; }
    const nextKey = spec.parts ? spec.key : JSON.stringify(spec.boxes || [spec.shape, spec.size]);
    if (key !== nextKey) {
      clear(); key = nextKey;
      if (spec.parts) {
        mesh = new THREE.Group();
        for (const source of spec.parts) {
          const part = new THREE.Mesh(source.geometry, material);
          part.matrixAutoUpdate = false;
          part.matrix.fromArray(source.matrix);
          mesh.add(part);
        }
      } else if (spec.boxes) {
        mesh = new THREE.Group();
        for (const box of spec.boxes) {
          const part = new THREE.Mesh(new THREE.BoxGeometry(...box.half.map((v) => v * 2)), material);
          part.userData.previewOwnGeometry = true;
          part.position.fromArray(box.center); part.rotation.y = box.pitch;
          mesh.add(part);
        }
      } else {
        const s = spec.size;
        const geometry = spec.shape === "sphere" ? new THREE.SphereGeometry(s[0] / 2, 12, 8)
          : spec.shape === "cylinder" ? new THREE.CylinderGeometry(s[0] / 2, s[0] / 2, s[1], 16)
            : new THREE.BoxGeometry(...s);
        if (spec.shape === "cylinder") geometry.rotateX(Math.PI / 2);
        mesh = new THREE.Mesh(geometry, material);
        mesh.userData.previewOwnGeometry = true;
      }
      mesh.name = "placement-size-preview";
      scene.add(mesh);
    }
    mesh.visible = true;
    mesh.rotation.z = spec.yaw * Math.PI / 180;
    mesh.position.set(point[0], point[1], point[2] + (spec.parts || spec.boxes ? 0 : spec.size.at(-1) / 2) + 0.002);
  }
  return { update, clear };
}
