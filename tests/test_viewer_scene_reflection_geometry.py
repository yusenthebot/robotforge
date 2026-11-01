"""CPU qualification of the fixed scene probe; no rendering or model compilation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
STATIC = Path(__file__).resolve().parents[1] / "src/robotforge/viewer/static"
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def run_js(body: str) -> None:
    """Use the viewer's vendored Three without installing a Node dependency."""
    three = (STATIC / "vendor/three/three.module.min.js").as_uri()
    hook = (
        "export async function resolve(specifier, context, next) {"
        f"if (specifier === 'three') return {{url: {json.dumps(three)}, shortCircuit: true}};"
        "return next(specifier, context); }"
    )
    source = (
        "import assert from 'node:assert/strict';\n"
        "import {register} from 'node:module';\n"
        f"register({json.dumps('data:text/javascript,' + hook)});\n"
        "const THREE = await import('three');\n"
        f"const G = await import({json.dumps((STATIC / 'scene-reflection-geometry.js').as_uri())});\n"
        "const cube = () => new THREE.Mesh(new THREE.BoxGeometry(2,2,2), new THREE.MeshStandardMaterial());\n"
        "const posed = (mesh) => { mesh.updateMatrixWorld(true); return mesh; };\n"
        "const state = (point, mesh, options) => G.qualifyProbeAnchor(point, [posed(mesh)], options);\n"
        f"{body}\n"
    )
    completed = subprocess.run(
        [NODE, "--experimental-default-type=module", "--input-type=module", "-"],
        input=source,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_box_fit_is_symmetric_fixed_and_camera_independent() -> None:
    run_js("""
    const center = [3, 2, 1], bounds = [{min: [-1,0,0], max: [4,5,3]}];
    const probe = G.fitReflectionProbe(center, bounds);
    assert.equal(probe.status, 'valid');
    assert.deepEqual(probe.min, [-1.1, -1.1, -1.1]);
    assert.deepEqual(probe.max, [7.1, 5.1, 3.1]);
    assert.equal(probe.near, .01);
    assert.ok(Math.abs(probe.far - (Math.hypot(8.2,6.2,4.2)+1)) < 1e-12);
    const camera = new THREE.PerspectiveCamera(); camera.position.set(50,30,100);
    assert.deepEqual(G.fitReflectionProbe(center,bounds), probe);
    center[0] = 100; assert.equal(probe.center[0], 3);
    assert.equal(G.fitReflectionProbe([0,0,0], [{min:[0,0,0],max:[2,0,0]}]).status, 'valid');
    """)


def test_empty_nonfinite_and_degenerate_bounds_are_explicit() -> None:
    run_js("""
    for (const bounds of [[], null, [{min:[0,0,0],max:[0,0,0]}],
      [{min:[0,0,0],max:[NaN,1,1]}], [{min:[1,0,0],max:[0,1,1]}]]) {
      assert.equal(G.fitReflectionProbe([0,0,0],bounds).status,'invalid');
    }
    assert.equal(G.fitReflectionProbe([Infinity,0,0], [{min:[0,0,0],max:[1,1,1]}]).status,'invalid');
    assert.equal(G.qualifyProbeAnchor([NaN,0,0], []).status,'invalid');
    assert.equal(G.qualifyProbeAnchor([0,0,0], [], {near:0}).status,'invalid');
    """)


def test_box_projection_exit_and_fallbacks() -> None:
    run_js("""
    const box = {center:[0,0,0],min:[-2,-2,-2],max:[2,2,2]};
    assert.deepEqual(G.projectReflectionRay([1,0,0],[0,5,0],box).hit,[1,2,0]);
    assert.deepEqual(G.projectReflectionRay([1,0,0],[-2,0,0],box).direction,[-2,0,0]);
    assert.deepEqual(G.projectReflectionRay([0,0,0],[1,1,1],box).hit,[2,2,2]);
    assert.equal(G.projectReflectionRay([2,1,0],[1,0,0],box).distance,0);
    assert.equal(G.projectReflectionRay([3,0,0],[-1,0,0],box).status,'outside');
    for (const direction of [[0,0,0],[Infinity,0,1],[NaN,1,1]]) {
      assert.equal(G.projectReflectionRay([0,0,0],direction,box).status,'invalid');
    }
    assert.equal(G.projectReflectionRay([0,0,0],[1,0,0],{...box,max:[-2,2,2]}).status,'invalid');
    """)


def test_cube_inside_outside_near_face_and_normals() -> None:
    run_js("""
    const mesh = cube();
    assert.equal(state([0,0,0],mesh).reason,'inside-closed-component');
    assert.equal(state([2,0,0],mesh).status,'clear');
    assert.equal(state([1.005,0,0],mesh).reason,'near-clip-intersects-triangle');
    assert.equal(state([1,0,0],mesh).status,'blocked');
    assert.equal(state([1.02,0,0],mesh).status,'clear');
    mesh.geometry.deleteAttribute('normal');
    assert.equal(state([0,0,0],mesh).reason,'inside-closed-component');
    assert.equal(state([2,0,0],mesh).status,'clear');
    mesh.geometry.setAttribute('normal', new THREE.Float32BufferAttribute(new Float32Array(72),3));
    assert.equal(state([0,0,0],mesh).reason,'inside-closed-component');
    assert.equal(state([2,0,0],mesh).status,'clear');
    """)


def test_transformed_nonuniform_and_mirrored_solids() -> None:
    run_js("""
    const parent = new THREE.Group(), mesh = cube(); parent.add(mesh);
    parent.position.set(3,-4,2); parent.rotation.set(.4,.3,.7); mesh.scale.set(-2,.5,3);
    parent.updateMatrixWorld(true);
    const transform = (p) => new THREE.Vector3(...p).applyMatrix4(mesh.matrixWorld).toArray();
    assert.equal(G.qualifyProbeAnchor(transform([0,0,0]),[mesh]).status,'blocked');
    assert.equal(G.qualifyProbeAnchor(transform([1.5,0,0]),[mesh]).status,'clear');
    mesh.scale.set(0,1,1); parent.updateMatrixWorld(true);
    assert.equal(G.qualifyProbeAnchor([3,-4,2],[mesh]).status,'unknown');
    """)


def test_merged_hollow_desk_uses_components_not_assembly_box() -> None:
    run_js("""
    // A single merged mesh with four separate legs and an apron. The anchor is
    // inside the assembly AABB and outside every actual closed solid.
    const positions = [];
    const part = (size, center) => {
      const g = new THREE.BoxGeometry(...size).toNonIndexed();
      g.translate(...center); positions.push(...g.attributes.position.array);
    };
    for (const x of [-1,1]) for (const y of [-1,1]) part([.2,.2,2],[x,y,0]);
    part([2.2,.2,.3],[0,-1,.7]);
    const g = new THREE.BufferGeometry(); g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
    const desk = new THREE.Mesh(g,new THREE.MeshStandardMaterial());
    const clear = state([0,0,0],desk);
    assert.equal(clear.status,'clear'); assert.equal(clear.receipts[0].components,5);
    assert.equal(state([1,1,0],desk).reason,'inside-closed-component');
    assert.equal(state([0,-1,.7],desk).status,'blocked');
    """)


def test_open_and_invalid_meshes_never_establish_a_solid() -> None:
    run_js("""
    const mesh = cube(); mesh.geometry = mesh.geometry.toNonIndexed();
    const a = mesh.geometry.attributes.position.array;
    mesh.geometry.setAttribute('position',new THREE.Float32BufferAttribute(a.slice(0,-18),3));
    assert.equal(state([0,0,0],mesh).status,'unknown');
    assert.equal(state([20,20,20],mesh).status,'clear');
    assert.equal(state([1.005,0,0],mesh).status,'blocked');
    const bad = cube(); bad.geometry.attributes.position.setX(0,NaN);
    assert.equal(state([0,0,0],bad).status,'unknown');
    const empty = new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshStandardMaterial());
    assert.equal(state([0,0,0],empty).status,'unknown');
    const partial = cube(); partial.geometry.setDrawRange(0,3);
    assert.equal(state([0,0,0],partial).status,'unknown');
    """)


def test_infinite_plane_clearance_after_transform_and_beyond_display_extent() -> None:
    run_js("""
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(2,2),new THREE.MeshStandardMaterial());
    plane.userData.isPlane = true;
    assert.equal(state([100,100,.005],plane).reason,'near-clip-intersects-plane');
    assert.equal(state([100,100,.02],plane).status,'clear');
    plane.rotation.y = Math.PI/2; plane.position.x = 3; plane.scale.set(2,3,4);
    assert.equal(state([3.005,20,20],plane).status,'blocked');
    assert.equal(state([3.02,20,20],plane).status,'clear');
    """)


def test_cache_invalidation_pose_and_version_and_distant_early_rejection() -> None:
    run_js("""
    const mesh = posed(cube()), cache = G.createSolidGeometryCache();
    const first = cache.snapshot(mesh); assert.equal(first.triangles,null);
    assert.equal(G.qualifyProbeAnchor([100,0,0],[mesh],{cache}).status,'clear');
    assert.equal(first.triangles,null);
    assert.equal(G.qualifyProbeAnchor([0,0,0],[mesh],{cache}).status,'blocked');
    assert.equal(cache.snapshot(mesh),first);
    mesh.position.x=5; mesh.updateMatrixWorld(true);
    assert.equal(G.qualifyProbeAnchor([0,0,0],[mesh],{cache}).status,'clear');
    assert.notEqual(cache.snapshot(mesh),first);
    const moved = cache.snapshot(mesh); mesh.geometry.attributes.position.needsUpdate=true;
    assert.notEqual(cache.snapshot(mesh),moved);
    const versioned = cache.snapshot(mesh); cache.invalidate(mesh);
    assert.notEqual(cache.snapshot(mesh),versioned);
    """)


def test_capture_ownership_excludes_fixed_robots_dynamic_unknown_and_alpha() -> None:
    run_js("""
    const mesh = cube(); mesh.userData={body:1,geom:7,group:2};
    const desc={bodies:[{id:1,dynamic:false}],geoms:[{id:7,body:1,group:2}],visibleGroupLimit:3};
    const context={desc,robotOfBody:new Map()};
    const reason=()=>G.classifyCaptureMesh(mesh,context).reason;
    assert.equal(G.classifyCaptureMesh(mesh,context).status,'eligible');
    context.robotOfBody.set(1,'fixed-arm'); assert.equal(reason(),'robot'); context.robotOfBody.clear();
    desc.bodies[0].dynamic=true; assert.equal(reason(),'dynamic');
    delete desc.bodies[0].dynamic; assert.equal(reason(),'unknown-body-motion'); desc.bodies[0].dynamic=false;
    mesh.material.transparent=true; assert.equal(reason(),'non-opaque'); mesh.material.transparent=false;
    mesh.material.opacity=.5; assert.equal(reason(),'non-opaque'); mesh.material.opacity=1;
    for (const flag of ['water','flex','rotor','overlay','helper']) {
      mesh.userData[flag]=true; assert.equal(reason(),flag); delete mesh.userData[flag];
    }
    desc.rotors=[{body:1}]; assert.equal(reason(),'rotor'); desc.rotors=[];
    desc.geoms[0].group=3; assert.equal(reason(),'collision-or-unknown-group'); desc.geoms[0].group=2;
    const parent=new THREE.Group(); parent.add(mesh); parent.visible=false; assert.equal(reason(),'hidden');
    parent.visible=true; context.excludedObjects=new Set([parent]); assert.equal(reason(),'excluded-subtree');
    context.excludedObjects.clear(); delete mesh.userData.geom; assert.equal(reason(),'unknown-geometry-ownership');
    """)


def test_rotated_solid_aabb_empty_corner_is_not_occupied() -> None:
    run_js("""
    const mesh=cube(); mesh.scale.set(2,.1,.3); mesh.rotation.z=Math.PI/4;
    assert.equal(state([1,-1,0],mesh).status,'clear');
    assert.equal(state([.5,.5,0],mesh).status,'blocked');
    """)


def test_float32_primitive_seams_and_ambiguous_opposed_nested_shells() -> None:
    run_js("""
    for (const geometry of [new THREE.SphereGeometry(1,16,8),new THREE.CylinderGeometry(1,1,2,16)]) {
      const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial());
      assert.equal(state([0,0,0],mesh).status,'blocked');
    }
    // An inner reversed shell might describe a cavity or an overlapping reversed
    // object. Do not silently claim either interpretation is an eligible anchor.
    const outer = new THREE.BoxGeometry(4,4,4).toNonIndexed().attributes.position.array;
    const inner = new THREE.BoxGeometry(2,2,2).toNonIndexed().attributes.position.array;
    const flipped=[];
    for(let i=0;i<inner.length;i+=9)
      flipped.push(...inner.slice(i,i+3),...inner.slice(i+6,i+9),...inner.slice(i+3,i+6));
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute([...outer,...flipped],3));
    const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial());
    const qualification=state([0,0,0],mesh);
    assert.equal(qualification.reason,'nested-or-overlapping-opposed-shells');
    assert.equal(qualification.status,'unknown');
    """)


def test_interleaved_position_updates_requalify_real_changed_solids() -> None:
    run_js("""
    const mesh=cube(), source=mesh.geometry.attributes.position;
    const values=new Float32Array(source.count*4);
    for(let i=0;i<source.count;i++) {
      values[i*4]=source.getX(i)+5;
      values[i*4+1]=source.getY(i);
      values[i*4+2]=source.getZ(i);
      values[i*4+3]=42; // unrelated interleaved field
    }
    const data=new THREE.InterleavedBuffer(values,4);
    const position=new THREE.InterleavedBufferAttribute(data,3,0);
    mesh.geometry.setAttribute('position',position); posed(mesh);
    const cache=G.createSolidGeometryCache();
    const qualify=()=>G.qualifyProbeAnchor([0,0,0],[mesh],{cache});
    assert.equal(qualify().status,'clear');
    for(let i=0;i<position.count;i++) position.setX(i,position.getX(i)-5);
    position.needsUpdate=true;
    const inside=qualify();
    assert.equal(inside.status,'blocked');
    assert.equal(inside.reason,'inside-closed-component');
    // A different backing buffer with the same version must invalidate too.
    const replacement=new THREE.InterleavedBuffer(data.array.slice(),data.stride);
    for(let i=0;i<position.count;i++) replacement.array[i*4]+=5;
    replacement.needsUpdate=true;
    assert.equal(replacement.version,data.version);
    position.data=replacement;
    assert.equal(qualify().status,'clear');
    """)
