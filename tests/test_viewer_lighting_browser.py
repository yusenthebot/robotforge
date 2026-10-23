"""Real GPU contact, shadow bounds and movement checks, independent of screenshot aesthetics."""

from __future__ import annotations

import asyncio

from test_viewer_layout import pytestmark, viewer  # noqa: F401

SETUP = """
const THREE=await import(new URL('static/vendor/three/three.module.min.js',location).href);
const {createLighting}=await import(new URL('static/lighting.js',location).href);
const {renderer}=window.__robotforge.three(), scene=new THREE.Scene(), rig=createLighting(scene,renderer);
const world=new THREE.Group(); scene.add(world);
const camera=new THREE.PerspectiveCamera(45,1,.001,100); camera.up.set(0,0,1);
const floorNode=new THREE.Group(), cubeNode=new THREE.Group(); world.add(floorNode,cubeNode);
const floor=new THREE.Mesh(new THREE.PlaneGeometry(40,40),new THREE.MeshStandardMaterial({color:0xaaaaaa,roughness:1}));
floor.receiveShadow=true; floorNode.add(floor);
const cube=new THREE.Mesh(new THREE.BoxGeometry(.1,.1,.1),new THREE.MeshStandardMaterial({color:0x557799}));
cube.castShadow=cube.receiveShadow=true; cube.userData.body=1; cubeNode.add(cube);
const desc={bodies:[{dynamic:false},{dynamic:false}],geoms:[{type:'plane',pos:[0,0,0],quat:[1,0,0,0],body:0}]};
const target=new THREE.WebGLRenderTarget(320,320), oldTarget=renderer.getRenderTarget();
function draw(){renderer.setRenderTarget(target);renderer.render(scene,camera);}
function check(value,message){if(!value)throw Error(message);}
function inside(point){scene.updateMatrixWorld(true);rig.sun.shadow.updateMatrices(rig.sun);
  const p=point.clone().project(rig.sun.shadow.camera);return [p.x,p.y,p.z].every(v=>v>=-1.00001&&v<=1.00001);}
function cleanup(){renderer.setRenderTarget(oldTarget);target.dispose();
  world.traverse(o=>{o.geometry?.dispose();o.material?.dispose()});
  scene.environment.dispose();rig.sun.shadow.map?.dispose();rig.bind(null);}
"""


def test_shadow_rig_covers_transformed_planes_mocap_rotors_and_flexes(viewer):  # noqa: F811
    async def scenario(tab, _server):
        result = await tab.run(
            SETUP
            + """
          try {
            const direction=rig.sun.position.clone().sub(rig.sun.target.position).normalize();
            check(direction.dot(new THREE.Vector3(-7,-9,11).normalize())>1-1e-10,'empty-world key direction');
            floorNode.position.z=-10; floorNode.rotation.x=.2; cubeNode.position.z=1;
            camera.position.set(2,-3,2);camera.lookAt(0,0,0);
            rig.bind(world,[floorNode,cubeNode],desc);rig.update();draw();
            const caster=new THREE.Vector3(0,0,1), normal=new THREE.Vector3(0,-Math.sin(.2),Math.cos(.2));
            const plane=new THREE.Plane().setFromNormalAndCoplanarPoint(normal,new THREE.Vector3(0,0,-10));
            const receiver=caster.addScaledVector(direction,-plane.distanceToPoint(caster)/normal.dot(direction));
            check(inside(receiver),'transformed floor receiver is clipped');
            const first=rig.state();cubeNode.position.x=8;rig.update();draw();
            check(rig.state()!==first&&inside(new THREE.Vector3(8,0,1)),'fixed-class/mocap movement clipped');
            const still=rig.state();camera.position.set(-2,3,5);camera.lookAt(0,0,0);rig.update();draw();
            check(rig.state()===still,'view camera moved world shadows');
            desc.rotors=[{body:1,center:[0,0,0],radius:.4}];
            rig.bind(world,[floorNode,cubeNode],desc);rig.update();draw();
            for(let i=0;i<16;i++){const a=i*Math.PI/8;
              check(inside(new THREE.Vector3(8+.4*Math.cos(a),.4*Math.sin(a),1)),'rotor sweep clipped');}
            const cloth=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshStandardMaterial());
            cloth.geometry.setAttribute('position',new THREE.Float32BufferAttribute([0,0,1,.1,0,1,0,.1,1],3));
            cloth.castShadow=true;cloth.userData.flex=true;world.add(cloth);
            rig.bind(world,[floorNode,cubeNode],desc);rig.update();
            cloth.geometry.attributes.position.setX(0,12);cloth.geometry.attributes.position.needsUpdate=true;
            rig.update();draw();check(inside(new THREE.Vector3(12,0,1)),'deformed cloth clipped');
            return {fit:rig.state(),gl:renderer.getContext().getError()};
          } finally {cleanup()}
        """
        )
        assert result["gl"] == 0
        assert 0 < result["fit"]["normalBias"] <= 0.004
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


def test_real_shadow_distinguishes_contact_from_a_raised_object_and_can_be_disabled(viewer):  # noqa: F811
    async def scenario(tab, _server):
        result = await tab.run(
            SETUP
            + """
          try {
            cubeNode.position.z=.05;camera.position.set(.5,.6,.5);camera.lookAt(0,0,0);
            rig.bind(world,[floorNode,cubeNode],desc);rig.update();
            camera.updateMatrixWorld();
            const probe=new THREE.Vector3(.06,.07,0).project(camera), pixels=new Uint8Array(4);
            function sample(shadows){rig.sun.castShadow=shadows;draw();
              renderer.readRenderTargetPixels(target,Math.floor((probe.x+1)*160),Math.floor((probe.y+1)*160),1,1,pixels);
              return Array.from(pixels).slice(0,3);}
            const on=sample(true),off=sample(false);
            cubeNode.position.z=.2;rig.update(true);const raisedOn=sample(true),raisedOff=sample(false);
            const diff=(a,b)=>Math.max(...a.map((v,i)=>Math.abs(v-b[i])));
            return {contact:diff(on,off),raised:diff(raisedOn,raisedOff),on,off,raisedOn,raisedOff,
              gl:renderer.getContext().getError()};
          } finally {cleanup()}
        """
        )
        assert result["contact"] >= 5, result
        assert result["raised"] <= 2, result
        assert result["gl"] == 0
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


def test_driving_beyond_initial_shadow_bounds_keeps_coverage_and_reset_refits(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.until("!!window.__robotforge.three().lighting.state()")
        initial = await tab.eval("window.__robotforge.three().lighting.state()")
        await tab.page.call("Input.dispatchKeyEvent", type="keyDown", key="w", code="KeyW", text="w")
        for _ in range(80):
            if server.pg.session.data.qpos[0] > 1.2:
                break
            await asyncio.sleep(0.1)
        await tab.page.call("Input.dispatchKeyEvent", type="keyUp", key="w", code="KeyW")
        assert server.pg.session.data.qpos[0] > 1.2
        moved = await tab.eval("window.__robotforge.three().lighting.state()")
        assert moved != initial
        assert await tab.run("""const {sun}=window.__robotforge.three();
          const THREE=await import(new URL('static/vendor/three/three.module.min.js',location).href);
          const id=window.__robotforge.scene.bodies.find(b=>b.name.startsWith('rover/')).id;
          const point=new THREE.Vector3().fromArray(window.__robotforge.bodyPosition(id)).project(sun.shadow.camera);
          return [point.x,point.y,point.z].every(v=>Math.abs(v)<=1.001);""")
        await tab.page.call("Input.dispatchKeyEvent", type="keyDown", key=" ", code="Space", text=" ")
        await tab.page.call("Input.dispatchKeyEvent", type="keyUp", key=" ", code="Space")
        for _ in range(40):
            if server.pg.session.paused:
                break
            await asyncio.sleep(0.05)
        assert server.pg.session.paused
        await tab.page.call("Input.dispatchKeyEvent", type="keyDown", key="r", code="KeyR", text="r")
        await tab.page.call("Input.dispatchKeyEvent", type="keyUp", key="r", code="KeyR")
        await tab.until("window.__robotforge.time < .2", timeout=5)
        reset = await tab.eval("window.__robotforge.three().lighting.state()")
        assert reset != moved
        assert abs(reset["x"] - initial["x"]) < 0.05
        assert abs(reset["y"] - initial["y"]) < 0.05
        assert server.pg.session.data.qpos[0] < 0.1
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("rover", template="rover"))
