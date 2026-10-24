"""Actual viewer rendering honors native PBR without replacing named visual contracts."""

from __future__ import annotations

import asyncio

import pytest
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_native_textures_browser import image_robot  # noqa: F401

READY = "window.__robotforge?.scene && window.__robotforge.renderedVersion===window.__robotforge.scene.version"
INSPECT = """(()=>{const rf=window.__robotforge,rows={};rf.three().scene.traverse(o=>{
  if(!o.isMesh||!Number.isInteger(o.userData.geom))return;
  const g=rf.scene.geoms.find(g=>g.id===o.userData.geom);
  if(!g?.name.startsWith('coated/'))return;
  const m=o.material;rows[g.name.split('/').pop()]={roughness:m.roughness,metalness:m.metalness,
    physical:!!m.isMeshPhysicalMaterial,coat:m.clearcoat??0,opacity:m.opacity,transparent:m.transparent,
    depthWrite:m.depthWrite};});return {rows,gl:rf.three().renderer.getContext().getError()};})()"""


@pytest.fixture
def native_robot(tmp_path):
    xml = tmp_path / "native.xml"
    names = ["white", "black", "rf_carbon", "rf_unknown", "rf_gloss", "constructor"]
    materials = "".join(f'<material name="{name}" roughness="0" metallic="1"/>' for name in names)
    geoms = "".join(
        f'<geom name="{name}" type="sphere" size=".15" material="{name}" pos="{i * 0.4} 0 .2"/>'
        for i, name in enumerate(names)
    )
    xml.write_text(
        f"<mujoco><asset>{materials}</asset><worldbody><body>{geoms}</body></worldbody></mujoco>", encoding="utf-8"
    )
    return xml


@pytest.mark.parametrize("size", ["laptop", "phone"])
def test_actual_materials_use_native_zero_values_and_keep_registered_and_known_rf_priority(viewer, native_robot, size):  # noqa: F811
    async def scenario(tab, _server):
        await tab.until(READY)
        result = await tab.eval(INSPECT)
        assert result["gl"] == 0
        for name in ["white", "black", "rf_unknown", "constructor"]:
            assert result["rows"][name] == {
                "roughness": 0,
                "metalness": 1,
                "physical": False,
                "coat": 0,
                "opacity": 1,
                "transparent": False,
                "depthWrite": True,
            }
        assert result["rows"]["rf_carbon"]["roughness"] == 0.42
        assert result["rows"]["rf_carbon"]["metalness"] == 0.1
        assert result["rows"]["rf_carbon"]["physical"] is True
        assert result["rows"]["rf_carbon"]["coat"] == 1
        assert result["rows"]["rf_gloss"]["roughness"] == 0.26
        assert result["rows"]["rf_gloss"]["metalness"] == 0
        assert result["rows"]["rf_gloss"]["coat"] == 0.85
        assert not tab.errors

    viewer(size, "en", scenario, world=lambda pg: pg.add_robot("coated", file=str(native_robot)))


def test_native_pbr_does_not_replace_interior_image_alpha_or_uv(viewer, image_robot):  # noqa: F811
    image_robot.write_text(
        image_robot.read_text(encoding="utf-8")
        .replace('name="printed" texture="label"', 'name="printed" roughness=".8" metallic="0" texture="label"')
        .replace('rgba="1 1 1 1"', 'rgba="1 1 1 .4"'),
        encoding="utf-8",
    )

    async def scenario(tab, _server):
        await tab.until(READY)
        result = await tab.run("""
          const rf=window.__robotforge;let mesh;rf.three().scene.traverse(o=>{
            if(o.isMesh&&rf.scene.geoms.find(g=>g.id===o.userData.geom)?.name==='coated/panel')mesh=o;});
          const m=mesh.material;return {roughness:m.roughness,metalness:m.metalness,physical:!!m.isMeshPhysicalMaterial,
            map:!!m.map,uv:mesh.geometry.attributes.uv.count,image:[m.map.image.width,m.map.image.height],
            opacity:m.opacity,transparent:m.transparent,depthWrite:m.depthWrite,vertexColors:m.vertexColors,
            gl:rf.three().renderer.getContext().getError()};
        """)
        assert result["roughness"] == pytest.approx(0.8) and result["metalness"] == 0
        assert result["map"] and result["uv"] > 0 and result["image"] == [64, 64]
        assert result["opacity"] == 0.4 and result["transparent"] and not result["depthWrite"]
        assert not result["physical"] and not result["vertexColors"] and result["gl"] == 0 and not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("coated", file=str(image_robot)))


def test_legacy_calls_partial_values_invalid_fields_and_exact_name_priority(viewer):  # noqa: F811
    async def scenario(tab, _server):
        result = await tab.run("""
          const T=await import('three'), {surfaceMaterial}=await import('./static/look.js');
          const color=new T.Color(.3,.4,.5), output=[];
          const cases=[['white',undefined,undefined],['glass',undefined,{roughness:.8}],
            ['white_metal',undefined,{metalness:.6}],['parent/rf_carbon',undefined,{roughness:0,metalness:1}],
            ['glass',undefined,{roughness:null,metalness:'1'}],
            ['__proto__',{}, {roughness:0,metalness:1}],
            ['custom',{'custom':{roughness:.7,metalness:.2,clearcoat:.3}},{roughness:0,metalness:1}]];
          for(const [name,surfaces,native] of cases){const m=surfaceMaterial(name,color,.4,surfaces,native);
            output.push({r:m.roughness,m:m.metalness,physical:!!m.isMeshPhysicalMaterial,coat:m.clearcoat??0,
              transparent:m.transparent,opacity:m.opacity,depthWrite:m.depthWrite});m.dispose();}return output;
        """)
        expected = [
            (0.34, 0, True, 0.55),
            (0.8, 0, False, 0),
            (0.5, 0.6, False, 0),
            (0.42, 0.1, True, 1),
            (0.06, 0, True, 1),
            (0, 1, False, 0),
            (0.7, 0.2, True, 0.3),
        ]
        for value, (roughness, metalness, physical, coat) in zip(result, expected, strict=True):
            assert value == {
                "r": roughness,
                "m": metalness,
                "physical": physical,
                "coat": coat,
                "transparent": True,
                "opacity": 0.4,
                "depthWrite": False,
            }
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


def test_scene_replacement_and_context_restore_keep_current_native_materials(viewer, native_robot):  # noqa: F811
    async def scenario(tab, server):
        await tab.until(READY)
        version = server.pg.session.version
        native_robot.write_text(
            native_robot.read_text(encoding="utf-8").replace(
                'roughness="0" metallic="1"', 'roughness=".8" metallic="0"'
            ),
            encoding="utf-8",
        )
        server.pg.new_world("replacement")
        server.pg.add_robot("coated", file=str(native_robot))
        server.pg.session.paused = True
        await tab.until(f"{READY} && window.__robotforge.scene.version>{version}")
        assert (await tab.eval(INSPECT))["rows"]["black"]["roughness"] == pytest.approx(0.8)
        await tab.run("window.__robotforge.lookAt([3,-4,3],[1,0,.2])")
        await asyncio.sleep(0.8)
        await tab.run("""
          const gl=window.__robotforge.three().renderer.getContext();
          window.__pbrExt=gl.getExtension('WEBGL_lose_context');
          if(!window.__pbrExt)throw Error('No context-loss extension');window.__pbrRestored=false;
          document.getElementById('view').addEventListener('webglcontextrestored',()=>window.__pbrRestored=true,{once:true});
          window.__pbrExt.loseContext();
        """)
        await asyncio.sleep(0.2)
        await tab.run("window.__pbrExt.restoreContext()")
        await tab.until(f"window.__pbrRestored && {READY} && document.getElementById('loading').hidden")
        restored = await tab.eval(INSPECT)
        assert restored["rows"]["black"]["roughness"] == pytest.approx(0.8)
        assert restored["rows"]["black"]["metalness"] == 0 and restored["gl"] == 0 and not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("coated", file=str(native_robot)))
