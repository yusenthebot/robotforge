# The asset engine

Robots in robotforge are drawn from parts that are generated, not modelled by hand.
A part is a named, parametric recipe (a *generator*) that makes three things which agree with each other:

- the drawn pieces: smooth, manufactured detail (filleted shells, parting lines, bolt circles, cable glands, tread) with a role and a surface material each;
- the collision shapes the physics touches: boxes, cylinders, capsules, spheres or convex meshes;
- the mass, centre of mass and inertia of those shapes at a density, so a part weighs what its shape says.

The same arguments always make the same asset.
Physics never reads the drawn pieces: dressing a robot in another style changes no mass, no joint and no collision shape (`tests/test_assets.py` compares them).

```python
from robotforge import assets

joint = assets.generate("joint_actuator", radius=0.04, length=0.09, style="lab")
joint.mass_properties(2700.0)  # kg, centre of mass and inertia of its collision shapes
joint.triangles, joint.bounds()  # what it costs to draw, how big it is
assets.catalog()  # every generator, its summary and its parameter ranges
```

## Generators

Every generator lists its parameters with a range.
A value outside the range is refused with the range in the message, so an agent that asks for `radius=5` is told what is allowed.

| Generator | Makes | Frame |
|---|---|---|
| `joint_actuator` | a rotary joint: barrel housing, rear cap with a nameplate, parting grooves, a dark output band, a flange with a bolt circle and a centre boss, a cable gland, a light ring or cooling ribs in some styles | axis z, output face on +z |
| `servo_case` | the case of a servo: rounded housing split by a parting line, a label, a connector, mounting ears with screws, a gear boss under the horn | centred, shaft on the +z face |
| `link_shell` | the shell between two joints: a tapered, filleted tube in two halves with a parting line, end plates, optional collars and a cable cover | along z from the root to the tip |
| `pedestal` | a robot's base: bolted mounting plate, column, groove, gland, light ring | z up from the mounting plane |
| `chassis` | a body shell: rounded edges, parting line, service hatch, vents, corner screws, rubber guards in rugged styles | centred, x forward |
| `wheel` | a tyre with slick, ribbed, lugged or knobby tread, a rim with spokes or a closed dish, a hub with bolts | axle z |
| `foot` | a rubber ball or flat pad foot with a metal collar | centre at the origin, shin from +z |
| `sensor_housing` | a camera, depth camera, lidar, IMU board or ultrasonic range finder | mounted on z = 0, looks along +z |
| `grip_pad` | a finger with a ridged rubber grip pad | root at z = 0, gripping face toward +x |
| `gripper_body` | the palm of a parallel gripper with side covers and guide rails on a bolted flange | flange on z = 0, closing along y |

`add_robot(name, template="asset", params={"generator": "wheel", "args": {"radius": 0.07}, "style": "rugged", "seed": 2})` puts one on a turntable in the viewer (the `turn` slider spins it), so what a generator makes can be looked at before a robot uses it.
`list_templates` lists the generators, their ranges, the styles and the materials.

## Styles

A style is one knob that changes how every generated part looks, not what it does.

| Style | Look | Shell | Joints and trim |
|---|---|---|---|
| `cobot` (default) | an industrial collaborative arm | white gloss, round sections | dark soft-touch bands, a thin light ring |
| `lab` | a research-lab machine | brushed and anodised aluminium, squarer sections | black end caps, many visible socket screws |
| `hobby` | a servo-and-bracket build | glossy printed plastic in bright colors, chunky | screws in every bracket |
| `rugged` | a field machine | painted cast housings with a rough skin, thick fillets | cooling ribs, rubber guards, bolts you can see from across the yard |

The fields of a style (`robotforge.assets.Style`): the surfaces of the shell, accent, structure, metal and rubber; the neutral colors; `fillet` (edge radius scale), `roundness` and `taper` of a link's section, `fasteners` (screw density), `panel_lines`, `cable_glands`, `ribs`, `bumpers`, `light_ring` and `wall` (how chunky the housings read).
`assets.for_material(style, "aluminum")` gives a style as it looks with a shell of bare aluminium, carbon or rubber; the builder and the arm template use it for their `material` field.
A plugin adds its own with `style(name, based_on="cobot", fillet=1.6)`.

The arm template takes `style` as a parameter: `add_robot("arm", template="arm", params={"dof": 3, "style": "lab"})`.

## Materials

A material is a surface the viewer draws.
MuJoCo's own material carries what its renderer reads (specular, roughness, metallic); the viewer is three.js and also gets a clear coat, sheen, glow and a procedural texture, in the scene description's `surfaces` table.
A plugin's new material therefore reaches the browser without a change to the viewer.

| Material | Surface | Texture |
|---|---|---|
| `gloss` | glossy painted shell | a faint orange peel |
| `soft_touch` | matte soft-touch plastic | a fine grain |
| `brushed` | brushed aluminium or stainless | streaks along the part's z axis |
| `anodized` | coloured anodised aluminium | micro-blasting |
| `cast` | painted cast housing | a rough skin |
| `powder` | powder-coated sheet metal | a fine texture |
| `tread` | moulded rubber | a fine grain |
| `knurled` | knurled steel | a diamond cross-hatch |

The older presets (`shell`, `dark`, `metal`, `aluminum`, `carbon`, `rubber`, `glass`, `led`, `wood`, ...) are still what imported models and props use.
Textures are drawn from the mesh's own coordinates: no image files and no UVs, they cost a few shader instructions, and where one feature is smaller than a pixel they fade out instead of shimmering.

## Collision shapes and mass

`asset.mass_properties(density)` adds up the colliders with the parallel-axis rule: a box, cylinder, capsule or sphere analytically, a closed mesh by the divergence theorem.
`assets.collide(spec, body, asset, density)` gives a MuJoCo body the asset's colliders at that density, and `tests/test_assets.py` checks that MuJoCo computes the same mass, centre and inertia for the body.
A concave part is split into convex pieces with `assets.hulls_of(mesh)` (CoACD, loaded only when asked).

## Drawing into a model

```python
from robotforge.assets import Palette, draw
from robotforge.design.visuals import Visuals

vis = Visuals(spec)
draw(vis, body, asset, Palette.of("lab", body=[0.9, 0.4, 0.1, 1.0]), pos, rot)
vis.finish()
```

`Palette` gives each role a color: `body` and `accent` are the robot's own, the rest follow the style.
`Visuals` merges the pieces per body, material and color into a handful of mesh geoms, makes them massless and contactless, and hides the primitives they cover.

## Extending

A plugin registers generators, materials and styles with `robotforge.extend` (the worked example is in [EXTENDING.md](EXTENDING.md)); they work everywhere a built-in one does: `assets.generate`, the `asset` template, the arm's `style`, `list_templates`, the viewer.
A generator is built once with its defaults when it is registered, so a mistake shows up with the plugin's file named.

## The viewer

The viewer draws every `rf_*` surface as a physically based material (`MeshPhysicalMaterial` where there is a clear coat or sheen), lit by a generated studio environment map, with soft shadows and neutral tone mapping.
`tests/test_viewer_js.py` parses the viewer's scripts and `scripts/bench.py` holds its frame time to a budget.

## Pictures

`scripts/assets_shots.py` has the shots that show the engine (`uv run python scripts/screenshots.py --only assets_arm3 assets_sheet --out DIR`): the 3-joint and 6-joint arms, a sheet of every generator in every style, the gallery designs, five robots made from a sentence, and a hexapod.
