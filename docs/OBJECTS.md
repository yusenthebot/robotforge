# Objects: real things to pick up, and tasks that are hard

A cube is a fine first thing to pick up, and a poor last one.
robotforge ships generators for things people actually handle: a mug with a handle, a bottle with a cap and a label, a can, a bowl, fruit, a cereal box, a book, a screwdriver, a hammer, a wrench, a spatula, a pen, a roll of tape, a sponge, a pitcher, a coil of cable.
Each is parametric and seeded, with a modelled mesh, convex collision pieces, a mass and friction from what it is made of, and grasp hints that the arm planner uses.

## Adding one

```
add_object(name="m", shape="mug")                                    # by name
add_object(name="m", shape="mug", seed=3)                             # a variation (the same seed is the same mug)
add_object(name="m", describe="a tall blue mug with a big handle")   # by description
add_object(name="m", shape="mug", params={"handle": "square"})        # fix what a shape lets vary
```

A description is read for the shape (a mug, a wine bottle, "a fruit" or "a tool", one of them by seed), a color (red, dark blue, "#9c6b3f"), a size (tall, tiny, "12 cm tall") and the parameters the shape has ("a half full soda bottle", "a paper coffee cup with a lid", "a standing book").
The answer's notes say what was understood, so a wrong reading is seen at once.
What the caller gives outright (`shape`, `size`, `color`, `params`) wins over the description.
A seed with no color picks one from the shape's palette.
`world.yaml` keeps `shape`, `size`, `color`, and, only for generated objects, `seed` and `params`, so a world saved before generators existed saves exactly as it did.

An object's frame has its origin at the middle of its base and z up, resting as it is meant to rest (a screwdriver and a hammer lie on their sides).

| shape | size | material | what varies (`params`) |
|---|---|---|---|
| mug | [diameter, height] | ceramic | handle round, square, tall; handle_size; taper; wall; stripe |
| cup | [diameter, height] | plastic, paper or glass | style tumbler, paper, glass; lid; taper |
| bowl | [diameter, height] | ceramic | round; wall |
| pitcher | [diameter, height] | plastic or ceramic | material; handle round, tall; wall; stripe |
| plate | [diameter, height] | ceramic | rim |
| bottle | [diameter, height] | plastic or glass, with water | style water, soda, wine; fill; label; cap |
| can | [diameter, height] | aluminium, with drink | style soda, soup, tuna; fill |
| squeeze_bottle | [length, width, height] | plastic | cap red, white, black (mustard or ketchup) |
| apple, orange, pear | [diameter] | fruit | squat, leaf; bumps; length |
| banana | [length] | fruit | curve; ripe |
| package | [length, width, height] | cardboard | style cereal, tea, parcel; density |
| tin | [length, width, height] | steel | |
| brick | [length, width, height] | foam | |
| sponge | [length, width, thickness] | foam | scour green, blue, white |
| tape | [diameter] | paper | width; style masking, duct, painter |
| book | [length, width, thickness] | paper | lying (flat) or standing |
| screwdriver | [length] | plastic and steel | tip flat, phillips; grip |
| hammer | [length] | steel head, wooden or rubber shaft | handle; head (mass) |
| wrench | [length] | steel | finish chrome, black; size_mm |
| spatula | [length] | plastic blade, wooden or plastic handle | handle; blade_width; slots; grip |
| pen | [length] | plastic | radius |
| cable | [coil diameter] | rubber, steel plug | turns; gauge; plug usb, barrel |

`objects.catalogue()` lists the same from the code, with each parameter's range or choices.

## What a generated object is

- **Look**: a mesh built from lathes, lofts and tubes, with a material preset (glossy ceramic, clear glass, brushed steel, rubber, paper).
  Labels are printed as geometry: a few hundred flat quads a fraction of a millimetre over the surface (text in a 5 by 7 pixel font, bars, a barcode), so they look the same in the viewer and in MuJoCo's renderer.
- **Collision**: convex pieces only.
  A hollow mug is a floor disc and a ring of boxes round its wall, with a handle of capsules, so a ball dropped in stays in it.
  `robotforge.objects.collision.decompose(mesh)` cuts a mesh a generator only drew into convex pieces with CoACD (seconds to a minute, kept on disk by content).
- **Mass and friction**: the volume of each mesh times the density of its material (`robotforge.objects.materials`), a bottle's and a can's water on top; friction is the material's, with torsional friction so a pinch can hold a mug by its handle.
  `mass=` on `add_object` replaces the total.
- **Grasp hints**: `objects.object_for("mug").grasps`: where the tool point goes, which way the jaws close, how wide they must open.
  A mug is taken by its handle, a bottle round its body, a tool across its shaft, a plate or a bowl by the rim, a package across its narrow side.

## Writing your own generator

See "Generators" in [EXTENDING.md](EXTENDING.md): a function over a builder, with a mesh, convex pieces, a material, grasp hints and parameters, saved as a plugin file; it is a shape for `add_object` and understood by `describe=` from its `keywords` and the choices of its `params`.

## Hard tasks

`add_task` reads these sentences, finds the things by what they are (and their color), adds nothing it cannot find (it says what to add), and saves a task with a check that looks at how things ended up.

| sentence | what the check wants |
|---|---|
| put the mug on the shelf upright | the mug on a board of the shelf, within 15 degrees of upright and at rest |
| stack the brick on the tin on the book | each thing standing on the one below, within 12 degrees of level and inside its top |
| insert the bottle into the rack | the bottle in a socket, down to the socket floor, standing |
| sort the things in the bin into the trays by kind | every thing in the tray named for its kind (fruit, cans, boxes, bottles, tools) |
| pour the balls from the cup into the bowl | three quarters of the balls that were in the cup lie in the bowl |
| hand the hammer to the other arm | the other arm holds it at least 4 cm up and the first arm has let go |

Fixtures that go with them are shapes too: `rack` (sockets for bottles; `params={"bottle": diameter}` sizes them), `tray` (a low tray; `params={"tag": "fruit"}` prints a name on it), `shelf`.
A thing is put on a slot of a fixture with the usual `pick_and_place(robot, "bottle", "rack")` (the first free socket, or `"rack/2"`), `"tray"` (spread out, a full tray says so) or `"shelf"` (a top board).
`pick_and_place(..., pour=true)` takes a cup (or a bowl) by its body, carries it over the target's mouth, tilts it about its lip in joint space (the lip stays over the target, the wrist shakes so a heap of balls cannot jam), tips it back and puts it down where it stood.
A cup with a lid, or one that can only be held by a handle, is refused before the arm moves, saying why.
A hand-over goes through the table: two grippers cannot share a shaft, so the first arm puts the tool down at a spot both arms reach (`pick_and_place("arm", "hammer", [0.32, 0.0])`), moves out of the way (`move_arm`), and the second takes it and keeps it lifted with `pick_and_place("partner", "hammer", "hold")` (`held`, `lifted_cm` in the answer).
`scripts/task_suite.py --hard` writes a suite of these worlds, one task each, in the layout `scripts/agent_eval.py --suite` runs.

### How often the scripted expert does them

`scripts/objects_tasks.py` plays each task as an agent would, with the real Menagerie models (a Franka Panda, and a UR5e with a Robotiq 2F-85) over seeds, and writes every failure with its cause (refused, never lifted, slipped, tipped over, landed outside, knocked something over, spilled).
Measured on 2026-10-20 with the real Menagerie models, six seeds each (the seed moves the things, picks the objects' variations and the number of balls):

| task | Franka Panda | UR5e with Robotiq 2F-85 |
|---|---|---|
| mug on the shelf upright | 6/6 | 6/6 |
| stack a brick, a tin and a book | 6/6 | 6/6 |
| bottle into the rack | 4/6 | 6/6 |
| sort a crowded bin into three trays | 4/6 | 6/6 |
| pour balls from a cup into a bowl | 6/6 | 4/6 |
| hand a hammer or a wrench to the other arm | 6/6 | 3/6 |

Why the rest failed, as the script reports it:

- Panda, rack: the bottle stands in its socket at release and falls over after the fingers leave, rolling out of the rack (2 of 6).
- Panda, bin: an apple in a corner was never lifted (the open fingers met a neighbour), and a tin tipped over 27 degrees in its tray.
- UR5e, pour: 3 of 6 and 3 of 8 balls reached the bowl; the rest stayed in the cup, jammed against its wall, through the tilt and the wrist shake.
- UR5e, hand-over: the wrench slips out of the Robotiq pads on the way or is never lifted (3 of 3 failures are the wrench; the hammer is 3/3).

Single objects, `scripts/objects_bench.py` (pick each shape from the table and put it in a tray, three seeds each): the Panda 53/55 (the two misses tipped over in the tray), the UR5e 49/55 (every plate, which slips out of the Robotiq pads on the way, and every wrench, which is never lifted).

The grip is where the arms differ: the Panda's pads have torsional friction, the Robotiq pads in the Menagerie have none, so a firm hold (pad torsion, squeeze, force range raised) is switched on while an object is carried and put back after.
