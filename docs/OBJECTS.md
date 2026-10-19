# Objects: real things to pick up, and tasks that are hard

A cube is a fine first thing to pick up, and a poor last one.
robotforge ships generators for things people actually handle: a mug with a handle, a bottle with a cap and a label, a can, a bowl, fruit, a cereal box, a book, a screwdriver, a hammer, a wrench.
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
| plate | [diameter, height] | ceramic | rim |
| bottle | [diameter, height] | plastic or glass, with water | style water, soda, wine; fill; label; cap |
| can | [diameter, height] | aluminium, with drink | style soda, soup, tuna; fill |
| squeeze_bottle | [length, width, height] | plastic | cap red, white, black (mustard or ketchup) |
| apple, orange, pear | [diameter] | fruit | squat, leaf; bumps; length |
| banana | [length] | fruit | curve; ripe |
| package | [length, width, height] | cardboard | style cereal, tea, parcel; density |
| tin | [length, width, height] | steel | |
| brick | [length, width, height] | foam | |
| book | [length, width, thickness] | paper | lying (flat) or standing |
| screwdriver | [length] | plastic and steel | tip flat, phillips; grip |
| hammer | [length] | steel head, wooden or rubber shaft | handle; head (mass) |
| wrench | [length] | steel | finish chrome, black; size_mm |

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
