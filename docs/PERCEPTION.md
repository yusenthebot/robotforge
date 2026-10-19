# Perception: maps, localization and going somewhere on the robot's own map

A robot can draw a map of where it goes, find itself in it, and plan its way by that map alone.
It needs only a lidar and an odometry, which `mapping` adds when a robot has none.
The code is in `src/robotforge/perception/` (grid, particle filter, mapper, exploring, scoring) and `behaviors/map_nav.py` (the planner on the robot's own grid).
No new dependency: numpy and MuJoCo, and the same A* and path follower as navigation on the scene (docs/NAVIGATION.md).

```text
mapping(robot="go2", action="start")                    # a 3D lidar and an odometry are added, the map starts
mapping(robot="go2", action="explore", options={"seconds": 120, "area": [-7.5, -3.4, 7.5, 5.4]})
mapping(robot="go2", action="status", options={"image": true})
mapping(robot="go2", action="localize", options={"init": "global"})   # find itself in the finished map from nowhere
move(robot="go2", to="rack_8", map="own")               # go there by its own map and its own pose estimate
read_sensors(robot="go2", names=["front_depth"], points=300)         # a depth camera's point cloud, thinned
```

## How it works

**The map.** An occupancy grid in the world's frame, 5 cm cells by default, grown as scans reach past its edge.
Every ray of a scan makes the cells it crosses a little more free and the cell where it ends a little more occupied (log odds, clamped), so a cell seen free from many scans and occupied from few settles on free.
A 3D lidar is flattened for this: hits between the floor and a hand above the lidar are obstacles, and its rays clear only the low air (half the lidar's height), because its level rays pass over a low crate and say nothing about the cells under it.
A flat 2D ring clears at its own height.

**The pose.** The odometry (an `odometry` sensor with its own noise, bias and drift) says how far the robot moved between two scans.
A particle filter (150 guesses by default) moves every guess by that step with its own noise, then weighs each guess by how well the scan's end points fall on the map's occupied cells (a likelihood field, 60 ends spread evenly in bearing, the nearly level rays only).
The weighted mean is the pose estimate; the scan is then added to the map at that pose.
The filter also learns a constant error of the odometry's turn (wheels of unequal radius, a bent gait) from how far the scans keep pulling the heading, and turns by it in advance, so a drift of several degrees a second is followed.
So the filter pulls the odometry's drift back wherever the map has walls to hold on to, and the map is drawn from the corrected poses (a particle-filter scan-matching mapper with one map, like Gmapping with one particle's map).
`pose="truth"` maps from the simulator's exact pose and `pose="odometry"` from the odometry alone: they exist to show what the filter buys.

**Localizing in a finished map.** `action="localize"` keeps the map as it is.
`init="pose"` starts the filter at the pose the robot believed.
`init="global"` tries the first scan at every place of a 20 cm lattice over the free cells within 9 m of a mapped wall, at every heading in 5 degree steps, and starts the 1500 particles where it fits best (drawn in proportion, softened).
While the robot has then traveled less than 6 m the scans' say is softened and a few new guesses are added at each resampling, so that a room like several others does not collapse the filter onto the wrong one.
A map with fewer than 30 wall cells cannot be matched against, and `localize` then adds a `warning` to its answer.

**Going somewhere.** `move(to=..., map="own")` plans on the robot's own grid: occupied cells are walls (inflated by the robot's size as on the scene's map), and cells it has not seen are walls too, so it plans only through space it has seen free.
The follower steers by the pose estimate, not the true pose.
While it goes, the path ahead is checked against the map once a second and planned again when something it has mapped since lies across it.
What the simulator still gives it is the destination: a name (`rack_8`) or a point is the task it was given, as an address is given to a courier; the way there is its own.
The result's `on_its_own_map` says where the robot believed it was when it stopped and how far off that was (the truth is read to say).
When the mapping was started with `pose="truth"` or `"odometry"` the robot steered by that, not by an estimate from its map, and the result says so in `pose_source`.

**Exploring.** `action="explore"` goes to the nearest frontier (a free cell beside unknown space, with enough unknown round it to matter) by the cost of its own path, again and again, until none is left that it can reach, the time is up (`seconds`, at most 300 per call; call again to go on) or it cannot get to the frontiers left.
`area` keeps it to part of the world: a scene with open sides has an unbounded frontier.
Exploring plans through unknown space as if it were open, since the frontier is where it ends; a trip to a named place does not.

## What it reports

`status` gives the map's size, the pose estimate (`localized`, `sigma_m`) with its `pose_error` against the true pose and the `odometry_error` beside it, and the map's `accuracy` against the scene.
The scene's shapes are rasterized onto the map's grid from the compiled model and used only to judge a map, never to build one.

| field | meaning |
|---|---|
| `precision` | share of mapped obstacle cells within 10 cm of a real obstacle |
| `wall_error_m` | mean and 95th percentile distance from a mapped obstacle cell to the nearest real one |
| `recall` | share of the real obstacle faces that border mapped free space that have a mapped obstacle within 10 cm |
| `free_correct` | share of mapped free cells that are free in the scene |
| `explored` | share of the scene's free cells, inside the area the robot has seen, that the map has seen |
| `unsafe_free` | share of what stands in the robot's way at any height that the map calls free: a 2D ring misses what is lower than it |

`accuracy.navigation` compares the robot's own map with the scene's for where it can go (`found`, `through_blocked`); when the robot stands in a cell its own map blocks it is scored from the nearest free cell within 0.8 m (`start_moved_m`), and when that is not possible the entry says why in `unscored`.

A 2D lidar is scored against the shapes within 5 cm of its plane, a 3D lidar against everything from the floor to a hand above it.

## The viewer

While any robot maps, the page draws the map on the floor (free space a teal about 30 percent opaque, what the robot found in the way hot pink, unknown clear), the particles as blue dots, the pose estimate as a blue arrow with the path it believes it walked, and the odometry's own pose as a grey arrow with its own path.
The grey one drifts away from the blue one until the filter pulls it back.
The drawer has a switch for it.
Seen from nearly straight above the map is drawn over the shelves and crates (it is laid on the floor, which they hide from there); from the side it stays under them.
When the last map goes (cleared, a reset, a mapper that failed) the page is told once and drops what it drew.
The message is `{"type": "map", "robots": [...]}` four times a second (viewer/server.py documents its fields); the grid is a deflated byte per cell, sent when it changed and at least every 3 s.

## Point clouds

A depth camera or a lidar gives its cloud with `read_sensors(points=N)`: world coordinates, every pixel or hit, thinned by voxel (the mean of the points in a cube) with the smallest cube that leaves at most N.
The camera's own robot is left out of its cloud.

## Measured: a Go2 in a warehouse it has never seen

`uv run python scripts/perception_bench.py --seed LAYOUT --filter-seed N --seconds 300` (2026-10-20, on the development Mac while other jobs shared it).
Nine runs: three layouts of the warehouse (`--seed`: it moves crates and parcels) times three seeds of the particle filter's random draws (`--filter-seed`), since one filter seed per layout hid how large the run-to-run spread is.
The Go2 starts at the warehouse's west end with a 16-ring 3D lidar (8 Hz, 1 cm noise) and an odometry that reads its turn 0.003 rad/s off (so after a minute it believes it has turned 10 degrees); the filter has 150 particles.
It explores the warehouse floor, then the map is kept and the robot is asked to find itself from nowhere (2500 particles, walking about 10 m), then goes to the rack furthest from where it stands, by its own map alone.
The numbers below are all that the benchmark reports; none is left out for looking bad.

**How exploring ended.** All nine runs ended "explored" (no frontier is left that the robot can reach) after 141 to 220 s of simulated time (11 to 13 trips, one frontier it could not reach each time, 65 to 118 s of computer time).
An earlier 150 s limit ended five of six runs "out_of_time" with about 70 percent of the floor seen; that was a time limit, not a finished map.
"Explored" is not "everything": the scene's free cells inside the area the robot has seen are 76 to 84 percent mapped (`explored`), because what it never reaches or sees is the inside of the racks, the cells behind the pallets and what lies beyond the open sides.
The map's free area (1330 to 1470 m2) is larger than the warehouse's floor (about 130 m2) since the lidar sees out through the open sides and clears the air there.

| over nine runs | min | median | max |
|---|---|---|---|
| pose error while exploring, rms | 0.11 m | 0.19 m | 0.30 m |
| pose error while exploring, max | 0.21 m | 0.37 m | 0.64 m |
| the odometry alone, rms | 1.50 m | 1.73 m | 2.65 m |
| the odometry alone, max | 2.83 m | 3.26 m | 8.01 m |
| mapped obstacle cells near a real obstacle (`precision`) | 0.81 | 0.91 | 1.00 |
| real obstacle faces that have a mapped obstacle (`recall`) | 0.40 | 0.45 | 0.59 |
| f1 of the two | 0.55 | 0.60 | 0.74 |
| mean distance of a mapped wall to the real one | 0.5 cm | 2.6 cm | 6.0 cm |
| what stands in the robot's way at any height, mapped free (`unsafe_free`) | 55 % | 67 % | 72 % |
| what the scene lets it reach that its map lets it reach (`found`) | 89 % | 92 % | 94 % |
| cells it would plan through that the scene says are blocked | 10 % | 13 % | 15 % |
| walked before the search from nowhere held (spread under 0.25 m and true error under 0.3 m, judged against the truth) | 0.0 m | 4.9 m | 11.5 m |
| walked before the filter's own spread said 0.25 m, which is all it can know | 0.0 m | 0.5 m | 7.3 m |
| true error at that moment | 0.02 m | 0.26 m | 0.85 m |
| error at the end of the search | 0.01 m | 0.05 m | 0.11 m |

Nine runs, nine arrivals at a named rack by the robot's own map and pose (rack_4 six times, rack_5 once, rack_8 twice): 3.0 to 6.9 m walked, the robot's middle 0.33 to 0.68 m from the rack's face (the scene's planner stops 0.55 to 0.57 m away), and where it believed it was off by 0.04 to 0.23 m when it stopped.

Reading it:

- The filter keeps the pose within a few decimeters where the odometry alone is 1.5 to 2.7 m off, and 8 m off in the worst run; where the walls are close (the aisle, the pallets) it holds to 2 to 5 cm.
  It drifts most on the open floor beside the racks, where the only things to match are far racks' uprights.
  Which run is good depends a lot on the filter's seed: on one layout the rms goes from 0.11 to 0.27 m between seeds.
- The map is as good as the pose: 0.5 cm to 6 cm mean distance to the real wall.
  When the pose wanders by 0.3 to 0.6 m for a stretch, the walls drawn then are doubled, which is what the lower precision (0.81) is.
- **Recall is low and `unsafe_free` is high, and that is the map's main weakness.**
  Racks are open shelving: the lidar's rays go in through the open front and stop at the back, so a rack is mapped as a hollow box (its outline occupied, its inside free).
  Of the cells that stand in the robot's way at any height, 55 to 72 percent are mapped free, and 10 to 15 percent of the cells the planner would go through are blocked in the scene, mostly rack pockets, so the planner could route into one (it did not in these runs).
  The loading dock is the other way round: a block to the lidar, a ramp to the scene's planner.
- The search from nowhere is not reliable at first sight: in a hall of identical racks the filter says its spread is under 0.25 m after 0 to 7.3 m of walking, and when it says so it is 0.26 m off at the median and 0.85 m off at worst (two of nine runs were 0.37 m off or more when they claimed to be sure).
  Judged by the truth it holds after 0 to 11.5 m of walking (median 4.9 m).
  The final error after the search walk is small (1 to 11 cm) in all nine, so it does find itself, later than it first claims.
- A separate check of nine searches on layout 1 (three filter seeds from three start points, run by the reviewer of this work) found the filter confident and right at first sight (0.0 to 0.1 m walked, error 0.02 to 0.25 m), so how soon it holds depends on where it starts and what it sees first.

Point clouds of the same view by a depth camera and by a 6 cm stereo pair (scripts/perception_figs.py): the stereo pair's depth falls in slabs a disparity step apart.
The cost of mapping while a Go2 walks the warehouse is about 6 ms per scan, independent of how large the map has grown (the distance field and the scan work on the box a change touches), about 10 percent of the wall time of a walking Go2 on the development Mac.

## Limits

- The mapper is 2D: one floor, no stairs.
  A ramp or loading dock is mapped as what the lidar sees of it, a solid block: the scene's own planner climbs it, the robot's own map plans round it.
- It needs walls or objects to hold on to: in a wide open space the filter has nothing to match and the pose drifts as the odometry does, until it sees something.
- A moved object stays where it was mapped; a person walking through a scan leaves little, but a crate pushed to another place leaves a ghost until rays clear it.
- Global localization can settle on the wrong one of two places that look alike; it needs the robot to move and see something that tells them apart.
- Another robot is a wall on the map while it stands still.
