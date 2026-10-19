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

**Going somewhere.** `move(to=..., map="own")` plans on the robot's own grid: occupied cells are walls (inflated by the robot's size as on the scene's map), and cells it has not seen are walls too, so it plans only through space it has seen free.
The follower steers by the pose estimate, not the true pose.
While it goes, the path ahead is checked against the map once a second and planned again when something it has mapped since lies across it.
What the simulator still gives it is the destination: a name (`rack_8`) or a point is the task it was given, as an address is given to a courier; the way there is its own.
The result's `on_its_own_map` says where the robot believed it was when it stopped and how far off that was (the truth is read to say).

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

A 2D lidar is scored against the shapes within 5 cm of its plane, a 3D lidar against everything from the floor to a hand above it.

## The viewer

While any robot maps, the page draws the map on the floor (free space a see-through teal, what the robot found in the way hot pink, unknown clear), the particles as blue dots, the pose estimate as a blue arrow with the path it believes it walked, and the odometry's own pose as a grey arrow with its own path.
The grey one drifts away from the blue one until the filter pulls it back.
The drawer has a switch for it.
The message is `{"type": "map", "robots": [...]}` four times a second (viewer/server.py documents its fields); the grid is a deflated byte per cell, sent when it changed and at least every 3 s.

## Point clouds

A depth camera or a lidar gives its cloud with `read_sensors(points=N)`: world coordinates, every pixel or hit, thinned by voxel (the mean of the points in a cube) with the smallest cube that leaves at most N.
The camera's own robot is left out of its cloud.

## Limits

- The mapper is 2D: one floor, no stairs.
  A ramp or loading dock is mapped as what the lidar sees of it, a solid block: the scene's own planner climbs it, the robot's own map plans round it.
- It needs walls or objects to hold on to: in a wide open space the filter has nothing to match and the pose drifts as the odometry does, until it sees something.
- A moved object stays where it was mapped; a person walking through a scan leaves little, but a crate pushed to another place leaves a ghost until rays clear it.
- Global localization can settle on the wrong one of two places that look alike; it needs the robot to move and see something that tells them apart.
- Another robot is a wall on the map while it stands still.
