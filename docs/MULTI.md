# Two robots in one world

Two robots in the same world can pass a thing from one to the other, and rovers that go different ways at the same time do not meet in the middle.

## Handing an object from one arm to another

```
pick_and_place(robot="left", object="cube", target="pad", hand_to="right")
```

The left arm picks the cube up and sets it down at a spot both arms reach.
It then moves out of the way, to a waiting pose short of the spot, and the right arm picks the cube up from there and puts it on the pad.
For two arms that stand apart, each out of reach of the other's side of the table.

- **The spot** (`behaviors/relay.py`) is a floor point both arms reach with the gripper pointing down, within 0.8 of each arm's reach and no nearer than 0.22 m to either base, away from other loose things and from the target.
  The easiest one comes first; each is tried by planning the giving arm's pick and set-down and a dry run of the receiving arm's pick and put, and the first that both plan is used.
  The result says where: `handed_over` has `from`, `to` and `at`.
- **Why not hand to hand.**
  I tried it first: the giver holds the cube up in the air and the receiver reaches in with its hand tilted and takes it.
  A Panda's hand is 20 cm wide and sits right over the cube, so the receiving hand met the giving hand and wrist: 3 to 5 cm into them with the receiving hand tilted 35 degrees, a 4 mm touch at 60 degrees with no way in, a pose at 75 degrees but no path to it; and held higher the Panda drops the cube.
  The relay does the same job with nothing in the air: the two arms are never near each other except for the one moment the giver is on its way to the waiting pose.
- **The waiting pose.**
  At its home pose a Panda holds its hand over the middle of the room, which is where the spot is; the receiving arm's hand met its forearm and stalled.
  The giver now goes to a point on the line from its base to the spot, 0.55 of the way, 0.45 m up, the tool pointing down.
- Both arms must stay where they are: an arm on a robot that drives or walks is refused (`pick_and_place` with `go` does the fetching for those).
  The call refuses before moving when the giving arm cannot reach the object (with the usual advice on where it can), when no spot suits both arms, when the receiving arm cannot put the cube on the target from any spot, or when the two arms are the same.

## Right of way at a crossing

Trips are planned as if the robot were alone, and a robot that finds another in its way plans round it (`navigate.py`, making way).
That is fine in the open, but two robots that both keep going to the same crossing meet in the middle and both swerve.
`behaviors/traffic.py` adds a rule for paths that cross:

1. Every robot on a trip publishes the path it has left, how fast it keeps to it and how big it is.
2. Two paths that cross make a conflict zone on each: the stretch within the two robots' radii plus 15 cm of the other's path.
3. The robot that gets to the zone first goes through. The other holds just short of the zone until it is clear.
   Which gets there first is read off the times to the zone; a tie goes to the trip that began first, then to the name that sorts first, so two robots never wait for each other.
4. A robot that is itself holding is not waited for, and after 25 s a robot goes on anyway and leaves the meeting to making way.
5. Paths that run side by side or head on are not crossings and are left to making way (a hold there would be a wait where a step aside costs nothing); a path that ends in the zone is left to it too, waiting for a robot that parks there would be for ever.

A trip's result says `gave_way_to` (the robots it held for) and `waited_s` when it did.

## Numbers

`uv run python scripts/multi_bench.py` (2026-10-20).

Handover: two Pandas 1.3 m apart, a 5 cm cube 0.2 to 0.45 m from the giving arm's base, a pad 0.35 m from the receiving arm's base on either side.

| run | on the pad | arms pressed into each other (over 5 mm) |
|---|---|---|
| seed 3, 24 handovers | 23 | 0 |
| seed 0, 16 handovers with the pad 0.2 m from the receiving base | 13 | 0 |

The misses are not in the relay.
A Panda's plain `pick_and_place` also misses a cube within about 0.3 m of its own base (six of six tried on the giving side) and a carry to a pad 0.2 m from its base on one side (from the middle to x 0.455, y 0.16 with the base at x 0.65: the cube is dragged along the floor); the relay inherits both.
It takes about 21 s of simulated time.

Crossing: rovers sent across each other's way to get to the middle together.

| meeting | closest, right of way | closest, making way only | time to both there, right of way | making way only |
|---|---|---|---|---|
| two rovers crossing | 0.71 m | 0.26 m | 9.1 s | 15.6 s |
| a T (one ends in the other's line) | 0.68 m | 0.25 m | 7.0 s | 10.6 s |
| three rovers from three sides | 0.56 m | 0.26 m | 11.7 s | 12.3 s |

Centres, 0.25 m being the rovers' touching distance.
One run each: they are deterministic, but start positions that change who gets there first change who waits.

## What it does not do

- Arms working at the same time in a shared space are not planned around each other: each arm's motion keeps off its own robot, the floor and the world, not off another arm.
  Two arms with disjoint jobs are fine; two that reach into the same space need the hand-off above, which keeps them apart by taking turns.
- The hand-off is made of two pick-and-places, so it has what they have: boxes and other things the gripper can close on, and the Panda's weak zone near its own base.
- Traffic covers rovers and walkers on `navigate` trips on the floor; flying and diving trips are not part of it, and a robot standing still is an obstacle on the map as before.
