"""A home of two rooms for mobile manipulation: a kitchen and a living room with a swing door between them.

Everything a robot dog with an arm works on is within its reach (from the floor to 0.45 m): the apple and a cup on
a low kitchen table, a chest of drawers with a block in its drawer, a button on a side table, the low living-room
table to bring things to. The counter, the sink and the fridge along the kitchen's north wall are for taller arms.
The rooms are named by the rugs on their floors ("kitchen", "living_room"), so "go to the kitchen" is move(robot,
to="kitchen"). The door between them swings both ways and closes by itself: a robot pushes it open as it drives
or walks through (navigation counts its leaf as something to push, not to go round).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotforge.design.scenes import Content

WALL = [0.88, 0.86, 0.82, 1]
WOOD = [0.62, 0.48, 0.34, 1]
LIGHT_WOOD = [0.78, 0.64, 0.46, 1]
WHITE = [0.92, 0.92, 0.9, 1]
KITCHEN_RUG = [0.8, 0.79, 0.74, 1]
LIVING_RUG = [0.64, 0.56, 0.5, 1]
SOFA = [0.42, 0.5, 0.58, 1]
WALL_HEIGHT = 2.1
SOUTH_WALL = 0.5  # m: no robot steps over it, and the camera sees in over it
LOW_TABLE = 0.36  # m: a Go2's arm points its gripper down 0.35 m out at 0.45 m high, further lower down
HALF_X, HALF_Y = 4.0, 2.5  # m, the home's floor: 8 by 5
DOOR_Y = -0.9  # where the doorway in the wall between the rooms is
DOOR_WIDTH = 1.0  # the frame's; its opening is 16 cm narrower (props.py)
FRAME_HEIGHT = 2.0


def home(rng: random.Random, opts: dict[str, float]) -> Content:
    from robotforge.design.scenes import Content, Slot, resting_z

    c = Content(floor={"style": "wood"})
    # Walls on three sides and one between the rooms; toward the south (-y), where the camera looks in from, a wall
    # only knee high, which keeps every robot inside and the rooms in view.
    c.add("north_wall", "wall", [0.0, HALF_Y + 0.05], size=[2 * HALF_X + 0.2, 0.1, WALL_HEIGHT], color=WALL)
    for name, x in (("west_wall", -HALF_X - 0.05), ("east_wall", HALF_X + 0.05)):
        c.add(name, "wall", [x, 0.0], size=[2 * HALF_Y, 0.1, WALL_HEIGHT], yaw=90, color=WALL)
    c.add("south_wall", "wall", [0.0, -HALF_Y - 0.05], size=[2 * HALF_X + 0.2, 0.1, SOUTH_WALL], color=WALL)
    top = DOOR_Y + DOOR_WIDTH / 2  # the partition north of the door frame, and south of it
    c.add("partition", "wall", [0.0, (top + HALF_Y) / 2], size=[HALF_Y - top, 0.1, WALL_HEIGHT], yaw=90, color=WALL)
    low = DOOR_Y - DOOR_WIDTH / 2
    c.add(
        "partition_south", "wall", [0.0, (low - HALF_Y) / 2], size=[low + HALF_Y, 0.1, WALL_HEIGHT], yaw=90, color=WALL
    )
    c.add("door_frame", "door_frame", [0.0, DOOR_Y], size=[DOOR_WIDTH, 0.12, FRAME_HEIGHT], yaw=90, color=WHITE)
    c.add(
        "door",
        "door",
        [0.0, DOOR_Y],
        # The frame's opening less a gap: its posts and lintel are 8 cm thick, and the leaf hangs 1.2 cm off the floor.
        size=[DOOR_WIDTH - 0.16 - 0.01, 0.035, FRAME_HEIGHT - 0.08 - 0.012 - 0.01],
        yaw=90,
        color=LIGHT_WOOD,
    )
    # Rugs name the rooms.
    c.add("kitchen", "marker", [-HALF_X / 2, -0.5], size=[2.6, 2.4], color=KITCHEN_RUG)
    c.add("living_room", "marker", [HALF_X / 2, -0.4], size=[2.6, 2.4], color=LIVING_RUG)

    # The kitchen: a counter with a sink and a fridge along the north wall (for tall arms), a low table with the
    # apple and a cup, and a chest of drawers with a block in its drawer.
    c.add("counter", "counter", [-2.0, HALF_Y - 0.3], size=[2.6, 0.6, 0.9], color=[0.3, 0.38, 0.47, 1])
    c.add("sink", "bin", [-1.4, HALF_Y - 0.3, 0.9], size=[0.5, 0.38, 0.12], color="gray", fixed=True)
    c.add("fridge", "box", [-3.6, HALF_Y - 0.37], size=[0.7, 0.7, 1.85], color=WHITE, fixed=True)
    # Things on it lie near its front edge: a Go2's arm points its gripper down at most 0.35 m out at that height.
    c.add("kitchen_table", "table", [-2.3, 0.6], size=[0.9, 0.5, LOW_TABLE], color=WOOD)
    front = 0.6 - 0.25
    for name, shape, x, size, color in (
        ("apple", "sphere", -2.5, [0.06], "red"),
        ("cup", "cylinder", -2.1, [0.06, 0.08], "yellow"),
    ):
        x, y = x + rng.uniform(-0.08, 0.08), front + rng.uniform(0.07, 0.12)
        c.add(name, shape, [x, y, resting_z(LOW_TABLE, shape, size)], size=size, color=color)
    chest = [0.32, 0.4, 0.36]  # narrow: a Go2 beside the drawer's way reaches its bar all the way out
    c.add("chest", "drawer", [-0.9, 1.4], size=chest, color=WHITE)
    # In the drawer, on its floor (the drawer's box sits 1.2 cm over the chest's bottom board), near its front and
    # its east side: a Go2 standing beside the open drawer reaches it there, and not in the drawer's middle, 0.38 m
    # from its arm's base.
    x, y = -0.9 + 0.07 + rng.uniform(-0.02, 0.02), 1.4 - 0.1 + rng.uniform(-0.02, 0.02)
    c.add("block", "box", [x, y, 0.012 + 0.003 + 0.006 + 0.02 + 0.002], size=[0.04, 0.04, 0.04], color="green")

    # The living room: a sofa, the low table to bring things to and a side table with a button.
    c.add("sofa", "box", [2.2, HALF_Y - 0.45, 0.21], size=[1.8, 0.75, 0.42], color=SOFA, fixed=True)
    c.add("sofa_back", "box", [2.2, HALF_Y - 0.12, 0.42], size=[1.8, 0.18, 0.84], color=SOFA, fixed=True)
    c.add("living_table", "table", [2.2, 0.5], size=[1.0, 0.6, LOW_TABLE], color=LIGHT_WOOD)
    c.add("side_table", "table", [1.0, 1.5], size=[0.45, 0.4, 0.3], color=WOOD)
    c.add("button", "button", [1.0, 1.5 - 0.1, 0.3], size=[0.05, 0.04], color="red")  # near its front edge

    c.ground_slots = [Slot((2.4, -1.5), 180), Slot((1.2, -1.6), 180), Slot((-2.2, -1.5), 0)]
    from robotforge.design import gallery

    c.starter = [{"name": "go2_arm", "build": gallery.build("go2_arm")}]
    c.tips = [
        "pick_and_place('go2_arm/arm', 'apple', 'living_table') goes to the kitchen through the door and brings the "
        "apple to the living-room table",
        "operate('go2_arm/arm', 'chest', 'open') walks over and pulls the drawer out; then pick_and_place takes the "
        "block out of it",
        "operate('go2_arm/arm', 'button', 'press') presses the button on the side table; move(robot, to='kitchen') "
        "goes to a room",
    ]
    return c
