# Sensors

Any robot can carry sensors on any of its bodies: a template, a robot from the library (Go2, Panda, G1, ...), an imported URDF or MJCF, or one built from parts.
A sensor is added in one call and kept on the robot's entry in world.yaml.
Every sensor is ideal unless it is given a rate, noise, bias, drift, latency or dropout, and an ideal one costs nothing while time passes.

```text
add_sensor(robot="go2", type="lidar3d")
add_sensor(robot="go2", type="camera", options={"mode": "depth", "noise": 0.01})
add_sensor(robot="go2", type="touch", on="FL", options={"rows": 2, "cols": 2})
add_sensor(robot="panda", type="force_torque", on="hand")
read_sensors(robot="go2")                      # every sensor, compact
read_sensors(robot="go2", names=["camera"], image=true)
remove("go2/lidar")
```

## Types

| type | measures | options (defaults) | reading |
|---|---|---|---|
| `camera` | a picture, a depth image or a segmentation image | `mode` rgb, depth or segmentation (rgb), `width` 320, `height` 240, `fov` 60 degrees, `range` 10 m | depth: `center_m`, `min_m`, `median_m` (depths along the axis), `beyond_range`, `own_share`, `nearest` (a distance); segmentation: `seen` (what, and how much of the picture); the picture with `image=true` |
| `lidar` | one ring of laser rays | `rays` 360, `hfov` 360, `range` 12 m, `vfov` tilts the ring | `hits`, `nearest`, `ahead`, `sectors_m` (36 numbers) |
| `lidar3d` | several rings over a vertical field | `channels` 16, `rays` 180 per ring, `vfov` [-15, 15], `hfov` 360, `range` 30 m | `points`, `ground_points`, `box_m`, `nearest` and `ahead` with `height_m` and `point_m`, `sectors_m` |
| `imu` | proper acceleration, turn rate, orientation | | `accel_m_s2`, `gyro_rad_s`, `orientation_deg` (roll, pitch, yaw) |
| `encoders` | joint angles | `joints` (all), `resolution` in rad | `rad` (and `m` for sliding joints) by joint |
| `force_torque` | force and torque between a body and the one it hangs from | `between` [parent, child] instead of `on` | `force_n`, `torque_nm` (sensor frame), `force_world_n`, `payload_kg` or `supported_kg` |
| `touch` | normal contact force on a pad | `rows`, `cols` (1 x 1), `size` [length, width] | `force_n`, `contact`, and for an array `cells_n` and `center` |
| `rangefinder` | distance along one beam | `range` 4 m | `distance_m` |
| `gps` | position in the world | | `position_m` |
| `odometry` | the body's speed and turn rate added up | | `position_m`, `heading_deg`, `traveled_m` |

Agents' words work as types: "depth camera", "3D lidar", "ft", "encoder", "gnss", "odom", "bumper".
A type refuses the options that mean nothing for it, and the error lists the ones it has.

Bearings are degrees from the sensor's forward, counter-clockwise, so +90 is its left.
`nearest` is the closest obstacle in any direction, `ahead` the closest within 15 degrees of forward.
`sectors_m[k]` is the nearest obstacle in the 10 degree sector centered on `k * 10` degrees (`sector_bearings_deg` gives the first center and the step).
The floor, terrain grids and flat ground marks are ground, not obstacles; a lidar's rays that meet its own robot are masked out, as a real lidar's self filter does.
A depth camera's `center_m`, `min_m` and `median_m` are depths, how far along the camera's axis a pixel's surface is, as a depth image holds them.
Its `nearest` is a distance: straight from the lens to the closest obstacle in the picture, with `bearing_deg`, `elevation_deg` and `point_m` (world).
Off to the side depth is shorter than distance by the cosine of the angle, so a pole 26 degrees off the axis is 10 percent nearer in depth than it is.
The camera's own robot is not an obstacle either (a lidar puck beside the lens, its own legs); `own_share` is how much of the picture shows it.
`payload_kg` is what a force-torque sensor carries beyond the weight of everything hanging below it, so a wrist sensor reads the weight of what the gripper holds.
Something holding it up from below (a leg on the ground) would make that negative, so then it reports `supported_kg` instead: the weight the ground takes through it beyond its own.

## Where a sensor goes

- `on`: a body, a part of a robot built from parts, a geom (Go2's feet are the geoms FL, FR, RL, RR), a site or a joint; the robot's base when left out.
- `at`: an anchor (`top`, `bottom`, `front`, `back`, `left`, `right`, `center`, `origin`) or `[x, y, z]` meters from the body's origin.
  Lidars default to `top`, cameras and rangefinders to `front`, touch pads to `bottom`, the others to `center`.
- `direction`: where it looks, a word (`forward`, `down`, `left`, ...) or three numbers.

Words mean the robot's own directions as it stands in its home pose, whatever the body's own axes are: the bottom of a Go2's foot is the side that meets the floor, though its calf's frame is tilted, and `down` from a Panda's hand is down.
A face is the middle of the outermost points of the body's shapes there, so a lidar stands on the crown of a Go2's back and a touch pad lies under its foot.
Touch pads use the shapes that collide and reach 1 cm past their edges.
The modules drawn round a sensor (a lidar puck, a board camera) weigh nothing and collide with nothing, so a robot moves the same with or without them.

## Realism

| knob | meaning |
|---|---|
| `rate_hz` | samples per second; between samples the last one is held (a lidar: at most 50, and 200,000 rays a second) |
| `latency_s` | a reading is the newest sample at least this old; none until the first has waited it out |
| `noise` | standard deviation in the reading's own unit; one number for every channel, or a mapping |
| `bias` | a constant offset (IMU: accel and gyro; odometry: turn) |
| `drift` | the bias's random walk, per square root of a second |
| `dropout` | the share of lidar rays (or rangefinder readings) that return nothing |
| `seed` | changes the noise; by default each sensor's noise starts from its name |

Channels: camera `depth` (m) and `rgb` (levels of 255), lidar and rangefinder `range` (m), imu `accel` (m/s2), `gyro` (rad/s) and `orientation` (deg), encoders `position` (rad), force_torque `force` (N) and `torque` (N m), touch `force` (N), gps `position` (m), odometry `speed` (m/s) and `turn` (rad/s).
A gyro bias makes the IMU's yaw drift as a real one's heading does.

An ideal sensor is measured when it is read.
Any other is sampled while time passes, at its rate, or a lidar at 10 Hz and a camera at 30 Hz when it was given none, and the others on every step.
Its first sample is taken when the world is built, reset or edited, and samples are due by the moment they show, so a run reads the same numbers however often it is read; a reset starts the noise again from its seed.
An edit that leaves a robot where it stands (a box added, another sensor) leaves its sensors going: an odometry's estimate and distance, an IMU's drift and the noise carry on.
A camera keeps the state of the world when its frame is due and draws it when read, so a delayed camera shows the world as it was without rendering frames nobody looks at.

## What lidars see

Rays are cast together with `mujoco.mj_multiRay` and see the shapes things collide with, not the visual meshes drawn over them: that is what a robot can bump into, and it is about ten times cheaper.
Cameras and rangefinders see what is drawn, which can stand a few millimetres proud of the collision shape (a test wall's plaster: 6 mm).

## Scripts

```python
from robotforge.script import connect

sim = connect()
scan = sim.sensor("go2", "lidar")  # the reading read_sensors gives
print(scan["ahead"]["distance_m"])
full = sim.sensor("go2", "lidar", full=True)  # plus every range, ring by ring, and a 3D lidar's points
```

## The robots' own sensors

The builder's part sensors (imu, touch, camera, rangefinder, lidar) and the templates' IMU, cameras and foot touch sensors are declared the same way, so `read_sensors` reads them too; part sensors take the same realism knobs.
A sensor is declared in the model itself, as a MuJoCo custom text `rf_sensor:<name>` holding its definition, so a model saved as MJCF keeps its sensors.
A library robot's own MuJoCo sensors (a G1's IMU) come back raw under `other_sensors`, also when it carries no other sensor.
A definition that does not load (a world.yaml edited by hand) is listed in `observe`'s `sensor_summary` and in `read_sensors` as `not_set_up`, with the reason.
A built robot's added sensors follow its parts: renamed with the part they sit on (update_part), and taken off with it (remove_part says which in `removed_sensors`).
add_sensor's own MuJoCo sensors are named `rf_...` and are left out of `observe`'s raw `sensors`; `observe` lists every sensor in one line in `sensor_summary`.

## The viewer

The viewer draws what the sensors sense.
In the world: a lidar's hit points colored by distance (near red, far blue, over the middle of the distances it sees) with some of its rays fading out from the sensor, a camera's frustum on the robot, touch pads that light up green with their force, contact arrows from the feet and a force arrow at a force-torque sensor, a rangefinder's beam.
Beside the view, for the selected robot: what one of its cameras sees and live plots of its IMUs (acceleration, turn rate, tilt), force-torque sensors (force, torque, payload, with what the wrist holds) and encoders (a leg, or a run of six joints, at a time).

The picture is rendered in the browser with three.js from the camera's pose, so it works where the server has no OpenGL.
A depth picture has the colors read_sensors(image=true) gives an agent, with its scale under it; a segmentation picture has the server's palette and names what it sees; an rgb picture is drawn as the view is, without its haze.
Side by side with the server's pictures of the same moment, the depth and segmentation pictures match.

The robots drawer lists the selected robot's sensors with a switch each, and one for the arrows.
Sensors put on with add_sensor show from the start; a robot's own (a template's IMU, camera and foot pads, a part's sensors) are listed and stay off until switched on, so a world without added sensors looks as it did.
A switch holds for that sensor in that world while the tab is open: another world's sensor of the same name, or the same sensor added again, starts from its default.
`open_viewer(show=["go2/front_cam"])` (or a robot's name for all of its sensors) lets an agent switch sensors on and bring a camera's picture up.
A tab applies each such request once (a reload does not apply it again), and a request ends when the sensors it names are removed or their world is closed.
In a lesson nothing is drawn.
Pads and arrows are drawn through the robot's own legs, but not through a rack or a wall standing between them and the eye.
On a phone the views are a strip at the top, one at a time in tabs; with the drawer or the world panel open beside a narrow window they step aside.

While the world has sensors the viewer's WebSocket carries a `sensors` message ten times a second; viewer/server.py documents its fields and `src/robotforge/viewer/static/sensorkit.js` lists the ones the page reads (a test checks the server sends them).
Each entry has the sensor's `name`, `robot`, `type`, whether it was `added`, its `uid` (the same while it stays on its robot), world `pos` and `quat`, the `body` it rides on with its place there (`local_pos`, `local_quat`), and what drawing it needs: a lidar's hit points (at most 1200 a lidar and 2400 a message, shared out), a camera's field of view, mode and range, a force-torque sensor's force and torque as world vectors and payload, an IMU's values, each touch zone's force and place on its body.
What sits on a robot follows its body in every frame drawn; a scan's points stay where they were measured.
A message is cheap to build: ideal lidars are scanned for it within 6000 rays, the stalest first, so with several big lidars each is redrawn a few times a second, and while time stands still the last message is sent again without measuring.

Measured in headless Chrome (`--use-angle=metal`, 1280 x 800 at 2x) with the Go2 in the warehouse carrying a 10 Hz 3D lidar (1200 points drawn, 150 rays), the depth camera's picture beside the view and four touch pads, on the development Mac while other jobs shared it:

| | frames a second | page main thread | render per frame (GPU work finished) |
|---|---|---|---|
| sensors drawn | 60 | 74 ms a second, 1.2 ms a frame | 0.86 ms |
| the same sensors switched off | 60 | 52 ms a second, 0.9 ms a frame | 0.47 ms |
| the same world without sensors | 60 | 45 to 56 ms a second | |
| sensors drawn, CPU slowed 4x | 60 | 302 ms a second, 5.0 ms a frame | |
| switched off, CPU slowed 4x | 60 | 176 ms a second, 2.9 ms a frame | |

The frame rate is the display's 60 everywhere; main-thread time comes from Chrome's own task counters over 8 s.
The depth picture is drawn ten times a second and reaches the panel 7 ms after it is rendered (read back without stalling the page).

## Measured

`uv run python scripts/sensor_bench.py` measures every claim against the truth (2026-10-18, on the development Mac while other jobs shared it).

| what | set up | measured |
|---|---|---|
| 3D lidar against a wall | Go2 in the warehouse, 1.72 m from the side wall, lidar noise 2 cm, 30 scans | `ahead` 2.3 mm long on average, 6.4 mm spread; a clean scan: 99.6 % of the 739 rays toward the wall within 1 mm of its plane |
| depth camera against the wall | same, noise 1 cm | `center_m` 1.5 mm long on average, 11.2 mm spread; `nearest` 1.3 mm long, 0.7 mm spread (2026-10-19) |
| depth camera, an obstacle off its axis | a pole 26 degrees to the side, 1.82 m from the lens, a lidar puck beside the lens | `nearest` 1.821 m at 25.8 degrees, the lidar beside it 1.809 m (1.808 from its own origin); before 2026-10-19 the depth along the axis (1.66 m), or the puck at 0.025 m |
| IMU on a dropped block | 2 kg block from 1 m, 200 Hz | 0.00 m/s2 while it falls, 9.81 at rest; lands at 0.46 s (free fall predicts 0.44) |
| force-torque at a Panda's wrist | a 100 g cube picked up and held 25 cm up | `payload_kg` 0.000 empty, 0.100 holding it |
| force-torque on a standing Go2's front-left calf | standing still, 1 s | `supported_kg` 3.576 (a quarter of the 15 kg dog, less the leg below the sensor); no negative `payload_kg` |
| touch arrays under a trotting Go2's feet | 2 x 2 pads, 0.4 m/s, 3 s at 100 Hz | agree with the true foot contacts 98.3 to 98.7 % of the time; diagonal feet correlate 0.96 to 0.97, side by side -0.51 to -0.52; stances FR+RL 34 %, FL+RR 32 %, all four 31 %; the pressure moves from the back row to the front row in each stance |
| noise | IMU accel 0.05, gyro 0.002, GPS 0.03, 1000 samples | 0.0485 to 0.0501, 0.00203 to 0.00204, 0.0287 to 0.0302 |
| latency | GPS 0.1 s on a falling block | 0.100 s |

Cost, as headless physics steps per second of CPU time for a Go2 standing in the warehouse (median of 7 shuffled trials):

| sensors | steps/s | change |
|---|---|---|
| none | 3666 | |
| an ideal 3D lidar (read only when asked) | 3665 | -0.1 % |
| a 10 Hz 2D lidar, 360 rays | 3574 | -2.5 % |
| a 10 Hz 3D lidar, 16 x 180 rays (the default) | 3396 | -7.4 % |
| a 10 Hz 3D lidar, 16 x 360 rays | 3059 | -16.6 % |
| the default 10 Hz 3D lidar, a 100 Hz noisy IMU and four touch pads | 3180 | -13.3 % |

One 16 x 180 scan takes 1.07 ms against 0.23 ms for a physics step: ten scans a second add about 9 % to the 500 steps of each simulated second.
A scan's cost grows with its rays and with the shapes in the world (the warehouse has 152 that rays can hit).
