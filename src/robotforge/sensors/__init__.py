"""The sensor suite: sensors any robot can carry, sampled as real ones are.

A sensor is added to a robot in one sentence (add_sensor) and kept on its world.yaml entry
(`sensors:`). The world compiler mounts it on a body of the robot (mount.py): a site for its pose,
MuJoCo's own sensors where they exist (accelerometer, gyro, force, torque, touch, rangefinder), a
camera for a camera, and the module drawn where it sits. Every mounted sensor, and every sensor the
builder or a template puts on a robot, is declared in the model itself as a custom text named
`rf_sensor:<name>` holding its definition, so whatever loads the model knows its sensors.

At run time a suite (runtime.py) reads those definitions back and samples each sensor at its rate,
with its noise, bias, drift, latency and dropout; the defaults are an ideal sensor that is read fresh
on demand and costs nothing while time passes. Readings come back small (readings.py): a lidar is its
nearest obstacle and a few dozen ranges, not thousands of points.

Units are meters, seconds, radians for joints and rates, degrees for angles an agent reads (bearing,
roll, pitch, yaw) and newtons. docs/SENSORS.md describes every type and knob.
"""

from robotforge.sensors.spec import SENSOR_TYPES, SensorEntry, without_sensors

__all__ = ["SENSOR_TYPES", "SensorEntry", "without_sensors"]
