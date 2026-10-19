"""MCP tools for sensors: add_sensor puts one on any robot, read_sensors reads them compactly.

Taking a sensor off is remove("robot/sensor"), the tool that removes everything else. Registered by
robotforge.mcp_server.build_server via `register(...)`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import ToolAnnotations

from robotforge.playground import Playground, PlaygroundError

PLACEMENT = ("on", "at", "direction", "name")


def register(server: MCPServer, playground: Playground) -> None:
    from robotforge.mcp_server import _json, _tool_errors  # here: mcp_server imports this module

    @server.tool()
    async def add_sensor(
        robot: str,
        type: str,
        on: str | None = None,
        at: str | list[float] | None = None,
        direction: str | list[float] | None = None,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Put a sensor on any robot (a template, a library robot like go2 or panda, an imported file, one built from
        parts), on any of its bodies. type: camera (options mode rgb, depth, segmentation, stereo (baseline), event
        (threshold, window_s), width, height, fov; lens fisheye reaches 200 degrees),
        lidar (one ring) or lidar3d (options channels, rays per ring, vfov, hfov, range), imu, encoders (joints,
        resolution), force_torque (on a body such as a wrist, e.g. on="hand", or between=[parent, child]), touch
        (rows and cols make an array; e.g. on a foot), rangefinder, gps, odometry, wheel_odometry (a wheeled robot's
        wheels added up; option slip), sonar (a 30 degree cone, options beam, range, min_range), skin (a tactile skin:
        taxels on faces of a body, options faces, rows, cols), temperature or gas (what heat_source and gas_source
        objects put into the world; option response_s, how slowly it follows). on: a body, part, geom, site or
        joint (default the base); at: top, bottom, front, back, left, right, center, origin or [x, y, z] m in that
        body's frame (lidars default to top, cameras to front); direction: where it looks (forward, down,
        [1, 0, -0.3]). Realism goes in options, ideal by default: rate_hz, noise (standard deviation in the reading's
        unit: m, m/s2, rad/s, N), bias and drift (imu, odometry), latency_s, dropout (lidar), seed. Saved on the
        robot in world.yaml; the robot stays where it is. Returns the sensor's name and first reading;
        remove("robot/sensor") takes it off."""
        with _tool_errors():
            fields = {"type": type, **(options or {})}
            given = {"on": on, "at": at, "direction": direction, "name": name}
            fields.update({k: v for k, v in given.items() if v is not None})
            return _json(await asyncio.to_thread(playground.add_sensor, robot, fields))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def read_sensors(
        robot: str, names: list[str] | None = None, image: bool = False, points: int | None = None
    ) -> list:
        """Read a robot's sensors (all, or those in names) as compact readings with units. lidar and lidar3d:
        nearest obstacle and nearest ahead (distance_m, bearing_deg: 0 is the sensor's forward, +90 its left),
        ranges as the nearest per 10 degree sector (sectors_m); a 3D scan also its point count and bounding box. Depth
        camera: center_m, min_m and median_m are depths along its axis; nearest is the closest obstacle's straight
        distance from the lens with its bearing (its own robot and the floor left out); segmentation camera: what it
        sees and how much. imu, encoders, force_torque (payload_kg is what it holds; a leg on the ground gives
        supported_kg), touch, rangefinder, gps, odometry: their values. sonar: distance_m of the nearest echo in its
        cone (blind up close, no echo at a grazing angle), echoes. wheel_odometry: pose from the wheels' speeds, as
        odometry. skin: total_n, contact, touched and faces (normal force on taxels over several faces of a body).
        temperature (temperature_c) and gas (gas_ppm): what the world's heat_source and gas_source objects make at the
        sensor, lagging by its response_s. Camera modes: stereo (depth worked out from a disparity, with a baseline),
        a fisheye lens (rgb through an equidistant lens), event (events: how many pixels got brighter or darker, their
        rate and centroid, from the picture against 30 ms before). A robot's own MuJoCo sensors (a G1's IMU) come
        as other_sensors. image=true adds each camera's picture; points=N adds each depth camera's and lidar's point
        cloud in world coordinates, thinned (voxel mean) to at most N points (1 to 2000). Readings follow each
        sensor's rate, noise and latency (age_s is how old a sample is)."""
        with _tool_errors():
            if points is not None and not 1 <= points <= 2000:
                raise PlaygroundError(f"points is how many cloud points to give back, 1 to 2000, got {points}")
            reading, pictures = playground.read_sensors(robot, names, image, points=points)
            content: list = [_json(reading)]
            content += [Image(data=png, format="png") for _, png in pictures if png]
            return content
