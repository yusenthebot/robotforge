"""Perception: point clouds, occupancy-grid mapping, particle-filter localization and exploring.

A robot with a lidar and an odometry draws a map of where it goes (mapper.py), keeps its pose estimate honest
against the odometry's drift with a particle filter (particles.py), explores toward the nearest unseen space
(explore.py) and goes somewhere on its own map (behaviors/map_nav.py). The hub (hub.py) steps the mappers with
the simulation and answers the `mapping` tool; truth.py scores a map against the scene. docs/PERCEPTION.md
describes all of it.
"""
