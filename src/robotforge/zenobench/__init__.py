"""ZenoBench on MuJoCo: the Isaac Sim benchmark for the Zeno Malo mobile manipulator, without Isaac.

ZenoBench (github.com/CMU-Agentic-Library/ZenoBench) is fetched as its own checkout and runs unchanged.
`usd_mjcf` turns its USD scenes into MuJoCo models, `world` simulates them the way PhysX would, and
`shim` provides the few Isaac Sim / Isaac Lab / Omniverse modules ZenoBench imports, backed by `world`.

Needs the optional extra: `uv sync --extra zenobench`.
"""
