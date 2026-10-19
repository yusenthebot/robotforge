# Extending robotforge

This page is the short version of docs/EXTENDING.md in the repository.

## Plugins

A plugin is a Python file that registers things with `robotforge.extend`.
Where it lives:

- `~/.robotforge/plugins/` (or the folders in `ROBOTFORGE_PLUGINS`): the user's own, loaded when robotforge starts.
- `<world>/plugins/`: travels with the world. `write_script(name, code, plugin=true)` writes one there and loads it.

Parts, sensors, controllers and behaviors are described below as each is added.
