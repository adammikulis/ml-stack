# Forest-search drones

The drone example uses [PyFlyt 0.29.0](https://github.com/jjshoots/PyFlyt) and native
Bullet collisions. Its [QuadX mode 7](https://taijunjet.com/PyFlyt/documentation/core/drones/quadx.html)
uses PyFlyt's cascaded PID controllers for position and yaw; ml-stack supplies
navigation setpoints and search-task rewards, not a flight controller or physics solver.
Unselected drones patrol with the same native PID. Selecting another drone preserves
all actors, the Bullet connection, and the world clock.

PyFlyt currently requires NumPy below 2. Use Python 3.12 for this example; the
Python 3.13 driving environment remains separate. Install `ml-stack[gym-drone,gym-rl]`
inside that environment and save its interpreter through the reusable Gym interpreter
setting. The Libraries screen installs the drone package into its own managed
`simulators/gym-drone/env` directory with Python 3.12, downloading a standalone
interpreter if a matching local Python is unavailable. Future launches automatically
rediscover that directory. Advanced existing environments can be remembered in
`settings.json` as `gym_pythons: {"drone": "/path/to/python"}`; `gym_python` remains
the default for other environments. Installing or removing drone packages affects
only the drone environment, including its NumPy and CPU PPO dependencies. On macOS, Bullet may need a source build with `SDKROOT` pointing at the
installed macOS SDK and `CFLAGS=-Dfdopen=fdopen`.

The regular camera is PyFlyt's native Bullet camera. The thermal camera is a
**synthetic visible-surface temperature sensor**, not a physical infrared/fire solver.
Visible hiker surfaces are 37°C and fire surfaces 180°C; ambient surfaces are 20°C.
The native segmentation buffer limits these temperatures to visible surfaces, so trees
and other opaque obstacles hide targets. Warm/hot detection labels come from sensor
temperatures, not hidden target coordinates. PPO receives sampled RGB/thermal pixels
and drone proprioception; decision models receive the same sensor data. World-debug
geometry contains target locations for the scene renderer and must never be passed
as a policy observation. Images are captured at five simulation frames per second.

Navigation actions are hold, north/south/east/west, ascend/descend, and yaw left/right.
The native controller follows bounded metre-scale setpoint increments. Separate report
hiker/report fire actions earn +5 per previously unreported visible target, or -0.5 for
an unsupported report. Every step costs -0.01; entering a new two-metre coverage cell
earns +0.1. A crashed aircraft costs -10. Each transition includes the exact summed
reward components, native setpoint, observation and simulation clock.

Episode mode rebuilds a seeded world when an episode resets, including held-out
seed evaluation. Persistent mode keeps the same Bullet world across learning-task
horizons and resets. Frozen PPO predicts without optimizer updates; online PPO uses
the existing live world's single clock. A crash does not silently restart a persistent
world; select another actor or create a new world.

Procedural controls set size, tree/hiker/fire counts, agent count and seed. Manual
worlds use a versioned JSON file beneath the daemon's trusted files root:

```json
{"version":1,"size":20,"n_agents":2,
 "trees":[{"x":3,"y":2,"height":5,"radius":0.5}],
 "hikers":[{"x":5,"y":0}],"fires":[{"x":6,"y":2}]}
```

Coordinates are native metres (x/y ground plane, z up). Trees and targets are native
Bullet collision bodies; the initial demo uses simple cylindrical scenery. This is a
search/control benchmark, not an atmospheric fire-spread or realistic human simulator.
The world definition and hashes are saved beside the trajectory manifest for reuse.
