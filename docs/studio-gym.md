# Studio and live Gym

Studio organizes chat, models, data, training, live environments, benchmarks, fleet
administration, tools, and settings in one sidebar. Running jobs remain visible as you
move between workspaces. The Gym combines a live viewport with the actual controller
inputs, action probabilities when available, applied actions, rewards, and episode outcomes.

## Installation

In the desktop app, open the environment library installer and select the Gym libraries
you need. Simulators run in that managed Python environment, separately from the desktop
binary. The app reports which dependencies are installed in the selected environment.

For a source checkout:

```sh
python -m pip install -e '.[gym]'
python -m pip install 'metadrive-simulator @ git+https://github.com/metadriverse/metadrive.git@85e5dadc6c7436d324348f6e3d8f8e680c06b4db'
ml-stack-gym catalogue
```

Install individual extras to select workloads:

| Extra | Workload |
| --- | --- |
| `gym-driving` | MetaDrive car physics, sensors, and native 3D preview |
| `gym-warehouse` | RWARE warehouse logistics |
| `gym-traffic` | SUMO-RL and the SUMO simulator |
| `gym-rl` | Stable-Baselines3 PPO and Gymnasium |
| `decide-pointer` | Local pointer decision models, including Strands 2B |

The `gym-driving` extra installs Gymnasium and image support. MetaDrive requires the
separate pinned source install shown above; the maintained native API used by the driving
adapter is available at that revision. The desktop library installer supplies this source
explicitly for both Smart car and All live environments. Published package dependencies
use PyPI packages only. MetaDrive's native renderer downloads official assets on first use. SUMO comes through the `eclipse-sumo`
package; the adapter resolves `SUMO_HOME` and the packaged SUMO-RL intersection scenario.
An existing SUMO installation can also supply `SUMO_HOME`.

Set `ML_STACK_GYM_PYTHON` to an installed Python interpreter to run simulator workers
outside the current process. `ML_STACK_CACHE` moves recordings and checkpoints together
with the other ml-stack caches. Their default location is `~/.cache/ml_stack/gym`.

Reuse an existing simulator installation without installing it again:

```sh
ml-stack-traind --root ~/.local/share/ml-stack-studio --gym-python /path/to/venv/bin/python
```

The launcher remembers that interpreter and keeps completed setup under the selected
root. On later launches, use the same `--root` and omit `--gym-python`:

```sh
ml-stack-traind --root ~/.local/share/ml-stack-studio
```

Changing the root selects a different workspace, with its own settings, setup state, files
and conversations. Reusing the root preserves completed setup and the interpreter selection.
An explicit
`ML_STACK_GYM_PYTHON` overrides the saved selection; otherwise the saved interpreter
takes precedence over the app's managed environment. A missing saved interpreter
reports an error and asks you to select its new location.

## Environments and ownership

| Environment ID | Simulation ownership | Control |
| --- | --- | --- |
| `car` | MetaDrive owns Bullet vehicle dynamics, contacts, roads, traffic, and numeric sensors | Nine steering/drive combinations |
| `warehouse` | RWARE owns grid movement, collisions, orders, and delivery rewards | Joint actions for two robots |
| `traffic` | SUMO owns traffic movement and routes; SUMO-RL owns signal control and rewards | Native signal phase actions |
| `traffic-driving` | SUMO owns demand/routes and SUMO-RL signal control; MetaDrive owns vehicle motion, collisions, and sensors | Signal actions over synchronized traffic |

Each simulator keeps its native task behavior. Gymnasium provides reset/step and space
interfaces; it does not replace the simulator. The combined traffic environment synchronizes
vehicle identities, coordinates, and clocks and feeds physical positions back to SUMO.
Its viewport reports backend ownership and synchronization telemetry. The combined task
uses SUMO's packaged single-intersection road network, rather than the car's winding-road map.
The combined example supports one signalized intersection. Native MetaDrive IDM
controllers drive its vehicles; the learned policy controls SUMO-RL signal phases.

The car opens in the Three.js sensor scene, with a selectable native offscreen 3D camera
and native lidar/lane sensor visualization. Both viewports display simulator state; the
camera is a preview: numeric sensor observations are the policy input. Warehouse and
traffic views render authoritative state snapshots with Three.js. Warehouse movement is
discrete; visual animation does not create extra simulator steps.

The sensor scene uses Three.js `MeshPhysicalMaterial` for clearcoat paint, metal and glass,
with environment reflections, tone mapping, sunlight and shadows. Native road geometry
supplies asphalt, shoulders and lane markings. Grass, hills and trees are visual scenery;
they add no simulator collisions or obstacles. Stop signs and stop lines follow the native
stop-task metadata.

Use **Follow car** for a chase camera; dragging the orbit controls turns follow off, and
checking it again resumes tracking. **Driver view** switches to eye height inside the selected
car; **Chase view** returns behind it. Press **C** to switch between these views. The selected
car exterior is hidden in driver view so its roof does not obstruct the camera. Movement is
interpolated for display only; sensor measurements and recorded trajectories retain native
values. **Whole map** fits the scene. **Sensor rays** toggles
lidar lines: cyan marks clear range, coral marks detected surfaces and hit markers.
**Heading / control** shows the native heading in blue and the normalized applied steering
vector in gold. The steering vector shows the current action, rather than a predicted path.
The **Viewport** selector switches to the native camera; **Native camera sensor lines**
controls its simulator lidar and road-detector overlays. **Expand scene** enters fullscreen.

Car seeds select native MetaDrive scenarios. Defaults cover seeds `0` through `19999`;
configure `start_seed` and `num_scenarios` to change this range. Out-of-range seeds are
rejected. The default car horizon is 1000 steps. Warehouse and traffic retain their finite
native episode limits, configurable through environment settings.
The car form defaults to `map: "SCSCS"` for native straight/curve segments and `traffic_density: 0.25` for surrounding native traffic. With `stop_signs: true`, a route checkpoint
requires speed at or below 0.5 m/s for one second with the front bumper between zero and
three metres before its stop line. Compliance adds five reward points; crossing without stopping subtracts ten,
once per checkpoint. The numeric policy observation adds distance, held time and
compliance; named decision inputs and recordings also include this rule. These policies
require checkpoints trained with the same stop-task configuration.

The nine manual commands combine steering left/straight/right with brake/coast/drive.
`steering_magnitude` defaults to `0.35` and must be greater than zero and at most one;
the native throttle values are `-1`, `0` and `1`. Session and training manifests record
the effective steering magnitude. PPO loading rejects a checkpoint trained with a
different magnitude.
`sensor_debug` controls the native camera's lidar and road detector overlays.
Session manifests record installed simulator versions and SHA256 hashes of the SUMO
network and route files. MetaDrive uses the pinned source revision shown in the installation command; its
installed package metadata records that source URL.

## Native procedural and manual worlds

See [Persistent simulator worlds](live-worlds.md) for live actor lifecycles, frozen and online
learning, agent inspection and control, and exact trajectory recording.

World construction is configured under `config.world`. Its `mode` is `procedural` or
`manual`, and its `seed` records the world definition independently of the controller's
random seed. World construction does not determine episode limits or reset behavior;
those belong to the runtime's simulation mode and native task configuration.

With `simulation_mode: "world"`, warehouse, traffic and traffic-driving initialize their
native environment once. `task_horizon` bounds a learning task in controller steps; its
reset returns the current observation without restarting the native world. SUMO time stays
monotonic and warehouse robots retain their identities and ongoing request queue. Native
SUMO vehicles still arrive and depart; actor identity is not held artificially after arrival.
Traffic's initial routes are followed by continued demand through native TraCI `route.add`
and `vehicle.add`, sampling the loaded route templates with the world seed. Override
`world_demand_period` to set the continuing vehicle interval in seconds. The resolved
native route templates and interval are recorded in `continuing-demand.json`. The
SUMO/MetaDrive bridge retains the same physics instance across learning-task boundaries.
Warehouse time is reported in native discrete steps, rather than physical seconds.

With `simulation_mode: "episode"`, the bounded native episode behavior remains available
for isolated training and evaluation. Model update policy is selected separately through
`learning_mode`; world construction does not train a model.

For traffic and traffic-driving, procedural construction runs SUMO's maintained
`netgenerate` and `randomTrips.py` tools. It creates one signal-controlled intersection
with attached approach roads. The seed selects road length and reproducible demand;
`arm_length`, `lanes`, `vehicle_period` and `demand_seconds` override those parameters.
The combined traffic-driving task keeps its one-intersection constraint. Procedural
construction generates new native network and route files, rather than selecting the
packaged intersection.

```json
{"world":{"mode":"procedural","seed":4,"arm_length":180,"lanes":1,"vehicle_period":3,"demand_seconds":3600}}
```

For manual traffic worlds, upload standalone native SUMO network and routes XML into
Data, then supply paths relative to the daemon files root. Both traffic tasks require
exactly one signal-controlled intersection. Imported XML cannot contain external includes,
entities or document types; paths cannot escape the files root. The selected files are
copied into the world artifact directory before use.

```json
{"world":{"mode":"manual","seed":4,"net_file":"worlds/junction.net.xml","route_file":"worlds/demand.rou.xml"}}
```

RWARE procedural worlds use its native lattice generator. The seed chooses odd
`shelf_columns`, `shelf_rows` and `column_height`; specifying these fields fixes the
native dimensions. `n_agents` selects one to four robots. Manual mode accepts RWARE's
rectangular ASCII layout: `x` denotes shelves, `.` aisles and `g` delivery goals.

```json
{"world":{"mode":"manual","seed":4,"n_agents":2,"layout":".......\n.xx.xx.\n.......\n.g...g."}}
```

World artifacts live under `gym/worlds` in the ml-stack cache. `world.json` records the
backend, resolved definition, seed, native package version and SHA256 hashes of the
network/routes or warehouse layout. Generated warehouse layouts are persisted as ASCII
from RWARE's native grid. Matching definitions reuse their generated files; the same
SUMO seed and parameters reproduce network and route content across fresh directories.

Native references: [SUMO netgenerate](https://sumo.dlr.de/docs/netgenerate.html),
[SUMO randomTrips](https://sumo.dlr.de/docs/Tools/Trip.html), and
[RWARE custom layouts](https://github.com/semitable/robotic-warehouse#custom-layout).

## Controllers and recordings

Basic controls keep environment choice and common task settings visible. Additional
native JSON and less common configuration sit in collapsed **Advanced** sections.

Choose manual control, a seeded random baseline, a PPO checkpoint, a decision model, or
an explicitly labelled native simulator policy. New car sessions default to **Strands
Decider 2B**. The decision model dropdown includes registered trained deciders; changing
it or the controller applies to the live world immediately, preserving its clock and
actors. PPO checkpoint setup retains an Apply button. Custom checkpoint paths and the
maximum input age are under **Advanced options**. The Live sessions selector reattaches
to an active worker after a page reload.

Pause, reset, single-step, and speed controls act on the physics worker. Decision loading
and inference run in a separate process with one outstanding request. Auto device selects an
available accelerator and waits for the maintained broker’s exclusive GPU claim before
loading weights; held serving models and training jobs are never displaced. Advanced
options allow CPU inference explicitly. Queued, loading and actual device status are visible. Native physics
continues while loading or waiting, using explicit straight braking for cars, hold/noop
for drones and warehouse robots, and the current signal phase for traffic. An abstaining,
stale or failed model never becomes a silently substituted native policy. Readiness and
the reason for fallback remain visible. Pause cancels the inference process; resume loads
it again. Pause also cancels camera perception and releases its serving lease; a paused
world retains neither model resource. Reset, actor selection and controller changes invalidate previous results.

The default model is `StrandsAgents/strands-decider-2B-hobson-v19` with auto device selection. Its files
must already be available through the existing decision-model installation workflow.
Loading verifies metadata and hashes and refuses pickled model files. The default maximum
decision input age is one second; `decision_max_age_s` accepts one through thirty seconds.
Slow CPU inference can exceed that limit, in which case the car continues braking and the
result is explicitly reported stale. See [decision models](decision-models.md).

Drone camera perception is optional and has a separate **Vision model** dropdown. It lists
all locally discovered vision models, with SmolVLM 256M as the speed default. Installed
GGUF models also require their vision projector. The official Apple FastVLM safetensors
bundle and MLX/CoreML formats are visible with runtime support marked unavailable until
their native runtime is implemented. The asynchronous vision child acquires a
normal serving-broker lease; it queues behind existing model workloads and never forces
GPU availability. It submits only actual RGB and synthetic visible-surface temperature
camera pixels, with an explicit synthetic-thermal disclaimer. This is not a thermal-trained
model or a real thermal camera. Simulator actor positions and target labels are excluded
from the vision request. Captured frame IDs, image hashes and camera poses identify the
source images; results expire after ten seconds and are discarded after actor/model/reset
changes. Accepted model text can accompany the next decision-model input.

For live PPO control, select `ppo` and provide `checkpoint` in the session configuration.
Only local Gym artifact checkpoints are accepted. Native SB3 checkpoints are trusted
training artifacts; do not move downloaded archives into that directory and load them.

Each step records the observation supplied to the controller, decision, applied action,
reward, next observation, termination flags, episode ID, and sequence. Reset starts a new
episode while sequence numbers keep increasing. Pending or rejected decisions record their
explicit fallback action. Completed model results retain their original input observation,
sequence, actor, revision, probability scores and latency in `decision_result`, separately
from the current transition.
Decision models receive named specialist state alongside the native numeric observation;
the submitted model input appears in `decision_result.model_state`; raw source state stays in
`decision_result.state`. Car inputs use native lane geometry, navigation, stop-rule state
and indexed obstacle ranges, with unobstructed lidar rays represented by range and count.
The raw sensor arrays remain in the recorded evidence. Camera pixels are replaced by capture metadata for the text-only
pointer model, while the vision model receives the image pixels. PPO receives the numeric observation.
Camera previews remain in the live stream and are omitted from saved trajectories by
default. Set `record_frames` to `true` to save separate PNG files with frame references.

```sh
ml-stack-gym run car --controller manual --action 5 --steps 100 \
  --config '{"map":"S","traffic_density":0.1}'
ml-stack-gym run warehouse --controller random --steps 100
ml-stack-gym replay /path/to/session/trajectory.jsonl
```

Sessions write `manifest.json`, `worker.log`, and `trajectory.jsonl` in their artifact
directory. The manifest identifies the environment, configuration, seed, and controller.
Decision-model manifests include model selection and CPU device.

## PPO training and evaluation

Training reuses Stable-Baselines3's PPO implementation, monitor, environment checker,
checkpoint callback, and native policies. Small numeric policies train on the CPU by
default. Car actions are discrete; warehouse policies use joint `MultiDiscrete` actions
and the sum of native individual rewards. Traffic policies use native SUMO-RL spaces.

```sh
ml-stack-gym train car --timesteps 2048 --seed 0 \
  --config '{"map":"S","traffic_density":0.1,"horizon":500}'
ml-stack-gym train warehouse --timesteps 2048 --seed 0
ml-stack-gym train traffic --timesteps 2048 --seed 0
ml-stack-gym evaluate car /path/to/gym/training-run/policy.zip \
  --episodes 5 --seed 10000 \
  --config '{"map":"S","traffic_density":0.1,"horizon":500}'
```

Use identical configuration and held-out seeds when comparing checkpoints. Evaluation
reports per-episode rewards and lengths. A short training run verifies the pipeline; it
does not establish that a policy learned useful behavior.

Training writes monitor data, periodic SB3 checkpoints, `policy.zip`, and a manifest with
dependency versions. Continue a run with `--checkpoint /path/to/gym/training-run/policy.zip`.
Continuation restores SB3 policy/optimizer state and starts new environment episodes;
it does not restore an exact physical rollout.

## Reviewed decision datasets

Label the action that should have been chosen for selected transitions. Save review rows
as JSONL using episode ID, sequence, and an action name from that session:

```json
{"episode_id":1,"sequence":12,"label":"straight brake"}
```

```sh
ml-stack-gym export /path/to/session/trajectory.jsonl reviews.jsonl cases.jsonl
```

Export includes only explicitly reviewed transitions and uses the observation before the
action, preserving the exact named controller state when recorded. Labels must name offered actions. Episode groups keep related transitions together
when splitting training/evaluation data. The output uses the existing labelled decision
dataset format and can feed [decision fine-tuning](decision-models.md). A trained pointer
directory can then be selected as the live Gym controller.

## Adding environments

Prefer a maintained task library with its own physics, sensors, rewards, and render assets.
Adapters expose its native spaces, state, and metrics to the workspace. Backends can
cooperate where their responsibilities are explicit; each state quantity has one owner.

Useful next integrations include [Gymnasium-Robotics](https://robotics.farama.org/) with
MuJoCo for robot manipulation, [MPE2](https://mpe2.farama.org/) with PettingZoo interfaces
for multiagent coordination, and [JoltPhysics.js](https://github.com/jrouwe/JoltPhysics.js)
for browser-native rigid-body examples. These are extension candidates, not installed Gym
environments. Jolt's upstream vehicle demos provide existing steering, suspension, and
collision examples; introducing it into a task requires an explicit ownership boundary.

Upstream references: [MetaDrive](https://metadrive-simulator.readthedocs.io/),
[RWARE](https://github.com/semitable/robotic-warehouse),
[SUMO-RL](https://github.com/LucasAlegre/sumo-rl),
[Gymnasium](https://gymnasium.farama.org/), and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
