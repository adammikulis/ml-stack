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

MetaDrive uses the verified source revision recorded in `pyproject.toml`. Its native
renderer downloads official assets on first use. SUMO comes through the `eclipse-sumo`
package; the adapter resolves `SUMO_HOME` and the packaged SUMO-RL intersection scenario.
An existing SUMO installation can also supply `SUMO_HOME`.

Set `ML_STACK_GYM_PYTHON` to an installed Python interpreter to run simulator workers
outside the current process. `ML_STACK_CACHE` moves recordings and checkpoints together
with the other ml-stack caches. Their default location is `~/.cache/ml_stack/gym`.

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
Its viewport reports backend ownership and synchronization telemetry.
The combined example supports one signalized intersection. Native MetaDrive IDM
controllers drive its vehicles; the learned policy controls SUMO-RL signal phases.

The car offers a native offscreen 3D camera and real lidar/lane sensor visualization. The
camera is a preview: numeric sensor observations are the policy input. Warehouse and
traffic views render authoritative state snapshots with Three.js. Warehouse movement is
discrete; visual animation does not create extra simulator steps.

Car seeds select native MetaDrive scenarios. Defaults cover seeds `0` through `19999`;
configure `start_seed` and `num_scenarios` to change this range. Out-of-range seeds are
rejected. The default car horizon is 1000 steps. Warehouse and traffic retain their finite
native episode limits, configurable through environment settings.
Session manifests record installed simulator versions and SHA256 hashes of the SUMO
network and route files. MetaDrive uses the pinned 0.4.3 source revision listed in the
`gym-driving` dependency; its installed package metadata records that source URL.

## Controllers and recordings

Choose manual control, a seeded random baseline, a PPO checkpoint, or a decision model.
Pause, reset, single-step, and speed controls act on the worker. Speed sets the requested
step rate; slow inference reduces the achieved rate. Simulation advancement waits for the
controller decision.

The default decision controller is `StrandsAgents/strands-decider-2B-hobson-v19` on the
CPU. Its files must already be available through the existing decision-model installation
workflow. Select a trained pointer-model directory with `decision_checkpoint` in the
session configuration. Native loading verifies model metadata and hashes and refuses
pickled model files. See [decision models](decision-models.md).

For live PPO control, select `ppo` and provide `checkpoint` in the session configuration.
Only local Gym artifact checkpoints are accepted. Native SB3 checkpoints are trusted
training artifacts; do not move downloaded archives into that directory and load them.

Each step records the observation supplied to the controller, decision, applied action,
reward, next observation, termination flags, episode ID, and sequence. Reset starts a new
episode while sequence numbers keep increasing. An inference failure or abstention pauses
the simulator and exposes the reason. No action is applied for an abstained decision.
Decision models receive named specialist state alongside the native numeric observation;
the exact model input appears in `decision.state`. PPO receives the numeric observation.
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
