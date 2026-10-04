# Drone search environment

The `drone` Gym environment uses PyFlyt's maintained quadrotor dynamics and Bullet
contacts. Native PID controllers fly a search patrol, while manual actions, a small
Stable-Baselines3 PPO policy, or a decision model can control the selected drone.
Other drones keep patrolling. Agent selection changes the inspected aircraft; **Take
control of selected agent** transfers the controller and camera to that aircraft.

## Installation and relaunch

Install the **Drone search** library from the environment installer. PyFlyt 0.29 requires
Python 3.12 and NumPy below 2; its environment is separate from the car's interpreter.
The installer keeps that environment for later launches.

For a source checkout, use a Python 3.12 environment:

```sh
python3.12 -m venv /path/to/drone-env
/path/to/drone-env/bin/python -m pip install -e '.[gym-drone]'
```

The environment's saved interpreter selection survives a daemon restart. Installed
simulators and downloaded models do not need to be fetched again.

## Worlds, training, and sensors

Choose a seeded procedural forest or a manual definition with trees, fires, and hikers.
World construction and learning mode are separate settings. An episodic run resets the
native world at episode boundaries. A live world keeps native time and aircraft alive
across learning tasks. Frozen policies keep their weights; online PPO updates train from
live transitions on the CPU.

The 3D view displays native aircraft and landmark positions. Rotor animation and tree
crowns decorate those positions; they add no native contacts or camera occlusion. RGB
and thermal panels show the selected aircraft's native camera samples, with capture time
and frame number. The debug camera footprint shows the down-facing sensor's view cone.

The thermal image is **synthetic visible-surface temperature**, using native Bullet
camera segmentation to retain occlusion: background is 20°C, visible hiker surfaces 37°C,
and visible fire surfaces 180°C. It does not simulate infrared radiometry, smoke transport,
heat diffusion, or temperatures behind trees. Heat-region markers describe warm/hot pixel
regions rather than supplying target identities to a policy.

The RL observation contains native aircraft state plus downsampled RGB and thermal pixels.
Reporting actions score only targets visible in the native camera. Session recordings keep
pre-action observations, actions, rewards, frame provenance, and model/policy versions.

## Small vision models

Downloaded model formats and backend compatibility are separate. The model selector shows
which installed models can be served. A vision result inspects a captured image; flight
physics continues while inference is pending. Older results are discarded after a camera,
controller, or world revision changes.

| Model | Intended use | Downloaded format |
| --- | --- | --- |
| SmolVLM-256M-Instruct | Speed-first RGB baseline | Q8 GGUF and Q8 vision projector |
| Qwen3.5-0.8B | Small general vision/language alternative | Q4_K_M GGUF and F16 vision projector |
| FastVLM-0.5B | Efficient Apple vision encoder comparison | Official MLX/Core ML model bundle |

These are selection candidates, not a measured throughput ranking on this machine.
General RGB vision training does not establish recognition accuracy on this synthetic
thermal representation. Test RGB and thermal inputs separately and retain false reports,
missed targets, frame age, and inference time in an evaluation.

Model references: [SmolVLM-256M model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct),
[Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[FastVLM code and Apple model downloads](https://github.com/apple-aiml-research/ml-fastvlm),
and [llama.cpp multimodal support](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).

Simulator references: [PyFlyt](https://github.com/jjshoots/PyFlyt) and
[Bullet](https://github.com/bulletphysics/bullet3).
