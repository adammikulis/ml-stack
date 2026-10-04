"""Stable-Baselines3 training, evaluation, and local checkpoints."""

import json
import uuid
from pathlib import Path

from ml_stack.files import write_json
from ml_stack.gym.adapters import make_environment
from ml_stack.gym.paths import artifact_root


def load_policy(checkpoint, env=None):
    """Load a trusted checkpoint produced in the Gym artifact directory."""
    from stable_baselines3 import PPO
    source = Path(checkpoint).expanduser().resolve(strict=True)
    if not source.is_relative_to(artifact_root().resolve()):
        raise ValueError("Only checkpoints in the local Gym artifact directory may be loaded")
    manifest_path = source.parent / "manifest.json"
    if env is not None and hasattr(env.unwrapped, "steering_magnitude") and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        trained = float(manifest.get("config", {}).get("steering_magnitude", 1.))
        if trained != env.unwrapped.steering_magnitude:
            raise ValueError(f"Car checkpoint requires steering_magnitude={trained}; use its training configuration")
    return PPO.load(source, env=env, device="cpu")


def train(environment, config=None, timesteps=2048, seed=0, checkpoint=None):
    """Train a CPU PPO policy and save its manifest and checkpoint."""
    from importlib.metadata import version

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor

    if timesteps < 1:
        raise ValueError("Training timesteps must be positive")
    config = dict(config or {})
    if environment == "car":
        config.setdefault("steering_magnitude", .35)
    path = artifact_root() / ("training-" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    env = make_environment(environment, config)
    try:
        check_env(env, warn=True)
        monitored = Monitor(env, str(path / "monitor.csv"))
        model = load_policy(checkpoint, monitored) if checkpoint else PPO(
            "MlpPolicy", monitored, device="cpu", seed=seed, verbose=1,
            n_steps=256, batch_size=64)
        model.learn(total_timesteps=timesteps, reset_num_timesteps=checkpoint is None,
                    callback=CheckpointCallback(save_freq=256, save_path=str(path), name_prefix="ppo"))
        model.save(path / "policy.zip")
        manifest = {"version": 1, "environment": environment, "config": config or {}, "seed": seed,
                        "timesteps": model.num_timesteps, "device": "cpu", "checkpoint": str(path / "policy.zip"),
                        "versions": {name: version(name) for name in ("gymnasium", "stable-baselines3", "torch")}}
        write_json(path / "manifest.json", manifest)
        return manifest
    finally:
        env.close()


def evaluate(environment, checkpoint, config=None, episodes=5, seed=10000):
    """Evaluate a local PPO policy using seeded independent episodes."""
    if episodes < 1:
        raise ValueError("Evaluation episodes must be positive")
    if (config or {}).get("simulation_mode") == "world":
        raise ValueError("Independent evaluation requires episode mode; inspect persistent-world trajectories separately")
    env = make_environment(environment, config)
    try:
        policy = load_policy(checkpoint, env)
        rows = []
        for index in range(episodes):
            observation, _ = env.reset(seed=seed + index)
            reward, steps = 0., 0
            while True:
                action, _ = policy.predict(observation, deterministic=True)
                observation, score, terminated, truncated, _ = env.step(action)
                reward += float(score)
                steps += 1
                if terminated or truncated:
                    break
            rows.append({"seed": seed + index, "reward": reward, "steps": steps,
                             "terminated": bool(terminated), "truncated": bool(truncated)})
        return {"environment": environment, "episodes": rows,
                    "mean_reward": sum(row["reward"] for row in rows) / episodes}
    finally:
        env.close()
