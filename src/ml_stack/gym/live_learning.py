"""CPU PPO rollouts over the live worker's single simulation clock."""

import json
import queue
import time

from ml_stack.gym.observations import decision_state
from ml_stack.gym.training import load_policy


def create_policy(simulation, checkpoint=None, learning_mode=None):
    """Bind PPO to the existing world without initializing another simulator."""
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO

    class LiveTask(gym.Env):
        def __init__(self):
            self.action_space = simulation.env.action_space
            self.observation_space = simulation.env.observation_space

        @property
        def unwrapped(self):
            return simulation.env.unwrapped

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return simulation.observation, dict(simulation.state.get("info", {})) if hasattr(simulation, "state") else {}

        def step(self, action):
            native = tuple(map(int, action)) if hasattr(action, "shape") and action.shape else int(action)
            state = decision_state(simulation.environment, simulation.env, simulation.observation)
            index = simulation.native_actions.index(native)
            decision = {"choice": simulation.names[index], "state": state, "probabilities": None,
                        "model": "PPO", "backend": "Stable-Baselines3", "learning_mode": "online"}
            simulation.advance(native, decision)
            transition = simulation.state["transition"]
            return (np.asarray(transition["next_observation"], dtype=self.observation_space.dtype),
                    transition["reward"], transition["terminated"], transition["truncated"],
                    dict(simulation.state["info"]))

    env = LiveTask()
    if checkpoint:
        return load_policy(checkpoint, env)
    if not (simulation.world and (learning_mode or simulation.learning_mode) == "online"):
        raise ValueError("Frozen PPO requires a checkpoint; choose online learning to create a policy")
    return PPO("MlpPolicy", env, device="cpu", seed=simulation.seed, n_steps=64, batch_size=32,
               n_epochs=4, verbose=0)


def learn_rollout(simulation, commands, updates, publish):
    """Train one native PPO rollout while handling worker commands between steps."""
    from stable_baselines3.common.callbacks import BaseCallback

    class LiveCallback(BaseCallback):
        closed = False
        revision = simulation.control_revision

        def _on_step(self):
            simulation.state["training_timesteps"] = self.model.num_timesteps
            publish(updates, simulation.state)
            deadline = time.monotonic() + 1 / simulation.speed
            while True:
                timeout = max(0, deadline - time.monotonic()) if simulation.running else .2
                try:
                    command, payload = commands.get(timeout=timeout)
                except queue.Empty:
                    if simulation.running:
                        return simulation.controller == "ppo" and simulation.learning_mode == "online"
                    continue
                if command == "close":
                    self.closed = True
                    return False
                if command == "step":
                    simulation.state["error"] = "Pause online learning and switch to frozen mode before single stepping"
                else:
                    try:
                        simulation.command(command, payload)
                    except (RuntimeError, ValueError, KeyError, TypeError, OSError, ImportError, AssertionError) as exc:
                        simulation.running = False
                        simulation.state.update(status="paused", error=str(exc))
                publish(updates, simulation.state)
                if (simulation.controller != "ppo" or simulation.learning_mode != "online"
                        or simulation.control_revision != self.revision):
                    return False

    callback = LiveCallback()
    model = simulation.policy
    old_updates = model._n_updates
    model._last_obs = None
    model.learn(total_timesteps=model.n_steps, callback=callback, reset_num_timesteps=False)
    simulation.state.update(training_timesteps=model.num_timesteps, optimizer_updates=model._n_updates)
    if model._n_updates != old_updates:
        simulation.state["policy_version"] += 1
        checkpoint = simulation.path / "policy.zip"
        model.save(checkpoint)
        simulation.settings["config"]["checkpoint"] = str(checkpoint)
        simulation.settings["model"] = str(checkpoint)
        simulation.settings.update(policy_version=simulation.state["policy_version"],
                                   training_timesteps=model.num_timesteps, optimizer_updates=model._n_updates)
        simulation.state.update(model=str(checkpoint), checkpoint=str(checkpoint))
        simulation.state["config"]["checkpoint"] = str(checkpoint)
        (simulation.path / "manifest.json").write_text(json.dumps(simulation.settings))
    publish(updates, simulation.state)
    return not callback.closed
