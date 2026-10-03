"""Named decision-model inputs from specialist simulators."""

from ml_stack.gym.road_rules import front_progress
from ml_stack.gym.values import json_value


def decision_state(name, env, observation):
    """Describe the current controller observation without rendering media."""
    if hasattr(env, "decision_state"):
        return env.decision_state(observation)
    native = env.unwrapped
    state = {"environment": name, "native_observation": json_value(observation)}
    if name == "car":
        vehicle = native.agent
        sensor = next(iter(native.observations.values()))
        state.update(ego={"speed_km_h": float(vehicle.speed_km_h),
                          "heading_radians": float(vehicle.heading_theta),
                          "position_m": json_value(vehicle.position)},
                     sensors={"lidar_normalized": json_value(sensor.cloud_points),
                              "lidar_range_m": vehicle.config["lidar"]["distance"]})
        if hasattr(native, "stop_checkpoint"):
            state["stop_rule"] = native.stop_checkpoint.state(front_progress(native))
    elif name == "warehouse":
        state.update(robots=[{"x": int(agent.x), "y": int(agent.y),
                              "direction": agent.dir.name,
                              "carrying_shelf": agent.carrying_shelf is not None}
                             for agent in native.agents], goals=json_value(native.goals),
                     requested_shelves=[{"x": int(shelf.x), "y": int(shelf.y)}
                                        for shelf in native.request_queue])
    else:
        state["signals"] = [{"id": key, "phase": int(signal.green_phase),
                             "seconds_since_change": float(signal.time_since_last_phase_change),
                             "queues": list(signal.get_lanes_queue()),
                             "densities": list(signal.get_lanes_density())}
                            for key, signal in native.traffic_signals.items()]
        if name == "traffic-driving":
            geometry = native.geometry()
            state.update(authority=geometry["authority"], time=geometry["time"],
                         vehicles=[{key: vehicle[key] for key in ("id", "position", "speed", "sensors")
                                    if key in vehicle} for vehicle in geometry["vehicles"]])
    return state
