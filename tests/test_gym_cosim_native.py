"""Native SUMO/MetaDrive shared-scene integration."""

import math

import pytest

from ml_stack.gym.cosim import make_cosim
from ml_stack.gym.cosim_physics import Physics
from ml_stack.gym.traffic import traffic_defaults

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def native_dependencies():
    pytest.importorskip("metadrive", reason="ml-stack[gym-driving]")
    pytest.importorskip("sumo", reason="ml-stack[gym-traffic]")


def test_native_signal_policy_controls_physical_vehicles_and_sensors():
    env = make_cosim({"num_seconds":30})
    try:
        observation, _info = env.reset(seed=4)
        assert env.observation_space.contains(observation)
        assert isinstance(env.physics, Physics)
        assert env.action_space.n >= 2
        for action in (0, 1, 1, 0, 0, 1):
            observation, reward, terminated, truncated, _info = env.step(action)
            scene = env.geometry()
            assert env.observation_space.contains(observation)
            assert math.isfinite(reward)
            assert scene["time"] == scene["physics_time"]
            assert scene["bridge"]["max_position_drift_m"] < 0.01
            assert set(env.physics.actors) == set(env.sumo.vehicle.getIDList())
            for vehicle in scene["vehicles"]:
                assert len(vehicle["sensors"]["lidar"]) == 36
                assert len(vehicle["sensors"]["rays"]) == 36
                native_speed = env.sumo.vehicle.getSpeed(vehicle["id"])
                assert abs(native_speed - vehicle["speed"]) < 0.001
            if terminated or truncated:
                break
        assert scene["bridge"]["spawned"] > 0
        assert scene["time"] == 30
        assert truncated
        assert all(math.isfinite(v) for car in scene["vehicles"] for v in car["sensors"]["lidar"])
        observation, _info = env.reset(seed=4)
        assert env.observation_space.contains(observation)
        assert env.geometry()["time"] == 0
        assert env.geometry()["vehicles"] == []
        assert env.geometry()["bridge"]["spawned"] == 0
    finally:
        env.close()


def test_red_phase_queues_physical_cars_and_green_releases_them():
    env = make_cosim({"num_seconds":60})
    try:
        env.reset(seed=4)
        for _ in range(7):
            _, _, _, _, info = env.step(0)
        scene = env.geometry()
        car = next(v for v in scene["vehicles"] if v["id"] == "flow_we.0")
        assert car["speed"] < 0.3, car
        assert car["position"][0] + car["dimensions"]["length"] / 2 < 142
        assert info["system_total_waiting_time"] > 0, {"speed":car["speed"], "info":info}
        red_x = car["position"][0]
        for _ in range(3):
            env.step(1)
        scene = env.geometry()
        car = next(v for v in scene["vehicles"] if v["id"] == "flow_we.0")
        assert car["position"][0] > red_x + 20
        assert car["speed"] > 1
        assert scene["bridge"]["max_position_drift_m"] < 0.01
    finally:
        env.close()



def two_way_scenario(tmp_path, route):
    from pathlib import Path

    import sumo_rl

    network = Path(sumo_rl.__file__).parent / "nets" / "2way-single-intersection" / "single-intersection.net.xml"
    routes = tmp_path / "finite.rou.xml"
    routes.write_text(f'<routes><route id="one" edges="{route}"/>'
                      '<vehicle id="finite" route="one" depart="0" departSpeed="max" departLane="best"/></routes>')
    return {"net_file": str(network), "route_file": str(routes), "num_seconds":80}


def test_native_curved_route_finishes_and_removes_both_actors(tmp_path):
    traffic_defaults({})
    env = make_cosim(two_way_scenario(tmp_path, "n_t t_e"))
    try:
        env.reset(seed=7)
        observed_turn = False
        for _ in range(15):
            env.step(1)
            scene = env.geometry()
            for car in scene["vehicles"]:
                observed_turn |= abs(car["heading"] - 180) > 15
            if scene["bridge"]["arrived"]:
                break
        assert observed_turn
        assert scene["bridge"]["spawned"] == 1
        assert scene["bridge"]["arrived"] == 1, {
            "cars": [{k:v for k,v in c.items() if k != "sensors"} for c in scene["vehicles"]],
            "route": [(key, policy.destination.tolist(),
                       policy.traj_to_follow.local_coordinates(env.physics.actors[key].position))
                      for key, policy in env.physics.policies.items()]}
        assert scene["vehicles"] == []
        assert env.sumo.vehicle.getIDList() == ()
        assert env.physics.actors == {}
    finally:
        env.close()


def test_green_straight_movement_ignores_red_turn_wall_on_shared_lane(tmp_path):
    traffic_defaults({})
    env = make_cosim(two_way_scenario(tmp_path, "e_t t_w"))
    try:
        env.reset(seed=7)
        state = "rrrrGrrrrrrr"
        crossed = False
        for _ in range(50):
            env.sumo.trafficlight.setRedYellowGreenState("t", state)
            env._sumo_step()
            if "finite" in env.physics.actors:
                actor = env.physics.actors["finite"]
                next_tls = env.sumo.vehicle.getNextTLS("finite")
                if next_tls:
                    assert next_tls[0][1] == 4
                crossed |= actor.position[0] < 140
            if env.arrived_count:
                break
        assert crossed
        assert env.arrived_count == 1
    finally:
        env.close()



def test_clock_failure_closes_both_native_simulators():
    from metadrive.engine.engine_utils import engine_initialized

    env = make_cosim({"num_seconds":10})
    env.reset(seed=4)
    env.physics.time += 0.1
    with pytest.raises(RuntimeError, match="clocks diverged"):
        env.step(0)
    assert env.sumo is None
    assert env.physics is None
    assert not engine_initialized()
    env.close()
