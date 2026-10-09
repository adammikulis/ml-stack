"""Native car world identity, actor control, and PG map import."""

import json

import pytest

from poolhouse.gym import simulation
from poolhouse.gym.adapters import make_environment
from poolhouse.gym.car_definition import build
from poolhouse.gym.training import evaluate, train
from poolhouse.gym.values import json_value


@pytest.mark.parametrize("road", [True, False, 0, 25, 3., None, [], "", "S" * 25])
def test_car_definition_rejects_non_native_road_values(road):
    with pytest.raises(ValueError, match="block sequence or integer block count"):
        build({"map": road}, 2)


@pytest.mark.slow
def test_native_numeric_road_count_keeps_integer_and_seeded_geometry():
    pytest.importorskip("metadrive")
    config = {"simulation_mode": "world", "num_agents": 2, "traffic_density": 0.,
              "world": {"map": 3, "seed": 2}}
    lanes = None
    for _ in range(2):
        env = make_environment("car", config)
        try:
            env.reset(seed=99)
            assert env.native.config["map"] == 3
            assert type(env.native.config["map"]) is int
            assert len(env.native.current_map.blocks) == 4
            assert env.native.current_seed == 2
            current = json_value(env.native.current_map.get_boundary_line_vector(3))
            if lanes is not None:
                assert current == lanes
            lanes = current
            env.step(None)
        finally:
            env.close()


@pytest.mark.slow
def test_native_car_world_continues_and_switches_actor_without_reset():
    pytest.importorskip("metadrive")
    env = make_environment("car", {"simulation_mode": "world", "horizon": 3,
                                   "num_agents": 4, "traffic_density": 0., "map": "SCSCS"})
    try:
        env.reset(seed=2)
        engine, road = env.native.engine, env.native.current_map
        assert env.native.engine.external_actions is None
        assert env.render_state()["vehicles"]
        actors = list(env.native.agents)
        first, second = actors[:2]
        for _ in range(5):
            _, _, ended, limited, info = env.step(None)
            if ended or limited:
                env.reset(seed=123)
        assert info["world_steps"] == 5
        assert info["world_time"] == pytest.approx(.5)
        assert info["world_reset_count"] == 1
        observation, selected = env.select_agent(second)
        assert selected["ego_actor_id"] == second
        assert env.observation_space.contains(observation)
        assert env.native.engine is engine and env.native.current_map is road
        assert env.ego_id != first
        with pytest.raises(ValueError, match="no longer active"):
            env.select_agent("missing-actor")
        _, _, _, _, following = env.step(None)
        assert following["transition_actor_id"] == second
        assert following["world_steps"] == 6
        assert len(following["applied_native_control"]) == 2
        assert env.render_state()["vehicles"]
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.parametrize("preview", [False, True])
def test_car_simulation_initial_snapshot_precedes_native_actions(tmp_path, monkeypatch, preview):
    pytest.importorskip("metadrive")
    monkeypatch.setattr(simulation, "artifact_root", lambda: tmp_path)
    live = simulation.Simulation({"id": "startup", "environment": "car", "seed": 2,
        "controller": "native-idm", "config": {"simulation_mode": "world", "render_preview": preview,
                                               "num_agents": 4, "traffic_density": 0., "map": "SCS"}})
    try:
        assert live.state["status"] == "paused"
        assert live.state["info"]["world_steps"] == 0
        assert live.state["info"]["render"]["vehicles"]
        assert live.env.native.engine.external_actions is None
        assert bool(live.state["frame"]) == preview
        live.step()
        assert live.state["info"]["world_steps"] == 1
        assert live.state["decision"]["choice"] == "Native IDM driving"
    finally:
        live.env.close()


@pytest.mark.slow
def test_native_manual_map_recreates_generated_lane_geometry(tmp_path, monkeypatch):
    pytest.importorskip("metadrive")
    env = make_environment("car", {"simulation_mode": "world", "num_agents": 2,
                                   "traffic_density": 0., "world": {"seed": 2, "map": "SCS"}})
    try:
        env.reset(seed=99)
        assert env.native.current_seed == 2
        metadata = env.native.current_map.get_meta_data()
        metadata["map_config"] = metadata["map_config"].get_serializable_dict()
        lanes = env.native.current_map.get_boundary_line_vector(3)
        source = tmp_path / "country.json"
        source.write_text(json.dumps(json_value(metadata)))
    finally:
        env.close()
    monkeypatch.setenv("POOLHOUSE_GYM_FILES_ROOT", str(tmp_path))
    imported = make_environment("car", {"simulation_mode": "world", "num_agents": 2,
                                        "traffic_density": 0., "world": {"mode": "manual", "map_file": "country.json"}})
    try:
        imported.reset(seed=2)
        assert json_value(imported.native.current_map.get_boundary_line_vector(3)) == json_value(lanes)
        assert imported.world_provenance["definition"]["mode"] == "manual"
        assert imported.world_provenance["files"]["map_file"]["sha256"]
    finally:
        imported.close()


@pytest.mark.slow
def test_native_car_arrivals_handoff_in_the_same_world():
    pytest.importorskip("metadrive")
    env = make_environment("car", {"simulation_mode": "world", "num_agents": 4,
                                   "traffic_density": 0., "map": "S", "stop_signs": False})
    try:
        env.reset(seed=2)
        road, engine = env.native.current_map, env.native.engine
        original = env.ego_id
        for _ in range(1000):
            _, _, ended, limited, info = env.step(None)
            if ended or limited:
                env.reset()
            if info["actor_handoffs"]:
                break
        assert info["actor_handoffs"] >= 1
        assert env.ego_id != original
        assert info["world_steps"] > 1
        assert env.native.current_map is road and env.native.engine is engine
        assert info["world_reset_count"] == 1
        assert env.native.agents
    finally:
        env.close()


@pytest.mark.slow
def test_procedural_episode_definitions_support_training_and_heldout_evaluation():
    pytest.importorskip("metadrive")
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    config = {"horizon": 20, "traffic_density": 0., "world": {"map": 3, "seed": 2}}
    env = make_environment("car", config)
    try:
        geometries = {}
        for seed in (0, 2, 10000):
            env.reset(seed=seed)
            assert env.unwrapped.current_seed == seed
            geometries[seed] = json_value(env.unwrapped.current_map.get_boundary_line_vector(3))
        assert geometries[0] != geometries[2] != geometries[10000]
        env.reset(seed=2)
        assert json_value(env.unwrapped.current_map.get_boundary_line_vector(3)) == geometries[2]
    finally:
        env.close()
    trained = train("car", config, timesteps=256, seed=2)
    assert trained["timesteps"] == 256
    evaluated = evaluate("car", trained["checkpoint"], config, episodes=2, seed=10000)
    assert [episode["seed"] for episode in evaluated["episodes"]] == [10000, 10001]
    assert all(0 < episode["steps"] <= 20 for episode in evaluated["episodes"])


@pytest.mark.slow
def test_manual_episode_geometry_is_independent_of_scenario_reset_seed(tmp_path, monkeypatch):
    pytest.importorskip("metadrive")
    source = make_environment("car", {"traffic_density": 0., "world": {"map": "SCS", "seed": 2}})
    try:
        source.reset(seed=2)
        metadata = source.unwrapped.current_map.get_meta_data()
        metadata["map_config"] = metadata["map_config"].get_serializable_dict()
        expected = json_value(source.unwrapped.current_map.get_boundary_line_vector(3))
        (tmp_path / "authored.json").write_text(json.dumps(json_value(metadata)))
    finally:
        source.close()
    monkeypatch.setenv("POOLHOUSE_GYM_FILES_ROOT", str(tmp_path))
    env = make_environment("car", {"traffic_density": 0., "world": {
        "mode": "manual", "map_file": "authored.json", "seed": 2}})
    try:
        for seed in (0, 2, 10000):
            env.reset(seed=seed)
            assert env.unwrapped.current_seed == seed
            assert json_value(env.unwrapped.current_map.get_boundary_line_vector(3)) == expected
            env.step(4)
    finally:
        env.close()
