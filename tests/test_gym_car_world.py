"""Native car world identity, actor control, and PG map import."""

import json

import pytest

from ml_stack.gym.adapters import make_environment
from ml_stack.gym.values import json_value


@pytest.mark.slow
def test_native_car_world_continues_and_switches_actor_without_reset():
    pytest.importorskip("metadrive")
    env = make_environment("car", {"simulation_mode": "world", "horizon": 3,
                                   "num_agents": 4, "traffic_density": 0., "map": "SCSCS"})
    try:
        env.reset(seed=2)
        engine, road = env.native.engine, env.native.current_map
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
    monkeypatch.setenv("ML_STACK_GYM_FILES_ROOT", str(tmp_path))
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
