"""MetaDrive Bullet actors, native IDM control and lidar for a SUMO map."""

import math

from poolhouse.gym.cosim_map import physics_heading


class Physics:
    def __init__(self, road_map, dt, lasers, distance, seed):
        from metadrive import MetaDriveEnv
        from metadrive.component.map.scenario_map import ScenarioMap
        from metadrive.engine.base_engine import BaseEngine
        from metadrive.engine.engine_utils import initialize_engine, initialize_global_config
        from metadrive.manager.base_manager import BaseManager

        class AgentManager(BaseManager):
            def __init__(self):
                super().__init__()
                self.active_agents = {}

        class MapManager(BaseManager):
            def __init__(self):
                super().__init__()
                self.current_map = None

            def reset(self):
                self.current_map = self.spawn_object(ScenarioMap, map_index=0,
                                                     map_data=road_map.features, auto_fill_random_seed=False)
                for key, lane in road_map.lanes.items():
                    self.current_map.road_network.get_lane(key).width = lane.getWidth()

        config = MetaDriveEnv.default_config()
        config.update({"use_render": False, "image_observation": False, "show_interface": False,
                       "show_terrain": False, "physics_world_step_size": dt,
                       "num_scenarios": 1, "start_seed": 0, "decision_repeat": round(0.1 / dt)})
        config["vehicle_config"].update({"navigation_module": None, "show_navi_mark": False,
                                          "show_dest_mark": False,
                                          "lidar": {"num_lasers": lasers, "distance": distance,
                                                    "num_others": 0, "gaussian_noise": 0.0,
                                                    "dropout_prob": 0.0, "add_others_navi": False}})
        initialize_global_config(config)
        BaseEngine.global_random_seed = seed
        success = False
        try:
            self.engine = initialize_engine(config)
            self.engine.register_manager("map_manager", MapManager())
            self.engine.register_manager("agent_manager", AgentManager())
            self.engine.reset()
            success = True
        finally:
            if not success:
                from metadrive.engine.engine_utils import close_engine
                close_engine()
        self.map = self.engine.current_map
        self.road_map = road_map
        self.dt = dt
        self.lasers = lasers
        self.distance = distance
        self.actors = {}
        self.lights = {}
        self.policies = {}
        self.time = 0.0

    def spawn(self, key, connection):
        import numpy as np
        from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
        from metadrive.component.vehicle.vehicle_type import DefaultVehicle
        from metadrive.policy.idm_policy import FrontBackObjects, TrajectoryIDMPolicy

        class SignalIDMPolicy(TrajectoryIDMPolicy):
            def act(self, do_speed_control=True):
                objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
                objects = [obj for obj in objects if not isinstance(obj, BaseTrafficLight)
                           or obj.movement == self.movement]
                front = FrontBackObjects.get_find_front_back_objs_single_lane(
                    objects, self.routing_target_lane, self.control_object.position, self.IDM_MAX_DIST)
                acceleration = (self.acceleration(front.front_object(), front.front_min_distance())
                                if do_speed_control else self.last_action[-1])
                action = [self.steering_control(self.routing_target_lane), acceleration]
                self.last_action = action
                self.action_info["action"] = action
                return action

        heading = physics_heading(connection.vehicle.getAngle(key))
        position = np.array(connection.vehicle.getPosition(key))
        actor = self.engine.spawn_object(DefaultVehicle, name=f"sumo:{key}",
                                         position=position, heading=heading,
                                         vehicle_config={"navigation_module": None})
        center = position - np.array([math.cos(heading), math.sin(heading)]) * actor.LENGTH / 2
        actor.set_position(center, height=actor.HEIGHT / 2)
        route = self.road_map.route(connection.vehicle.getLaneID(key), connection.vehicle.getRoute(key))
        policy = self.engine.add_policy(actor.id, SignalIDMPolicy, actor, actor.random_seed, route)
        policy.target_speed = min(policy.NORMAL_SPEED, connection.vehicle.getMaxSpeed(key) * 3.6, route.speed_limit)
        policy.movement = None
        actor.set_velocity(actor.heading, connection.vehicle.getSpeed(key))
        connection.vehicle.setLength(key, actor.LENGTH)
        connection.vehicle.setWidth(key, actor.WIDTH)
        connection.vehicle.setSpeedMode(key, 0)
        connection.vehicle.setLaneChangeMode(key, 0)
        self.actors[key] = actor
        self.policies[key] = policy

    def synchronize_lights(self, connection):
        from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight

        for key, policy in self.policies.items():
            upcoming = connection.vehicle.getNextTLS(key)
            policy.movement = tuple(upcoming[0][:2]) if upcoming else None
        for signal_id in connection.trafficlight.getIDList():
            states = connection.trafficlight.getRedYellowGreenState(signal_id)
            for index, links in enumerate(connection.trafficlight.getControlledLinks(signal_id)):
                for incoming, outgoing, _via in links:
                    key = (signal_id, index, incoming, outgoing)
                    if key not in self.lights:
                        lane = self.map.road_network.get_lane(incoming)
                        self.lights[key] = self.engine.spawn_object(
                            BaseTrafficLight, name=f"light:{signal_id}:{index}:{incoming}:{outgoing}", lane=lane,
                            position=lane.position(lane.length - 0.5, 0), show_model=False)
                    light = self.lights[key]
                    light.movement = (signal_id, index)
                    state = states[index]
                    if state in "gG":
                        light.set_green()
                    elif state in "yY":
                        light.set_yellow()
                    else:
                        light.set_red()

    def advance(self, duration):
        steps = round(duration / self.dt)
        if not math.isclose(steps * self.dt, duration, abs_tol=1e-8):
            raise ValueError("Physics timestep must divide the SUMO timestep")
        completed = []
        repeat = self.engine.global_config["decision_repeat"]
        for _ in range(steps // repeat):
            self.engine.before_step({})
            for key, actor in self.actors.items():
                actor.before_step(self.policies[key].act(True))
            self.engine.step(repeat)
            for key, actor in list(self.actors.items()):
                actor.after_step()
                if self.policies[key].arrive_destination:
                    completed.append(key)
                    self.remove(key)
            self.engine.after_step()
        self.time += duration
        return completed

    def remove(self, key):
        actor = self.actors.pop(key)
        self.policies.pop(key)
        self.engine.clear_objects([actor.id], force_destroy=True)

    def sensors(self, key):
        actor = self.actors[key]
        rays, objects = actor.lidar.perceive(actor, self.engine.physics_world.dynamic_world,
                                           self.lasers, self.distance, show=False)
        from metadrive.utils.math import get_laser_end
        angles = actor.lidar._get_lidar_range(self.lasers, actor.lidar.start_phase_offset)
        endpoints = [get_laser_end(angles, self.distance * float(value), index,
                                  actor.heading_theta, *actor.position)
                     for index, value in enumerate(rays)]
        start = [float(v) for v in actor.position]
        return {"lidar": [float(v) for v in rays], "range_m": self.distance,
                "distance_unit": "normalized_fraction_of_range", "ray_coordinate_unit": "metres",
                "rays": [{"start": start, "end": [float(v) for v in endpoint]}
                         for endpoint in endpoints],
                "detected_objects": len(objects), "crash_vehicle": bool(actor.crash_vehicle),
                "crash_object": bool(actor.crash_object)}

    def close(self):
        from metadrive.engine.engine_utils import close_engine

        close_engine()
        self.actors.clear()
        self.policies.clear()
        self.lights.clear()
