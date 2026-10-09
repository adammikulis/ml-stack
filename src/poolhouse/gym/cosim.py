"""SUMO-RL signal control with SUMO demand and MetaDrive vehicle dynamics."""

import math

from poolhouse.gym.cosim_map import RoadMap, sumo_heading
from poolhouse.gym.cosim_physics import Physics
from poolhouse.gym.traffic import traffic_defaults

AUTHORITY = {"signals": "sumo-rl", "demand_routes": "sumo",
             "vehicle_motion_collisions_sensors": "metadrive-bullet"}


class CosimulationMixin:
    def __init__(self, cfg, dt, lasers, distance, remove_arrived):
        self.physics_dt = dt
        self.lidar_num_lasers = lasers
        self.lidar_distance = distance
        self.remove_arrived = remove_arrived
        self.physics = None
        self.road_map = RoadMap(cfg["net_file"])
        self.bridge_steps = 0
        self.spawned_count = 0
        self.arrived_count = 0
        self.max_drift = 0.0
        self._ready = False
        self.departed_ids = []
        self.arrived_ids = []
        super().__init__(**cfg)
        if len(self.ts_ids) != 1:
            raise ValueError("The traffic-driving environment requires one signal-controlled intersection")

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Configure the shared scenario when creating the environment")
        success = False
        try:
            observation, info = super().reset(seed=seed)
            self.physics = Physics(self.road_map, self.physics_dt, self.lidar_num_lasers, self.lidar_distance, seed or 0)
            if not math.isclose(self.sumo.simulation.getDeltaT(), 1, abs_tol=1e-8):
                raise ValueError("Co-simulation requires a one-second SUMO timestep")
            self.physics.time = self.sim_step
            self.bridge_steps = self.spawned_count = self.arrived_count = 0
            self.max_drift = 0.0
            self.departed_ids = self.arrived_ids = []
            self._synchronize_ids()
            self.physics.synchronize_lights(self.sumo)
            self._ready = True
            success = True
            return observation, {**info, "cosimulation": self._bridge_info()}
        finally:
            if not success:
                self.close()

    def _synchronize_ids(self):
        current = set(self.sumo.vehicle.getIDList())
        for key in sorted(current - self.physics.actors.keys()):
            self.physics.spawn(key, self.sumo)
            self.spawned_count += 1
        for key in sorted(set(self.physics.actors) - current):
            if key not in self.sumo.simulation.getArrivedIDList():
                raise RuntimeError(f"SUMO removed physical vehicle {key} without a route arrival")
            self.physics.remove(key)
            self.arrived_count += 1

    def _sumo_step(self):
        if self.physics is None:
            raise RuntimeError("Reset the co-simulation before stepping")
        self._synchronize_ids()
        self.physics.synchronize_lights(self.sumo)
        physical_arrivals = self.physics.advance(1.0)
        for key in physical_arrivals:
            self.sumo.vehicle.remove(key, reason=self.remove_arrived)
            self.arrived_count += 1
        expected = {}
        for key, actor in self.physics.actors.items():
            direction = [math.cos(actor.heading_theta), math.sin(actor.heading_theta)]
            front = [float(actor.position[i] + actor.LENGTH / 2 * direction[i]) for i in range(2)]
            expected[key] = front
            self.sumo.vehicle.setSpeed(key, float(actor.speed))
            self.sumo.vehicle.moveToXY(key, "", -1, *front, angle=sumo_heading(actor.heading_theta),
                                      keepRoute=3, matchThreshold=100)
        super()._sumo_step()
        self.departed_ids = list(self.sumo.simulation.getDepartedIDList())
        self.arrived_ids = sorted(set(physical_arrivals) | set(self.sumo.simulation.getArrivedIDList()))
        teleported = self.sumo.simulation.getStartingTeleportIDList()
        if teleported:
            raise RuntimeError(f"SUMO teleported physical vehicles: {teleported}")
        active = set(self.sumo.vehicle.getIDList())
        for key, front in expected.items():
            if key in active:
                drift = math.dist(front, self.sumo.vehicle.getPosition(key))
                self.max_drift = max(self.max_drift, drift)
                if drift > 0.01:
                    raise RuntimeError(f"Vehicle {key} authority drifted by {drift:.3f}m")
        if not math.isclose(self.physics.time, self.sim_step, abs_tol=1e-8):
            raise RuntimeError("SUMO and MetaDrive clocks diverged")
        self.bridge_steps += 1
        self._synchronize_ids()

    def step(self, action):
        if not self._ready:
            raise RuntimeError("Reset the co-simulation before stepping")
        success = False
        try:
            observation, reward, terminated, truncated, info = super().step(action)
            success = True
            return observation, reward, terminated, truncated, {**info, "cosimulation": self._bridge_info()}
        finally:
            if not success:
                self.close()

    def _bridge_info(self):
        return {"authority": AUTHORITY, "steps": self.bridge_steps, "spawned": self.spawned_count,
                "arrived": self.arrived_count, "max_position_drift_m": self.max_drift,
                "physics_dt": self.physics_dt, "physics_time": self.physics.time if self.physics else None,
                "departed_ids": self.departed_ids, "arrived_ids": self.arrived_ids}

    def geometry(self):
        if not self._ready:
            raise RuntimeError("Reset the co-simulation before reading its geometry")
        vehicles = [{"id": key, "position": [float(v) for v in actor.position],
                     "heading": sumo_heading(actor.heading_theta), "heading_unit": "degrees",
                     "speed": float(actor.speed), "speed_unit": "metres_per_second",
                     "dimensions": {"length": actor.LENGTH, "width": actor.WIDTH, "height": actor.HEIGHT},
                     "sensors": self.physics.sensors(key)} for key, actor in self.physics.actors.items()]
        lights = [{"id": key, "state": self.sumo.trafficlight.getRedYellowGreenState(key),
                   "position": list(self.sumo.junction.getPosition(key))}
                  for key in self.sumo.trafficlight.getIDList()]
        return {"kind": "traffic-driving", "authority": AUTHORITY, "time": self.sim_step,
                "physics_time": self.physics.time, "coordinate_unit": "metres",
                "coordinate_system": "SUMO_XY", "vehicle_controller": "native_trajectory_idm",
                "roads": self.road_map.roads,
                "vehicles": vehicles, "lights": lights, "bridge": self._bridge_info()}

    def close(self):
        self._ready = False
        try:
            super().close()
        finally:
            if self.physics is not None:
                self.physics.close()
                self.physics = None


def make_cosim(config=None):
    """Create the traffic-driving Gymnasium environment."""

    cfg = dict(config or {})
    dt = float(cfg.pop("physics_dt", 0.02))
    lasers = int(cfg.pop("lidar_num_lasers", 36))
    distance = float(cfg.pop("lidar_distance", 50.0))
    if dt <= 0 or not math.isfinite(dt) or not math.isclose(round(0.1 / dt) * dt, 0.1, abs_tol=1e-8):
        raise ValueError("physics_dt must be positive and divide the native 0.1-second control interval")
    if not 4 <= lasers <= 360 or not 0 < distance <= 200:
        raise ValueError("lidar requires 4-360 rays and a range between 0 and 200 metres")
    cfg = traffic_defaults(cfg)
    cfg.update(single_agent=True, use_gui=False, time_to_teleport=-1)
    cfg.setdefault("sumo_seed", 0)
    cfg.setdefault("additional_sumo_cmd", "--step-length 1 --collision.action warn")
    from sumo_rl import SumoEnvironment
    from traci.constants import REMOVE_ARRIVED

    environment_type = type("Cosimulation", (CosimulationMixin, SumoEnvironment), {})
    return environment_type(cfg, dt, lasers, distance, REMOVE_ARRIVED)
