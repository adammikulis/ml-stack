"""MetaDrive lane geometry and numeric sensor telemetry."""

import math

from ml_stack.gym.values import json_value


def geometry(native):
    """Read authoritative road, vehicle, and lidar geometry."""
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    from metadrive.utils.math import get_laser_end

    vehicle = native.vehicle
    observation = next(iter(native.observations.values()))
    distances = observation.cloud_points
    count = len(distances)
    angles = [2 * math.pi * index / count for index in range(count)]
    maximum = vehicle.config["lidar"]["distance"]
    rays = [list(get_laser_end(angles, float(distance) * maximum, index, vehicle.heading_theta,
                              vehicle.position[0], vehicle.position[1]))
            for index, distance in enumerate(distances)]
    roads = []
    for start, destinations in native.current_map.road_network.graph.items():
        for end, lanes in destinations.items():
            for index, lane in enumerate(lanes):
                segments = max(2, int(lane.length / 5))
                points = [json_value(lane.position(lane.length * fraction / segments, 0))
                          for fraction in range(segments + 1)]
                roads.append({"id": f"{start}:{end}:{index}", "points": points,
                              "width": float(lane.width_at(0)),
                              "line_types": json_value(lane.line_types), "line_colors": json_value(lane.line_colors)})
    vehicles = [{"id": key, "position": json_value(obj.position),
                 "heading": float(obj.heading_theta), "speed": float(obj.speed_km_h) / 3.6,
                 "length": float(obj.LENGTH), "width": float(obj.WIDTH), "height": float(obj.HEIGHT),
                 "last_action": json_value(obj.last_action)}
                for key, obj in native.engine.get_objects().items() if isinstance(obj, BaseVehicle)]
    stop = getattr(native, "stop_checkpoint", None)
    stop_state = stop.state(vehicle.navigation.travelled_length) if stop else None
    rule_sensors = [{"name": "upcoming stop", "value": stop_state["distance_m"], "unit": "m"},
                    {"name": "stop hold", "value": stop_state["held_seconds"], "unit": "s"},
                    {"name": "stop compliance", "value": stop_state["state"], "unit": "status"}] if stop_state else []
    return {"heading_unit": "radians", "vehicle": {"id": vehicle.id, "position": json_value(vehicle.position), "heading": float(vehicle.heading_theta),
                        "speed": float(vehicle.speed_km_h)}, "vehicles": vehicles, "roads": roads,
            "lidar": {"origin": json_value(vehicle.position), "endpoints": rays,
                      "distances_m": [float(distance) * maximum for distance in distances],
                      "range_m": maximum},
            "sensors": [{"name": "speed", "value": float(vehicle.speed_km_h), "unit": "km/h"},
                        {"name": "heading", "value": float(vehicle.heading_theta), "unit": "rad"},
                        {"name": "lidar", "value": json_value(distances), "unit": "normalized range"}, *rule_sensors],
            "stop_signs": [stop_state] if stop_state else [],
            "camera_is_policy_input": False}
