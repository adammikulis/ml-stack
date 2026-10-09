"""SUMO lane geometry and routes for MetaDrive scenario maps."""

import math


class RoadMap:
    def __init__(self, filename):
        import numpy as np
        import sumolib

        self.net = sumolib.net.readNet(filename, withInternal=True)
        self.lanes = {lane.getID(): lane for edge in self.net.getEdges(withInternal=True)
                      for lane in edge.getLanes()}
        self.features = {}
        self.roads = []
        from metadrive.type import MetaDriveType

        for key, lane in self.lanes.items():
            points = np.array(lane.getShape(), dtype=float)
            if len(points) < 2:
                raise ValueError(f"SUMO lane {key} has no usable centerline")
            width = lane.getWidth()
            left = sumolib.geomhelper.move2side(points.tolist(), width / 2)
            right = sumolib.geomhelper.move2side(points.tolist(), -width / 2)
            self.features[key] = {"type": MetaDriveType.LANE_SURFACE_STREET, "polyline": points,
                                  "polygon": np.array([*left, *reversed(right)]),
                                  "speed_limit_kmh": lane.getSpeed() * 3.6,
                                  "entry_lanes": [], "exit_lanes": []}
            self.roads.append({"id": key, "points": points.tolist(), "width": width})
        for key, lane in self.lanes.items():
            for link in lane.getOutgoing():
                target = link.getViaLaneID() or link.getToLane().getID()
                if target in self.features:
                    self.features[key]["exit_lanes"].append(target)
                    self.features[target]["entry_lanes"].append(key)

    def route(self, start_lane, edges):
        import numpy as np

        lane = self.lanes[start_lane]
        selected = [lane]
        current_edge = lane.getEdge().getID()
        remaining = list(edges)
        if current_edge in remaining:
            remaining = remaining[remaining.index(current_edge) + 1:]
        else:
            target_edges = {link.getToLane().getEdge().getID() for link in lane.getOutgoing()}
            candidates = [edge for edge in remaining if edge in target_edges]
            if not candidates:
                raise ValueError(f"SUMO internal lane {start_lane} cannot join its vehicle route")
            remaining = remaining[remaining.index(candidates[0]):]
        for edge in remaining:
            links = [c for c in lane.getOutgoing() if c.getToLane().getEdge().getID() == edge]
            if not links:
                raise ValueError(f"SUMO route cannot connect lane {lane.getID()} to edge {edge}")
            link = links[0]
            via = link.getViaLaneID()
            visited = set()
            while via:
                if via in visited:
                    raise ValueError(f"SUMO internal lane cycle at {via}")
                visited.add(via)
                internal = self.lanes[via]
                selected.append(internal)
                outgoing = internal.getOutgoing()
                via = outgoing[0].getViaLaneID() if outgoing else ""
            lane = link.getToLane()
            selected.append(lane)
        points = []
        for selected_lane in selected:
            for point in selected_lane.getShape():
                if not points or math.dist(points[-1], point) > 0.01:
                    points.append(point)
        from metadrive.component.lane.point_lane import PointLane

        return PointLane(np.array(points), width=selected[0].getWidth(),
                         speed_limit=selected[0].getSpeed() * 3.6)


def sumo_heading(radians):
    return (90 - math.degrees(radians)) % 360


def physics_heading(degrees):
    return math.radians(90 - degrees)
