"""Visibility-limited native RGB and synthetic thermal observations."""

import base64
import io


def png(array):
    from PIL import Image
    output = io.BytesIO()
    Image.fromarray(array).save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode('ascii')


def instrument_camera(native, drone):
    """Keep pose and clock from the instant the maintained camera captures."""
    original = drone.camera.capture_image

    def recorded_capture():
        images = original()
        drone.sensor_metadata = {'pose': {'position': drone.state[3].tolist(),
                                           'rotation': drone.state[1].tolist()},
                                 'frame_id': native.physics_steps + 1,
                                 'world_time': (native.physics_steps + 1) / native.physics_hz}
        return images

    drone.camera.capture_image = recorded_capture


def capture(env, fresh=False):
    import numpy as np
    drone = env.native.drones[env.actor]
    if hasattr(drone, 'rgbaImg') and not fresh:
        rgba, depth, segmentation = drone.rgbaImg, drone.depthImg, drone.segImg

    else:
        rgba, depth, segmentation = drone.camera.capture_image()
        drone.rgbaImg, drone.depthImg, drone.segImg = rgba, depth, segmentation
        drone.sensor_metadata['frame_id'] = env.native.physics_steps
        drone.sensor_metadata['world_time'] = env.native.elapsed_time
    ids = segmentation[:, :, 0].astype(np.int64)
    ids = np.where(ids < 0, -1, ids & ((1 << 24) - 1))
    thermal = np.full(ids.shape, 20., dtype=np.float32)
    visible = []
    for body, target in env.targets.items():
        mask = ids == body
        thermal[mask] = 37. if target['kind'] == 'hiker' else 180.
        if np.count_nonzero(mask) >= 2:
            visible.append(body)
    # This is a temperature sensor, not an oracle semantic detector.
    detections = []
    for label, mask in [('warm surface', (thermal >= 30) & (thermal < 80)),
                        ('hot surface', thermal >= 80)]:
        yy, xx = np.nonzero(mask)
        if len(xx) >= 2:
            detections.append({'label': label, 'pixels': len(xx),
                               'centroid': [float(xx.mean() / ids.shape[1]), float(yy.mean() / ids.shape[0])]})
    env.rgb = rgba[:, :, :3].astype(np.uint8)
    env.thermal = thermal
    env.visible_targets = visible
    normalized = np.clip((thermal - 20) / 160, 0, 1)
    env.thermal_rgb = np.stack([normalized * 255, normalized**.5 * 180,
                                (1 - normalized) * 80], axis=-1).astype(np.uint8)
    env.camera = {'rgb': png(env.rgb), 'thermal': png(env.thermal_rgb),
                  'agent_id': f'drone-{env.actor}',
                  'thermal_kind': 'synthetic visible-surface temperature',
                  'intrinsics': {'width': 128, 'height': 128, 'fov_degrees': 90},
                  **drone.sensor_metadata,
                  'detections_visible': detections}
    # Actual camera pixels, not target locations/classes, are PPO's visual input.
    env.visual_observation = np.concatenate([env.rgb[::8, ::8].ravel() / 255,
                                            normalized[::8, ::8].ravel()]).astype(np.float32)
    env.depth = depth


def tree_visual(native, height, radius):
    """Layer the maintained Bullet cone mesh above a short trunk."""
    scales = [[radius * (3.2 - i * .6) / .06] * 2 + [height * .48 / .165] for i in range(3)]
    positions = [[0, 0, height * (.225 - .5)]] + [
        [0, 0, height * (.24 + i * .17 - .5) - .005 * height * .48 / .165] for i in range(3)]
    return native.createVisualShapeArray(
        shapeTypes=[native.GEOM_CYLINDER] + [native.GEOM_MESH] * 3,
        radii=[radius * .3, 0, 0, 0], lengths=[height * .45, 0, 0, 0],
        fileNames=[''] + ['racecar/meshes/cone.obj'] * 3,
        meshScales=[[1, 1, 1], *scales], visualFramePositions=positions,
        rgbaColors=[[.36, .25, .14, 1], [.14, .30, .22, 1], [.21, .40, .29, 1], [.26, .47, .33, 1]])


def forest(native, definition):
    """Construct scenery with maintained Bullet shapes and collision handling."""
    bodies, targets = [], {}
    native.changeVisualShape(native.planeId, -1, textureUniqueId=-1, rgbaColor=[.34, .39, .22, 1])
    for kind in ['trees', 'hikers', 'fires']:
        for row in definition[kind]:
            if kind == 'trees':
                height, radius, color = row['height'], row['radius'], [.15, .3, .12, 1]
            elif kind == 'hikers':
                height, radius, color = 1.6, .25, [.95, .35, .15, 1]
            else:
                height, radius, color = .4, .8, [1, .15, .02, 1]
            shape = native.createCollisionShape(native.GEOM_CYLINDER, radius=radius, height=height)
            visual = (tree_visual(native, height, radius) if kind == 'trees' else
                      native.createVisualShape(native.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color))
            body = native.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                          baseVisualShapeIndex=visual,
                                          basePosition=[row['x'], row['y'], height / 2])
            bodies.append({'body_id': body, 'kind': kind, **row, 'height': height, 'radius': radius})
            if kind != 'trees':
                targets[body] = {'kind': 'hiker' if kind == 'hikers' else 'fire', **row}
    return bodies, targets
