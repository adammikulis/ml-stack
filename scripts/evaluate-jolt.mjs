import initJolt from '../app/node_modules/jolt-physics/dist/jolt-physics.wasm-compat.js';

export async function evaluate() {
  const J = await initJolt();
  const settings = new J.JoltSettings();
  settings.mMaxBodies = 128;
  const layers = new J.ObjectLayerPairFilterTable(2);
  layers.EnableCollision(0, 1);
  layers.EnableCollision(1, 1);
  settings.mObjectLayerPairFilter = layers;
  const broad = new J.BroadPhaseLayerInterfaceTable(2, 2);
  for (let i = 0; i < 2; i++) {
    const layer = new J.BroadPhaseLayer(i);
    broad.MapObjectToBroadPhaseLayer(i, layer);
    J.destroy(layer);
  }
  settings.mBroadPhaseLayerInterface = broad;
  settings.mObjectVsBroadPhaseLayerFilter = new J.ObjectVsBroadPhaseLayerFilterTable(broad, 2, layers, 2);
  const world = new J.JoltInterface(settings);
  J.destroy(settings);
  const bodies = world.GetPhysicsSystem().GetBodyInterface();
  function add(shape, y, motion, layer) {
    const position = new J.RVec3(0, y, 0);
    const rotation = new J.Quat(0, 0, 0, 1);
    const config = new J.BodyCreationSettings(shape, position, rotation, motion, layer);
    const body = bodies.CreateBody(config);
    bodies.AddBody(body.GetID(), J.EActivation_Activate);
    J.destroy(config);
    J.destroy(position);
    J.destroy(rotation);
    return body;
  }
  const extent = new J.Vec3(10, 0.5, 10);
  const ground = add(new J.BoxShape(extent, 0.05), -0.5, J.EMotionType_Static, 0);
  J.destroy(extent);
  const sphere = add(new J.SphereShape(0.5), 5, J.EMotionType_Dynamic, 1);
  const began = performance.now();
  for (let i = 0; i < 300; i++) world.Step(1 / 60, 1);
  const elapsed = performance.now() - began;
  const height = sphere.GetPosition().GetY();
  if (height < 0.45 || height > 0.55) throw Error(`Contact failed: sphere at ${height}`);
  for (const body of [sphere, ground]) {
    const id = body.GetID();
    bodies.RemoveBody(id);
    bodies.DestroyBody(id);
  }
  J.destroy(world);
  return { steps: 300, height, elapsed_ms: elapsed, contact: 'passed' };
}

if (typeof window !== 'undefined') {
  window.joltEvaluation = evaluate();
} else {
  process.stdout.write(JSON.stringify(await evaluate()) + '\n');
}
