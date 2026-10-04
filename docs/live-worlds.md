# Persistent simulator worlds

The Gym workspace offers **Training / evaluation episode** and **Live world**. Training episodes
end at the task horizon and require a reset. A persistent world initializes its native map
and simulator once. Learning task boundaries reset counters and select the next active actor;
they preserve the road, physics engine, simulator connection, other actors, and world clock.
Create a new world to change its seed or definition.

Each example supports procedural and manual definitions using its simulator's formats:

| Example | Procedural definition | Manual definition |
| --- | --- | --- |
| Smart car | Seeded MetaDrive PG blocks, such as `SCSCS` | Native PGMap metadata JSON |
| Warehouse | Seeded RWARE shelves and aisles | RWARE ASCII layout |
| Traffic | Seeded SUMO network and demand | SUMO network and route XML |
| Smart traffic city | Seeded SUMO intersection imported into MetaDrive | SUMO network and route XML |

Manual files use paths relative to the workspace files directory. Imports reject absolute
paths and paths escaping that directory. World manifests retain the definition and file
hashes. The combined traffic example supports one signal-controlled intersection: SUMO owns
signals, demand, and routes; MetaDrive owns vehicle dynamics and sensors.

In the car world, MetaDrive manages active vehicles and respawns arrivals. **Native driving policy** uses IDM to drive
the watched car and the other cars with native continuous steering and acceleration. The
recorded native control remains separate from the discrete nine-action space used by manual,
decision-model, and PPO controllers. Left/right arrows select an agent for inspection. **Take control of selected agent** changes the controlled actor
at a simulation step boundary and retains the world. A completed actor hands control to another
active car. Stop rules require a low-speed hold before the stop line and record compliance.

**Frozen** PPO uses predictions without optimizer updates. It requires a trusted checkpoint
inside the local Gym artifact directory. **Online** PPO can create a CPU policy or continue a
trusted policy in a persistent world. Stable-Baselines3 collects 64 live transitions per rollout
and runs four optimizer epochs. The worker owns the single simulation clock throughout both
collection and learning. Pause stops world advancement between steps; switch to Frozen for
single stepping. Switching agents or resetting a learning task interrupts the current rollout
before its optimizer update. Policy versions, training timesteps, optimizer updates, world time,
and the latest saved checkpoint appear in session snapshots. This establishes the learning
loop; it does not establish a policy's quality.

Trajectories retain each decision's named input, its selected action, reward, and native next
observation. Task termination is recorded even while the surrounding world continues. Native
IDM records its continuous applied control without inventing action probabilities. Reviewed
trajectories can supply decision-model training examples. Independent policy evaluation uses
episode mode; persistent-world recordings measure behavior in the continuing world.

To author a manual car map, generate a native MetaDrive `PGMap` and call
`env.current_map.get_meta_data()`. Serialize its `map_config` with
`get_serializable_dict()` and save the returned metadata as JSON. The importer reads the native
`block_sequence` and `map_config`, including lane dimensions and block socket configurations.
The native map's internal seed is excluded from constructor settings; its exact saved block
geometry defines the imported road.
