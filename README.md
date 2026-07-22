# Traffic

A Vite + TypeScript WebGPU traffic simulator using microscopic vehicle behavior.

The current simulator supports IDM car following, MOBIL-style lane changes,
traffic lights, connected road networks, configurable road geometry, and a
signalized four-way intersection with turns and approach closures.

The important architectural point: the existing simulation has been migrated to a
lane-backed graph model. Vehicles still carry a scalar position `s` along their
current lane, but topology is now expressed at lane level through `LaneNode` and
`LaneConnection` instead of hardcoded road-level special cases.

## Run It

```sh
npm install
npm run dev
```

Open the local URL printed by Vite in a current browser with WebGPU support, such
as Chrome or Edge.

## Commands

- `npm run dev` starts the Vite development server.
- `npm run build` type-checks and creates a production build in `dist/`.
- `npm run check:sim` runs the simulator regression checks in `src/sim/idm.check.ts`.
- `npm run check` runs both the production build and simulator checks.
- `npm run preview` serves the production build locally.

Run `npm run check` before and after topology or simulation changes. It is the
main regression gate for the migration.

## Scenes

- Scene 1: circular two-lane loop with a bridge and signal.
- Scene 2: square loop with rounded corners and traffic lights.
- Scene 3: configurable connected roads.
- Scene 4: configurable signalized intersection with closures and turns.

Scenes 1 and 2 are closed loops, but they still use a lane-backed `Network`.
Their loop-specific stepping lives in `src/sim/loop.ts`. Scenes 3 and 4 use the
open-road network stepper in `src/sim/network.ts`.

## Source Map

- `src/sim/idm.ts`
  Shared IDM primitives, vehicle state types, obstacle type, and lateral
  lane-change easing.

- `src/sim/road.ts`
  Road and lane geometry. `Road` is a container for one or more `Lane` objects.
  A `Lane` has direction, lateral offset, explicit lateral neighbors, and a
  lane-center `point(s)` accessor. `RoadConfig.customPath` can provide custom
  centerline geometry.

- `src/sim/network.ts`
  Lane-level graph and open-road simulation. `Network.lanes` contains derived
  `LaneNode` objects keyed by global lane id. Route choices are represented by
  `LaneConnection` objects.

- `src/sim/loop.ts`
  Closed-loop lane-backed stepping, leader-gap lookup, and spawn-slot selection
  for loop scenes.

- `src/sim/intersection.ts`
  Four-way intersection builder. It authors roads, internal turn connector
  lanes, route tables, signal stop lines, closures, and left-turn yield
  obstacles.

- `src/renderer/scenes.ts`
  Scene definitions, scene-specific geometry setup, car pose mapping, and scene
  step/spawn hooks.

- `src/renderer/renderer.ts`
  WebGPU rendering and simulation loop integration.

- `src/gui/settings_gui.ts`
  `lil-gui` controls for scene, road, car, signal, and intersection parameters.

- `src/sim/idm.check.ts`
  Executable regression checks covering IDM behavior, lane geometry, network
  seams, intersections, turn geometry, closures, and lane commit behavior.

## Lane Graph Model

Use `Lane` as the first-class topology unit.

`Road` is a high-level geometric/grouping container:

- Owns the road length and centerline path.
- Owns a list of local `Lane` objects.
- Provides `road.point(s)` for the road centerline.
- Provides `road.lanePoint(lane, s)` for lane center geometry.

`Lane` is the simulation path:

- Has a local lane index inside its road.
- Has `direction` (`1` forward, `-1` backward).
- Has a lateral `offset`.
- Has explicit `leftNeighbor` and `rightNeighbor` links by local lane index.
- Has `point(s)` for the lane centerline.

`Network` is the lane graph:

- Uses global lane ids: `global = net.laneOffsets[road] + localLane`.
- Exposes `net.lanes: LaneNode[]`.
- Uses route-indexed connection tables through helper functions.
- Keeps `net.exit` as the backing route table for compatibility, but new code
  should prefer lane helpers.

Important helpers in `src/sim/network.ts`:

- `laneNode(net, global)`
- `laneConnectionFor(net, global, route)`
- `laneRouteConnections(net, global)`
- `setLaneRouteConnections(net, global, conns)`
- `availableRouteIndices(net, global)`
- `connectionTargetLane(net, conn)`
- `lateralNeighbors(net, global)`
- `lanePathPoint(net, global, s)`
- `rebuildLaneGraph(net)`

`LaneConnection.priority` currently supports:

- `PriorityType.DIRECT`
- `PriorityType.MERGE`
- `PriorityType.DIVERGE`
- `PriorityType.YIELD`

The priority metadata is present so merge/diverge/yield behavior can be layered
onto the topology. The current mature behavior is direct continuation, signal
stops, intersection turns, and left-turn yield obstacles.

## Adding Complex Roads

Prefer representing complexity as lane graph data, not scene-specific conditionals.

For a new complex road type:

1. Create one or more `Road` objects.
2. Give each road the needed lane count, direction, offsets, and geometry.
3. Use `customPath` when the road centerline is not one of the built-in shapes.
4. Adjust lane offsets or lane `point(s)` accessors only when lane geometry must
   differ from the road centerline plus offset convention.
5. Build route tables with `LaneConnection` objects.
6. Set explicit `leftNeighbor` and `rightNeighbor` only where lane changes should
   be allowed.
7. Call `rebuildLaneGraph(net)` after mutating route tables or lane topology.
8. Add focused checks in `src/sim/idm.check.ts`.

Examples:

- Lane drop: connect two upstream lanes into one downstream lane, with the dropped
  lane using `PriorityType.MERGE` or `PriorityType.YIELD`.
- Diverge or exit ramp: give a lane multiple route-indexed outgoing connections.
- Left-side merge: connect the ramp lane to the target left lane and set only the
  lateral neighbors that are legal.
- Solid-line separation: omit the lateral neighbor link between adjacent lanes.
- Dedicated lane: keep it as a normal lane but restrict route/lateral connections.

## Migration Status

The lane-backed migration is complete for the existing feature set.

Completed:

- Existing scenes run through lane-backed topology.
- Scene 1 and Scene 2 loops use `Network` plus `src/sim/loop.ts`.
- Scene 3 and Scene 4 use lane graph helpers for routing and simulation.
- Car pose mapping uses lane geometry where vehicle position depends on lane
  centerlines.
- The old `stepRing` simulation path has been removed.
- Regression checks pass through `npm run check`.

Intentional remaining road-level usage:

- Static road placement, road markings, and intersection overlay drawing may still
  use `road.point(s)`.
- That is acceptable because those calls describe visual road centerlines and
  layout, not vehicle topology or lane-level simulation behavior.

## Development Rules For Future Work

- Do not reintroduce road-level branching for traffic behavior when lane graph
  connections can express the same thing.
- Prefer `laneNode`, `laneConnectionFor`, `lanePathPoint`, and related helpers over
  direct `net.exit` reads outside graph-building code.
- Keep `Road` responsible for geometry and grouping; keep `Network` responsible
  for topology and simulation.
- Preserve the existing scenes when adding new road types. Add regression checks
  before making broad topology changes.
- After mutating lane connections or neighbor links, call `rebuildLaneGraph(net)`.
- Before handing off a change, run `npm run check`.

## Current Limits

The graph structure is ready for more complex roads, but some behavior layers are
still basic:

- Merge and yield priority metadata exists, but general right-of-way handling for
  arbitrary merge conflict zones is not fully implemented.
- Route planning is still route-index based, not a full graph search over long
  paths.
- Junctions are authored procedurally in `intersection.ts`; there is no visual
  road-network editor yet.
- Spawn logic is scene-specific and conservative.

These are natural next steps now that the topology is lane-backed.
