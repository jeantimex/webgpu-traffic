# Traffic

A Vite + TypeScript WebGPU traffic simulator using microscopic car-following behavior.

The simulation includes IDM car following, MOBIL-style lane changes, traffic lights,
open road networks, and a configurable four-way intersection. The renderer draws the
current scene with WebGPU, while `lil-gui` exposes controls for scene selection, road
shape, vehicle parameters, signal timing, and intersection closures.

## Run It

```sh
npm install
npm run dev
```

Open the local URL printed by Vite in a current browser with WebGPU support, such as
Chrome or Edge.

## Commands

- `npm run dev` starts the Vite development server.
- `npm run build` type-checks and creates a production build in `dist/`.
- `npm run check:sim` runs the simulator regression checks in `src/sim/idm.check.ts`.
- `npm run check` runs both the production build and simulator checks.
- `npm run preview` serves the production build locally.

## Scenes

- Scene 1: circular two-lane ring with a bridge and signal.
- Scene 2: square loop with rounded corners and traffic lights.
- Scene 3: configurable connected roads.
- Scene 4: configurable signalized intersection with closures and turns.

## Architecture Notes

The project is being migrated toward lane-level graph modeling:

- `Road` remains the high-level geometric container.
- `Lane` now carries explicit lateral neighbor links and lane-center geometry access.
- `Network` exposes a derived `LaneNode[]` graph using global lane ids.
- `LaneConnection` carries priority metadata for future `DIRECT`, `MERGE`, `DIVERGE`,
  and `YIELD` behavior.
- Existing routing still uses `Network.exit`, so the migration remains compatible with
  the current scenes.

The current lane geometry reproduces the existing road-centerline plus lane-offset
convention. Future work can migrate rendering and more complex roads to consume lane
geometry directly.
