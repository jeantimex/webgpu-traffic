import { type Car, type IdmParams, type Obstacle } from '../sim/idm';
import { laneLoopGapAhead, laneLoopSpawnSlot, stepLaneLoop } from '../sim/loop';
import {
  buildNetwork,
  laneNode,
  lanePathPoint,
  networkGapAhead,
  networkSpawnSlot,
  stepNetwork,
  type NetObstacle,
  type Network,
} from '../sim/network';
import {
  buildIntersection,
  intersectionLampColor,
  intersectionObstacles,
  intersectionPhaseAt,
  leftTurnYieldObstacles,
  STOP_BACK,
  type IntersectionConfig,
  type IntersectionPhase,
  type IntersectionState,
} from '../sim/intersection';
import { Road, type RoadConfig } from '../sim/road';

export type Vec3 = [number, number, number];

/** Traffic-light timing + override from the GUI. */
export interface LightSettings {
  green: number;
  yellow: number;
  red: number;
  override: string;
}

export interface Pose {
  x: number;
  y: number;
  z: number;
  angle: number;
  pitch: number;
}

export interface Palette {
  sky: Vec3;
  ground: Vec3;
  asphalt: Vec3;
  wall: Vec3;
  ambient: number;
  diffuse: number;
}

export const PALETTES: Record<'day' | 'night', Palette> = {
  day: {
    sky: [0.53, 0.75, 0.95],
    ground: [0.32, 0.47, 0.25],
    asphalt: [0.38, 0.39, 0.41],
    wall: [0.3, 0.3, 0.32],
    ambient: 0.55,
    diffuse: 0.7,
  },
  night: {
    sky: [0.05, 0.06, 0.09],
    ground: [0.1, 0.12, 0.1],
    asphalt: [0.24, 0.25, 0.27],
    wall: [0.17, 0.17, 0.19],
    ambient: 0.35,
    diffuse: 0.65,
  },
};

/** Everything the renderer needs from a scene: layout, geometry, car placement, and topology logic. */
export interface SceneDef {
  readonly c: number; // loop circumference (m); total length for open networks
  lamps: { x: number; z: number }[]; // signal pole positions
  buildStatic(palette: Palette): number[];
  /** World position, heading angle (around +Y), and pitch for a car. */
  carPose(car: Car): Pose;
  /** Current signal phase id (honors the GUI override). */
  phaseAt(clock: number, light: LightSettings): string;
  /** Active stop-line obstacles for a phase. */
  obstaclesFor(phase: string): Obstacle[];
  /** What one lamp stack shows in a phase. */
  lampColor(lamp: number, phase: string): 'red' | 'yellow' | 'green';
  /** Advances the sim one step; obstacles are the active signal stop lines (may be empty). */
  step(cars: Car[], params: IdmParams[], carLength: number, dt: number, obstacles: Obstacle[]): void;
  /** Bumper gap to the nearest car ahead in the same lane, or null when alone. */
  leaderGap(cars: Car[], i: number, carLength: number): number | null;
  /** Best safe spawn slot, or null when the road is too full. */
  findSpawnSlot(
    cars: Car[],
    carParams: IdmParams[],
    params: IdmParams,
    carLength: number,
  ): { s: number; lane: number } | null;
}

function lightPhase(clock: number, green: number, yellow: number, red: number): string {
  const t = clock % (green + yellow + red);
  return t < green ? 'green' : t < green + yellow ? 'yellow' : 'red';
}

/** Signal behavior for the ring scenes: one light, all lamps show the same phase. */
function ringSignal(stops: Obstacle[]): Pick<SceneDef, 'phaseAt' | 'obstaclesFor' | 'lampColor'> {
  return {
    phaseAt: (clock, light) =>
      light.override === 'auto' ? lightPhase(clock, light.green, light.yellow, light.red) : light.override,
    obstaclesFor: (phase) => (phase === 'green' ? [] : stops),
    lampColor: (_lamp, phase) => phase as 'red' | 'yellow' | 'green',
  };
}

function laneLoopTopology(net: Network, c: number): Pick<SceneDef, 'step' | 'leaderGap' | 'findSpawnSlot'> {
  return {
    step: (cars, params, carLength, dt, obstacles) => stepLaneLoop(net, cars, params, c, carLength, dt, obstacles),
    leaderGap: (cars, i, carLength) => laneLoopGapAhead(cars, i, c, carLength),
    findSpawnSlot: (cars, carParams, params, carLength) =>
      laneLoopSpawnSlot(net, cars, carParams, params, c, carLength),
  };
}

/** Appends a quad (6 vertices, interleaved position/normal/color). Corners must be CCW seen from outside. */
export function pushQuad(out: number[], corners: [Vec3, Vec3, Vec3, Vec3], color: Vec3): void {
  const [a, b, c, d] = corners;
  const ux = b[0] - a[0];
  const uy = b[1] - a[1];
  const uz = b[2] - a[2];
  const vx = c[0] - a[0];
  const vy = c[1] - a[1];
  const vz = c[2] - a[2];
  const nx = uy * vz - uz * vy;
  const ny = uz * vx - ux * vz;
  const nz = ux * vy - uy * vx;
  const len = Math.hypot(nx, ny, nz);
  for (const p of [a, b, c, a, c, d]) {
    out.push(p[0], p[1], p[2], nx / len, ny / len, nz / len, color[0], color[1], color[2]);
  }
}

/** Appends a box (5 faces, bottom omitted: the camera stays above) to a vertex list. */
export function pushBox(out: number[], min: Vec3, max: Vec3, color: Vec3): void {
  const [x0, y0, z0] = min;
  const [x1, y1, z1] = max;
  pushQuad(out, [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]], color); // top
  pushQuad(out, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], color); // +x
  pushQuad(out, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], color); // −x
  pushQuad(out, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], color); // +z
  pushQuad(out, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], color); // −z
}

// ---------------------------------------------------------------------------
// Scene 1: the ring with a bridge.
// ---------------------------------------------------------------------------

const TRACK_RADIUS = 40;
const ROAD_HALF_WIDTH = 4;
const RING_C = 2 * Math.PI * TRACK_RADIUS;
/** Arc position of the stop line / pedestrian crossing (quarter lap, nearest the camera). */
const STOP_S = RING_C / 4;
/** The signal hangs just inside the inner road edge at the stop line. */
const RING_LAMP = {
  x: (TRACK_RADIUS - ROAD_HALF_WIDTH - 1.2) * Math.cos(STOP_S / TRACK_RADIUS),
  z: (TRACK_RADIUS - ROAD_HALF_WIDTH - 1.2) * Math.sin(STOP_S / TRACK_RADIUS),
};

function ringPathPoint(s: number): PathPoint {
  const theta = s / TRACK_RADIUS;
  const hx = -Math.sin(theta);
  const hz = Math.cos(theta);
  return {
    x: TRACK_RADIUS * Math.cos(theta),
    z: TRACK_RADIUS * Math.sin(theta),
    hx,
    hz,
    rx: -hz,
    rz: hx,
  };
}

export const scene1State: { net: Network } = {
  net: buildLoopNetwork(RING_C, ringPathPoint, [2, -2]),
};

/** The bridge is a raised-cosine bump on the far side of the ring (clear of the cars' start and the crossing). */
const BRIDGE_LENGTH = 60; // m along the arc
const BRIDGE_HEIGHT = 4; // m
const BRIDGE_START = (3 * RING_C) / 4 - BRIDGE_LENGTH / 2;

/** Road surface height above the ground at arc position s (0 off the bridge). */
function roadHeight(s: number): number {
  if (s < BRIDGE_START || s > BRIDGE_START + BRIDGE_LENGTH) return 0;
  const u = (s - BRIDGE_START) / BRIDGE_LENGTH;
  return (BRIDGE_HEIGHT / 2) * (1 - Math.cos(2 * Math.PI * u));
}

/** Slope (dh/ds) of the road at arc position s. */
function roadGrade(s: number): number {
  if (s < BRIDGE_START || s > BRIDGE_START + BRIDGE_LENGTH) return 0;
  const u = (s - BRIDGE_START) / BRIDGE_LENGTH;
  return ((BRIDGE_HEIGHT * Math.PI) / BRIDGE_LENGTH) * Math.sin(2 * Math.PI * u);
}

/** Appends a rectangular road patch covering arc [s0, s1] and radius [r0, r1]. */
function pushRoadPatch(out: number[], s0: number, s1: number, r0: number, r1: number, color: Vec3): void {
  // ponytail: a single straight quad per patch; over a few meters of arc it sags ~4 cm off the
  // circle, invisible at this zoom. Segment along the arc if patches get much longer.
  const at = (s: number, r: number): Vec3 => {
    const theta = s / TRACK_RADIUS;
    return [r * Math.cos(theta), 0.03, r * Math.sin(theta)];
  };
  pushQuad(out, [at(s0, r0), at(s1, r0), at(s1, r1), at(s0, r1)], color);
}

/** Appends a lane-divider dash centered between the two lanes, following the road height. */
function pushLaneDash(out: number[], s0: number, s1: number, color: Vec3): void {
  const r0 = TRACK_RADIUS - 0.075;
  const r1 = TRACK_RADIUS + 0.075;
  const at = (s: number, r: number): Vec3 => {
    const theta = s / TRACK_RADIUS;
    return [r * Math.cos(theta), 0.03 + roadHeight(s), r * Math.sin(theta)];
  };
  pushQuad(out, [at(s0, r0), at(s1, r0), at(s1, r1), at(s0, r1)], color);
}

/** Ground plane + ring road + crossing paint + light pole, vertex colors baked in. */
function ringStatic(palette: Palette): number[] {
  const verts: number[] = [];
  const G = 300;
  pushQuad(verts, [[-G, 0, -G], [-G, 0, G], [G, 0, G], [G, 0, -G]], palette.ground);

  const asphalt = palette.asphalt;
  const wall = palette.wall;
  const inner = TRACK_RADIUS - ROAD_HALF_WIDTH;
  const outer = TRACK_RADIUS + ROAD_HALF_WIDTH;
  const SEGMENTS = 128;
  for (let i = 0; i < SEGMENTS; i++) {
    const t0 = (i / SEGMENTS) * 2 * Math.PI;
    const t1 = ((i + 1) / SEGMENTS) * 2 * Math.PI;
    const h0 = 0.02 + roadHeight(t0 * TRACK_RADIUS);
    const h1 = 0.02 + roadHeight(t1 * TRACK_RADIUS);
    pushQuad(
      verts,
      [
        [inner * Math.cos(t0), h0, inner * Math.sin(t0)],
        [inner * Math.cos(t1), h1, inner * Math.sin(t1)],
        [outer * Math.cos(t1), h1, outer * Math.sin(t1)],
        [outer * Math.cos(t0), h0, outer * Math.sin(t0)],
      ],
      asphalt,
    );
    // Side walls under the elevated section so the bridge reads as solid.
    if (h0 > 0.05 || h1 > 0.05) {
      pushQuad(
        verts,
        [
          [outer * Math.cos(t0), 0, outer * Math.sin(t0)],
          [outer * Math.cos(t0), h0, outer * Math.sin(t0)],
          [outer * Math.cos(t1), h1, outer * Math.sin(t1)],
          [outer * Math.cos(t1), 0, outer * Math.sin(t1)],
        ],
        wall,
      );
      pushQuad(
        verts,
        [
          [inner * Math.cos(t0), h0, inner * Math.sin(t0)],
          [inner * Math.cos(t0), 0, inner * Math.sin(t0)],
          [inner * Math.cos(t1), 0, inner * Math.sin(t1)],
          [inner * Math.cos(t1), h1, inner * Math.sin(t1)],
        ],
        wall,
      );
    }
  }

  // Solid stop line across the lane, then a zebra crossing after it: stripes run
  // parallel to travel, packed across the lane width.
  const paint: Vec3 = [0.9, 0.9, 0.9];
  pushRoadPatch(verts, STOP_S - 0.125, STOP_S + 0.125, inner, outer, paint);
  for (let i = 0; i < 7; i++) {
    const r0 = inner + 0.6 + i * 1.0;
    pushRoadPatch(verts, STOP_S + 0.8, STOP_S + 4.3, r0, r0 + 0.5, paint);
  }

  // Dashed divider between the lanes, skipping the crossing.
  for (let s = 0; s < RING_C; s += 6) {
    if (s > STOP_S - 2 && s < STOP_S + 6) continue;
    pushLaneDash(verts, s, s + 2, paint);
  }

  // Traffic-light pole beside the road, tall enough for the three lamps.
  pushBox(
    verts,
    [RING_LAMP.x - 0.1, 0, RING_LAMP.z - 0.1],
    [RING_LAMP.x + 0.1, 4, RING_LAMP.z + 0.1],
    [0.4, 0.4, 0.42],
  );
  return verts;
}

function ringScene(): SceneDef {
  return {
    c: RING_C,
    lamps: [RING_LAMP],
    buildStatic: ringStatic,
    ...laneLoopTopology(scene1State.net, RING_C),
    ...ringSignal([{ s: STOP_S }]),
    carPose(car) {
      const fromGlobal = Math.round(car.laneFrom);
      const from = lanePathPoint(scene1State.net, fromGlobal, car.s);
      const to = lanePathPoint(scene1State.net, car.lane, car.s);
      const span = car.lane - car.laneFrom;
      const t = span === 0 ? 1 : (car.lateral - car.laneFrom) / span;
      const x = from.x + (to.x - from.x) * t;
      const z = from.z + (to.z - from.z) * t;
      // While sliding sideways, yaw the body along the actual velocity direction.
      const yaw = Math.atan2(4 * car.lateralVel, Math.max(car.v, 1));
      return {
        x,
        y: 0.02 + roadHeight(car.s),
        z,
        angle: Math.atan2(-to.hz, to.hx) + yaw,
        pitch: Math.atan(roadGrade(car.s)),
      };
    },
  };
}

// ---------------------------------------------------------------------------
// Scene 2: a square loop, clockwise. 4 straights, 4 rounded corners, and a
// stop line + zebra + traffic light at the end of each straight.
// ---------------------------------------------------------------------------

const SQ_STRAIGHT = 100; // m per side
const SQ_CORNER_R = 8; // m, centerline corner radius
const SQ_HALF = SQ_STRAIGHT / 2 + SQ_CORNER_R;
const SQ_SEG = SQ_STRAIGHT + (Math.PI / 2) * SQ_CORNER_R;
const SQ_C = 4 * SQ_SEG;
/** Stop line arc offset within each side: 5 m before the corner. */
const SQ_STOP_OFFSET = SQ_STRAIGHT - 5;

// Clockwise sides: straight start point, heading h, and right normal r (toward the loop center).
const SQ_SIDES = [
  { ex: -SQ_STRAIGHT / 2, ez: -SQ_HALF, hx: 1, hz: 0, rx: 0, rz: 1 },
  { ex: SQ_HALF, ez: -SQ_STRAIGHT / 2, hx: 0, hz: 1, rx: -1, rz: 0 },
  { ex: SQ_STRAIGHT / 2, ez: SQ_HALF, hx: -1, hz: 0, rx: 0, rz: -1 },
  { ex: -SQ_HALF, ez: SQ_STRAIGHT / 2, hx: 0, hz: -1, rx: 1, rz: 0 },
];

interface PathPoint {
  x: number;
  z: number;
  hx: number; // heading (unit)
  hz: number;
  rx: number; // right normal (unit, toward the turn's center)
  rz: number;
}

/** Centerline point + heading + right normal at arc position s on the square loop. */
function squarePathPoint(s: number): PathPoint {
  const wrapped = ((s % SQ_C) + SQ_C) % SQ_C;
  const seg = Math.min(Math.floor(wrapped / SQ_SEG), 3);
  const d = wrapped - seg * SQ_SEG;
  const side = SQ_SIDES[seg];
  if (d <= SQ_STRAIGHT) {
    return {
      x: side.ex + d * side.hx,
      z: side.ez + d * side.hz,
      hx: side.hx,
      hz: side.hz,
      rx: side.rx,
      rz: side.rz,
    };
  }
  // Rounded corner: quarter circle turning right, from heading h to heading r.
  const phi = (d - SQ_STRAIGHT) / SQ_CORNER_R;
  const ax = side.ex + SQ_STRAIGHT * side.hx;
  const az = side.ez + SQ_STRAIGHT * side.hz;
  const sin = Math.sin(phi);
  const cos = Math.cos(phi);
  return {
    x: ax + SQ_CORNER_R * (sin * side.hx + (1 - cos) * side.rx),
    z: az + SQ_CORNER_R * (sin * side.hz + (1 - cos) * side.rz),
    hx: cos * side.hx + sin * side.rx,
    hz: cos * side.hz + sin * side.rz,
    rx: -sin * side.hx + cos * side.rx,
    rz: -sin * side.hz + cos * side.rz,
  };
}

/** Appends a path patch covering arc [s0, s1] and lateral offsets [o0, o1] (o+ = along the right normal). */
function pushPathPatch(
  out: number[],
  path: (s: number) => { x: number; z: number; rx: number; rz: number },
  s0: number,
  s1: number,
  o0: number,
  o1: number,
  y: number,
  color: Vec3,
): void {
  const at = (s: number, o: number): Vec3 => {
    const p = path(s);
    return [p.x + p.rx * o, y, p.z + p.rz * o];
  };
  const [a0, b0, c0, d0] = [at(s0, o1), at(s1, o1), at(s1, o0), at(s0, o0)];
  // Road surfaces must face up; swap corners to flip the normal if the path's
  // normal convention would produce a downward face (culled otherwise).
  const e1x = b0[0] - a0[0];
  const e1z = b0[2] - a0[2];
  const e2x = c0[0] - a0[0];
  const e2z = c0[2] - a0[2];
  const ny = e1z * e2x - e1x * e2z;
  pushQuad(out, ny >= 0 ? [a0, b0, c0, d0] : [a0, d0, c0, b0], color);
}

const SQ_STOPS = [0, 1, 2, 3].map((k) => k * SQ_SEG + SQ_STOP_OFFSET);
/** Poles just inside the inner road edge at each stop line. */
const SQ_LAMPS = SQ_STOPS.map((s) => {
  const p = squarePathPoint(s);
  return {
    x: p.x + p.rx * (ROAD_HALF_WIDTH + 1.2),
    z: p.z + p.rz * (ROAD_HALF_WIDTH + 1.2),
  };
});

export const scene2State: { net: Network } = {
  net: buildLoopNetwork(SQ_C, squarePathPoint, [2, -2]),
};

function buildLoopNetwork(length: number, path: (s: number) => PathPoint, offsets: [number, number]): Network {
  const road = new Road({
    shape: 'straight',
    length,
    radius: 50,
    angle: 90,
    lanesForward: 2,
    lanesBackward: 0,
    customPath: path,
  });
  road.lanes[0].offset = offsets[0];
  road.lanes[1].offset = offsets[1];
  return buildNetwork([road], [[0, 0]]);
}

function squareStatic(palette: Palette): number[] {
  const verts: number[] = [];
  const G = 300;
  pushQuad(verts, [[-G, 0, -G], [-G, 0, G], [G, 0, G], [G, 0, -G]], palette.ground);

  // Road ribbon, 1 m patches so the corner arcs are smooth.
  for (let s = 0; s < SQ_C; s += 1) {
    pushPathPatch(verts, squarePathPoint, s, Math.min(s + 1, SQ_C), -ROAD_HALF_WIDTH, ROAD_HALF_WIDTH, 0.02, palette.asphalt);
  }

  const paint: Vec3 = [0.9, 0.9, 0.9];
  // Stop lines + zebra crossings (stripes parallel to travel) at every intersection.
  for (const stop of SQ_STOPS) {
    pushPathPatch(verts, squarePathPoint, stop - 0.125, stop + 0.125, -ROAD_HALF_WIDTH, ROAD_HALF_WIDTH, 0.03, paint);
    for (let i = 0; i < 7; i++) {
      const o0 = -ROAD_HALF_WIDTH + 0.6 + i * 1.0;
      pushPathPatch(verts, squarePathPoint, stop + 0.8, stop + 4.3, o0, o0 + 0.5, 0.03, paint);
    }
  }

  // Dashed lane divider, skipping the crossings.
  for (let s = 0; s < SQ_C; s += 6) {
    if (SQ_STOPS.some((stop) => s > stop - 2 && s < stop + 6)) continue;
    pushPathPatch(verts, squarePathPoint, s, s + 2, -0.075, 0.075, 0.03, paint);
  }

  // Signal poles.
  for (const lamp of SQ_LAMPS) {
    pushBox(verts, [lamp.x - 0.1, 0, lamp.z - 0.1], [lamp.x + 0.1, 4, lamp.z + 0.1], [0.4, 0.4, 0.42]);
  }
  return verts;
}

function squareScene(): SceneDef {
  return {
    c: SQ_C,
    lamps: SQ_LAMPS,
    buildStatic: squareStatic,
    ...laneLoopTopology(scene2State.net, SQ_C),
    ...ringSignal(SQ_STOPS.map((s) => ({ s }))),
    carPose(car) {
      const fromGlobal = Math.round(car.laneFrom);
      const from = lanePathPoint(scene2State.net, fromGlobal, car.s);
      const to = lanePathPoint(scene2State.net, car.lane, car.s);
      const span = car.lane - car.laneFrom;
      const t = span === 0 ? 1 : (car.lateral - car.laneFrom) / span;
      const x = from.x + (to.x - from.x) * t;
      const z = from.z + (to.z - from.z) * t;
      const yaw = Math.atan2(4 * car.lateralVel, Math.max(car.v, 1));
      return {
        x,
        y: 0.02,
        z,
        angle: Math.atan2(-to.hz, to.hx) + yaw,
        pitch: 0,
      };
    },
  };
}

// ---------------------------------------------------------------------------
// Scene 3: two roads connected end-to-start (the network building block demo).
// Road A keeps its own frame; road B is rotated + translated to start exactly at
// A's end with heading continuity.
// ---------------------------------------------------------------------------

/** Rotation (about +Y) + translation placing a road's local path into the world. */
export interface Transform {
  tx: number;
  tz: number;
  cos: number;
  sin: number;
}

function applyTransform(p: PathPoint, t: Transform): PathPoint {
  return {
    x: p.x * t.cos + p.z * t.sin + t.tx,
    z: -p.x * t.sin + p.z * t.cos + t.tz,
    hx: p.hx * t.cos + p.hz * t.sin,
    hz: -p.hx * t.sin + p.hz * t.cos,
    rx: p.rx * t.cos + p.rz * t.sin,
    rz: -p.rx * t.sin + p.rz * t.cos,
  };
}

/** The active scene-3 network, rebuilt by the renderer on scene switch or config change. */
export const scene3State: { net: Network | null; transforms: Transform[]; handed: number } = {
  net: null,
  transforms: [],
  handed: 1,
};

function requireNet(): Network {
  if (!scene3State.net) throw new Error('scene 3 network not built yet');
  return scene3State.net;
}

/** Builds the two-road network and places road B at road A's end. */
export function buildScene3(cfgA: RoadConfig, cfgB: RoadConfig, handed = 1): void {
  const roadA = new Road(cfgA, handed);
  const roadB = new Road(cfgB, handed);
  const net = buildNetwork([roadA, roadB], [[0, 1]]);
  const tA: Transform = { tx: 0, tz: 0, cos: 1, sin: 0 };
  // Rotate B so its start heading matches A's end heading, then translate B's start to A's end.
  // rotY(phi) rotates the xz-plane angle by -phi, so phi = b0Angle - endAngle.
  const end = roadA.point(roadA.length);
  const b0 = roadB.point(0);
  const phi = Math.atan2(b0.hz, b0.hx) - Math.atan2(end.hz, end.hx);
  const cos = Math.cos(phi);
  const sin = Math.sin(phi);
  const tB: Transform = {
    tx: end.x - (b0.x * cos + b0.z * sin),
    tz: end.z - (-b0.x * sin + b0.z * cos),
    cos,
    sin,
  };
  scene3State.net = net;
  scene3State.transforms = [tA, tB];
  scene3State.handed = handed;
}

function buildRoadStatic(road: Road, palette: Palette, t: Transform): number[] {
  const verts: number[] = [];
  const path = (s: number): PathPoint => applyTransform(road.point(s), t);
  const lanePath = (lane: number) => (s: number): PathPoint => applyTransform(road.lanePoint(lane, s), t);
  const f = road.config.lanesForward;
  const b = road.config.lanesBackward;
  // A single centered lane (turn arcs): narrow ribbon, no center line, no dashes.
  const single = road.lanes.length === 1 && road.lanes[0].offset === 0;
  const L = road.length;

  // Lane ribbons in 1 m patches (smooth on arcs). For current road-offset lanes this
  // exactly tiles the old full road ribbon while allowing future lanes to own shape.
  if (single) {
    for (let s = 0; s < L; s += 1) {
      pushPathPatch(verts, lanePath(0), s, Math.min(s + 1, L), -2, 2, 0.02, palette.asphalt);
    }
  } else {
    road.lanes.forEach((_, lane) => {
      for (let s = 0; s < L; s += 1) {
        pushPathPatch(verts, lanePath(lane), s, Math.min(s + 1, L), -2, 2, 0.02, palette.asphalt);
      }
    });
  }

  // Solid yellow line separating the directions (the left edge on a one-way road).
  if (!single) {
    const yellow: Vec3 = [0.8, 0.65, 0.1];
    for (let s = 0; s < L; s += 2) {
      pushPathPatch(verts, path, s, Math.min(s + 2, L), -0.075, 0.075, 0.03, yellow);
    }
  }

  // White dashes between same-direction lanes.
  const paint: Vec3 = [0.9, 0.9, 0.9];
  const dashOffsets: number[] = [];
  if (!single) {
    for (let i = 1; i < f; i++) dashOffsets.push(4 * i);
    for (let j = 1; j < b; j++) dashOffsets.push(-4 * j);
  }
  for (const o of dashOffsets) {
    for (let s = 0; s < L; s += 6) {
      pushPathPatch(verts, path, s, Math.min(s + 2, L), o - 0.075, o + 0.075, 0.03, paint);
    }
  }
  return verts;
}

function networkCarPose(net: Network, transforms: Transform[], car: Car, handed: number): Pose {
  const fromGlobal = Math.round(car.laneFrom);
  const to = laneNode(net, car.lane);
  const from = laneNode(net, fromGlobal);
  const span = car.lane - car.laneFrom;
  const t = span === 0 ? 1 : (car.lateral - car.laneFrom) / span;
  const offsetVel = span === 0 ? 0 : ((to.offset - from.offset) * car.lateralVel) / span;
  const fromPoint = applyTransform(lanePathPoint(net, fromGlobal, car.s), transforms[from.road]);
  const toPoint = applyTransform(lanePathPoint(net, car.lane, car.s), transforms[to.road]);
  const x = fromPoint.x + (toPoint.x - fromPoint.x) * t;
  const z = fromPoint.z + (toPoint.z - fromPoint.z) * t;
  const dir = to.direction;
  // Nose along the true velocity: travel direction plus the lateral slide (the body's
  // local +z is the right normal; handedness flips which side the lanes sit on).
  const yaw = Math.atan2(-offsetVel * dir * handed, Math.max(car.v, 1));
  return {
    x,
    y: 0.02,
    z,
    angle: Math.atan2(-toPoint.hz, toPoint.hx) + (dir < 0 ? Math.PI : 0) + yaw,
    pitch: 0,
  };
}

function roadScene(): SceneDef {
  return {
    get c() {
      return requireNet().roads.reduce((sum, road) => sum + road.length, 0);
    },
    lamps: [],
    ...ringSignal([]),
    buildStatic: (palette) => {
      const net = requireNet();
      const groundVerts: number[] = [];
      const G = 300;
      pushQuad(groundVerts, [[-G, 0, -G], [-G, 0, G], [G, 0, G], [G, 0, -G]], palette.ground);
      return net.roads
        .flatMap((road, r) => buildRoadStatic(road, palette, scene3State.transforms[r]))
        .concat(groundVerts);
    },
    carPose: (car) => networkCarPose(requireNet(), scene3State.transforms, car, scene3State.handed),
    step: (cars, params, carLength, dt) => stepNetwork(requireNet(), cars, params, carLength, dt),
    leaderGap: (cars, i, carLength) => networkGapAhead(requireNet(), cars, i, carLength),
    findSpawnSlot: (cars, _carParams, params, carLength) =>
      networkSpawnSlot(requireNet(), cars, params, carLength),
  };
}

// ---------------------------------------------------------------------------
// Scene 4: a 4-way signalized intersection (straight-through only). Approach
// roads meet two connectors inside the zone; right-of-way is the 2-phase signal
// from sim/intersection.ts.
// ---------------------------------------------------------------------------

export const scene4State: {
  state: IntersectionState | null;
  transforms: Transform[];
  lamps: { x: number; z: number; ns: boolean }[];
  handed: number;
  closed: IntersectionConfig['closed'];
} = {
  state: null,
  transforms: [],
  lamps: [],
  handed: 1,
  closed: { n: 'open', e: 'open', s: 'open', w: 'open' },
};

function requireScene4(): IntersectionState {
  if (!scene4State.state) throw new Error('scene 4 intersection not built yet');
  return scene4State.state;
}

/** Rotates/translates a road so road.point(atS) lands at the target pose. */
function placeAt(road: Road, atS: number, target: { x: number; z: number; hx: number; hz: number }): Transform {
  const p = road.point(atS);
  // rotY(phi) rotates the xz-plane angle by -phi, so phi = sourceAngle - targetAngle.
  const phi = Math.atan2(p.hz, p.hx) - Math.atan2(target.hz, target.hx);
  const cos = Math.cos(phi);
  const sin = Math.sin(phi);
  return {
    tx: target.x - (p.x * cos + p.z * sin),
    tz: target.z - (-p.x * sin + p.z * cos),
    cos,
    sin,
  };
}

/** Builds the intersection network and places every existing road around the zone. */
export function buildScene4(cfg: IntersectionConfig, handed = 1): void {
  const state = buildIntersection(cfg, handed);
  const zh = state.zoneHalf;
  const L = cfg.approach;
  const edge = 4 * cfg.lanesEachWay + 1.2; // lamp offset from the approach centerline
  const ri = state.roadIndex;
  const has = (key: string): boolean => ri[key] !== undefined;
  const canEnter = (way: 'n' | 'e' | 's' | 'w'): boolean =>
    cfg.closed[way] === 'open' || cfg.closed[way] === 'out';
  const transforms: Transform[] = [];
  const put = (key: string, road: Road, atS: number, target: { x: number; z: number; hx: number; hz: number }): void => {
    if (has(key)) transforms[ri[key]] = placeAt(road, atS, target);
  };
  put('s', state.net.roads[ri.s], L, { x: 0, z: -zh, hx: 0, hz: 1 });
  put('n', state.net.roads[ri.n], 0, { x: 0, z: zh, hx: 0, hz: 1 });
  put('e', state.net.roads[ri.e], 0, { x: zh, z: 0, hx: 1, hz: 0 });
  put('w', state.net.roads[ri.w], L, { x: -zh, z: 0, hx: 1, hz: 0 });
  put('nsConn', state.net.roads[ri.nsConn], 0, { x: 0, z: -zh, hx: 0, hz: 1 });
  put('ewConn', state.net.roads[ri.ewConn], 0, { x: -zh, z: 0, hx: 1, hz: 0 });
  // Turn arcs are pinned at their authored entry lane positions. Forced L-corner
  // turns may include extra lane-specific arcs beyond the normal dedicated lanes.
  state.turns.forEach((turn) => put(turn.key, state.net.roads[ri[turn.key]], 0, turn.spec.entry));

  // Lamp stacks on the right side of each entering approach, just before its stop line.
  scene4State.lamps = [
    ...(canEnter('s') ? [{ x: -edge * handed, z: -zh - 1, ns: true }] : []),
    ...(canEnter('n') ? [{ x: edge * handed, z: zh + 1, ns: true }] : []),
    ...(canEnter('e') ? [{ x: zh + 1, z: -edge * handed, ns: false }] : []),
    ...(canEnter('w') ? [{ x: -zh - 1, z: edge * handed, ns: false }] : []),
  ];
  scene4State.state = state;
  scene4State.transforms = transforms;
  scene4State.handed = handed;
  scene4State.closed = cfg.closed;
}

function buildIntersectionStatic(palette: Palette): number[] {
  const state = requireScene4();
  const net = state.net;
  const zh = state.zoneHalf;
  const verts: number[] = [];
  const G = 300;
  pushQuad(verts, [[-G, 0, -G], [-G, 0, G], [G, 0, G], [G, 0, -G]], palette.ground);

  // The zone box sits 1 cm under everything; all existing roads draw ribbons on top.
  pushQuad(verts, [[-zh, 0.01, -zh], [-zh, 0.01, zh], [zh, 0.01, zh], [zh, 0.01, -zh]], palette.asphalt);
  net.roads.forEach((road, r) => {
    verts.push(...buildRoadStatic(road, palette, scene4State.transforms[r]));
  });

  // Stop lines + zebra crossings for ways that may enter the intersection.
  const paint: Vec3 = [0.9, 0.9, 0.9];
  const paint_patch = (
    road: Road,
    t: Transform,
    s0: number,
    s1: number,
    o0: number,
    o1: number,
    y = 0.03,
  ): void => {
    pushPathPatch(verts, (s) => applyTransform(road.point(s), t), s0, s1, o0, o1, y, paint);
  };
  const handed = scene4State.handed;
  const stopSide = (a: number, b: number): [number, number] => (handed > 0 ? [a, b] : [-b, -a]);
  const laneSpan = (o0: number, o1: number): [number, number] =>
    o0 * handed < o1 * handed ? [o0 * handed, o1 * handed] : [o1 * handed, o0 * handed];
  const paintApproach = (way: 'n' | 'e' | 's' | 'w'): void => {
    const road = state.net.roads[state.roadIndex[way]];
    const t = scene4State.transforms[state.roadIndex[way]];
    const L = road.length;
    const lanes = road.config.lanesForward;
    const oMin = -4 * lanes;
    const oMax = 4 * lanes;
    const enteringFwd = way === 's' || way === 'w';
    const stopS = enteringFwd ? L - STOP_BACK : STOP_BACK;
    const wayState = scene4State.closed[way];

    // Closed lanes are tinted light red (crosswalk stays): entering side for
    // entry-closed ways, exiting side for exit-closed ways.
    const closedSpan =
      wayState === 'in'
        ? enteringFwd
          ? laneSpan(0, oMax)
          : laneSpan(oMin, 0)
        : wayState === 'out'
          ? enteringFwd
            ? laneSpan(oMin, 0)
            : laneSpan(0, oMax)
          : null;
    if (closedSpan) {
      pushPathPatch(verts, (s) => applyTransform(road.point(s), t), 0, L, closedSpan[0], closedSpan[1], 0.025, [0.72, 0.38, 0.38]);
    }

    // Stop line only where traffic may enter — right lanes for RHT, left for LHT.
    if (wayState === 'open' || wayState === 'out') {
      const [a, b] = enteringFwd ? stopSide(0, oMax) : stopSide(oMin, 0);
      paint_patch(road, t, stopS - 0.125, stopS + 0.125, a, b);
    }
    // Scene-1 layout: a 3.5 m zebra band across the full road width after the line,
    // stripe count derived from the road width.
    const [z0, z1] = enteringFwd ? [stopS + 0.8, stopS + 4.3] : [STOP_BACK - 4.3, STOP_BACK - 0.8];
    for (let o0 = oMin + 0.4; o0 + 0.5 <= oMax - 0.4 + 1e-6; o0 += 1.0) {
      paint_patch(road, t, z0, z1, o0, o0 + 0.5);
    }
  };
  (['s', 'n', 'e', 'w'] as const).forEach((way) => {
    if (state.roadIndex[way] !== undefined) paintApproach(way);
  });
  return verts;
}

export function buildScene4TurnCurveOverlay(cars: Car[]): number[] {
  const state = requireScene4();
  const turnRoads = new Set(state.turns.map((turn) => state.roadIndex[turn.key]));
  const active = new Set<number>();
  cars.forEach((car) => {
    const road = laneNode(state.net, car.lane).road;
    if (turnRoads.has(road)) active.add(road);
  });
  const verts: number[] = [];
  const color: Vec3 = [0.05, 0.95, 1.0];
  active.forEach((roadIndex) => {
    const road = state.net.roads[roadIndex];
    const t = scene4State.transforms[roadIndex];
    const path = (s: number): PathPoint => applyTransform(road.lanePoint(0, s), t);
    for (let s = 0; s < road.length; s += 1) {
      pushPathPatch(verts, path, s, Math.min(s + 1, road.length), -0.18, 0.18, 0.08, color);
    }
  });
  return verts;
}

function intersectionScene(): SceneDef {
  return {
    get c() {
      return requireScene4().net.roads.reduce((sum, road) => sum + road.length, 0);
    },
    get lamps() {
      return scene4State.lamps;
    },
    buildStatic: buildIntersectionStatic,
    carPose: (car) => networkCarPose(requireScene4().net, scene4State.transforms, car, scene4State.handed),
    phaseAt: (clock, light) => intersectionPhaseAt(clock, light),
    obstaclesFor: (phase) => intersectionObstacles(requireScene4(), phase as IntersectionPhase),
    lampColor: (lamp, phase) =>
      intersectionLampColor(requireScene4(), scene4State.lamps[lamp].ns ? 0 : 2, phase as IntersectionPhase),
    step: (cars, params, carLength, dt, obstacles) => {
      const state = requireScene4();
      const yieldObstacles = leftTurnYieldObstacles(state, cars);
      stepNetwork(state.net, cars, params, carLength, dt, [
        ...(obstacles as NetObstacle[]),
        ...yieldObstacles,
      ]);
    },
    leaderGap: (cars, i, carLength) => networkGapAhead(requireScene4().net, cars, i, carLength),
    findSpawnSlot: (cars, _carParams, params, carLength) =>
      networkSpawnSlot(requireScene4().net, cars, params, carLength),
  };
}

export const SCENES: SceneDef[] = [ringScene(), squareScene(), roadScene(), intersectionScene()];
