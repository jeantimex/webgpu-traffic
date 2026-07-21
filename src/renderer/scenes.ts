import {
  B_SAFE,
  idmAcceleration,
  stepRing,
  type Car,
  type IdmParams,
  type Obstacle,
} from '../sim/idm';
import {
  buildNetwork,
  locate,
  networkGapAhead,
  networkSpawnSlot,
  stepNetwork,
  type Network,
} from '../sim/network';
import { Road, type RoadConfig } from '../sim/road';

export type Vec3 = [number, number, number];

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
  readonly c: number; // loop circumference (m); road length for the open-road scene
  obstacles: Obstacle[]; // stop lines, active while the light is not green
  lamps: { x: number; z: number }[]; // signal pole positions
  buildStatic(palette: Palette): number[];
  /** World position, heading angle (around +Y), and pitch for a car. */
  carPose(car: Car): Pose;
  /** Advances the sim one step; obstacles are the red-light stop lines (may be empty). */
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

/**
 * Topology logic for closed loops (ring and square): same-direction lanes 0/1,
 * wrap-around leader search, and safe-spawn evaluation over the whole loop.
 */
function ringTopology(c: number): Pick<SceneDef, 'step' | 'leaderGap' | 'findSpawnSlot'> {
  return {
    step: (cars, params, carLength, dt, obstacles) => stepRing(cars, params, c, carLength, dt, obstacles),
    leaderGap: (cars, i, carLength) => {
      const car = cars[i];
      let gap = Infinity;
      for (let j = 0; j < cars.length; j++) {
        if (j === i || cars[j].lane !== car.lane) continue;
        gap = Math.min(gap, (((cars[j].s - car.s) % c) + c) % c);
      }
      return Number.isFinite(gap) ? gap - carLength : null;
    },
    findSpawnSlot: (cars, carParams, params, carLength) => {
      let best: { s: number; lane: number } | null = null;
      let bestScore = -Infinity;
      for (let lane = 0; lane <= 1; lane++) {
        for (let k = 0; k < 32; k++) {
          const s = (k * c) / 32;
          let leaderGap = Infinity;
          let leaderV = params.v0;
          let followerGap = Infinity;
          let followerV = params.v0;
          let followerParams: IdmParams | null = null;
          cars.forEach((car, j) => {
            if (car.lane !== lane && car.laneProgress >= 1) return;
            const fwd = (((car.s - s) % c) + c) % c;
            if (fwd < leaderGap) {
              leaderGap = fwd;
              leaderV = car.v;
            }
            const back = (c - fwd) % c;
            if (back < followerGap) {
              followerGap = back;
              followerV = car.v;
              followerParams = carParams[j];
            }
          });
          if (idmAcceleration(params.v0, leaderGap - carLength, params.v0 - leaderV, params) < -B_SAFE)
            continue;
          if (
            followerParams !== null &&
            idmAcceleration(followerV, followerGap - carLength, followerV - params.v0, followerParams) <
              -B_SAFE
          )
            continue;
          const score = Math.min(leaderGap, followerGap);
          if (score > bestScore) {
            bestScore = score;
            best = { s, lane };
          }
        }
      }
      return best;
    },
  };
}

function ringScene(): SceneDef {
  return {
    c: RING_C,
    obstacles: [{ s: STOP_S }],
    lamps: [RING_LAMP],
    buildStatic: ringStatic,
    ...ringTopology(RING_C),
    carPose(car) {
      const theta = car.s / TRACK_RADIUS;
      // Lane centers are 2 m either side of the track radius; lateral eases between them.
      const r = TRACK_RADIUS - 2 + 4 * car.lateral;
      // While sliding sideways, yaw the body along the actual velocity direction.
      const yaw = Math.atan2(4 * car.lateralVel, Math.max(car.v, 1));
      return {
        x: r * Math.cos(theta),
        y: 0.02 + roadHeight(car.s),
        z: r * Math.sin(theta),
        angle: -theta - Math.PI / 2 + yaw,
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
  pushQuad(out, [at(s0, o1), at(s1, o1), at(s1, o0), at(s0, o0)], color);
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
    obstacles: SQ_STOPS.map((s) => ({ s })),
    lamps: SQ_LAMPS,
    buildStatic: squareStatic,
    ...ringTopology(SQ_C),
    carPose(car) {
      const p = squarePathPoint(car.s);
      // Lane centers are 2 m either side of the centerline (o+ = inner lane, toward the center).
      const o = 2 * (1 - 2 * car.lateral);
      const yaw = Math.atan2(4 * car.lateralVel, Math.max(car.v, 1));
      return {
        x: p.x + p.rx * o,
        y: 0.02,
        z: p.z + p.rz * o,
        angle: Math.atan2(-p.hz, p.hx) + yaw,
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
export const scene3State: { net: Network | null; transforms: Transform[] } = {
  net: null,
  transforms: [],
};

function requireNet(): Network {
  if (!scene3State.net) throw new Error('scene 3 network not built yet');
  return scene3State.net;
}

/** Builds the two-road network and places road B at road A's end. */
export function buildScene3(cfgA: RoadConfig, cfgB: RoadConfig): void {
  const roadA = new Road(cfgA);
  const roadB = new Road(cfgB);
  const net = buildNetwork([roadA, roadB], [[0, 1]]);
  const tA: Transform = { tx: 0, tz: 0, cos: 1, sin: 0 };
  // Rotate B so its start heading matches A's end heading, then translate B's start to A's end.
  const end = roadA.point(roadA.length);
  const b0 = roadB.point(0);
  const phi = Math.atan2(end.hz, end.hx) - Math.atan2(b0.hz, b0.hx);
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
}

function buildRoadStatic(road: Road, palette: Palette, t: Transform): number[] {
  const verts: number[] = [];
  const path = (s: number): PathPoint => applyTransform(road.point(s), t);
  const f = road.config.lanesForward;
  const b = road.config.lanesBackward;
  const oMin = -4 * b;
  const oMax = 4 * f;
  const L = road.length;

  // Road ribbon in 1 m patches (smooth on arcs).
  for (let s = 0; s < L; s += 1) {
    pushPathPatch(verts, path, s, Math.min(s + 1, L), oMin, oMax, 0.02, palette.asphalt);
  }

  // Solid yellow line separating the directions (the left edge on a one-way road).
  const yellow: Vec3 = [0.8, 0.65, 0.1];
  for (let s = 0; s < L; s += 2) {
    pushPathPatch(verts, path, s, Math.min(s + 2, L), -0.075, 0.075, 0.03, yellow);
  }

  // White dashes between same-direction lanes.
  const paint: Vec3 = [0.9, 0.9, 0.9];
  const dashOffsets: number[] = [];
  for (let i = 1; i < f; i++) dashOffsets.push(4 * i);
  for (let j = 1; j < b; j++) dashOffsets.push(-4 * j);
  for (const o of dashOffsets) {
    for (let s = 0; s < L; s += 6) {
      pushPathPatch(verts, path, s, Math.min(s + 2, L), o - 0.075, o + 0.075, 0.03, paint);
    }
  }
  return verts;
}

function networkCarPose(net: Network, transforms: Transform[], car: Car): Pose {
  const from = locate(net, Math.round(car.laneFrom));
  const to = locate(net, car.lane);
  const road = net.roads[to.road];
  const fromLane = road.lanes[from.lane]; // from/to are the same road during a slide
  const toLane = road.lanes[to.lane];
  const span = car.lane - car.laneFrom;
  const t = span === 0 ? 1 : (car.lateral - car.laneFrom) / span;
  const offset = fromLane.offset + (toLane.offset - fromLane.offset) * t;
  const offsetVel = span === 0 ? 0 : ((toLane.offset - fromLane.offset) * car.lateralVel) / span;
  const dir = toLane.direction;
  const p = applyTransform(road.point(car.s), transforms[to.road]);
  // Nose along the true velocity: travel direction plus the lateral slide (negated:
  // local +z points along +offset), flipped for backward lanes.
  const yaw = Math.atan2(-offsetVel * dir, Math.max(car.v, 1));
  return {
    x: p.x + p.rx * offset,
    y: 0.02,
    z: p.z + p.rz * offset,
    angle: Math.atan2(-p.hz, p.hx) + (dir < 0 ? Math.PI : 0) + yaw,
    pitch: 0,
  };
}

function roadScene(): SceneDef {
  return {
    get c() {
      return requireNet().roads.reduce((sum, road) => sum + road.length, 0);
    },
    obstacles: [],
    lamps: [],
    buildStatic: (palette) => {
      const net = requireNet();
      const groundVerts: number[] = [];
      const G = 300;
      pushQuad(groundVerts, [[-G, 0, -G], [-G, 0, G], [G, 0, G], [G, 0, -G]], palette.ground);
      return net.roads
        .flatMap((road, r) => buildRoadStatic(road, palette, scene3State.transforms[r]))
        .concat(groundVerts);
    },
    carPose: (car) => networkCarPose(requireNet(), scene3State.transforms, car),
    step: (cars, params, carLength, dt) => stepNetwork(requireNet(), cars, params, carLength, dt),
    leaderGap: (cars, i, carLength) => networkGapAhead(requireNet(), cars, i, carLength),
    findSpawnSlot: (cars, _carParams, params, carLength) =>
      networkSpawnSlot(requireNet(), cars, params, carLength),
  };
}

export const SCENES: SceneDef[] = [ringScene(), squareScene(), roadScene()];
