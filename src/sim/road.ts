/**
 * Road: the foundation building block of the traffic system. A road is an open
 * (non-looping) segment with a centerline path of any shape — straight, arc, or
 * S-curve — carrying N forward lanes and M backward lanes (M = 0 for one-way).
 * The path's offset 0 is the yellow line separating the directions.
 *
 * The IDM sim runs 1D along the road; the open ends act as permanent stop signs
 * until roads can be connected into a network.
 */
import {
  advanceLateral,
  B_SAFE,
  DELTA_A,
  idmAcceleration,
  LANE_CHANGE_COOLDOWN,
  type Car,
  type IdmParams,
} from './idm';

export interface PathPoint {
  x: number;
  z: number;
  hx: number; // heading (unit vector)
  hz: number;
  rx: number; // right normal of the heading (unit vector)
  rz: number;
}

export type RoadShape = 'straight' | 'arc' | 'scurve';

export interface RoadConfig {
  shape: RoadShape;
  length: number; // straight only (m)
  radius: number; // arc & scurve (m)
  angle: number; // arc & scurve (degrees; per arc for scurve)
  lanesForward: number; // 1..3
  lanesBackward: number; // 0..3 (0 = one-way)
}

export interface Lane {
  direction: 1 | -1;
  offset: number; // signed lateral offset from the path, + = right of forward heading
}

const LANE_WIDTH = 4;

function straightPath(length: number): (s: number) => PathPoint {
  return (s) => ({ x: s - length / 2, z: 0, hx: 1, hz: 0, rx: 0, rz: 1 });
}

/** Circular arc of `radius` sweeping `angleDeg` counterclockwise, centered in view. */
function arcPath(radius: number, angleDeg: number): (s: number) => PathPoint {
  const theta = (angleDeg * Math.PI) / 180;
  const zOffset = (radius * (1 - Math.cos(theta / 2))) / 2;
  return (s) => {
    const phi = s / radius - theta / 2;
    return {
      x: radius * Math.sin(phi),
      z: radius * (1 - Math.cos(phi)) - zOffset,
      hx: Math.cos(phi),
      hz: Math.sin(phi),
      rx: -Math.sin(phi),
      rz: Math.cos(phi),
    };
  };
}

/** S-curve: an arc of `angleDeg` followed by an equal opposite arc, tangent-continuous. */
function scurvePath(radius: number, angleDeg: number): (s: number) => PathPoint {
  const theta = (angleDeg * Math.PI) / 180;
  const leg = radius * theta; // arc length of one side
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  // End of the first arc.
  const p1x = radius * sinT;
  const p1z = radius * (1 - cosT);
  // Center the whole S in view.
  const cx = p1x;
  const cz = p1z;
  return (s) => {
    if (s <= leg) {
      const phi = s / radius;
      return {
        x: radius * Math.sin(phi) - cx,
        z: radius * (1 - Math.cos(phi)) - cz,
        hx: Math.cos(phi),
        hz: Math.sin(phi),
        rx: -Math.sin(phi),
        rz: Math.cos(phi),
      };
    }
    const psi = (s - leg) / radius;
    // Second arc turns right back to the original heading (derived by rotating the
    // first arc's local frame by theta at P1).
    const lx = radius * Math.sin(psi);
    const lz = -radius * (1 - Math.cos(psi));
    return {
      x: p1x + lx * cosT - lz * sinT - cx,
      z: p1z + lx * sinT + lz * cosT - cz,
      hx: Math.cos(theta - psi),
      hz: Math.sin(theta - psi),
      rx: -Math.sin(theta - psi),
      rz: Math.cos(theta - psi),
    };
  };
}

export class Road {
  readonly lanes: Lane[] = [];
  readonly length: number;
  private readonly path: (s: number) => PathPoint;

  constructor(readonly config: RoadConfig) {
    switch (config.shape) {
      case 'straight':
        this.length = config.length;
        this.path = straightPath(this.length);
        break;
      case 'arc':
        this.length = (config.radius * config.angle * Math.PI) / 180;
        this.path = arcPath(config.radius, config.angle);
        break;
      case 'scurve':
        this.length = (2 * config.radius * config.angle * Math.PI) / 180;
        this.path = scurvePath(config.radius, config.angle);
        break;
    }
    for (let i = 0; i < config.lanesForward; i++) {
      this.lanes.push({ direction: 1, offset: LANE_WIDTH / 2 + i * LANE_WIDTH });
    }
    for (let j = 0; j < config.lanesBackward; j++) {
      this.lanes.push({ direction: -1, offset: -(LANE_WIDTH / 2 + j * LANE_WIDTH) });
    }
  }

  /** Centerline point at arc position s (clamped to the road). */
  point(s: number): PathPoint {
    return this.path(Math.min(Math.max(s, 0), this.length));
  }
}

function clamp(x: number, min: number, max: number): number {
  return Math.min(Math.max(x, min), max);
}

/** Nearest car in `laneIndex` ahead of (or behind) car `me`, in travel direction. */
function nearestOnRoad(
  cars: Car[],
  me: number,
  laneIndex: number,
  direction: number,
  ahead: boolean,
): { index: number; gap: number; v: number } | null {
  let best: { index: number; gap: number; v: number } | null = null;
  for (let j = 0; j < cars.length; j++) {
    if (j === me || cars[j].lane !== laneIndex) continue;
    const d = (cars[j].s - cars[me].s) * direction;
    const gap = ahead ? d : -d;
    if (gap > 0 && (best === null || gap < best.gap)) best = { index: j, gap, v: cars[j].v };
  }
  return best;
}

/** Lanes a car on `laneIndex` may change to: adjacent lanes going the same direction. */
function adjacentLanes(road: Road, laneIndex: number): number[] {
  const lane = road.lanes[laneIndex];
  const targets: number[] = [];
  road.lanes.forEach((other, i) => {
    if (i !== laneIndex && other.direction === lane.direction && Math.abs(other.offset - lane.offset) === LANE_WIDTH) {
      targets.push(i);
    }
  });
  return targets;
}

function accelToward(car: Car, leader: { gap: number; v: number } | null, carLength: number, p: IdmParams): number {
  return leader
    ? idmAcceleration(car.v, leader.gap - carLength, car.v - leader.v, p)
    : idmAcceleration(car.v, 1e6, 0, p);
}

/** MOBIL-lite lane changes on the road: same rules as the ring, adjacent same-direction lanes only. */
function updateRoadLanes(road: Road, cars: Car[], params: IdmParams[], carLength: number): void {
  for (let i = 0; i < cars.length; i++) {
    const car = cars[i];
    if (car.cooldown > 0) continue;
    const dir = road.lanes[car.lane].direction;
    const accelHere = accelToward(car, nearestOnRoad(cars, i, car.lane, dir, true), carLength, params[i]);
    let bestTarget = -1;
    let bestAccel = accelHere + DELTA_A;
    for (const target of adjacentLanes(road, car.lane)) {
      const accelThere = accelToward(car, nearestOnRoad(cars, i, target, dir, true), carLength, params[i]);
      if (accelThere <= bestAccel) continue;
      // Safety first: no hard braking for me or for the car behind me in the target lane.
      if (accelThere < -B_SAFE) continue;
      const follower = nearestOnRoad(cars, i, target, dir, false);
      if (follower) {
        const followerAccel = idmAcceleration(
          follower.v,
          follower.gap - carLength,
          follower.v - car.v,
          params[follower.index],
        );
        if (followerAccel < -B_SAFE) continue;
      }
      bestTarget = target;
      bestAccel = accelThere;
    }
    if (bestTarget >= 0) {
      car.lane = bestTarget;
      car.cooldown = LANE_CHANGE_COOLDOWN;
      car.laneFrom = car.lateral;
      car.laneProgress = 0;
    }
  }
}

/**
 * Advances every car on the road by one fixed step. Cars follow their lane leader;
 * the open ends act as permanent stop signs (a virtual standing car at the stop line).
 */
export function stepRoad(road: Road, cars: Car[], params: IdmParams[], carLength: number, dt: number): void {
  const L = road.length;
  const accels = cars.map((car, i) => {
    const dir = road.lanes[car.lane].direction;
    let accel = idmAcceleration(car.v, 1e6, 0, params[i]); // free road
    const leader = nearestOnRoad(cars, i, car.lane, dir, true);
    if (leader) accel = Math.min(accel, accelToward(car, leader, carLength, params[i]));
    // End of the road is a full stop: the front bumper stops at the stop line.
    const dEnd = dir > 0 ? L - car.s : car.s;
    return Math.min(accel, idmAcceleration(car.v, dEnd - carLength / 2, car.v, params[i]));
  });

  updateRoadLanes(road, cars, params, carLength);

  for (let i = 0; i < cars.length; i++) {
    const car = cars[i];
    car.a = accels[i];
    car.v = Math.max(0, car.v + car.a * dt);
    car.s = clamp(car.s + road.lanes[car.lane].direction * car.v * dt, 0, L);
    advanceLateral(car, dt);
  }
}

/** Bumper gap to the nearest car ahead in the same lane, or null when alone. */
export function roadGapAhead(road: Road, cars: Car[], i: number, carLength: number): number | null {
  const dir = road.lanes[cars[i].lane].direction;
  const leader = nearestOnRoad(cars, i, cars[i].lane, dir, true);
  return leader ? leader.gap - carLength : null;
}

/**
 * Best lane entrance to spawn at (s = 0 for forward lanes, s = L for backward lanes),
 * or null when no entrance is safe. Safe = the new car would not brake harder than
 * B_SAFE behind its leader. A car mid-lane-change counts as occupying both lanes.
 */
export function roadSpawnSlot(
  road: Road,
  cars: Car[],
  params: IdmParams,
  carLength: number,
): { s: number; lane: number } | null {
  let best: { s: number; lane: number } | null = null;
  let bestGap = -Infinity;
  road.lanes.forEach((lane, laneIndex) => {
    const s = lane.direction > 0 ? 0 : road.length;
    let leaderGap = Infinity;
    let leaderV = params.v0;
    for (const car of cars) {
      const inLane =
        car.lane === laneIndex ||
        (car.laneProgress < 1 && adjacentLanes(road, car.lane).includes(laneIndex));
      if (!inLane) continue;
      const gap = (car.s - s) * lane.direction;
      if (gap > 0 && gap < leaderGap) {
        leaderGap = gap;
        leaderV = car.v;
      }
    }
    if (idmAcceleration(params.v0, leaderGap - carLength, params.v0 - leaderV, params) < -B_SAFE) return;
    if (leaderGap > bestGap) {
      bestGap = leaderGap;
      best = { s, lane: laneIndex };
    }
  });
  return best;
}
