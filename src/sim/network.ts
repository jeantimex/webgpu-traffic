/**
 * Network: connects roads at their ends. A link joins road A's end to road B's
 * start; lane connections are derived automatically — A's forward lanes feed B's
 * forward lanes, B's backward lanes feed A's backward lanes, clamped by index when
 * lane counts differ.
 *
 * Sim model: a car's constraint is its lane leader, plus a "downstream" constraint
 * at the road end — either the stop sign (unconnected end) or a virtual leader:
 * the tail car on the connected road. Crossing the seam remaps the car to the next
 * road's lane, preserving speed.
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
import type { Road } from './road';

/** A lane exit connected to another road's lane entrance. */
export interface LaneConnection {
  toRoad: number;
  toLane: number; // local lane index on the target road
  entranceS: number; // arc position where traffic enters the target road
}

export interface Network {
  roads: Road[];
  /** Prefix sums of lane counts: global lane index = laneOffsets[road] + localLane. */
  laneOffsets: number[];
  numLanes: number;
  /** exit[road][lane] = the connection at that lane's travel end, or null for a stop. */
  exit: (LaneConnection | null)[][];
}

export function globalLane(net: Network, road: number, lane: number): number {
  return net.laneOffsets[road] + lane;
}

/** Resolves a global lane index to (road, localLane). Linear scan: networks are tiny. */
export function locate(net: Network, global: number): { road: number; lane: number } {
  for (let r = net.roads.length - 1; r >= 0; r--) {
    if (global >= net.laneOffsets[r]) return { road: r, lane: global - net.laneOffsets[r] };
  }
  return { road: 0, lane: 0 };
}

/**
 * Builds a network from roads and end→start links [fromRoad, toRoad].
 * Lane mapping is automatic per direction of travel.
 */
export function buildNetwork(roads: Road[], links: [number, number][]): Network {
  const laneOffsets: number[] = [];
  let total = 0;
  for (const road of roads) {
    laneOffsets.push(total);
    total += road.lanes.length;
  }
  const exit: (LaneConnection | null)[][] = roads.map((r) => r.lanes.map(() => null));

  for (const [a, b] of links) {
    const roadA = roads[a];
    const fwdA = roadA.lanes.flatMap((lane, i) => (lane.direction > 0 ? [i] : []));
    const fwdB = roads[b].lanes.flatMap((lane, i) => (lane.direction > 0 ? [i] : []));
    const backA = roadA.lanes.flatMap((lane, i) => (lane.direction < 0 ? [i] : []));
    const backB = roads[b].lanes.flatMap((lane, i) => (lane.direction < 0 ? [i] : []));
    // A's forward lanes exit at A's end and enter B's forward lanes at B's start.
    fwdA.forEach((lane, i) => {
      exit[a][lane] = { toRoad: b, toLane: fwdB[Math.min(i, fwdB.length - 1)], entranceS: 0 };
    });
    // B's backward lanes exit at B's start and enter A's backward lanes at A's end.
    backB.forEach((lane, j) => {
      exit[b][lane] = { toRoad: a, toLane: backA[Math.min(j, backA.length - 1)], entranceS: roadA.length };
    });
  }
  return { roads, laneOffsets, numLanes: total, exit };
}

/** Nearest car in `laneIndex` (local, same road) ahead of (or behind) car `me`, in travel direction. */
function nearestInLane(
  net: Network,
  cars: Car[],
  me: number,
  road: number,
  lane: number,
  direction: number,
  ahead: boolean,
): { index: number; gap: number; v: number } | null {
  const g = globalLane(net, road, lane);
  let best: { index: number; gap: number; v: number } | null = null;
  for (let j = 0; j < cars.length; j++) {
    if (j === me || cars[j].lane !== g) continue;
    const d = (cars[j].s - cars[me].s) * direction;
    const gap = ahead ? d : -d;
    if (gap > 0 && (best === null || gap < best.gap)) best = { index: j, gap, v: cars[j].v };
  }
  return best;
}

interface Constraint {
  gap: number;
  vLead: number;
}

/**
 * What a car must brake for at its lane's end: the stop sign (unconnected), or the
 * tail car on the connected road — walking the chain while lanes are empty, so a
 * stop at the far end of the network propagates upstream.
 */
function downstream(
  net: Network,
  cars: Car[],
  me: number,
  road: number,
  lane: number,
  distToExit: number,
  carLength: number,
): Constraint {
  let gap = distToExit;
  let r = road;
  let l = lane;
  for (let hop = 0; hop < 8; hop++) {
    const conn = net.exit[r][l];
    if (!conn) return { gap: gap - carLength / 2, vLead: 0 }; // stop sign at this end
    const tRoad = net.roads[conn.toRoad];
    const tDir = tRoad.lanes[conn.toLane].direction;
    const tGlobal = globalLane(net, conn.toRoad, conn.toLane);
    let best: { dist: number; v: number } | null = null;
    for (let j = 0; j < cars.length; j++) {
      if (j === me || cars[j].lane !== tGlobal) continue;
      const dist = tDir > 0 ? cars[j].s : tRoad.length - cars[j].s;
      if (best === null || dist < best.dist) best = { dist, v: cars[j].v };
    }
    if (best) return { gap: gap + best.dist - carLength, vLead: best.v };
    gap += tRoad.length;
    r = conn.toRoad;
    l = conn.toLane;
  }
  return { gap: 1e6, vLead: 0 }; // closed loop with nobody downstream: free road
}

function accelToward(
  car: Car,
  leader: { gap: number; v: number } | null,
  carLength: number,
  p: IdmParams,
): number {
  return leader
    ? idmAcceleration(car.v, leader.gap - carLength, car.v - leader.v, p)
    : idmAcceleration(car.v, 1e6, 0, p);
}

/** MOBIL-lite lane changes, scoped to adjacent same-direction lanes on the same road. */
function updateNetworkLanes(net: Network, cars: Car[], params: IdmParams[], carLength: number): void {
  for (let i = 0; i < cars.length; i++) {
    const car = cars[i];
    if (car.cooldown > 0) continue;
    const { road: ri, lane: li } = locate(net, car.lane);
    const road = net.roads[ri];
    const dir = road.lanes[li].direction;
    const accelHere = accelToward(car, nearestInLane(net, cars, i, ri, li, dir, true), carLength, params[i]);
    let bestTarget = -1;
    let bestAccel = accelHere + DELTA_A;
    road.lanes.forEach((other, target) => {
      if (other.direction !== dir || Math.abs(other.offset - road.lanes[li].offset) !== 4) return;
      const accelThere = accelToward(car, nearestInLane(net, cars, i, ri, target, dir, true), carLength, params[i]);
      if (accelThere <= bestAccel || accelThere < -B_SAFE) return;
      const follower = nearestInLane(net, cars, i, ri, target, dir, false);
      if (follower) {
        const followerAccel = idmAcceleration(
          follower.v,
          follower.gap - carLength,
          follower.v - car.v,
          params[follower.index],
        );
        if (followerAccel < -B_SAFE) return;
      }
      bestTarget = target;
      bestAccel = accelThere;
    });
    if (bestTarget >= 0) {
      car.lane = globalLane(net, ri, bestTarget);
      car.cooldown = LANE_CHANGE_COOLDOWN;
      car.laneFrom = car.lateral;
      car.laneProgress = 0;
    }
  }
}

/** Advances every car in the network by one fixed step, crossing seams where connected. */
export function stepNetwork(net: Network, cars: Car[], params: IdmParams[], carLength: number, dt: number): void {
  const accels = cars.map((car, i) => {
    const { road: ri, lane: li } = locate(net, car.lane);
    const road = net.roads[ri];
    const dir = road.lanes[li].direction;
    let accel = idmAcceleration(car.v, 1e6, 0, params[i]); // free road
    const leader = nearestInLane(net, cars, i, ri, li, dir, true);
    if (leader) accel = Math.min(accel, accelToward(car, leader, carLength, params[i]));
    const distToExit = dir > 0 ? road.length - car.s : car.s;
    const down = downstream(net, cars, i, ri, li, distToExit, carLength);
    return Math.min(accel, idmAcceleration(car.v, down.gap, car.v - down.vLead, params[i]));
  });

  updateNetworkLanes(net, cars, params, carLength);

  for (let i = 0; i < cars.length; i++) {
    const car = cars[i];
    const { road: ri, lane: li } = locate(net, car.lane);
    const road = net.roads[ri];
    const dir = road.lanes[li].direction;
    car.a = accels[i];
    car.v = Math.max(0, car.v + car.a * dt);
    car.s += dir * car.v * dt;

    // Seam crossing: remap to the connected road's lane, keeping speed and overshoot.
    const exitS = dir > 0 ? road.length : 0;
    const overshoot = (car.s - exitS) * dir;
    if (overshoot > 0) {
      const conn = net.exit[ri][li];
      if (conn) {
        const tDir = net.roads[conn.toRoad].lanes[conn.toLane].direction;
        car.s = conn.entranceS + overshoot * tDir;
        car.lane = globalLane(net, conn.toRoad, conn.toLane);
        car.lateral = car.lane;
        car.laneFrom = car.lane;
        car.laneProgress = 1;
        car.lateralVel = 0;
      } else {
        car.s = exitS; // stop sign clamp
      }
    }
    advanceLateral(car, dt);
  }
}

/** Bumper gap to the nearest car ahead in the same lane, or null when alone. */
export function networkGapAhead(net: Network, cars: Car[], i: number, carLength: number): number | null {
  const { road: ri, lane: li } = locate(net, cars[i].lane);
  const dir = net.roads[ri].lanes[li].direction;
  const leader = nearestInLane(net, cars, i, ri, li, dir, true);
  return leader ? leader.gap - carLength : null;
}

/**
 * Best lane entrance to spawn at, or null when none is safe. Only unfed entrances
 * (no connection flowing into them) are candidates. Safe = the new car would not
 * brake harder than B_SAFE behind its leader.
 */
export function networkSpawnSlot(
  net: Network,
  cars: Car[],
  params: IdmParams,
  carLength: number,
): { s: number; lane: number } | null {
  // Lanes already fed by a connection are not spawn entrances.
  const fed = new Set<number>();
  net.roads.forEach((_, r) => {
    net.exit[r].forEach((conn) => {
      if (conn) fed.add(globalLane(net, conn.toRoad, conn.toLane));
    });
  });

  let best: { s: number; lane: number } | null = null;
  let bestGap = -Infinity;
  net.roads.forEach((road, ri) => {
    road.lanes.forEach((lane, li) => {
      const g = globalLane(net, ri, li);
      if (fed.has(g)) return;
      const s = lane.direction > 0 ? 0 : road.length;
      let leaderGap = Infinity;
      let leaderV = params.v0;
      for (const car of cars) {
        if (car.lane !== g) continue;
        const gap = (car.s - s) * lane.direction;
        if (gap > 0 && gap < leaderGap) {
          leaderGap = gap;
          leaderV = car.v;
        }
      }
      if (idmAcceleration(params.v0, leaderGap - carLength, params.v0 - leaderV, params) < -B_SAFE)
        return;
      if (leaderGap > bestGap) {
        bestGap = leaderGap;
        best = { s, lane: g };
      }
    });
  });
  return best;
}
